import logging
import os
from typing import Dict

from pdb import set_trace as st
import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from losses import (
    ReconstructionLoss,
    PairwiseSimilarityLoss,
    TopKPairwiseSimilarityLoss,
    RankLoss,
    TemporalLoss
)
from utils import get_params_info
from pathlib import Path
import numpy as np
from torch.nn import ConstantPad2d


from tevatron.retriever.arguments import ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.driver.encode import DenseModel, EncoderOutput
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, cast_mixed_precision_params
from transformers.activations import ACT2FN

temporal_max_length = 16

logger = logging.getLogger(__name__)


def _convert_to_tensor(a: list | np.ndarray | Tensor) -> Tensor:
    """
    Converts the input `a` to a PyTorch tensor if it is not already a tensor.
    Handles lists of sparse tensors by stacking them.

    Args:
        a (Union[list, np.ndarray, Tensor]): The input array or tensor.

    Returns:
        Tensor: The converted tensor.
    """
    if isinstance(a, list):
        # Check if list contains sparse tensors
        if all(isinstance(x, Tensor) and x.is_sparse for x in a):
            # Stack sparse tensors while preserving sparsity
            return torch.stack([x.coalesce().to(dtype=torch.float32) for x in a])
        else:
            a = torch.tensor(a)
    elif not isinstance(a, Tensor):
        a = torch.tensor(a)
    if a.is_sparse:
        return a.to(dtype=torch.float32)
    return a


def pairwise_angle_sim(x: Tensor, y: Tensor) -> Tensor:
    """
    Computes the absolute normalized angle distance. See :class:`~sentence_transformers.losses.AnglELoss`
    or https://arxiv.org/abs/2309.12871v1 for more information.

    Args:
        x (Tensor): The first tensor.
        y (Tensor): The second tensor.

    Returns:
        Tensor: Vector with res[i] = angle_sim(a[i], b[i])
    """
    if x.is_sparse:
        # logger.warning_once("Pairwise angle similarity does not support sparse tensors. Converting to dense.")
        x = x.coalesce().to_dense()
        y = y.coalesce().to_dense()

    x = _convert_to_tensor(x)
    y = _convert_to_tensor(y)

    # modified from https://github.com/SeanLee97/AnglE/blob/main/angle_emb/angle.py
    # chunk both tensors to obtain complex components
    a, b = torch.chunk(x, 2, dim=1)
    c, d = torch.chunk(y, 2, dim=1)

    z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z

    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True) ** 0.5
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True) ** 0.5
    re /= dz / dw
    im /= dz / dw

    norm_angle = torch.sum(torch.concat((re, im), dim=1), dim=1)
    return torch.abs(norm_angle)


class Adaptor(nn.Module):

    def __init__(self, hidden_size):
        super(Adaptor, self).__init__()
        self.down_project = nn.Linear(hidden_size, 256)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(256, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, inputs):
        down_projected = self.activation(self.down_project(inputs))
        up_projected = self.up_project(down_projected)
        output = inputs + up_projected
        output = self.layernorm(output)
        return output


class UnsupervisedMAdaptor(DenseModel):
    def __init__(self, *args, **kwargs):
        super(UnsupervisedMAdaptor, self).__init__(*args, **kwargs)

        self.loss_pair_fn = PairwiseSimilarityLoss()
        self.loss_rec_fn = ReconstructionLoss()
        self.loss_topk_fn = TopKPairwiseSimilarityLoss(k=10)

        self.matryoshka_dim_list = sorted([384, 512, 768])
        self.truncated_dim = 768 # self.matryoshka_dim_list[1] # Default to the smallest dimension
        self.adaptor = Adaptor(hidden_size=self.matryoshka_dim_list[-1])

    def compute_loss(self, original_embeddings, adapted_embeddings):
        loss_pair, loss_pair_partial = self.loss_pair_fn(
            original_embeddings,
            adapted_embeddings,
            m_list=self.matryoshka_dim_list,
        )
        loss_rec = self.loss_rec_fn(original_embeddings, adapted_embeddings)
        loss_topk, loss_topk_partial = self.loss_topk_fn(
            original_embeddings,
            adapted_embeddings,
            m_list=self.matryoshka_dim_list,
        )

        losses ={
            "loss": loss_pair + loss_rec + loss_topk,
            "loss_pair": loss_pair.detach().clone(),
            "loss_topk": loss_topk.detach().clone(),
            "loss_rec": loss_rec.detach().clone(),
        }
        losses.update(loss_pair_partial)
        losses.update(loss_topk_partial)
        return losses

    def encode_query(self, qry):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        if self.training:
            return self._pooling(query_hidden_states, qry["attention_mask"])
        else:
            return self.adaptor(
                self._pooling(query_hidden_states, qry["attention_mask"])
            )[:, : self.truncated_dim]

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        if self.training:
            docid, chunkid, doc = passage
            original_embeddings = self.encode_passage(doc)

            adapted_embeddings = self.adaptor(original_embeddings)

            loss = self.compute_loss(
                original_embeddings=original_embeddings,
                adapted_embeddings=adapted_embeddings,
            )
            return loss
        else:
            # Copy from EncoderModel's forward
            q_reps = self.encode_query(query) if query else None
            p_reps = self.encode_passage(passage) if passage else None

            # for inference
            if q_reps is None or p_reps is None:

                # if q_reps is not None:
                #     print("Encoded queries shape", q_reps.shape)
                # if p_reps is not None:
                #     print("Encoded passages shape", p_reps.shape)

                return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

            # for eval
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

            return EncoderOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps,
            )

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        train_args: TrainingArguments,
        **hf_kwargs,
    ):
        if not os.path.exists(model_args.model_name_or_path):
            model = super().build(
                model_args, train_args, **hf_kwargs
            )
        else:
            model = cls.load(
                model_args.model_name_or_path,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                lora_name_or_path=model_args.lora_name_or_path,
                cache_dir=model_args.cache_dir,
                # torch_dtype=torch_dtype,
                attn_implementation=model_args.attn_implementation,
            )
        for name, param in model.encoder.named_parameters():
            param.requires_grad = False

        get_params_info(model)

        return model

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        pooling: str = "cls",
        normalize: bool = False,
        lora_name_or_path: str = None,
        **hf_kwargs,
    ):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(
            model_name_or_path, **hf_kwargs
        )
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0

        model = cls(encoder=base_model, pooling=pooling, normalize=normalize)

        if os.path.exists(os.path.join(model_name_or_path, "adaptor.safetensors")):
            adaptor_state_dict = safetensors.torch.load_file(
                os.path.join(model_name_or_path, "adaptor.safetensors"), device="cpu"
            )
            model.adaptor.load_state_dict(adaptor_state_dict)
            print("Loaded adaptor")
        else:
            print("There is no adaptor state dict")

        return model


class UnsupervisedTemporalMAdaptor(UnsupervisedMAdaptor):
    def __init__(self, *args, **kwargs):
        super(UnsupervisedTemporalMAdaptor, self).__init__(*args, **kwargs)

        self.matryoshka_dim_list = sorted([512, 768])
        # self.truncated_dim = self.matryoshka_dim_list[0]  # Default to the smallest dimension
        self.adaptor = Adaptor(hidden_size=self.matryoshka_dim_list[-1])

        self.loss_temporal_fn = TemporalLoss()

    def compute_loss(self, original_embeddings, adapted_embeddings, temporal_embeddings, adapted_temporal_embeddings):
        sim_target = F.cosine_similarity(
            original_embeddings.unsqueeze(1), 
            original_embeddings.unsqueeze(0), 
            dim=-1
        )
        loss_pair, loss_pair_partial = self.loss_pair_fn(
            original_embeddings,
            adapted_embeddings,
            m_list=self.matryoshka_dim_list,
            sim_target=sim_target,
        )
        loss_rec = self.loss_rec_fn(original_embeddings, adapted_embeddings)
        loss_topk, loss_topk_partial = self.loss_topk_fn(
            original_embeddings,
            adapted_embeddings,
            m_list=self.matryoshka_dim_list,
            sim_target=sim_target,
        )
        loss_temporal, loss_temporal_partial = self.loss_temporal_fn(
            temporal_embeddings,
            adapted_temporal_embeddings,
            adapted_embeddings,
            temp_dim=256, # Assume that we are going to use the second largest matryoshka-dim embedding,
            base_dim=768,
        )

        losses = {
            "loss": loss_pair + loss_rec + loss_topk + loss_temporal,
            "loss_pair": loss_pair.detach().clone(),
            "loss_topk": loss_topk.detach().clone(),
            "loss_rec": loss_rec.detach().clone(),
            "loss_temporal": loss_temporal.detach().clone()
        }
        losses.update(loss_pair_partial)
        losses.update(loss_topk_partial)
        losses.update(loss_temporal_partial)

        return losses

    def encode_query(self, qry):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state

        if self.training:
            return self._pooling(query_hidden_states, qry["attention_mask"])
        else:
            return self.adaptor(
                self._pooling(query_hidden_states, qry["attention_mask"])
            )

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        if self.training:
            docid, chunkid, doc, temporal = passage

            original_embeddings = self.encode_passage(doc)

            adapted_embeddings = self.adaptor(original_embeddings)

            temporal_embeddings = self.encode_passage(temporal)

            adapted_temporal_embeddings = self.adaptor(temporal_embeddings)

            F.cosine_similarity(adapted_embeddings, adapted_temporal_embeddings, dim=-1)

            loss = self.compute_loss(
                original_embeddings=original_embeddings,
                adapted_embeddings=adapted_embeddings,
                temporal_embeddings=temporal_embeddings,
                adapted_temporal_embeddings=adapted_temporal_embeddings,
            )
            return loss
        else:
            # Copy from EncoderModel's forward
            q_reps = self.encode_query(query) if query else None
            p_reps = self.encode_passage(passage) if passage else None

            # for inference
            if q_reps is None or p_reps is None:

                if q_reps is not None:
                    print("Encoded queries shape", q_reps.shape)
                if p_reps is not None:
                    print("Encoded passages shape", p_reps.shape)

                return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

            # for eval
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

            return EncoderOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps,
            )


class SupervisedMAdaptor(UnsupervisedMAdaptor):
    def __init__(self, *args, **kwargs):
        super(SupervisedMAdaptor, self).__init__(*args, **kwargs)

        self.loss_rank_fn = RankLoss()
        
    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        if self.training:

            q_reps = self.encode_query(query)
            adapted_q_reps = self.adaptor(q_reps)

            """
            passage: the positive-negative (contrastive) temporal questions
            docid: corresponding question's docid
            chunkid: corresponding question's chunkid
            corpus_doc: corresponding question's document
            """
            passage, docid, chunkid, corpus_doc = passage

            p_reps = self.encode_passage(passage)
            adapted_p_reps = self.adaptor(p_reps)

            corpus_doc_reps = self.encode_passage(corpus_doc)
            adapted_corpus_doc_reps = self.adaptor(corpus_doc_reps)

            loss = self.compute_loss(
                q_reps,
                adapted_q_reps,
                p_reps,
                adapted_p_reps,
                corpus_doc_reps,
                adapted_corpus_doc_reps,
            )
            return loss
        else:
            # Copy from EncoderModel's forward
            q_reps = self.encode_query(query) if query else None
            p_reps = self.encode_passage(passage) if passage else None

            # for inference
            if q_reps is None or p_reps is None:

                # if q_reps is not None:
                #     print("Encoded queries shape", q_reps.shape)
                # if p_reps is not None:
                #     print("Encoded passages shape", p_reps.shape)

                return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

            # for eval
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

            return EncoderOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps,
            )

    def compute_loss(
        self,
        q_reps,
        adapted_q_reps,
        p_reps,
        adapted_p_reps,
        corpus_doc_reps,
        adapted_corpus_doc_reps,
    ):
        # losses = super().compute_loss(p_reps, adapted_p_reps)
        losses = super().compute_loss(corpus_doc_reps, adapted_corpus_doc_reps)

        loss_rank = self.loss_rank_fn(
            adapted_q_reps, 
            adapted_p_reps,
            m_list=self.matryoshka_dim_list,
        )
        losses["loss"] += loss_rank
        losses["loss_rank"] = loss_rank.detach().clone()
        return losses


class NaiveMAdaptor(SupervisedMAdaptor):
    def __init__(self, *args, **kwargs):
        super(NaiveMAdaptor, self).__init__(*args, **kwargs)

    def compute_loss(self, q_reps, adapted_q_reps, p_reps, adapted_p_reps):

        losses = {"loss": torch.tensor(0.0, device=q_reps.device)}
        target = None
        for m in self.matryoshka_dim_list:

            scores = self.compute_similarity(
                adapted_q_reps[:, :m], adapted_p_reps[:, :m]
            )
            scores = scores.view(q_reps.size(0), -1)

            if target is None:
                target = torch.arange(
                    scores.size(0), device=scores.device, dtype=torch.long
                )
                target = target * (p_reps.size(0) // q_reps.size(0))

                loss = self.cross_entropy(scores / self.temperature, target)
                losses["loss"] += loss
                losses[f"loss_partial_{m}"] = loss.detach().clone()

        return losses


class NaiveSupervisedMAdaptor(UnsupervisedMAdaptor):
    def __init__(self, *args, **kwargs):
        super(NaiveSupervisedMAdaptor, self).__init__(*args, **kwargs)

    def compute_loss(self, q_reps, adapted_q_reps, p_reps, adapted_p_reps):
        losses = super().compute_loss(p_reps, adapted_p_reps)

        target = None
        for m in self.matryoshka_dim_list:

            scores = self.compute_similarity(
                adapted_q_reps[:, :m], adapted_p_reps[:, :m]
            )
            scores = scores.view(q_reps.size(0), -1)

            if target is None:
                target = torch.arange(
                    scores.size(0), device=scores.device, dtype=torch.long
                )
                target = target * (p_reps.size(0) // q_reps.size(0))

                loss = self.cross_entropy(scores / self.temperature, target)
                losses["loss"] += loss
                losses[f"loss_matryoshka_{m}"] = loss.detach().clone()
        return losses

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        if self.training:

            q_reps = self.encode_query(query)
            adapted_q_reps = self.adaptor(q_reps)

            p_reps = self.encode_passage(passage)
            adapted_p_reps = self.adaptor(p_reps)

            loss = self.compute_loss(
                q_reps,
                adapted_q_reps,
                p_reps,
                adapted_p_reps,
            )
            return loss
        else:
            # Copy from EncoderModel's forward
            q_reps = self.encode_query(query) if query else None
            p_reps = self.encode_passage(passage) if passage else None

            # for inference
            if q_reps is None or p_reps is None:

                # if q_reps is not None:
                #     print("Encoded queries shape", q_reps.shape)
                # if p_reps is not None:
                #     print("Encoded passages shape", p_reps.shape)

                return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

            # for eval
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

            return EncoderOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps,
            )


class TemporalProjector(nn.Module):
    
    def __init__(self, embed_dim=768, temporal_dim=256, vocab_size=30522) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.intermeditate_dim = temporal_dim
        # self.output_dim = vocab_size # Equivalent to bert vocab size
        self.output_dim = 256
        
        self.input_layer = nn.Linear(self.embed_dim, self.intermeditate_dim, bias=False)
        self.act = nn.GELU()
        self.output_layer = nn.Linear(self.intermeditate_dim, self.output_dim, bias=False)
        
        # nn.init.eye_(self.input_layer.weight)
        # nn.init.eye_(self.output_layer.weight)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.act(x)
        x = self.output_layer(x)
        return x


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config, temporal_dim=256):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, temporal_dim)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.
        # layer_norm_eps)
        self.LayerNorm = nn.LayerNorm(temporal_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class TemporalProjectorWithReconstructionLoss(nn.Module):
    """
    Copy from modeling_bert.py's BertLMPredictionHead
    """
    def __init__(self, config, temporal_dim=256):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder = nn.Linear(temporal_dim, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        temporal_hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(temporal_hidden_states)
        return temporal_hidden_states, hidden_states
     

class NaiveTemporal(DenseModel):
    def __init__(self, *args, **kwargs):
        super(NaiveTemporal, self).__init__(*args, **kwargs)

        self.matryoshka_dim_list = sorted([512, 768], reverse=True)
        self.truncated_dim = self.matryoshka_dim_list[1]  # Default to the second largest dimension
        self.temporal = self.training_args.temporal
        self.truncated_normalize = self.training_args.truncated_normalize
        self.filter_false_negatives = self.training_args.filter_false_negatives
        
        self.kl_loss: nn.KLDivLoss = None
        if self.training_args is not None and self.training_args.kl_loss:
            self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    @torch.no_grad()
    def _get_base_model_embeddings(self, query, passage):
        self.base_model.eval()
        q_reps_base = self.base_model(**query, return_dict=True).last_hidden_state
        q_reps_base = self._pooling(q_reps_base, query["attention_mask"])
        
        p_reps_base = self.base_model(**passage, return_dict=True).last_hidden_state
        p_reps_base = self._pooling(p_reps_base, passage["attention_mask"])

        return q_reps_base, p_reps_base

    def compute_loss(
        self,
        q_reps,
        p_reps,
        p_temporal_reps=None,
        q_reps_base=None,
        p_reps_base=None,
        temporal_tokens_input_ids=None,
        temporal_span_list=None,
        truncated_normalize=False,
        filter_false_positives=False,
        temporal=True,
    ):
        total_loss = torch.tensor(0.0, device=q_reps.device)
        partial_losses = {}

        target = None
        row_idx = None
        no_filter_mask = None
        num_neg = p_reps.size(0) // q_reps.size(0)

        for m in self.matryoshka_dim_list:
            q_reps_trunc = q_reps[:, :m]
            p_reps_trunc = p_reps[:, :m]
            
            if truncated_normalize and m < q_reps.size(1):
                q_reps_trunc = F.normalize(q_reps_trunc, dim=-1)
                p_reps_trunc = F.normalize(p_reps_trunc, dim=-1)

            scores_semantic = self.compute_similarity(q_reps_trunc, p_reps_trunc)
            scores_semantic = scores_semantic.view(q_reps.size(0), -1)

            if target is None:
                target, row_idx, no_filter_mask = self.get_index_and_masks(num_neg, scores_semantic)
                
            # (Optional) Filter potential false negatives by filtering samples with simialrty greater than 95% compared to the ground truth
            if filter_false_positives:
                scores_semantic = self.mask_inbatch_negative_percentile(
                    target, row_idx, no_filter_mask, scores_semantic,   percentile=0.95
                )

            loss_semantic = self.cross_entropy(
                scores_semantic / self.temperature, target
            )
            
            # KL loss regularization to preserve the pre-trained embedding
            # Only work for the full-dim embedding
            if self.kl_loss is not None and m == q_reps.size(1) and (q_reps_base is not None and p_reps_base is not None):
                kl_loss = self._calc_kl_loss(
                    q_reps_base, p_reps_base, q_reps_trunc, p_reps_trunc
                )
                total_loss += kl_loss
                partial_losses[f"loss_kl"] = kl_loss.detach().clone() 

            if m != q_reps.size(1) and temporal and p_temporal_reps is not None:
                # Train the right-left part of embedding to explicitly represent temporal
                t = q_reps.size(1) - m
                qt_reps = q_reps[:, -t:]
                pt_reps = p_reps[:, -t:]
                p_temporal_reps = p_temporal_reps[:, -t:] # Doesn't really make sense
                
                if truncated_normalize:
                    qt_reps = F.normalize(qt_reps, dim=-1)
                    pt_reps = F.normalize(pt_reps, dim=-1)
                    p_temporal_reps = F.normalize(p_temporal_reps, dim=-1)

                # score_qp = self.compute_similarity(
                #     qt_reps, pt_reps
                # )
                score_qt = self.compute_similarity(
                    qt_reps, p_temporal_reps
                )
                # score_pt = self.compute_similarity(
                #     pt_reps, p_temporal_reps
                # )
                
                # arange_idx = torch.arange(score_temporal_pt.size(0), device=scores_semantic.device)
                
                if filter_false_positives:
                    # score_qp = self.mask_inbatch_negative_percentile(
                    #     target, row_idx, no_filter_mask, score_qp,   percentile=0.95
                    # )
                    score_qt = self.mask_inbatch_negative_percentile(
                        target, row_idx, no_filter_mask, score_qt,   percentile=0.95
                    )
                    # score_pt = self.mask_inbatch_negative_percentile(
                    #     arange_idx, arange_idx, no_filter_mask, score_pt,   percentile=0.95
                    # )
                
                # loss_qp = self.cross_entropy(score_qp / self.temperature, target)
                loss_qt = self.cross_entropy(score_qt / self.temperature, target)
                # loss_pt = self.cross_entropy(score_pt / self.temperature, arange_idx)
                loss_temporal = loss_qt # + loss_qp + loss_pt

                # partial_losses[f"loss_temporal_qp_{m}"] = loss_qp.detach().clone()
                partial_losses[f"loss_temporal_qt_{m}"] = loss_qt.detach().clone()
                # partial_losses[f"loss_temporal_pt_{m}"] = loss_pt.detach().clone()
                partial_losses[f"loss_temporal_{m}"] = loss_temporal.detach().clone()

            total_loss += loss_semantic
            partial_losses[f"loss_semantic_{m}"] = loss_semantic.detach().clone()

        return total_loss, partial_losses

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        if self.training:
            q_reps_base, p_reps_base = None, None
            q_reps = self.encode_query(query) if query else None
            
            if self.temporal:
                passage, passage_temporal = passage
                p_temporal_reps = self.encode_passage(passage_temporal)
            else:
                p_temporal_reps = None

            p_reps = self.encode_passage(passage)
            if self.kl_loss is not None:
                q_reps_base, p_reps_base = self._get_base_model_embeddings(query, passage)

            loss, loss_partial = self.compute_loss(
                q_reps,
                p_reps,
                p_temporal_reps,
                q_reps_base=q_reps_base,
                p_reps_base=p_reps_base,
                filter_false_positives=self.filter_false_negatives,
                truncated_normalize=self.truncated_normalize,
                temporal=self.temporal,
            )
            losses = {"loss": loss}
            losses.update(loss_partial)

            return losses
        else:
            # Copy from EncoderModel's forward
            q_reps = self.encode_query(query) if query else None
            p_reps = self.encode_passage(passage) if passage else None

            # for inference
            if q_reps is None or p_reps is None:
                return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

            # for eval
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

            return EncoderOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps,
            )

    def _calc_kl_loss(self, q_reps_base, p_reps_base, q_reps_trunc, p_reps_trunc):
        kl_loss = self.kl_loss(
            F.log_softmax(q_reps_trunc, dim=1), 
            F.softmax(q_reps_base, dim=1),
        ) + self.kl_loss(
            F.log_softmax(p_reps_trunc, dim=1), 
            F.softmax(p_reps_base, dim=1),
        )
        return kl_loss

    def get_index_and_masks(self, num_neg, scores_semantic):
        # The positive passages
        target = torch.arange(
            scores_semantic.size(0),
            device=scores_semantic.device,
            dtype=torch.long,
        )
        target = target * num_neg
        
        # A mask to filter out the positive and annotated negatives
        no_filter_mask = torch.stack(
            [
                torch.arange(
                    x,
                    x + num_neg,
                    device=scores_semantic.device,
                    dtype=torch.long,
                )
                for x in target
            ]
        )
        row_idx = torch.arange(
            scores_semantic.size(0), device=scores_semantic.device
        )
        
        return target,row_idx,no_filter_mask


    def mask_inbatch_negative_percentile(self, target, row_idx, no_filter_mask, scores_semantic, percentile=0.95):
        if scores_semantic.shape[0] == 60:
            st()
        semantic_thresholds = scores_semantic[row_idx, target] * percentile  # shape: [B]

        mask = scores_semantic > semantic_thresholds.unsqueeze(1)  # Mask out those greater than the threshold
        mask[row_idx.unsqueeze(1), no_filter_mask] = False  # type: ignore # don't mask the positive and annotated negatives, only consider the other in-batch negatives

        # Apply the mask
        scores_semantic = scores_semantic.masked_fill(mask, float('-inf'))
        return scores_semantic


class NaiveTemporalProjector(NaiveTemporal):

    def __init__(self, *args, **kwargs):
        super(NaiveTemporalProjector, self).__init__(*args, **kwargs)

        embed_dim = self.matryoshka_dim_list[0]
        temporal_dim = self.matryoshka_dim_list[0] - self.truncated_dim
        self.temporal_projector = TemporalProjector(embed_dim, temporal_dim)

    def compute_loss(
        self,
        q_reps,
        p_reps,
        p_temporal_reps=None,
        q_reps_base=None,
        p_reps_base=None,
        temporal_tokens_input_ids=None,
        temporal_span_list=None,
        truncated_normalize=False,
        filter_false_positives=False,
        temporal=True,
    ):
        total_loss = torch.tensor(0.0, device=q_reps.device)
        partial_losses = {}

        target = None
        row_idx = None
        no_filter_mask = None
        num_neg = p_reps.size(0) // q_reps.size(0)

        for m in self.matryoshka_dim_list:
            q_reps_trunc = q_reps[:, :m]
            p_reps_trunc = p_reps[:, :m]
            
            if truncated_normalize and m < q_reps.size(1):
                q_reps_trunc = F.normalize(q_reps_trunc, dim=-1)
                p_reps_trunc = F.normalize(p_reps_trunc, dim=-1)

            scores_semantic = self.compute_similarity(q_reps_trunc, p_reps_trunc)

            scores_semantic = scores_semantic.view(q_reps.size(0), -1)
            # Create useful masks for calculating different losses
            # Which will be reused later on
            if target is None:
                target, row_idx, no_filter_mask = self.get_index_and_masks(num_neg, scores_semantic)

            # (Optional) Filter potential false negatives by filtering samples with simialrty greater than 95% compared to the ground truth
            if filter_false_positives:
                scores_semantic = self.mask_inbatch_negative_percentile(
                    target, row_idx, no_filter_mask, scores_semantic,   percentile=0.95
                )

            loss_semantic = self.cross_entropy(
                scores_semantic / self.temperature, target
            )
            
            # KL loss regularization to preserve the pre-trained embedding
            # Only work for the full-dim embedding
            if self.kl_loss is not None and m == q_reps.size(1) and (q_reps_base is not None and p_reps_base is not None):
                kl_loss = self._calc_kl_loss(
                    q_reps_base, p_reps_base, q_reps_trunc, p_reps_trunc
                )
                total_loss += kl_loss
                partial_losses[f"loss_kl"] = kl_loss.detach().clone()  

            if p_temporal_reps is not None and m != q_reps.size(1):
                # Train the temporal part in [semantic|temporal] to explicitly represent temporal
                t = q_reps.size(1) - m
                qt_reps = q_reps[:, -t:]
                pt_reps = p_reps[:, -t:]
                p_temporal_reps_projected = self.temporal_projector(p_temporal_reps)
                
                if truncated_normalize:
                    qt_reps = F.normalize(qt_reps, dim=-1)
                    pt_reps = F.normalize(pt_reps, dim=-1)
                    p_temporal_reps_projected = F.normalize(p_temporal_reps_projected, dim=-1)
                
                # score_qp = self.compute_similarity(
                #     qt_reps, pt_reps
                # )
                score_qt = self.compute_similarity(
                    qt_reps, p_temporal_reps_projected
                )
                # score_pt = self.compute_similarity(
                #     pt_reps, p_temporal_reps_projected
                # )
                
                # arange_idx = torch.arange(score_temporal_pt.size(0), device=scores_semantic.device)
                
                if filter_false_positives:
                    # score_qp = self.mask_inbatch_negative_percentile(
                    #     target, row_idx, no_filter_mask, score_qp,   percentile=0.95
                    # )
                    score_qt = self.mask_inbatch_negative_percentile(
                        target, row_idx, no_filter_mask, score_qt,   percentile=0.95
                    )
                    # score_pt = self.mask_inbatch_negative_percentile(
                    #     arange_idx, arange_idx, arange_idx, score_pt,   percentile=0.95
                    # )
                
                # loss_qp = self.cross_entropy(score_qp / self.temperature, target)
                loss_qt = self.cross_entropy(score_qt / self.temperature, target)
                # loss_pt = self.cross_entropy(score_pt / self.temperature, arange_idx)
                loss_temporal = loss_qt # + loss_qp + loss_pt
                total_loss += loss_temporal

                # partial_losses[f"loss_temporal_qp_{m}"] = loss_qp.detach().clone()
                partial_losses[f"loss_temporal_qt_{m}"] = loss_qt.detach().clone()
                # partial_losses[f"loss_temporal_pt_{m}"] = loss_pt.detach().clone()
                partial_losses[f"loss_temporal_{m}"] = loss_temporal.detach().clone()

            total_loss += loss_semantic
            partial_losses[f"loss_semantic_{m}"] = loss_semantic.detach().clone()

        return total_loss, partial_losses

    def encode_passage(self, psg, temporal_span_list=None):
        # encode passage is the same as encode query
        if self.encoder.name_or_path != "jinaai/jina-embeddings-v3":

            if self.training and temporal_span_list is not None:
                query_hidden_states = self.encoder(**psg, return_dict=True)
                query_hidden_states = query_hidden_states.last_hidden_state
                pooled_hidden_states = self._pooling(
                    query_hidden_states, psg["attention_mask"]
                )

                # Processing temporal spans
                # Create a pre-average embedding list and an average embedding list
                temporal_hidden_states = []

                for temporal_span_sublist, hidden_state in zip(
                    temporal_span_list, query_hidden_states
                ):
                    temp = []

                    for start_token_index, end_token_index in temporal_span_sublist:
                        temp.append(
                            hidden_state[start_token_index : end_token_index + 1].mean(dim=0)
                        )

                    if temp:
                        avg_span = torch.stack(temp, dim=0).mean(dim=0)
                    else:
                        # use zero hidden states if no temporal span
                        avg_span = torch.zeros_like(hidden_state[0])  

                    temporal_hidden_states.append(avg_span)

                return pooled_hidden_states, torch.stack(temporal_hidden_states)
            else:
                return self.encode_query(psg)
        else:
            task = "retrieval.passage"
            task_id = self.encoder._adaptation_map[task]
            adapter_mask = torch.full(
                (psg["input_ids"].size(0),),
                task_id,
                dtype=torch.int32,
                device=psg["input_ids"].device,
            )
            query_hidden_states = self.encoder(
                **psg,
                return_dict=True,
                adapter_mask=adapter_mask,
            )
            query_hidden_states = query_hidden_states.last_hidden_state[:, :, :768]

            return self._pooling(query_hidden_states, psg["attention_mask"])

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        if self.training:
            q_reps_base, p_reps_base = None, None
            q_reps = self.encode_query(query) if query else None
            
            if self.temporal:
                passage, temporal_span_list = passage
                p_reps, p_temporal_reps = self.encode_passage(passage, temporal_span_list)
            else:
                p_temporal_reps = None

            p_reps, p_temporal_reps = self.encode_passage(passage, temporal_span_list)

            if self.kl_loss is not None:
                q_reps_base, p_reps_base = self._get_base_model_embeddings(query, passage)

            loss, loss_partial = self.compute_loss(
                q_reps,
                p_reps,
                p_temporal_reps,
                q_reps_base=q_reps_base,
                p_reps_base=p_reps_base,
                filter_false_positives=self.filter_false_negatives,
                truncated_normalize=self.truncated_normalize,
                temporal=self.temporal,
            )
            losses = {"loss": loss}
            losses.update(loss_partial)

            return losses
        else:
            # Copy from EncoderModel's forward
            q_reps = self.encode_query(query) if query else None
            p_reps = self.encode_passage(passage) if passage else None

            # for inference
            if q_reps is None or p_reps is None:
                return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

            # for eval
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

            return EncoderOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps,
            )


class NaiveTemporalProjectorReconstruction(NaiveTemporal):
    """
    Add a temporal FFN
    """

    def __init__(self, *args, **kwargs):
        super(NaiveTemporalProjectorReconstruction, self).__init__(*args, **kwargs)
        
        self.temporal_projector = TemporalProjectorWithReconstructionLoss(self.config)

    def compute_loss(
        self,
        q_reps,
        p_reps,
        p_temporal_reps=None,
        p_temporal_reps_raw=None,
        q_reps_base=None,
        p_reps_base=None,
        temporal_tokens_input_ids=None,
        temporal_span_list=None,
        truncated_normalize=False,
        filter_false_positives=False,
        temporal=True,
    ):
        total_loss = torch.tensor(0.0, device=q_reps.device)
        partial_losses = {}

        target = None
        row_idx = None
        no_filter_mask = None
        num_neg = p_reps.size(0) // q_reps.size(0)

        for m in self.matryoshka_dim_list:
            q_reps_trunc = q_reps[:, :m]
            p_reps_trunc = p_reps[:, :m]
            
            if truncated_normalize and m < q_reps.size(1):
                q_reps_trunc = F.normalize(q_reps_trunc, dim=-1)
                p_reps_trunc = F.normalize(p_reps_trunc, dim=-1)

            scores_semantic = self.compute_similarity(q_reps_trunc, p_reps_trunc)

            scores_semantic = scores_semantic.view(q_reps.size(0), -1)
            # Create useful masks for calculating different losses
            # Which will be reused later on
            if target is None:
                target, row_idx, no_filter_mask = self.get_index_and_masks(num_neg, scores_semantic)

            # (Optional) Filter potential false negatives by filtering samples with simialrty greater than 95% compared to the ground truth
            if filter_false_positives:
                scores_semantic = self.mask_inbatch_negative_percentile(
                    target, row_idx, no_filter_mask, scores_semantic,   percentile=0.95
                )

            loss_semantic = self.cross_entropy(
                scores_semantic / self.temperature, target
            )
            
            # KL loss regularization to preserve the pre-trained embedding
            # Only work for the full-dim embedding
            if self.kl_loss is not None and m == q_reps.size(1) and (q_reps_base is not None and p_reps_base is not None):
                kl_loss = self._calc_kl_loss(
                    q_reps_base, p_reps_base, q_reps_trunc, p_reps_trunc
                )
                total_loss += kl_loss
                partial_losses[f"loss_kl"] = kl_loss.detach().clone()  

            if p_temporal_reps is not None and m != q_reps.size(1):
                # Train the temporal part in [semantic|temporal] to explicitly represent temporal
                t = q_reps.size(1) - m
                qt_reps = q_reps[:, -t:]
                pt_reps = p_reps[:, -t:]
                
                # Temporal reconstruction loss
                if p_temporal_reps_raw is not None and temporal_tokens_input_ids is not None and temporal_span_list is not None:
                    p_temporal_reps_raw = [x for xs in p_temporal_reps_raw for x in xs] # Flatten the list of lists
                    temporal_tokens_input_ids = torch.stack(
                        [x for xs in temporal_tokens_input_ids for x in xs]
                    ) # Flatten the list of lists
                    max_temporal_length = temporal_tokens_input_ids[0].size(0)

                    # try:
                    padded = torch.stack([
                        F.pad(x, (0, 0, 0, max_temporal_length-x.size(0)), "constant", 0) for x in p_temporal_reps_raw]
                    ) # Pad the sequence length to perform batch forward
                    
                    p_temporal_reps_projected, temporal_reconstructed_pred_scores = self.temporal_projector(padded)

                    x = temporal_reconstructed_pred_scores.view(-1, self.encoder.config.vocab_size)
                    y = temporal_tokens_input_ids.view(-1)
                    
                    # Take the average of the embeddings in case there are multiple temporal expressions: (*, dim) -> (bs, dim)
                    temp = []
                    start = 0
                    for i in [len(x) for x in temporal_span_list]:
                        temp.append(
                            p_temporal_reps_projected[start:start+i].view(-1, t).mean(dim=0)
                        )
                        start = start + i
                    p_temporal_reps_projected = torch.stack(temp)

                    temporal_reconstruction_loss = F.cross_entropy(x, y, ignore_index=0)
                else:
                    temporal_reconstruction_loss = torch.tensor(0.0, device=q_reps.device)
                    p_temporal_reps_projected = self.temporal_projector(p_temporal_reps)
                # except Exception as e:
                #     print(e)
                #     st()
                #     temporal_reconstruction_loss = 0.0
                # # Temporal reconstruction loss

                if truncated_normalize:
                    qt_reps = F.normalize(qt_reps, dim=-1)
                    pt_reps = F.normalize(pt_reps, dim=-1)
                    p_temporal_reps_projected = F.normalize(p_temporal_reps_projected, dim=-1)
                
                # score_qp = self.compute_similarity(
                #     qt_reps, pt_reps
                # )
                score_qt = self.compute_similarity(
                    qt_reps, p_temporal_reps_projected
                )
                # score_pt = self.compute_similarity(
                #     pt_reps, p_temporal_reps_projected
                # )
                
                # arange_idx = torch.arange(score_temporal_pt.size(0), device=scores_semantic.device)
                
                if filter_false_positives:
                    # score_qp = self.mask_inbatch_negative_percentile(
                    #     target, row_idx, no_filter_mask, score_qp,   percentile=0.95
                    # )
                    score_qt = self.mask_inbatch_negative_percentile(
                        target, row_idx, no_filter_mask, score_qt,   percentile=0.95
                    )
                    # score_pt = self.mask_inbatch_negative_percentile(
                    #     arange_idx, arange_idx, arange_idx, score_pt,   percentile=0.95
                    # )
                
                # loss_qp = self.cross_entropy(scores_temporal_qp / self.temperature, target)
                loss_qt = self.cross_entropy(score_qt / self.temperature, target)
                # loss_pt = self.cross_entropy(score_temporal_pt / self.temperature, arange_idx)
                
                loss_temporal = loss_qt + temporal_reconstruction_loss # + loss_qp + loss

                total_loss += loss_temporal
                partial_losses[f"loss_temporal_{m}"] = loss_temporal.detach().clone()
                partial_losses[f"loss_temporal_reconstruction"] = temporal_reconstruction_loss.detach().clone()

            total_loss += loss_semantic
            partial_losses[f"loss_semantic_{m}"] = loss_semantic.detach().clone()

        return total_loss, partial_losses

    def encode_passage(self, psg, temporal_span_list=None):
        # encode passage is the same as encode query
        if self.encoder.name_or_path != "jinaai/jina-embeddings-v3":

            if self.training and temporal_span_list is not None:
                query_hidden_states = self.encoder(**psg, return_dict=True)
                query_hidden_states = query_hidden_states.last_hidden_state
                pooled_hidden_states = self._pooling(
                    query_hidden_states, psg["attention_mask"]
                )

                # Processing temporal spans
                # Create a pre-average embedding list and an average embedding list
                temporal_hidden_states, temporal_average_hidden_states = [], []

                for temporal_span_sublist, hidden_state in zip(
                    temporal_span_list, query_hidden_states
                ):
                    temp, temp_avg = [], []
                    
                    for start_token_index, end_token_index in temporal_span_sublist:
                        temp.append(
                            hidden_state[start_token_index : end_token_index + 1]
                        )
                        temp_avg.append(temp[-1].mean(dim=0))

                    if temp_avg:
                        avg_span = torch.stack(temp_avg, dim=0).mean(dim=0)
                    else:
                        # use zero hidden states if no temporal span
                        avg_span = torch.zeros_like(hidden_state[0])
                        
                    if temp:
                        temporal_hidden_states.append(temp)
                    else:
                        # use zero hidden states if no temporal span
                        temporal_hidden_states.append(torch.zeros_like(hidden_state[0]))

                    temporal_average_hidden_states.append(avg_span)

                return pooled_hidden_states, torch.stack(temporal_average_hidden_states), temporal_hidden_states
            else:
                return self.encode_query(psg)
        else:
            task = "retrieval.passage"
            task_id = self.encoder._adaptation_map[task]
            adapter_mask = torch.full(
                (psg["input_ids"].size(0),),
                task_id,
                dtype=torch.int32,
                device=psg["input_ids"].device,
            )
            query_hidden_states = self.encoder(
                **psg,
                return_dict=True,
                adapter_mask=adapter_mask,
            )
            query_hidden_states = query_hidden_states.last_hidden_state[:, :, :768]

            return self._pooling(query_hidden_states, psg["attention_mask"])
        
    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        if self.training:
            q_reps_base, p_reps_base, temporal_span_list, temporal_tokens_input_ids = None, None, None, None
            q_reps = self.encode_query(query) if query else None
            
            if self.temporal:
                passage, temporal_span_list, temporal_tokens_input_ids = passage
                p_reps, p_temporal_reps, p_temporal_reps_raw = self.encode_passage(passage, temporal_span_list)
            else:
                p_temporal_reps = None

            if self.kl_loss is not None:
                q_reps_base, p_reps_base = self._get_base_model_embeddings(query, passage)

            loss, loss_partial = self.compute_loss(
                q_reps,
                p_reps,
                p_temporal_reps,
                p_temporal_reps_raw,
                q_reps_base,
                p_reps_base,
                temporal_tokens_input_ids,
                temporal_span_list,
                filter_false_positives=self.filter_false_negatives,
                truncated_normalize=self.truncated_normalize,
                temporal=self.temporal,
            )
            losses = {"loss": loss}
            losses.update(loss_partial)

            return losses
        else:
            # Copy from EncoderModel's forward
            q_reps = self.encode_query(query) if query else None
            p_reps = self.encode_passage(passage) if passage else None

            # for inference
            if q_reps is None or p_reps is None:
                return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

            # for eval
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

            return EncoderOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps,
            )