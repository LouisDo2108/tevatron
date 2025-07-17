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


from tevatron.retriever.arguments import ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.driver.encode import DenseModel, EncoderOutput

logger = logging.getLogger(__name__)


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
            "loss_pair": loss_pair.clone().detach(),
            "loss_topk": loss_topk.clone().detach(),
            "loss_rec": loss_rec.clone().detach(),
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
            "loss_pair": loss_pair.clone().detach(),
            "loss_topk": loss_topk.clone().detach(),
            "loss_rec": loss_rec.clone().detach(),
            "loss_temporal": loss_temporal.clone().detach()
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
        losses["loss_rank"] = loss_rank.clone().detach()
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
                losses[f"loss_partial_{m}"] = loss.clone().detach()

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
                losses[f"loss_matryoshka_{m}"] = loss.clone().detach()
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


class NaiveTemporal(DenseModel):
    def __init__(self, *args, **kwargs):
        super(NaiveTemporal, self).__init__(*args, **kwargs)

        self.matryoshka_dim_list = sorted([512, 768])
        self.truncated_dim = self.matryoshka_dim_list[
            -2
        ]  # Default to the second largest dimension

    def compute_loss(
        self,
        q_reps,
        p_reps,
        p_temporal_reps,
    ):
        total_loss = 0.0
        partial_losses = {}

        target = None

        for m in self.matryoshka_dim_list:
            q_reps_trunc = q_reps[:, :m]
            p_reps_trunc = p_reps[:, :m]

            scores_semantic = self.compute_similarity(q_reps_trunc, p_reps_trunc)
            scores_semantic = scores_semantic.view(q_reps.size(0), -1)

            if target is None:
                target = torch.arange(
                    scores_semantic.size(0),
                    device=scores_semantic.device,
                    dtype=torch.long,
                )
                target = target * (p_reps.size(0) // q_reps.size(0))

            loss_semantic = self.cross_entropy(
                scores_semantic / self.temperature, target
            )

            if p_temporal_reps is not None and m != q_reps.shape[-1]:
                # Train the right-left part of embedding to explicitly represent temporal
                t = q_reps.shape[-1] - m
                q_reps_temporal_trunc = q_reps[:, -t:]
                p_temporal_reps_trunc = p_temporal_reps[:, -t:]

                scores_temporal = self.compute_similarity(
                    q_reps_temporal_trunc, p_temporal_reps_trunc
                )

                loss_temporal = self.cross_entropy(
                    scores_temporal / self.temperature,
                    target,
                )

                total_loss += loss_temporal
                partial_losses[f"loss_temporal_{m}"] = loss_temporal.clone().detach()

            total_loss += loss_semantic
            partial_losses[f"loss_semantic_{m}"] = loss_semantic.clone().detach()

        return total_loss, partial_losses

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        if self.training:

            q_reps = self.encode_query(query) if query else None

            passage, passage_temporal = passage
            p_reps = self.encode_passage(passage)
            p_temporal_reps = self.encode_passage(passage_temporal)

            loss, loss_partial = self.compute_loss(
                q_reps,
                p_reps,
                p_temporal_reps,
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


class NaiveTemporalv2(DenseModel):

    def __init__(self, *args, **kwargs):
        super(NaiveTemporalv2, self).__init__(*args, **kwargs)

        self.matryoshka_dim_list = sorted([512, 768], reverse=True)
        self.truncated_dim = self.matryoshka_dim_list[1]  # Default to the second largest dimension

    def compute_loss(
        self,
        q_reps,
        p_reps,
        p_temporal_reps,
    ):
        total_loss = 0.0
        partial_losses = {}

        # original_shape_scores = None

        target = None

        for m in self.matryoshka_dim_list:
            q_reps_trunc = q_reps[:, :m]
            p_reps_trunc = p_reps[:, :m]

            scores_semantic = self.compute_similarity(q_reps_trunc, p_reps_trunc)

            scores_semantic = scores_semantic.view(q_reps.size(0), -1)

            # if m == q_reps.shape[-1]:
            #     original_shape_scores = scores_semantic.diag().clone().detach()

            # if original_shape_scores is not None and m < q_reps.shape[-1]:
            #     # Regularization with MSE loss
            #     mse_loss = F.mse_loss(scores_semantic.diag(), original_shape_scores)
            #     total_loss += mse_loss
            #     partial_losses[f"loss_mse"] = mse_loss.clone().detach()

            if target is None:
                target = torch.arange(
                    scores_semantic.size(0),
                    device=scores_semantic.device,
                    dtype=torch.long,
                )
                target = target * (p_reps.size(0) // q_reps.size(0))

            loss_semantic = self.cross_entropy(scores_semantic / self.temperature, target)

            if p_temporal_reps is not None and m != q_reps.shape[-1]:
                # Train the right-left part of embedding to explicitly represent temporal
                t = q_reps.shape[-1] - m
                q_reps_temporal_trunc = q_reps[:, -t:]
                p_temporal_reps_trunc = p_temporal_reps[:, -t:]

                scores_temporal = self.compute_similarity(
                    q_reps_temporal_trunc, p_temporal_reps_trunc
                )

                loss_temporal = self.cross_entropy(
                    scores_temporal / self.temperature,
                    target,
                )

                total_loss += loss_temporal
                partial_losses[f"loss_temporal_{m}"] = loss_temporal.clone().detach()

            total_loss += loss_semantic 
            partial_losses[f"loss_semantic_{m}"] = loss_semantic.clone().detach()

        return total_loss, partial_losses

    def encode_passage(self, psg, temporal_span_list=None):
        # encode passage is the same as encode query
        if self.encoder.name_or_path != "jinaai/jina-embeddings-v3":

            if self.training and temporal_span_list is not None:
                query_hidden_states = self.encoder(**psg, return_dict=True)
                query_hidden_states = query_hidden_states.last_hidden_state
                pooled_hidden_states = self._pooling(query_hidden_states, psg["attention_mask"])

                # Processing temporal spans
                temporal_hidden_states = []

                for temporal_span_sublist, hidden_state in zip(temporal_span_list, query_hidden_states):
                    temp = []
                    for start_token_index, end_token_index in temporal_span_sublist:
                        temp.append(hidden_state[start_token_index : end_token_index + 1].mean(dim=0))

                    if temp:
                        avg_span = torch.stack(temp, dim=0).mean(dim=0)
                    else:
                        avg_span = torch.zeros_like(hidden_state[0])  # fallback if no temporal span

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

            q_reps = self.encode_query(query) if query else None

            passage, temporal_span_list = passage
            p_reps, p_temporal_reps = self.encode_passage(passage, temporal_span_list)

            loss, loss_partial = self.compute_loss(
                q_reps,
                p_reps,
                p_temporal_reps,
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
