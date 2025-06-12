import json
import logging
import os
from typing import Dict, Optional

from pdb import set_trace as st
import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch import Tensor, nn

from transformers import AutoConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.trainer import TRAINING_ARGS_NAME

from tevatron.retriever.arguments import ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.driver.encode import DenseModel, EncoderOutput

logger = logging.getLogger(__name__)


def get_params_info(model):
    all_param = 0
    trainable_param = 0

    print("\nAll trainable parameters:")
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_param += param.numel()
            print(name, param.numel())


class PairwiseSimilarityLossBase(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_similarity_matrix(self, x):
        return F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)

    def forward(self, embeddings, adapted_embeddings, m_list, reduce=True):
        raise NotImplementedError("Must be implemented in subclass.")


class PairwiseSimilarityLoss(PairwiseSimilarityLossBase):
    def forward(self, embeddings, adapted_embeddings, m_list, reduce=True):
        sim_target = self.compute_similarity_matrix(embeddings)
        total_loss = 0.0
        partial_losses = {}

        for m in m_list:
            sim_adapted = self.compute_similarity_matrix(adapted_embeddings[:, :m])
            diff = torch.abs(sim_target - sim_adapted)
            loss_m = diff.sum()
            total_loss += loss_m
            partial_losses[f"loss_pair_partial_{m}"] = loss_m.clone().detach()
            partial_losses[f"loss_pair_partial_{m}"] /= embeddings.shape[0]

        total_loss /= embeddings.shape[0]

        return total_loss, partial_losses


class TopKPairwiseSimilarityLoss(PairwiseSimilarityLossBase):
    def __init__(self, k=10):
        super().__init__()
        self.k = k

    def forward(self, embeddings, adapted_embeddings, m_list, reduce=True):
        N = embeddings.size(0)
        total_loss = 0.0
        partial_losses = {}

        if N <= self.k:
            # If the batch is too small, ignore this loss
            for m in m_list:
                partial_losses[m] = torch.tensor(0.0)
            return torch.tensor(0.0), partial_losses

        sim_target = self.compute_similarity_matrix(embeddings)
        sim_target.fill_diagonal_(-float("inf"))

        topk_values, topk_indices = torch.topk(
            sim_target,
            k=self.k,
            dim=1,
        )

        for m in m_list:
            adapted_trunc = adapted_embeddings[:, :m]
            adapted_i = adapted_trunc.unsqueeze(1).expand(-1, self.k, -1)  # (N, k, m)
            adapted_j = adapted_trunc[topk_indices]  # (N, k, m)
            sim_adapted = F.cosine_similarity(adapted_i, adapted_j, dim=-1)  # (N, k)

            diff = torch.abs(sim_adapted - topk_values)
            loss_m = diff.sum()
            total_loss += loss_m
            partial_losses[f"loss_topk_partial_{m}"] = loss_m.clone().detach()
            partial_losses[f"loss_topk_partial_{m}"] /= embeddings.shape[0]

        total_loss /= embeddings.shape[0]
        return total_loss, partial_losses


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, embeddings, adapted_embeddings):
        return torch.abs(embeddings - adapted_embeddings).mean()


class RankLoss(nn.Module):
    def __init__(self):
        super(RankLoss, self).__init__()

    # def compute_similarity(self, q_reps, p_reps):
    #     return torch.matmul(
    #         F.normalize(q_reps, p=2, dim=-1),
    #         F.normalize(p_reps, p=2, dim=-1).transpose(0, 1),
    #     )
    
    def compute_similarity(self, q, p):
        # q: (B, D), p: (B, K, D)
        q_norm = F.normalize(q, p=2, dim=-1)        # (B, D)
        p_norm = F.normalize(p, p=2, dim=-1)        # (B, K, D)
        # Compute dot product per query with its K passages
        # Result shape: (B, K)
        sim = torch.bmm(p_norm, q_norm.unsqueeze(2)).squeeze(2).transpose(0,1)
        return sim.T  # (B, K)

    def forward(
        self, adapted_q_reps, adapted_p_reps, m_list
    ):
        # total_loss = 0.0
        # batch_size = adapted_q_reps.size(0)

        # for m in m_list:
        #     scores = self.compute_similarity(adapted_q_reps[:, :m], adapted_p_reps[:, :m])
        #     scores = scores.view(adapted_q_reps.size(0), -1)
        #     target = torch.arange(
        #         scores.size(0), device=scores.device, dtype=torch.long
        #     )
        #     target = target * (adapted_p_reps.size(0) // adapted_q_reps.size(0))
        #     positive_pair_scores = scores[torch.arange(batch_size), target].unsqueeze(1).expand(*scores.shape)

        #     loss = F.softplus(scores - positive_pair_scores)
        #     loss[torch.arange(batch_size), target] = 0

        #     total_loss += loss.sum() / batch_size

        # return total_loss

        B = adapted_q_reps.size(0)
        K = adapted_p_reps.size(0) // B
        total_loss = 0.0

        # reshape p_reps to (B, K, D)
        adapted_p_reps = adapted_p_reps.view(B, K, -1)

        for m in m_list:
            q = adapted_q_reps[:, :m]  # (B, m)
            p = adapted_p_reps[:, :, :m]  # (B, K, m)

            sim = self.compute_similarity(q, p)  # (B, K)
            pos_scores = sim[:, 0]  # positive scores (B,)

            # Broadcast pos scores for comparison
            pos_scores_exp = pos_scores.unsqueeze(1).expand_as(sim)

            loss = F.softplus(sim - pos_scores_exp)
            loss[:, 0] = 0  # zero loss for positive pairs

            total_loss += loss.sum() / B  # average per query

        return total_loss


class Adaptor(nn.Module):

    def __init__(self, hidden_size):
        super(Adaptor, self).__init__()
        self.down_project = nn.Linear(hidden_size, 256)
        # self.ffn = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(256, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, inputs):
        down_projected = self.activation(self.down_project(inputs))
        # ffn_output = self.activation(self.ffn(down_projected))
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
        self.matryoshka_dim_list = [384]
        self.adaptor = Adaptor(hidden_size=768)

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
            )[:, : self.matryoshka_dim_list[0]]

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        if self.training:
            docid, doc = passage
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


class SupervisedMAdaptor(UnsupervisedMAdaptor):
    def __init__(self, *args, **kwargs):
        super(SupervisedMAdaptor, self).__init__(*args, **kwargs)

        self.loss_rank_fn = RankLoss()

    def compute_loss(self, q_reps, adapted_q_reps, p_reps, adapted_p_reps):
        losses = super().compute_loss(p_reps, adapted_p_reps)

        loss_rank = self.loss_rank_fn(
            adapted_q_reps,
            adapted_p_reps,
            m_list=self.matryoshka_dim_list,
        )
        losses["loss"] += loss_rank
        losses["loss_rank"] = loss_rank.clone().detach()
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
