import logging
import os
from pathlib import Path
from pdb import set_trace as st
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ConstantPad2d
from transformers.activations import ACT2FN
from utils import get_params_info

from tevatron.retriever.arguments import ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.driver.encode import DenseModel as TevatronDenseModel
from tevatron.retriever.driver.encode import EncoderOutput

logger = logging.getLogger(__name__)


def norm(x):
    return F.normalize(x, p=2, dim=-1)


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


class DenseModel(TevatronDenseModel):
    def forward(
        self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None
    ):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(
                scores.size(0), device=scores.device, dtype=torch.long
            )
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores / self.temperature, target)
            # if self.is_ddp:
            #     loss = loss * self.world_size  # counter average weight reduction
            losses = {
                "loss": loss,
                "loss_semantic_768": loss.clone().detach(),
            }
            return losses
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
            return EncoderOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps,
            )


class PredictionHeadTransform(nn.Module):
    def __init__(self, config, temporal_dim=64):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, temporal_dim)
        # if isinstance(config.hidden_act, str):
        if hasattr(config, "hidden_act"):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        elif hasattr(config, "activation_function"):
            # For nomic-ai/nomic-embed-text-v1.5
            if config.activation_function == "swiglu":
                self.transform_act_fn = ACT2FN["silu"]
        else:
            self.transform_act_fn = config.hidden_act

        print(f"The adaptor uses {self.transform_act_fn} activation")
        
        if hasattr(config, "layer_norm_eps"):
            self.LayerNorm = nn.LayerNorm(temporal_dim, eps=config.layer_norm_eps)
        elif hasattr(config, "layer_norm_epsilon"):
            # nomic-ai/nomic-embed-text-v1.5
            self.LayerNorm = nn.LayerNorm(temporal_dim, eps=config.layer_norm_epsilon)
        elif hasattr(config, "rms_norm_eps"):
            from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
            self.LayerNorm = Qwen3RMSNorm(temporal_dim, eps=config.rms_norm_eps)
        
        print(f"The adaptor uses {self.LayerNorm} normalization")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class TemporalProjector(nn.Module):
    """
    Copy from modeling_bert.py's BertLMP redictionHead
    """
    def __init__(self, config, temporal_dim=64, reconstruction=False):
        super().__init__()
        self.transform = PredictionHeadTransform(config, temporal_dim)
        self.reconstruction = reconstruction

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        if reconstruction:
            self.decoder = nn.Linear(temporal_dim, config.vocab_size, bias=False)

            self.bias = nn.Parameter(torch.zeros(config.vocab_size))

            # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
            self.decoder.bias = self.bias
        else:
            self.decoder = nn.Linear(temporal_dim, temporal_dim, bias=False)

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        temporal_hidden_states = self.transform(hidden_states)
        if self.reconstruction:
            hidden_states = self.decoder(temporal_hidden_states)
            return temporal_hidden_states, hidden_states
        else:
            temporal_hidden_states = self.decoder(temporal_hidden_states)
            return temporal_hidden_states, None


class NaiveTemporal(DenseModel):
    def __init__(self, *args, **kwargs):
        super(NaiveTemporal, self).__init__(*args, **kwargs)

        self.matryoshka_dim_list = sorted(self.training_args.matryoshka_dim_list, reverse=True)
        self.temporal = self.training_args.temporal
        self.temporal_dim = self.training_args.temporal_dim
        self.max_temporal_length = self.training_args.max_temporal_length
        self.temporal_reconstruction = self.training_args.temporal_reconstruction
        self.filter_false_negatives = self.training_args.filter_false_negatives
        self.qt = self.training_args.qt
        self.pt = self.training_args.pt
        self.qt_recon = self.training_args.qt_recon
        self.pt_recon = self.training_args.pt_recon

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
        *args, **kwargs
    ):
        total_loss = torch.tensor(0.0)
        partial_losses = {}

        return total_loss, partial_losses

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
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

        return target,row_idx, no_filter_mask

    def mask_inbatch_negative_percentile(self, scores_semantic, target, row_idx, no_filter_mask=None, percentile=0.95):
        semantic_thresholds = scores_semantic[row_idx, target] * percentile  # shape: [B]

        mask = scores_semantic > semantic_thresholds.unsqueeze(1)  # Mask out those greater than the threshold
        if no_filter_mask is not None:
            mask[row_idx.unsqueeze(1), no_filter_mask] = False  # type: ignore # don't mask the positive and annotated negatives, only consider the other in-batch negatives

        # Apply the mask
        scores_semantic = scores_semantic.masked_fill(mask, float('-inf'))
        return scores_semantic


class TemporalProjectorReconstruction(NaiveTemporal):
    def __init__(self, *args, **kwargs):
        super(TemporalProjectorReconstruction, self).__init__(*args, **kwargs)
        
        if self.temporal or self.temporal_reconstruction:
            self.temporal_projector = TemporalProjector(
                self.config, self.temporal_dim, reconstruction=self.temporal_reconstruction
            )
        else:
            self.temporal_projector = nn.Identity()

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        """
        query: tuple(
            transformers.tokenization_utils_base.BatchEncoding, 
            List[List[int]]: (bs, (num_temporal_span, 2)) a list contains lists of temporal spans' start and end indices, such as [ [[4, 6]], [[21, 22], [10, 13], [76, 78]] ]
            List[List[tensor]]: (bs, (num_temporal_span, max_temporal_len)) a list contains lists of tensor that stores the token labels of the temporal spans for reconstruction, such as: 
            [
                [tensor([12518,  1516, 13206,     0,     0,], device='cuda:0')], 
                [
                    tensor([2251, 2249,    0,    0,    0,    0,], device='cuda:0'), 
                    tensor([2028, 1011, 2095, 3206,    0,    0], device='cuda:0'), 
                    tensor([2459, 2233, 2286,    0,    0,    0,], device='cuda:0')]
                ]
        )
        
        transformers.tokenization_utils_base.BatchEncoding's keys: 
            input_ids: (bs, seq_len), 
            attention_mask: (bs, seq_len)
        """
        if self.training:

            q, qt_token_spans_list, qt_tokens_input_ids_list = query

            p, pt_token_spans_list, pt_tokens_input_ids_list, pt_query_type_list, p_allen_relation_list = passage

            q_reps_base, p_reps_base = None, None

            num_neg = p["input_ids"].size(0) // q["input_ids"].size(0)

            q_reps, qt_reps = self.encode(q, qt_token_spans_list, num_neg, is_query=True)

            p_reps, pt_reps = self.encode(p, pt_token_spans_list, num_neg, is_query=False)

            if self.kl_loss is not None:
                q_reps_base, p_reps_base = self._get_base_model_embeddings(q, p)

            loss, loss_partial = self.compute_loss(
                q_reps,
                qt_reps,
                p_reps,
                pt_reps,
                q_reps_base,
                p_reps_base,
                qt_tokens_input_ids_list,
                pt_tokens_input_ids_list, pt_query_type_list, p_allen_relation_list,
            )
            losses = {"loss": loss}
            losses.update(loss_partial)

            return losses
        else:
            # Copy from EncoderModel's forward
            q_reps = self.encode(query, temporal_span_list=None, is_query=True) if query else None
            p_reps = self.encode(query, temporal_span_list=None, is_query=False) if passage else None

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

    def compute_loss(
        self,
        q_reps,
        qt_reps,
        p_reps,
        pt_reps,
        q_reps_base,
        p_reps_base,
        # qt_token_spans_list, 
        qt_tokens_input_ids_list,
        # pt_token_spans_list, 
        pt_tokens_input_ids_list, 
        pt_query_type_list=None, 
        p_allen_relation_list=None,
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

            if m != q_reps.size(-1):
                q_reps_trunc = norm(q_reps_trunc)
                p_reps_trunc = norm(p_reps_trunc)

            scores_semantic = self.compute_similarity(q_reps_trunc, p_reps_trunc).view(q_reps.size(0), -1)

            # Create useful masks for calculating different losses
            # Which will be reused later on
            if target is None:
                target, row_idx, no_filter_mask = self.get_index_and_masks(num_neg, scores_semantic)

            # (Optional) Filter potential false negatives by filtering samples with simialrty greater than 95% compared to the ground truth
            if self.filter_false_negatives:
                scores_semantic = self.mask_inbatch_negative_percentile(
                    scores_semantic, target, row_idx, no_filter_mask,   percentile=0.95
                )

            loss_semantic = self.cross_entropy(
                scores_semantic / self.temperature, target
            )
            total_loss += loss_semantic
            partial_losses[f"loss_{m}"] = loss_semantic.detach().clone()

            # KL loss regularization to preserve the pre-trained embedding
            # Only work for the full-dim embedding
            if self.kl_loss is not None and m == q_reps.size(1) and (q_reps_base is not None and p_reps_base is not None):
                kl_loss = self._calc_kl_loss(
                    q_reps_base, p_reps_base, q_reps_trunc, p_reps_trunc
                )
                total_loss += kl_loss
                partial_losses[f"loss_kl"] = kl_loss.detach().clone()

        # Temporal contrastive loss
        if self.temporal:

            # Query inputs
            # Query side
            qt_reps_padded = torch.stack([
                F.pad(x, (0, 0, 0, self.max_temporal_length-x.size(0)), "constant", 0) for x in qt_reps
            ])
            qt_tokens_input_ids = torch.stack(
                [x for xs in qt_tokens_input_ids_list for x in xs if len(xs) > 0]
            )  # Flatten the list of lists

            # Passage inputs
            q_reps_new = [] # A list of query rep that correspond to a pair of (pos, neg(s)) passage for contrastive learning
            p_reps_new = []  # The list of corresponding pairs (pos, neg(s)) passag
            new_target = []  # the label for contrastive learning according to q_reps_new and p_reps_new

            qt_reps_mask = [] # the mask to filter out the corresponding qt_reps_projected for temporal contrastive learning with p_reps
            # p_reps_mask = [] # the mask to filter out the corresponding p_reps after batch forwarding
            pt_reps_mask = [] # the mask to filter out pt_reps_projected for temporal contrastive learning with q_reps

            for q_rep, p_rep, pt_rep_list in zip(
                q_reps, p_reps.view(q_reps.size(0), -1, p_reps.size(-1)), pt_reps):

                """
                For all temporal in the passage side,
                if there exists a pair of positive and negative explicit/implicit temporal expreessions,
                we should perform Temporal contrastive learning for this sample.
                """
                if pt_rep_list != [] and pt_rep_list[0] != [] and any(z != [] for z in pt_rep_list[1:]):
                    # There must be 1 positive and at least 1 negative sample
                    q_reps_new.append(q_rep)
                    qt_reps_mask.append(1)
                    new_target.append(len(p_reps_new))

                    for i, j in zip(p_rep, pt_rep_list):
                        if isinstance(j, Tensor):
                            p_reps_new.append(i)
                            pt_reps_mask.append(1)
                        # else:
                        #     pt_reps_mask.append(0)
                else:
                    # No positive sample
                    # or all negative samples have no temporal expression
                    qt_reps_mask.append(0)
                    for x in pt_rep_list:
                        if isinstance(x, Tensor):
                            pt_reps_mask.append(0)

            q_reps_new = torch.stack(q_reps_new)
            qt_reps_mask = torch.as_tensor(qt_reps_mask, dtype=torch.long, device=q_reps.device)

            p_reps_new = torch.stack(p_reps_new)
            pt_reps_mask = torch.as_tensor(
                pt_reps_mask, dtype=torch.long, device=q_reps.device
            )
            new_target = torch.as_tensor(new_target, dtype=torch.long, device=q_reps.device)

            pt_reps_padded = torch.stack([F.pad(x, (0, 0, 0, self.max_temporal_length-x.size(0)), "constant", 0) for xs in pt_reps for x in xs if x != []])

            pt_tokens_input_ids = torch.stack([
                x for xs in pt_tokens_input_ids_list for x in xs if x != []
            ]) # Flatten the list of lists

            if self.qt or self.qt_recon:
                if self.temporal_reconstruction and self.qt_recon != 0:
                    qt_reconstruction_loss = torch.tensor(0.0, device=q_reps.device)

                    qt_reps_projected, qt_reconstructed_pred_scores = self.temporal_projector(qt_reps_padded)

                    qx = qt_reconstructed_pred_scores.view(-1, self.encoder.config.vocab_size)
                    qy = qt_tokens_input_ids.view(-1)

                    qt_reconstruction_loss = F.cross_entropy(qx, qy, ignore_index=0)

                    qt_recon_loss = self.qt_recon * qt_reconstruction_loss
                    total_loss += qt_recon_loss
                    partial_losses[f"loss_qt_recon"] = qt_recon_loss.detach().clone()
                else:
                    qt_reps_projected, _ = self.temporal_projector(qt_reps_padded)

                # mask out the padded tokens
                mask = (qt_tokens_input_ids != 0).unsqueeze(-1)

                # Take the average of (non-padded) temporal tokens
                qt_reps_projected = (qt_reps_projected * mask).sum(dim=1) / mask.sum(dim=1)

            if self.temporal_reconstruction and self.pt_recon != 0:
                pt_reconstruction_loss = torch.tensor(0.0, device=p_reps.device)

                pt_reps_projected, pt_reconstructed_pred_scores = self.temporal_projector(pt_reps_padded)

                px = pt_reconstructed_pred_scores.view(-1, self.encoder.config.vocab_size)
                # There are temporal answer type in pasasge, therefore, we need to check for empty lists, for reconstruction purpose
                py = pt_tokens_input_ids.view(-1)
                pt_reconstruction_loss = F.cross_entropy(px, py, ignore_index=0)

                pt_recon_loss = self.pt_recon * pt_reconstruction_loss
                total_loss += pt_recon_loss
                partial_losses[f"loss_pt_recon"] = pt_recon_loss.detach().clone()
            elif self.pt != 0:
                pt_reps_projected, _ = self.temporal_projector(pt_reps_padded)

            # mask out the padded tokens
            mask = (pt_tokens_input_ids != 0).unsqueeze(-1)

            # Take the average of (non-padded) temporal tokens
            pt_reps_projected = (pt_reps_projected * mask).sum(dim=1) / mask.sum(dim=1)

            # # Contrastive learning for projected temporal embeddings
            # if self.qt != 0:
            #     score_qt_projected = self.compute_similarity(
            #         norm(q_reps[:, :temporal_dim]),
            #         norm(qt_reps_projected)
            #     )
            #     score_qt_projected = score_qt_projected.view(q_reps.size(0), -1)

            #     loss_qt_projected = self.cross_entropy(
            #         score_qt_projected / self.temperature,
            #         torch.arange(score_qt_projected.size(0), dtype=torch.long, device=score_qt_projected.device)
            #     )
            #     total_loss += self.qt * loss_qt_projected
            #     partial_losses[f"loss_qt_projected_original"] = (
            #         self.qt * loss_qt_projected.detach().clone()
            #     )

            # if self.pt != 0:
            #     score_pt_projected = self.compute_similarity(
            #         norm(p_reps_new[:, :temporal_dim]),
            #         norm(pt_reps_projected[p_reps_contrastive_mask.nonzero(as_tuple=True)]),
            #     )
            #     score_pt_projected = score_pt_projected.view(p_reps_new.size(0), -1)

            #     loss_pt_projected = self.cross_entropy(
            #         score_pt_projected / self.temperature,
            #         torch.arange(score_pt_projected.size(0), dtype=torch.long, device=score_pt_projected.device)
            #     )
            #     total_loss += self.pt * loss_pt_projected
            #     partial_losses[f"loss_pt_projected_original"] = (
            #         self.pt * loss_pt_projected.detach().clone()
            #     )

            if self.qt != 0:
                score_qt_projected = self.compute_similarity(
                    norm(qt_reps_projected)[qt_reps_mask.nonzero(as_tuple=True)],
                    norm(p_reps_new[:, :self.temporal_dim]),
                )
                loss_qt_projected = self.cross_entropy(
                    score_qt_projected / self.temperature, new_target
                )
                total_loss += self.qt * loss_qt_projected
                partial_losses[f"loss_qt_proj"] = (
                    self.qt * loss_qt_projected.detach().clone()
                )

            if self.pt != 0:
                score_pt_projected = self.compute_similarity(
                    norm(q_reps_new[:, : self.temporal_dim]),
                    norm(pt_reps_projected[pt_reps_mask.nonzero(as_tuple=True)]),
                )
                loss_pt_projected = self.cross_entropy(
                    score_pt_projected / self.temperature,
                    new_target,
                )

                total_loss += self.pt * loss_pt_projected
                partial_losses[f"loss_pt_proj"] = self.pt * loss_pt_projected.detach().clone()

        return total_loss, partial_losses

    def encode(self, q, temporal_span_list=None, num_neg=1, is_query=True):

        # if self.encoder.name_or_path != "jinaai/jina-embeddings-v3":
        #   query_hidden_states = self.encoder(**q, return_dict=True)
        # else:
        #     task = "retrieval.query" if is_query else "retrieval.passage"
        #     task_id = self.encoder._adaptation_map[task]
        #     adapter_mask = torch.full(
        #         (q["input_ids"].size(0),),
        #         task_id,
        #         dtype=torch.int32,
        #         device=q["input_ids"].device,
        #     )
        #     query_hidden_states = self.encoder(
        #         **q,
        #         return_dict=True,
        #         adapter_mask=adapter_mask,
        #     )

        query_hidden_states = self.encoder(**q, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        pooled_hidden_states = self._pooling(query_hidden_states, q["attention_mask"])

        if self.training:
            temporal_hidden_states = []
            temp = []

            # if is_query:
            for ix, (temporal_span_sublist, hidden_state) in enumerate(
                zip(temporal_span_list, query_hidden_states)
            ):
                if len(temporal_span_sublist) == 0:
                    if is_query:
                        temporal_hidden_states.append([])
                    else:
                        temp.append([])
                else:
                    if is_query:
                        start_token_index, end_token_index = temporal_span_sublist[0]
                        temporal_hidden_states.append(
                            hidden_state[start_token_index : end_token_index + 1]
                        )
                    else:
                        start_token_index, end_token_index = temporal_span_sublist[0]
                        temp.append(
                            hidden_state[start_token_index : end_token_index + 1]
                        )
                # Split according to the number of negatives
                if not is_query and ((ix + 1) % num_neg) == 0:
                    temporal_hidden_states.append(temp)
                    temp = []

            if not is_query and temp != []:
                temporal_hidden_states.append(temp)
            # else:
            #     for ix, (temporal_span_sublist, hidden_state) in enumerate(zip(
            #         temporal_span_list, query_hidden_states
            #     )):
            #         if len(temporal_span_sublist) == 0:
            #             temp.append([])
            #         else:
            #             start_token_index, end_token_index = temporal_span_sublist[0]
            #             temp.append(
            #                 hidden_state[start_token_index : end_token_index + 1]
            #             )

            #         # Split according to the number of negatives
            #         if ((ix+1) % num_neg) == 0:
            #             temporal_hidden_states.append(temp)
            #             temp = []
            #     if temp != []:
            #         temporal_hidden_states.append(temp)
            return pooled_hidden_states, temporal_hidden_states
        else:
            return pooled_hidden_states


class TempRetriever(NaiveTemporal):
    def __init__(self, *args, **kwargs):
        super(TempRetriever, self).__init__(*args, **kwargs)

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        if self.training:
            q, qt = query
            p, pt = passage

            q_reps, qt_reps = self.encode(q, qt)
            p_reps, pt_reps = self.encode(p, pt)

            loss, loss_partial = self.compute_loss(
                q_reps,
                qt_reps,
                p_reps,
                pt_reps,
            )
            losses = {"loss": loss}
            losses.update(loss_partial)

            return losses
        else:
            # Copy from EncoderModel's forward
            q_reps = self.encode(query, query_flag=True) if query else None
            p_reps = self.encode(passage, query_flag=False) if passage else None

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

    def encode(self, q, qt):
        query_hidden_states = self.encoder(**q, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        pooled_hidden_states = self._pooling(query_hidden_states, q["attention_mask"])

        qt_hidden_states = self.base_model(**qt, return_dict=True)
        qt_hidden_states = qt_hidden_states.last_hidden_state
        qt_pooled_hidden_states = self._pooling(query_hidden_states, qt["attention_mask"])

        return pooled_hidden_states, qt_pooled_hidden_states

    def compute_loss(
        self,
        q_reps,
        qt_reps,
        p_reps,
        pt_reps,
    ):
        total_loss = torch.tensor(0.0, device=q_reps.device)
        partial_losses = {}

        q_reps = torch.cat([q_reps, qt_reps], dim=1)
        p_reps = torch.cat([p_reps, pt_reps], dim=1)

        scores = self.compute_similarity(q_reps, p_reps)
        scores = scores.view(q_reps.size(0), -1)

        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (p_reps.size(0) // q_reps.size(0))

        loss = self.cross_entropy(scores / self.temperature, target)

        total_loss += loss
        partial_losses["loss"] = loss

        return total_loss, partial_losses
