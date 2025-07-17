import torch
import torch.nn.functional as F
from torch import Tensor, nn
from pdb import set_trace as st


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, embeddings, adapted_embeddings):
        return torch.abs(embeddings - adapted_embeddings).mean()


class PairwiseSimilarityLossBase(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_similarity_matrix(self, x):
        return F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)

    def forward(self, embeddings, adapted_embeddings, m_list, reduce=True, sim_target=None):
        raise NotImplementedError("Must be implemented in subclass.")


class PairwiseSimilarityLoss(PairwiseSimilarityLossBase):
    def forward(self, embeddings, adapted_embeddings, m_list, reduce=True, sim_target=None):
        if sim_target is None:
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

    def forward(
        self, embeddings, adapted_embeddings, m_list, reduce=True, sim_target=None
    ):
        N = embeddings.size(0)
        total_loss = 0.0
        partial_losses = {}

        if N <= self.k:
            # If the batch is too small, ignore this loss
            for m in m_list:
                partial_losses[m] = torch.tensor(0.0)
            return torch.tensor(0.0), partial_losses

        if sim_target is None:
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


class RankLoss(nn.Module):
    def __init__(self):
        super(RankLoss, self).__init__()

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(
            F.normalize(q_reps, p=2, dim=-1),
            F.normalize(p_reps, p=2, dim=-1).transpose(0, 1),
        )

    def forward(self, adapted_q_reps, adapted_p_reps, m_list):
        total_loss = 0.0
        batch_size = adapted_q_reps.size(0)
        partial_losses = {}

        for m in m_list:
            scores = self.compute_similarity(
                adapted_q_reps[:, :m], adapted_p_reps[:, :m]
            )
            scores = scores.view(adapted_q_reps.size(0), -1)
            target = torch.arange(
                scores.size(0), device=scores.device, dtype=torch.long
            )
            target = target * (adapted_p_reps.size(0) // adapted_q_reps.size(0))
            loss = F.softplus(
                scores
                - scores[torch.arange(batch_size), target]
                .unsqueeze(1)
                .expand(-1, adapted_p_reps.size(0))
            )  # Pairwise loss
            loss[torch.arange(batch_size), target] = 0  # zero loss for positive pairs

            partial_losses[f"loss_rank_partial_{m}"] = loss.mean().clone().detach()

            total_loss += loss.mean()

        return total_loss


def CoSentLossFunction(x, y, labels, scale=1.0):

    scores = F.cosine_similarity(x, y, dim=-1)
    scores = scores * scale
    scores = scores[:, None] - scores[None, :]

    # label matrix indicating which pairs are relevant
    labels = labels[:, None] < labels[None, :]
    labels = labels.float()

    # mask out irrelevant pairs so they are negligible after exp()
    scores = scores - (1 - labels) * 1e12

    # append a zero as e^0 = 1
    scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
    loss = torch.logsumexp(scores, dim=0)
    return loss


class TemporalLoss(PairwiseSimilarityLossBase):

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(
            F.normalize(q_reps, p=2, dim=-1),
            F.normalize(p_reps, p=2, dim=-1).transpose(0, 1),
        )

    def forward(
        self,
        temporal_embeddings,
        adapted_temporal_embeddings,
        adapted_embeddings,
        temp_dim=256,
        base_dim=768,
    ):
        total_loss = 0.0
        partial_losses = {}

        sim_target = self.compute_similarity_matrix(temporal_embeddings)
        sim_adapted = self.compute_similarity_matrix(
            adapted_temporal_embeddings[:, base_dim-temp_dim:]
        )
        pairwiseloss = torch.abs(sim_target - sim_adapted).mean()
        total_loss += pairwiseloss
        partial_losses["loss_pairwise_temporal"] = pairwiseloss.clone().detach()

        recon_loss = torch.abs(temporal_embeddings - adapted_temporal_embeddings).mean()

        # total_loss += recon_loss
        # partial_losses["loss_rec_temporal"] = recon_loss.clone().detach()

        # Should align with the adapted semantic embeddings
        semantic_temporal_alignment_loss = 0.1 * torch.abs(
            self.compute_similarity_matrix(adapted_embeddings)[:, : base_dim - temp_dim] - sim_adapted,
        ).mean()
        total_loss += semantic_temporal_alignment_loss
        partial_losses["loss_semantic_temporal_alignment"] = (
            semantic_temporal_alignment_loss.clone().detach()
        )

        return total_loss, partial_losses
