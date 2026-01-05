import torch
import torch.nn.functional as F
from torch import nn
from pdb import set_trace as st
from tevatron.louis.src.utils import norm


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
            
        sim_target.fill_diagonal_(0.0)
        N = embeddings.size(0)
        total_loss = 0.0
        partial_losses = {}
        

        for m in m_list:
            sim_adapted = self.compute_similarity_matrix(adapted_embeddings[:, :m])
            diff = torch.abs(sim_target - sim_adapted)
            loss_m = diff.sum()
            total_loss += loss_m
            partial_losses[f"loss_pair_partial_{m}"] = loss_m.detach().clone()
            partial_losses[f"loss_pair_partial_{m}"] /= embeddings.shape[0]

        total_loss /= (N * len(m_list))

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
            partial_losses[f"loss_topk_partial_{m}"] = loss_m.detach().clone()
            partial_losses[f"loss_topk_partial_{m}"] /= embeddings.shape[0]

        total_loss /= (N * len(m_list))
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
    
            partial_losses[f"loss_rank_partial_{m}"] = loss.mean().detach().clone()

            total_loss += loss.mean()

        return total_loss, partial_losses


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
        partial_losses["loss_pairwise_temporal"] = pairwiseloss.detach().clone()

        recon_loss = torch.abs(temporal_embeddings - adapted_temporal_embeddings).mean()

        # total_loss += recon_loss
        # partial_losses["loss_rec_temporal"] = recon_loss.detach().clone()

        # Should align with the adapted semantic embeddings
        semantic_temporal_alignment_loss = 0.1 * torch.abs(
            self.compute_similarity_matrix(adapted_embeddings)[:, : base_dim - temp_dim] - sim_adapted,
        ).mean()
        total_loss += semantic_temporal_alignment_loss
        partial_losses["loss_semantic_temporal_alignment"] = (
            semantic_temporal_alignment_loss.detach().clone()
        )

        return total_loss, partial_losses


class DistillLoss(PairwiseSimilarityLossBase):
    def __init__(self, k=10, m_list=[64, 128, 256, 512]):
        super().__init__()
        self.k = k
        self.m_list = m_list
        
    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def forward(
        self, embeddings, target_dim, q=True,
    ):
        N = embeddings[target_dim].size(0)
        total_loss = 0.0
        partial_losses = {}

        if N <= self.k:
            # If the batch is too small, ignore this loss
            for m in self.m_list:
                partial_losses[m] = torch.tensor(0.0)
            return torch.tensor(0.0), partial_losses

        with torch.no_grad():
            sim_target = self.compute_similarity(embeddings[target_dim], embeddings[target_dim])
            topk_values, topk_indices = torch.topk(sim_target, k=self.k, dim=1)

        for m in self.m_list:
            adapted_trunc = embeddings[m]

            sim_adapted = self.compute_similarity(adapted_trunc, adapted_trunc)
            sim_adapted = sim_adapted.gather(dim=1, index=topk_indices)

            loss_m = torch.abs(sim_adapted - topk_values).sum()
            
            total_loss += loss_m
            partial_losses[f'loss_{"q" if q else "p"}_topk_partial_{m}'] = loss_m.detach().clone()
            partial_losses[f'loss_{"q" if q else "p"}_topk_partial_{m}'] /= (embeddings[target_dim].shape[0] * len(self.m_list))
            
        del sim_target, topk_values

        total_loss /= (embeddings[target_dim].shape[0] * len(self.m_list))
        return total_loss, partial_losses


class LinearCKALoss(nn.Module):
    def __init__(self, m_list=[64, 128, 256, 512]):
        super().__init__()
        self.m_list = m_list
        # self.cache_var1 = None

    # ---------- Centering ----------
    @staticmethod
    def centering(K):
        """
        Double-center a Gram matrix: H K H
        H = I - 1/n
        """
        n = K.size(0)
        unit = torch.ones((n, n), device=K.device)
        I = torch.eye(n, device=K.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    # ---------- HSIC ----------
    def linear_HSIC(self, X, Y):
        Lx = self.centering(torch.matmul(X, X.T))
        Ly = self.centering(torch.matmul(Y, Y.T))
        return torch.sum(Lx * Ly)
    # ---------- CKA ----------
    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))
        
        return hsic / (var1 * var2)
    
    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    # ---------- Loss ----------
    def forward(self, embeddings, target_dim, q=True):
        total_loss = 0.0
        partial_losses = {}

        for m in self.m_list:
            cka_loss = 1.0 - self.linear_CKA(
                embeddings[m],
                embeddings[target_dim], 
            )
            partial_losses[f"loss_{'q' if q else 'p'}_cka_partial_{m}"] = cka_loss.detach().clone()
            total_loss += cka_loss
        
        return total_loss, partial_losses
