"""Differentiable Task Weighter (Sparsemax & Softmax) with Joint Subject Loss.

References:
    - Martins & Astudillo, "From Softmax to Sparsemax: A Sparse Model of Attention
      and Multi-Label Classification", ICML 2016.
    - Reference open-source: deep-spin/entmax & KrisKorrel/sparsemax-pytorch (MIT License).
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class SparsemaxFunction(torch.autograd.Function):
    """Exact Sparsemax forward and backward implementation following Martins et al. (ICML 2016)."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        ctx.dim = dim
        # Translation invariance: subtract max along dim for numerical stability
        input_shift = input - input.max(dim=dim, keepdim=True)[0]

        # Sort descending
        z_sorted, _ = torch.sort(input_shift, descending=True, dim=dim)

        k_size = input_shift.size(dim)
        shape = [1] * input_shift.dim()
        shape[dim] = k_size
        range_tensor = torch.arange(1, k_size + 1, device=input.device, dtype=input.dtype).view(*shape)

        # Support condition: 1 + k * z_(k) > sum_{j=1}^k z_(j)
        bound = 1.0 + range_tensor * z_sorted
        cumsum = torch.cumsum(z_sorted, dim=dim)
        support = bound > cumsum

        # k(z) is the maximum index where the support condition holds (at least 1)
        k_max = torch.max(
            torch.where(support, range_tensor, torch.ones_like(range_tensor)),
            dim=dim,
            keepdim=True,
        )[0]

        # Threshold tau(z) = (sum_{j=1}^{k(z)} z_(j) - 1) / k(z)
        cumsum_k = torch.gather(cumsum, dim, (k_max - 1).long())
        tau = (cumsum_k - 1.0) / k_max

        output = torch.clamp(input_shift - tau, min=0.0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output, = ctx.saved_tensors
        dim = ctx.dim
        nonzeros = (output > 0).to(grad_output.dtype)
        sum_grad = torch.sum(grad_output * nonzeros, dim=dim, keepdim=True)
        count = torch.sum(nonzeros, dim=dim, keepdim=True).clamp(min=1.0)
        grad_input = nonzeros * (grad_output - sum_grad / count)
        return grad_input, None


def sparsemax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax activation function."""
    return SparsemaxFunction.apply(input, dim)


class DifferentiableTaskWeighter(nn.Module):
    """Learnable task weighter parameterized by theta in R^num_tasks.

    Weights are projected via Sparsemax or Softmax:
        w = Sparsemax(theta / tau)  or  Softmax(theta / tau),
    satisfying w_t >= 0 and sum(w_t) = 1.0, with exact 0.0 assignment for harmful tasks under Sparsemax.

    Joint Subject Loss:
        L_total = L_win(win_logits, win_labels) + lambda_subj * BCE(P_hat_s, y_s) + lambda_ent * sum_t w_t * log(w_t).
    """

    def __init__(
        self,
        num_tasks: int = 8,
        mode: str = "sparsemax",
        tau: float = 1.0,
        lambda_subj: float = 1.0,
        lambda_ent: float = 0.01,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.num_tasks = int(num_tasks)
        if mode not in ("sparsemax", "softmax"):
            raise ValueError(f"Unknown mode {mode!r}. Valid modes are 'sparsemax', 'softmax'.")
        self.mode = mode
        self.tau = float(tau)
        self.lambda_subj = float(lambda_subj)
        self.lambda_ent = float(lambda_ent)
        self.eps = float(eps)

        # Learnable task parameter vector theta in R^num_tasks (initialized to zeros -> uniform)
        self.theta = nn.Parameter(torch.zeros(self.num_tasks, dtype=torch.float32))

    def forward_weights(self) -> torch.Tensor:
        """Compute simplex-constrained weight vector w in R^num_tasks."""
        scaled_theta = self.theta / max(self.tau, 1e-6)
        if self.mode == "sparsemax":
            return sparsemax(scaled_theta, dim=-1)
        else:
            return torch.softmax(scaled_theta, dim=-1)

    def get_task_weights(self) -> Dict[int, float]:
        """Return learned task weights as a Python dictionary {t: float} for evaluation."""
        with torch.no_grad():
            w = self.forward_weights().detach().cpu().numpy()
        return {t: float(w[t]) for t in range(self.num_tasks)}

    def forward(
        self,
        win_logits: torch.Tensor,
        tasks: torch.Tensor,
        labels: torch.Tensor,
        sids: Union[List[str], np.ndarray, torch.Tensor],
        win_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute joint total loss: L_win + lambda_subj * L_subj + lambda_ent * L_ent.

        Args:
            win_logits: Model window logits [B, 2] or [B].
            tasks: Task ID per window [B].
            labels: True ground-truth label per window [B] (0=HC, 1=MCI).
            sids: Subject ID per window (length B).
            win_loss: Window-level CrossEntropyLoss scalar tensor.

        Returns:
            total_loss: Differentiable scalar tensor.
            loss_dict: Dictionary with 'total_loss', 'win_loss', 'subj_loss', 'ent_loss'.
        """
        w = self.forward_weights()  # [num_tasks]

        if win_logits.dim() == 2 and win_logits.size(1) == 2:
            win_probs = torch.softmax(win_logits, dim=-1)[:, 1]
        else:
            win_probs = torch.sigmoid(win_logits.view(-1))

        # Group window indices by subject to form subject-level predicted probabilities
        unique_sids: List[str] = []
        sid_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, sid in enumerate(sids):
            sid_str = str(sid)
            if sid_str not in sid_to_indices:
                unique_sids.append(sid_str)
            sid_to_indices[sid_str].append(i)

        subj_losses = []
        for sid_str in unique_sids:
            idxs = sid_to_indices[sid_str]
            sub_tasks = tasks[idxs]
            sub_probs = win_probs[idxs]
            sub_weights = w[sub_tasks]

            w_sum = sub_weights.sum()
            weighted_prob_sum = (sub_weights * sub_probs).sum()
            mean_prob = sub_probs.mean()

            # Smooth fallback to unweighted mean if w_sum == 0 (avoids 1/eps gradient spike)
            if w_sum > 1e-6:
                p_hat_s = weighted_prob_sum / w_sum
            else:
                p_hat_s = mean_prob
            p_hat_s = torch.clamp(p_hat_s, min=self.eps, max=1.0 - self.eps)

            y_s = labels[idxs[0]].to(dtype=torch.float32)
            bce_s = -(y_s * torch.log(p_hat_s) + (1.0 - y_s) * torch.log(1.0 - p_hat_s))
            subj_losses.append(bce_s)

        if subj_losses:
            loss_subj = torch.stack(subj_losses).mean()
        else:
            loss_subj = torch.tensor(0.0, device=win_logits.device)

        # Entropy penalty: sum_t w_t * log(w_t)
        if self.mode == "sparsemax":
            pos_mask = w > 1e-12
            if pos_mask.any():
                loss_ent = (w[pos_mask] * torch.log(w[pos_mask])).sum()
            else:
                loss_ent = torch.tensor(0.0, device=w.device)
        else:
            loss_ent = (w * torch.log(torch.clamp(w, min=1e-12))).sum()

        total_loss = win_loss + self.lambda_subj * loss_subj + self.lambda_ent * loss_ent

        loss_dict = {
            "total_loss": float(total_loss.detach().item()),
            "win_loss": float(win_loss.detach().item()),
            "subj_loss": float(loss_subj.detach().item()),
            "ent_loss": float(loss_ent.detach().item()),
        }
        return total_loss, loss_dict
