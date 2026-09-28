"""Hierarchical Gated Attention Multiple Instance Learning (MIL) Pooler.

References:
    - Ilse, Tomczak, & Welling, "Attention-based Deep Multiple Instance Learning",
      ICML 2018 (AMLab-Amsterdam/AttentionDeepMIL, MIT License).
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class GatedAttentionMILPooler(nn.Module):
    """Hierarchical task-level Gated Attention MIL Pooler.

    Given 672-dim representations per window:
    1) Windows within task t are pooled to h_{s, t} in R^672.
    2) Gated Attention: a_{s, t} = Softmax_t(w^T (tanh(V h_{s, t}) * sigmoid(U h_{s, t}))).
    3) Subject embedding: z_s = sum_t a_{s, t} h_{s, t}.
    4) Subject classification: P_hat_s = sigmoid(W_cls z_s + b).
    """

    def __init__(
        self,
        in_features: int = 672,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_dim = int(hidden_dim)

        # Gated attention layers: a = softmax(w^T (tanh(V h) * sigmoid(U h)))
        self.attention_V = nn.Linear(self.in_features, self.hidden_dim, bias=False)
        self.attention_U = nn.Linear(self.in_features, self.hidden_dim, bias=False)
        self.attention_w = nn.Linear(self.hidden_dim, 1, bias=False)

        # Subject classification head
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.in_features, 1)

    def forward(
        self,
        h_task: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute subject logits and gated task attention weights.

        Args:
            h_task: [B, num_tasks, in_features] or [num_tasks, in_features].
            mask: Optional boolean tensor [B, num_tasks] or [num_tasks], True where task is present.

        Returns:
            logits: [B, 1] (or [1] if single input).
            attn_weights: [B, num_tasks] (or [num_tasks] if single input), sum(attn_weights) == 1.
        """
        is_single = (h_task.dim() == 2)
        if is_single:
            h_task = h_task.unsqueeze(0)  # [1, num_tasks, D]
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)  # [1, num_tasks]

        param = next(self.parameters(), None)
        if param is not None:
            if h_task.device != param.device:
                h_task = h_task.to(param.device)
            if mask is not None and mask.device != param.device:
                mask = mask.to(param.device)

        # V(h): [B, T, L], U(h): [B, T, L]
        v_out = torch.tanh(self.attention_V(h_task))
        u_out = torch.sigmoid(self.attention_U(h_task))
        gated = v_out * u_out  # [B, T, L]

        # Attention scores: [B, T]
        scores = self.attention_w(gated).squeeze(-1)  # [B, T]

        if mask is not None:
            # Missing tasks receive very large negative score so attention is 0.0
            scores = scores.masked_fill(~mask, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)  # [B, T]

        if mask is not None:
            # Handle edge case where a subject has all tasks masked
            all_masked = (~mask).all(dim=-1, keepdim=True)
            if all_masked.any():
                uniform = torch.ones_like(attn_weights) / attn_weights.size(-1)
                attn_weights = torch.where(all_masked, uniform, attn_weights)

        # Subject embedding: z_s = sum_t a_{s, t} h_{s, t} -> [B, D]
        z_s = torch.bmm(attn_weights.unsqueeze(1), h_task).squeeze(1)

        z_drop = self.dropout(z_s)
        logits = self.classifier(z_drop)  # [B, 1]

        if is_single:
            logits = logits.squeeze(0)
            attn_weights = attn_weights.squeeze(0)

        return logits, attn_weights

    def predict_proba(
        self,
        h_task: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute subject probability P_hat_s = sigmoid(logits) in [0, 1]."""
        logits, _ = self.forward(h_task, mask=mask)
        return torch.sigmoid(logits)


def pool_window_features(
    features: torch.Tensor,
    tasks: torch.Tensor,
    sids: Union[List[str], np.ndarray, torch.Tensor],
    num_tasks: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Pool window-level representations to subject-task representations h_{s, t}.

    Args:
        features: [N_win, D] window feature representations.
        tasks: [N_win] task integer IDs.
        sids: [N_win] subject identifiers.
        num_tasks: Number of unique tasks (default 8).

    Returns:
        h_task: [N_subj, num_tasks, D] pooled representations.
        mask: [N_subj, num_tasks] boolean tensor indicating task presence.
        subj_list: List of sorted unique subject IDs.
    """
    sid_strs = [str(s) for s in sids]
    subj_list = sorted(list(set(sid_strs)))
    n_subjs = len(subj_list)
    d = features.size(1)
    device = features.device

    h_task = torch.zeros(n_subjs, num_tasks, d, device=device, dtype=features.dtype)
    mask = torch.zeros(n_subjs, num_tasks, device=device, dtype=torch.bool)

    subj_task_indices = defaultdict(lambda: defaultdict(list))
    for idx, (sid_str, t) in enumerate(zip(sid_strs, tasks)):
        subj_task_indices[sid_str][int(t)].append(idx)

    for i, sid in enumerate(subj_list):
        for t in range(num_tasks):
            idxs = subj_task_indices[sid].get(t, [])
            if idxs:
                idx_tensor = torch.as_tensor(idxs, dtype=torch.long, device=device)
                h_task[i, t] = features[idx_tensor].mean(dim=0)
                mask[i, t] = True

    return h_task, mask, subj_list
