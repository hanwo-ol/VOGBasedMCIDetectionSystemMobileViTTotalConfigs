"""Full-experiment (8-task) Monte-Carlo Group CV evaluator.

Mirrors the 2-experiment evaluator's layout/conventions: DI'd probe,
per-fold checkpoint dir, logger, optional AugmentedSubset on the train
split. Uses fully-qualified package imports so this works as long as
`src/` is on sys.path (set up by the entry-point caller).
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.optimize import nnls
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from four_error_using.data_processor.data_engineer import AugmentedSubset
from four_error_using.models.task_conditioned_classifier import TransferMobileViTClassifier
from four_error_using.models.layers.task_weighter import DifferentiableTaskWeighter
from four_error_using.models.layers.mil_pooler import GatedAttentionMILPooler, pool_window_features
from four_error_using.model_trainers.model_trainer import ModelTrainer, _auroc

logger = logging.getLogger(__name__)


class RepetitiveGroupValidator:
    def __init__(
        self,
        dataset,
        max_epochs: int = 500,
        batch_size: int = 32,
        n_splits: int = 30,
        probe=None,
        checkpoint_dir: Optional[Path] = None,
        num_tasks: int = 8,
        augment: bool = False,
        early_stop_patience: int = 40,
        dropout: float = 0.3,
        task_weights: Optional[dict] = None,
        vote_mode: str = "manual",
        eval_batch_size: Optional[int] = None,
        in_channels: int = 4,
        stratified: bool = False,
        strat_test_counts: Optional[dict] = None,
        kin_store: Optional[dict] = None,
        kin_weights: Optional[dict] = None,
        fuse_alphas: Optional[list] = None,
        dump_probs_path: Optional[Path] = None,
        entropy_dim: int = 0,
        backbone: str = "mobilevit-small",
    ):
        self.entropy_dim = int(entropy_dim)
        self.backbone = backbone
        # If set, every test window's out-of-fold probability is written to this CSV
        # (columns: rep, subject, task, prob, label) so any vote scheme can be
        # re-applied later without retraining.
        self.dump_probs_path = Path(dump_probs_path) if dump_probs_path is not None else None
        self._prob_rows = []
        # Optional late-fusion with a per-fold kinematic logistic gate. When
        # kin_store is given ({sid: [(feat_vec, task_id), ...]}), each fold also
        # trains a logistic on the train subjects' task-weighted kinematic vectors
        # and reports fused metrics P = alpha*P_cwt + (1-alpha)*P_kin for each
        # alpha in fuse_alphas (CWT weight). Purely additive: the CWT path is
        # unchanged when kin_store is None.
        self.kin_store = kin_store
        self.kin_weights = dict(kin_weights) if kin_weights is not None else None
        self.fuse_alphas = list(fuse_alphas) if fuse_alphas else [0.9]
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # A6000 / Ampere throughput. Inputs are a fixed 256x256 after interpolate,
        # so cuDNN autotune converges to the fastest conv kernels on the first
        # fold and every fold thereafter reuses them. TF32 accelerates the frozen
        # MobileViT matmuls/convs with negligible accuracy impact. These were left
        # off on the Jetson AGX Orin (Ampere-mobile, unified memory) build.
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self.dataset = dataset
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        # Inference runs in eval() under no_grad, so a larger batch changes nothing
        # numerically — it just saturates the 48 GB A6000. Defaults to 8x the train
        # batch (min 256) unless overridden.
        self.eval_batch_size = int(eval_batch_size) if eval_batch_size else max(256, batch_size * 8)
        self.n_splits = n_splits
        self.probe = probe
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.num_tasks = num_tasks
        self.in_channels = int(in_channels)
        self.augment = augment
        self.early_stop_patience = early_stop_patience
        self.dropout = float(dropout)
        # None → plain (unweighted) soft-vote. Dict {task_id: weight} → weighted vote.
        self.task_weights = dict(task_weights) if task_weights is not None else None

        valid_modes = (
            "manual",
            "auto_auroc",
            "auto_stacking",
            "auto_true_stacking",
            "auto_nnls",
            "auto_diff_sparsemax",
            "auto_diff_softmax",
            "auto_attention_mil",
        )
        if vote_mode not in valid_modes:
            raise ValueError(
                f"Unknown vote_mode: {vote_mode!r}. Valid modes are {valid_modes}."
            )
        self.vote_mode = vote_mode

        # Fold sampling. stratified=False → GroupShuffleSplit (random subject share,
        # class ratio drifts fold to fold). stratified=True → fixed per-class TEST
        # subject counts each fold with random membership (grouped, no leakage), so
        # every fold holds the same HC/MCI train/test composition. strat_test_counts
        # maps {class_label: n_test_subjects}; None uses a sensible per-class ~30%.
        self.stratified = bool(stratified)
        self.strat_test_counts = dict(strat_test_counts) if strat_test_counts is not None else None

    def _aggregate_subject_prob(self, pairs, weights: Optional[dict] = None):
        """pairs: list of (prob, task_id) for one subject's windows.

        If weights is set (or self.task_weights when weights is None), returns the weighted mean
        Σ(w_t · p) / Σ(w_t); otherwise the plain mean. Falls back to the
        plain mean when the weighted denominator is 0 (subject has windows
        only in zero-weight tasks — rare; logged once per occurrence)."""
        if not pairs:
            return 0.5
        active_weights = weights if weights is not None else self.task_weights
        if active_weights is None:
            return float(np.mean([p for p, _ in pairs]))
        num = sum(active_weights.get(t, 0.0) * p for p, t in pairs)
        den = sum(active_weights.get(t, 0.0) for _, t in pairs)
        if np.isnan(den) or den <= 0.0:
            logger.warning("Subject had only zero-weight tasks; falling back to unweighted mean.")
            return float(np.mean([p for p, _ in pairs]))
        return float(num / den)

    def _predict_train_windows(self, model, train_idx, subj_ids):
        """Evaluate trained model on train_idx windows under torch.no_grad().

        Returns:
            train_records: list of dict(sid=str, prob=float, task=int, label=int)
        """
        model.eval()
        train_records = []
        loader = DataLoader(
            Subset(self.dataset, train_idx),
            batch_size=self.eval_batch_size,
            shuffle=False,
            pin_memory=(self.device.type == "cuda"),
        )
        offset = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self.device, non_blocking=True)
                tasks = batch[1].to(self.device, non_blocking=True)
                labels = batch[2]
                probs = torch.softmax(model(inputs, tasks), dim=1)[:, 1].cpu().numpy()

                tasks_np = tasks.cpu().numpy()
                labels_np = labels.numpy()
                sid_slice = subj_ids[train_idx[offset:offset + len(labels)]]

                for sid, p, l, t in zip(sid_slice, probs, labels_np, tasks_np):
                    train_records.append({
                        "sid": str(sid),
                        "prob": float(p),
                        "task": int(t),
                        "label": int(l),
                    })
                offset += len(labels)
        return train_records

    def _compute_auto_auroc_weights(self, train_records, tau: float = 0.1) -> dict:
        """Option A: Performance-based temperature softmax mapping.

        For each task t in {0..num_tasks-1}:
          Collect (probs_t, labels_t). Compute roc_auc_score(labels_t, probs_t).
          Edge case: If task has only 1 class or no samples in train split, AUROC = 0.5.
        w_t = exp((AUROC_t - 0.5) / tau) / sum_j(exp((AUROC_j - 0.5) / tau))
        """
        aurocs = np.zeros(self.num_tasks, dtype=np.float64)
        tau = max(1e-6, float(tau))
        for t in range(self.num_tasks):
            t_probs = [r["prob"] for r in train_records if r["task"] == t]
            t_labels = [r["label"] for r in train_records if r["task"] == t]
            if len(t_labels) == 0 or len(np.unique(t_labels)) < 2:
                aurocs[t] = 0.5
            else:
                try:
                    score = float(roc_auc_score(t_labels, t_probs))
                    aurocs[t] = 0.5 if np.isnan(score) else score
                except Exception:
                    aurocs[t] = 0.5

        logits = (aurocs - 0.5) / tau
        exp_logits = np.exp(logits - np.max(logits))
        sum_exp = np.sum(exp_logits)
        if sum_exp <= 0.0 or np.isnan(sum_exp):
            weights = np.ones(self.num_tasks, dtype=np.float64) / self.num_tasks
        else:
            weights = exp_logits / sum_exp
        return {t: float(weights[t]) for t in range(self.num_tasks)}

    def _prepare_subj_feature_matrix(
        self, train_records, train_subjs, subj_label_map: Optional[dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct subject-level mean predicted probability matrix X [n_train_subjs, num_tasks],
        target vector y [n_train_subjs], and training column means [num_tasks].
        """
        subj_list = sorted([str(s) for s in train_subjs])
        n_subjs = len(subj_list)
        if n_subjs == 0:
            return (
                np.empty((0, self.num_tasks), dtype=np.float64),
                np.empty(0, dtype=int),
                np.full(self.num_tasks, 0.5, dtype=np.float64),
            )

        X = np.full((n_subjs, self.num_tasks), np.nan, dtype=np.float64)
        y = np.zeros(n_subjs, dtype=int)

        by_subj = defaultdict(list)
        for r in train_records:
            by_subj[r["sid"]].append(r)

        for i, sid in enumerate(subj_list):
            records_s = by_subj.get(sid, [])
            if subj_label_map is not None and sid in subj_label_map:
                y[i] = int(subj_label_map[sid])
            elif subj_label_map is not None and str(sid) in subj_label_map:
                y[i] = int(subj_label_map[str(sid)])
            elif records_s:
                y[i] = int(records_s[0]["label"])
            for t in range(self.num_tasks):
                p_st = [r["prob"] for r in records_s if r["task"] == t]
                if p_st:
                    X[i, t] = float(np.mean(p_st))

        col_means = np.full(self.num_tasks, 0.5, dtype=np.float64)
        for t in range(self.num_tasks):
            col = X[:, t]
            valid_col = col[~np.isnan(col)]
            col_mean = float(np.mean(valid_col)) if len(valid_col) > 0 else 0.5
            col_means[t] = col_mean
            X[np.isnan(col), t] = col_mean

        return X, y, col_means

    def _compute_auto_stacking_weights(
        self, train_records, train_subjs, subj_label_map: Optional[dict] = None
    ) -> dict:
        """Option B: L2-regularized logistic regression stacking with Softmax mapping.

        Extracts coefficients beta in R^num_tasks and maps via Softmax:
            w_t = exp(beta_t) / sum_j(exp(beta_j)).
        """
        X, y, _ = self._prepare_subj_feature_matrix(train_records, train_subjs, subj_label_map=subj_label_map)
        n_subjs = len(y)
        if n_subjs == 0 or len(np.unique(y)) < 2:
            weights = np.ones(self.num_tasks, dtype=np.float64) / self.num_tasks
            return {t: float(weights[t]) for t in range(self.num_tasks)}

        try:
            clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
            clf.fit(X, y)
            beta = clf.coef_[0]
            if np.any(np.isnan(beta)):
                beta = np.zeros(self.num_tasks, dtype=np.float64)
        except Exception as e:
            logger.warning("LogisticRegression stacking failed (%s); falling back to uniform weights.", e)
            beta = np.zeros(self.num_tasks, dtype=np.float64)

        exp_beta = np.exp(beta - np.max(beta))
        sum_exp = np.sum(exp_beta)
        if sum_exp <= 0.0 or np.isnan(sum_exp):
            weights = np.ones(self.num_tasks, dtype=np.float64) / self.num_tasks
        else:
            weights = exp_beta / sum_exp
        return {t: float(weights[t]) for t in range(self.num_tasks)}

    def _compute_auto_nnls_weights(
        self, train_records, train_subjs, subj_label_map: Optional[dict] = None
    ) -> dict:
        """Non-Negative Least Squares (NNLS) sparse weight estimation.

        Solves min_w ||X w - y||_2^2 s.t. w >= 0, then normalizes sum(w) = 1.
        Allows harmful tasks to receive an exact weight of 0.0.
        """
        X, y, _ = self._prepare_subj_feature_matrix(train_records, train_subjs, subj_label_map=subj_label_map)
        n_subjs = len(y)
        if n_subjs == 0 or len(np.unique(y)) < 2:
            weights = np.ones(self.num_tasks, dtype=np.float64) / self.num_tasks
            return {t: float(weights[t]) for t in range(self.num_tasks)}

        try:
            w, _ = nnls(X, y.astype(np.float64))
            sum_w = float(np.sum(w))
            if np.isnan(sum_w) or sum_w <= 0.0:
                weights = np.ones(self.num_tasks, dtype=np.float64) / self.num_tasks
            else:
                weights = w / sum_w
        except Exception as e:
            logger.warning("NNLS weight optimization failed (%s); falling back to uniform weights.", e)
            weights = np.ones(self.num_tasks, dtype=np.float64) / self.num_tasks

        return {t: float(weights[t]) for t in range(self.num_tasks)}

    def _fit_true_stacking(
        self, train_records, train_subjs, subj_label_map: Optional[dict] = None
    ) -> Tuple[Optional[LogisticRegression], np.ndarray]:
        """Fit unconstrained L2 LogisticRegression meta-classifier on training subjects (X, y).

        Returns:
            (clf, train_col_means): Fitted classifier and column means for test imputation.
        """
        X, y, col_means = self._prepare_subj_feature_matrix(train_records, train_subjs, subj_label_map=subj_label_map)
        n_subjs = len(y)
        if n_subjs == 0 or len(np.unique(y)) < 2:
            return None, col_means

        try:
            clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
            clf.fit(X, y)
            return clf, col_means
        except Exception as e:
            logger.warning("True stacking fit failed (%s); fallback to unweighted voting.", e)
            return None, col_means

    def _fit_attention_mil(
        self,
        mil_pooler: GatedAttentionMILPooler,
        model: torch.nn.Module,
        train_idx: np.ndarray,
        subj_ids: np.ndarray,
        y_arr: np.ndarray,
        epochs: int = 60,
        lr: float = 1e-3,
    ) -> dict:
        """Fit GatedAttentionMILPooler on train subjects using frozen model features.

        Returns:
            mean_attn_weights: dict mapping task t -> average attention weight across train subjects.
        """
        model.eval()
        loader = DataLoader(
            Subset(self.dataset, train_idx),
            batch_size=self.eval_batch_size,
            shuffle=False,
            pin_memory=(self.device.type == "cuda"),
        )
        all_feats = []
        all_tasks = []
        offset = 0
        sids_train = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self.device, non_blocking=True)
                tasks = batch[1].to(self.device, non_blocking=True)
                if hasattr(model, "extract_features"):
                    feats = model.extract_features(inputs, tasks).detach().to(self.device)
                else:
                    feats = torch.zeros(inputs.size(0), getattr(mil_pooler, "in_features", 672), device=self.device)
                all_feats.append(feats)
                all_tasks.append(tasks)
                sid_slice = subj_ids[train_idx[offset:offset + len(inputs)]]
                sids_train.extend([str(s) for s in sid_slice])
                offset += len(inputs)

        if not all_feats:
            return {t: 1.0 / self.num_tasks for t in range(self.num_tasks)}

        feats_cat = torch.cat(all_feats, dim=0)
        tasks_cat = torch.cat(all_tasks, dim=0)

        H_train, mask_train, subjs = pool_window_features(
            feats_cat, tasks_cat, sids_train, num_tasks=self.num_tasks
        )
        H_train = H_train.to(self.device)
        mask_train = mask_train.to(self.device)
        s2l = {str(sid): int(lab) for sid, lab in zip(subj_ids[train_idx], y_arr[train_idx])}
        targets = torch.tensor([s2l[s] for s in subjs], dtype=torch.float32, device=self.device)

        if len(subjs) == 0 or len(torch.unique(targets)) < 2:
            return {t: 1.0 / self.num_tasks for t in range(self.num_tasks)}

        optimizer = torch.optim.AdamW(mil_pooler.parameters(), lr=lr, weight_decay=1e-2)
        criterion = nn.BCEWithLogitsLoss()

        mil_pooler.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits, _ = mil_pooler(H_train, mask=mask_train)
            loss = criterion(logits.squeeze(-1), targets)
            loss.backward()
            optimizer.step()

        mil_pooler.eval()
        with torch.no_grad():
            _, attn = mil_pooler(H_train, mask=mask_train)
            mean_attn = attn.mean(dim=0).cpu().numpy()
            sum_attn = float(np.sum(mean_attn))
            if sum_attn > 0:
                mean_attn = mean_attn / sum_attn
            else:
                mean_attn = np.ones(self.num_tasks) / self.num_tasks

        return {t: float(mean_attn[t]) for t in range(self.num_tasks)}

    def _infer_subjects(
        self,
        model,
        test_idx,
        subj_ids,
        fold_idx,
        weights: Optional[dict] = None,
        stacking_model: Optional[LogisticRegression] = None,
        train_col_means: Optional[np.ndarray] = None,
        mil_pooler: Optional[GatedAttentionMILPooler] = None,
    ):
        """Score every test window, then collapse to one probability per subject.

        Runs the model on the fold's test windows, maps each window back to its
        subject (via subj_ids at the true test index — the fix for the old
        offset bug), optionally records rows for window_probs.csv / the probe,
        and returns {subject: (label, predicted prob)}.

        If mil_pooler is supplied (auto_attention_mil), the subject probability
        is evaluated by hierarchical gated attention pooling over test window representations.
        If stacking_model and train_col_means are supplied (auto_true_stacking),
        the subject probability is directly evaluated via logistic meta-regression.
        Otherwise, uses weighted soft-voting via _aggregate_subject_prob.
        """
        model.eval()
        subj_probs: dict = defaultdict(list)  # sid -> list of (prob, task_id)
        subj_labels: dict = {}
        test_feats = []
        test_tasks = []
        test_sids = []

        loader = DataLoader(
            Subset(self.dataset, test_idx),
            batch_size=self.eval_batch_size,
            shuffle=False,
            pin_memory=(self.device.type == "cuda"),
        )
        offset = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self.device, non_blocking=True)
                tasks = batch[1].to(self.device, non_blocking=True)
                labels = batch[2]
                probs = torch.softmax(model(inputs, tasks), dim=1)[:, 1].cpu().numpy()

                tasks_np = tasks.cpu().numpy()
                labels_np = labels.numpy()
                sid_slice = subj_ids[test_idx[offset:offset + len(labels)]]

                if mil_pooler is not None:
                    if hasattr(model, "extract_features"):
                        feats = model.extract_features(inputs, tasks).detach()
                    else:
                        feats = torch.zeros(inputs.size(0), getattr(mil_pooler, "in_features", 672), device=self.device)
                    test_feats.append(feats)
                    test_tasks.append(tasks)
                    test_sids.extend([str(s) for s in sid_slice])

                if self.probe is not None:
                    for i in range(len(labels_np)):
                        self.probe.add_window(
                            fold_idx=fold_idx,
                            subject_id=str(sid_slice[i]),
                            task_id=int(tasks_np[i]),
                            true_label=int(labels_np[i]),
                            prob=float(probs[i]),
                        )

                for sid, p, l, t in zip(sid_slice, probs, labels_np, tasks_np):
                    subj_probs[sid].append((float(p), int(t)))
                    subj_labels[sid] = int(l)
                    if self.dump_probs_path is not None:
                        self._prob_rows.append((fold_idx, str(sid), int(t), float(p), int(l)))
                offset += len(labels)

        if mil_pooler is not None and test_feats:
            res = {}
            feats_cat = torch.cat(test_feats, dim=0)
            tasks_cat = torch.cat(test_tasks, dim=0)
            H_test, mask_test, test_subjs_order = pool_window_features(
                feats_cat, tasks_cat, test_sids, num_tasks=self.num_tasks
            )
            mil_pooler.eval()
            with torch.no_grad():
                probs = mil_pooler.predict_proba(H_test, mask=mask_test).cpu().squeeze(-1).numpy()
                if probs.ndim == 0:
                    probs = np.array([float(probs)])
            for sid, prob in zip(test_subjs_order, probs):
                res[sid] = (subj_labels[sid], float(prob))
            for sid in subj_probs:
                if sid not in res:
                    res[sid] = (subj_labels[sid], self._aggregate_subject_prob(subj_probs[sid], weights=weights))
            return res

        if stacking_model is not None and train_col_means is not None:
            res = {}
            for sid in subj_probs:
                p_vec = np.array(train_col_means, dtype=np.float64).copy()
                for t in range(self.num_tasks):
                    p_st = [p for p, task in subj_probs[sid] if task == t]
                    if p_st:
                        p_vec[t] = float(np.mean(p_st))
                try:
                    c1_idx = int(np.where(stacking_model.classes_ == 1)[0][0]) if 1 in stacking_model.classes_ else 1
                    p_pred = float(stacking_model.predict_proba(p_vec[None, :])[0, c1_idx])
                except Exception as e:
                    logger.warning("Stacking predict_proba failed (%s); fallback to unweighted mean", e)
                    p_pred = float(np.mean([p for p, _ in subj_probs[sid]]))
                res[sid] = (subj_labels[sid], p_pred)
            return res

        return {
            sid: (subj_labels[sid], self._aggregate_subject_prob(subj_probs[sid], weights=weights))
            for sid in subj_probs
        }

    def _kin_vec(self, trials):
        """Task-weighted mean kinematic vector for one subject's trials."""
        feats = np.stack([f for f, _ in trials]).astype(np.float64)
        if self.kin_weights is None:
            return feats.mean(0)
        w = np.array([self.kin_weights.get(t, 0.0) for _, t in trials])
        return (feats * w[:, None]).sum(0) / w.sum() if w.sum() > 0 else feats.mean(0)

    def _stratified_group_splits(self, y_arr, subj_ids):
        """Yield (train_idx, test_idx) window-index arrays for each fold with a
        FIXED per-class test-subject count and random membership.

        Subjects are grouped (all a subject's windows move together → no leakage)
        and, per class label, a fixed number of subjects (self.strat_test_counts,
        defaulting to round(30% ) per class) is drawn into the test split each
        fold. A single seeded RNG advanced across folds keeps the 30 folds
        distinct yet fully reproducible."""
        # Each subject has one label; map subject -> class.
        subj_to_label = {}
        for sid, y in zip(subj_ids, y_arr):
            subj_to_label[sid] = int(y)
        per_class = defaultdict(list)
        for sid, lab in subj_to_label.items():
            per_class[lab].append(sid)
        for lab in per_class:
            per_class[lab] = np.array(sorted(per_class[lab]))  # deterministic base order

        # Resolve test counts per class (default ~30%, min 1 kept in each side).
        counts = {}
        for lab, subs in per_class.items():
            if self.strat_test_counts is not None and lab in self.strat_test_counts:
                n_test = int(self.strat_test_counts[lab])
            else:
                n_test = int(round(0.3 * len(subs)))
            n_test = max(1, min(n_test, len(subs) - 1))  # never empty train or test side
            counts[lab] = n_test

        rng = np.random.RandomState(42)
        for _ in range(self.n_splits):
            test_subjs = set()
            for lab, subs in per_class.items():
                perm = rng.permutation(subs)
                test_subjs.update(perm[:counts[lab]].tolist())
            test_mask = np.array([sid in test_subjs for sid in subj_ids])
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            yield train_idx, test_idx

    def run(self):
        """The main loop: for each of n_splits stratified folds, train a fresh model,
        score subjects, apply the weighted vote (+ optional kinematic fusion), and
        average AUROC / sensitivity / specificity ± SD across folds. Dumps
        window_probs.csv at the end if a path was given.
        """
        y_arr = self.dataset.y.numpy()
        subj_ids = np.array(self.dataset.subject_ids)
        indices = np.arange(len(self.dataset))

        fold_metrics = []
        fold_learned_weights = []
        # optional kinematic-gate fusion accumulators
        _fuse = self.kin_store is not None
        fuse_metrics = {a: [] for a in self.fuse_alphas} if _fuse else None
        kin_only_metrics = [] if _fuse else None
        subj_label_map = {str(sid): int(lab) for sid, lab in zip(subj_ids, y_arr)}
        sampling = "stratified-group" if self.stratified else "grouped (unstratified)"
        logger.info(
            "MC Group CV [%s] — %d folds | device=%s | num_tasks=%d | augment=%s | dropout=%.2f | vote_mode=%s | weighted_vote=%s",
            sampling, self.n_splits, self.device, self.num_tasks, self.augment, self.dropout,
            self.vote_mode,
            (self.task_weights if self.task_weights is not None else "off"),
        )

        if self.stratified:
            splits = list(self._stratified_group_splits(y_arr, subj_ids))
        else:
            gss = GroupShuffleSplit(n_splits=self.n_splits, test_size=0.3, random_state=42)
            splits = list(gss.split(indices, y_arr, groups=subj_ids))

        for fold, (train_idx, test_idx) in enumerate(splits):
            train_subjs = set(subj_ids[train_idx])
            test_subjs = set(subj_ids[test_idx])
            assert train_subjs.isdisjoint(test_subjs), "Subject leakage detected!"

            # Realized per-class (HC=0, MCI=1) SUBJECT counts, so the stratification
            # is verifiable in the log rather than merely asserted.
            def _cls_subj_counts(idx):
                s2l = {sid: int(lab) for sid, lab in zip(subj_ids[idx], y_arr[idx])}
                hc = sum(1 for v in s2l.values() if v == 0)
                return hc, len(s2l) - hc
            tr_hc, tr_mci = _cls_subj_counts(train_idx)
            te_hc, te_mci = _cls_subj_counts(test_idx)

            logger.info(
                "=== Fold %02d/%d | train: %d subj (HC=%d MCI=%d) / %d win | test: %d subj (HC=%d MCI=%d) / %d win ===",
                fold + 1, self.n_splits,
                len(train_subjs), tr_hc, tr_mci, len(train_idx),
                len(test_subjs), te_hc, te_mci, len(test_idx),
            )

            model = TransferMobileViTClassifier(
                num_classes=2, in_channels=self.in_channels, num_tasks=self.num_tasks,
                dropout=self.dropout, entropy_dim=self.entropy_dim, backbone=self.backbone,
            )
            task_weighter = None
            if self.vote_mode in ("auto_diff_sparsemax", "auto_diff_softmax"):
                mode = "sparsemax" if self.vote_mode == "auto_diff_sparsemax" else "softmax"
                task_weighter = DifferentiableTaskWeighter(
                    num_tasks=self.num_tasks,
                    mode=mode,
                    tau=1.0,
                    lambda_subj=1.0,
                    lambda_ent=0.01,
                )

            trainer = ModelTrainer(
                model,
                device=self.device,
                checkpoint_dir=self.checkpoint_dir,
                fold_idx=fold,
                task_weighter=task_weighter,
            )

            train_subset = (
                AugmentedSubset(self.dataset, train_idx)
                if self.augment
                else Subset(self.dataset, train_idx)
            )
            test_subset = Subset(self.dataset, test_idx)

            trained_model = trainer.train_model(
                train_subset,
                test_subset,
                max_epochs=self.max_epochs,
                batch_size=self.batch_size,
                early_stop_patience=self.early_stop_patience,
                eval_batch_size=self.eval_batch_size,
            )

            fold_task_weights = None
            stacking_model = None
            train_col_means = None
            mil_pooler = None

            if self.vote_mode == "manual":
                fold_task_weights = self.task_weights
            elif self.vote_mode in ("auto_auroc", "auto_stacking", "auto_nnls"):
                train_records = self._predict_train_windows(trained_model, train_idx, subj_ids)
                if self.vote_mode == "auto_auroc":
                    fold_task_weights = self._compute_auto_auroc_weights(train_records, tau=0.1)
                elif self.vote_mode == "auto_stacking":
                    fold_task_weights = self._compute_auto_stacking_weights(
                        train_records, train_subjs, subj_label_map=subj_label_map
                    )
                elif self.vote_mode == "auto_nnls":
                    fold_task_weights = self._compute_auto_nnls_weights(
                        train_records, train_subjs, subj_label_map=subj_label_map
                    )
                fold_learned_weights.append(fold_task_weights)
                w_str = ", ".join(f"T{t}:{fold_task_weights[t]:.3f}" for t in range(self.num_tasks))
                logger.info("   Fold %02d Learned Task Weights (%s): [%s]", fold + 1, self.vote_mode, w_str)
            elif self.vote_mode == "auto_true_stacking":
                train_records = self._predict_train_windows(trained_model, train_idx, subj_ids)
                stacking_model, train_col_means = self._fit_true_stacking(
                    train_records, train_subjs, subj_label_map=subj_label_map
                )
                if stacking_model is not None:
                    b0 = float(stacking_model.intercept_[0])
                    coefs = {t: float(stacking_model.coef_[0][t]) for t in range(self.num_tasks)}
                    fold_learned_weights.append({"intercept": b0, **coefs})
                    coef_str = ", ".join(f"T{t}:{coefs[t]:+.3f}" for t in range(self.num_tasks))
                    logger.info("   Fold %02d True Stacking Model: b0=%+.3f, Coefs: [%s]",
                                fold + 1, b0, coef_str)
                else:
                    fallback_weights = {"intercept": 0.0, **{t: 0.0 for t in range(self.num_tasks)}}
                    fold_learned_weights.append(fallback_weights)
                    logger.warning("   Fold %02d True Stacking Model fit failed; falling back to unweighted voting.", fold + 1)
            elif self.vote_mode in ("auto_diff_sparsemax", "auto_diff_softmax"):
                if task_weighter is not None:
                    fold_task_weights = task_weighter.get_task_weights()
                else:
                    fold_task_weights = {t: 1.0 / self.num_tasks for t in range(self.num_tasks)}
                fold_learned_weights.append(fold_task_weights)
                w_str = ", ".join(f"T{t}:{fold_task_weights[t]:.3f}" for t in range(self.num_tasks))
                logger.info("   Fold %02d Learned Task Weights (%s): [%s]", fold + 1, self.vote_mode, w_str)
            elif self.vote_mode == "auto_attention_mil":
                mil_pooler = GatedAttentionMILPooler(
                    in_features=getattr(trained_model, "feature_dim", 672),
                    hidden_dim=64,
                ).to(self.device)
                fold_task_weights = self._fit_attention_mil(
                    mil_pooler, trained_model, train_idx, subj_ids, y_arr
                )
                fold_learned_weights.append(fold_task_weights)
                w_str = ", ".join(f"T{t}:{fold_task_weights[t]:.3f}" for t in range(self.num_tasks))
                logger.info("   Fold %02d Learned Task Attention Weights (%s): [%s]", fold + 1, self.vote_mode, w_str)
            else:
                raise ValueError(f"Unknown vote_mode: {self.vote_mode!r}")

            subj_preds = self._infer_subjects(
                trained_model,
                test_idx,
                subj_ids,
                fold_idx=fold,
                weights=fold_task_weights,
                stacking_model=stacking_model,
                train_col_means=train_col_means,
                mil_pooler=mil_pooler,
            )

            true_arr = np.array([v[0] for v in subj_preds.values()])
            prob_arr = np.array([v[1] for v in subj_preds.values()])
            pred_arr = (prob_arr > 0.5).astype(int)

            tp = int(np.sum((true_arr == 1) & (pred_arr == 1)))
            tn = int(np.sum((true_arr == 0) & (pred_arr == 0)))
            fp = int(np.sum((true_arr == 0) & (pred_arr == 1)))
            fn = int(np.sum((true_arr == 1) & (pred_arr == 0)))

            acc = (tp + tn) / len(true_arr)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            auroc = _auroc(true_arr, prob_arr)

            fold_metrics.append({'acc': acc, 'sens': sens, 'spec': spec, 'auroc': auroc})
            logger.info(
                "-> Fold %02d Result | Acc=%.3f Sens=%.3f Spec=%.3f AUROC=%.3f",
                fold + 1, acc, sens, spec, auroc,
            )

            # ---- optional kinematic-gate late fusion (additive; CWT path above unchanged) ----
            if _fuse:
                tr_sids = [s for s in train_subjs if s in self.kin_store]
                Xtr = np.stack([self._kin_vec(self.kin_store[s]) for s in tr_sids])
                ytr = np.array([subj_label_map[s] for s in tr_sids])
                sc = StandardScaler().fit(Xtr)
                clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(sc.transform(Xtr), ytr)
                pkin = {sid: (float(clf.predict_proba(sc.transform(self._kin_vec(self.kin_store[sid])[None]))[0, 1])
                              if sid in self.kin_store else 0.5) for sid in subj_preds}

                def _fold_m(prob_by_sid):
                    yy = np.array([subj_preds[s][0] for s in subj_preds])
                    pp = np.array([prob_by_sid[s] for s in subj_preds])
                    prd = (pp > 0.5).astype(int)
                    tp = int(((yy == 1) & (prd == 1)).sum()); tn = int(((yy == 0) & (prd == 0)).sum())
                    fp = int(((yy == 0) & (prd == 1)).sum()); fn = int(((yy == 1) & (prd == 0)).sum())
                    return {'acc': (tp + tn) / len(yy), 'sens': tp / (tp + fn) if tp + fn else 0.0,
                            'spec': tn / (tn + fp) if tn + fp else 0.0, 'auroc': _auroc(yy, pp)}

                kin_only_metrics.append(_fold_m(pkin))
                for a in self.fuse_alphas:
                    fused = {s: a * subj_preds[s][1] + (1 - a) * pkin[s] for s in subj_preds}
                    fuse_metrics[a].append(_fold_m(fused))
                logger.info("   Fold %02d Fusion | KIN AUROC=%.3f | fused@%.2f AUROC=%.3f",
                            fold + 1, kin_only_metrics[-1]['auroc'],
                            self.fuse_alphas[0], fuse_metrics[self.fuse_alphas[0]][-1]['auroc'])

        accs = [m['acc'] for m in fold_metrics]
        senss = [m['sens'] for m in fold_metrics]
        specs = [m['spec'] for m in fold_metrics]
        aurocs = [m['auroc'] for m in fold_metrics]

        logger.info("=" * 60)
        logger.info("  MC-CV Results [%s sampling] (%d folds)", sampling, self.n_splits)
        logger.info("  Accuracy    : %.3f ± %.3f", float(np.mean(accs)), float(np.std(accs)))
        logger.info("  Sensitivity : %.3f ± %.3f", float(np.mean(senss)), float(np.std(senss)))
        logger.info("  Specificity : %.3f ± %.3f", float(np.mean(specs)), float(np.std(specs)))
        logger.info("  AUROC       : %.3f ± %.3f", float(np.mean(aurocs)), float(np.std(aurocs)))
        if self.vote_mode != "manual" and fold_learned_weights:
            if self.vote_mode == "auto_true_stacking":
                b0_vals = [fw["intercept"] for fw in fold_learned_weights if "intercept" in fw]
                task_strs = []
                if b0_vals:
                    task_strs.append(f"Intercept (b0): {float(np.mean(b0_vals)):+.3f} +/- {float(np.std(b0_vals)):.3f}")
                for t in range(self.num_tasks):
                    w_t = [fw[t] for fw in fold_learned_weights if t in fw]
                    if w_t:
                        task_strs.append(f"Task {t} (b{t}): {float(np.mean(w_t)):+.3f} +/- {float(np.std(w_t)):.3f}")
                logger.info("  Learned Stacking Coefficients across folds:\n    %s", "\n    ".join(task_strs))
            else:
                task_strs = []
                for t in range(self.num_tasks):
                    w_t = [fw[t] for fw in fold_learned_weights if t in fw]
                    if w_t:
                        task_strs.append(f"Task {t}: {float(np.mean(w_t)):.3f} +/- {float(np.std(w_t)):.3f}")
                logger.info("  Learned Task Weights across folds: %s", ", ".join(task_strs))
        logger.info("=" * 60)

        if _fuse:
            def _agg(ms):
                return {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms])))
                        for k in ('acc', 'sens', 'spec', 'auroc')}
            ko = _agg(kin_only_metrics)
            logger.info("  [FUSION] weights=%s  vote=%s", self.kin_weights, self.task_weights)
            logger.info("  [FUSION] CWT-only    AUROC %.3f ± %.3f", float(np.mean(aurocs)), float(np.std(aurocs)))
            logger.info("  [FUSION] KIN-only    AUROC %.3f ± %.3f  Sens %.3f  Spec %.3f",
                        ko['auroc'][0], ko['auroc'][1], ko['sens'][0], ko['spec'][0])
            best_a, best_au = None, -1.0
            for a in self.fuse_alphas:
                fa = _agg(fuse_metrics[a])
                logger.info("  [FUSION] alpha=%.2f (CWT:%.2f/KIN:%.2f)  AUROC %.3f ± %.3f  Sens %.3f  Spec %.3f  Acc %.3f",
                            a, a, 1 - a, fa['auroc'][0], fa['auroc'][1], fa['sens'][0], fa['spec'][0], fa['acc'][0])
                if fa['auroc'][0] > best_au:
                    best_a, best_au = a, fa['auroc'][0]
            logger.info("  [FUSION] best fused AUROC %.3f at alpha=%.2f", best_au, best_a)
            logger.info("=" * 60)

        if self.dump_probs_path is not None and self._prob_rows:
            self.dump_probs_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.dump_probs_path, "w") as fh:
                fh.write("rep,subject,task,prob,label\n")
                for rep, sid, t, p, l in self._prob_rows:
                    fh.write(f"{rep},{sid},{t},{p:.6f},{l}\n")
            logger.info("Saved %d window probabilities -> %s",
                        len(self._prob_rows), self.dump_probs_path)

        if self.probe is not None:
            logger.info("Evaluator lifecycle end: writing probe report...")
            self.probe.generate_markdown_report()

        res = {
            'accuracy': (float(np.mean(accs)), float(np.std(accs))),
            'sensitivity': (float(np.mean(senss)), float(np.std(senss))),
            'specificity': (float(np.mean(specs)), float(np.std(specs))),
            'auroc': (float(np.mean(aurocs)), float(np.std(aurocs))),
            'fold_metrics': fold_metrics,
        }
        if self.vote_mode != "manual" and fold_learned_weights:
            res['learned_task_weights'] = fold_learned_weights
        return res
