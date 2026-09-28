"""Unit tests for Option A (auto_auroc) and Option B (auto_stacking) learnable voting."""

import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from unittest.mock import MagicMock

# Ensure src/ and root are in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from four_error_using.evaluators.repetitive_validator import RepetitiveGroupValidator
from four_error_using.data_processor.data_engineer import (
    EventLockedCWTPipeline,
    DatasetItem,
    AugmentedSubset,
)
from four_error_using.models.layers.task_weighter import (
    DifferentiableTaskWeighter,
    sparsemax,
    SparsemaxFunction,
)
from four_error_using.models.layers.mil_pooler import (
    GatedAttentionMILPooler,
    pool_window_features,
)
from four_error_using.model_trainers.model_trainer import ModelTrainer
from tools.track_runs import parse_log_file


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=20):
        self.X = torch.randn(n_samples, 4, 32, 32)
        self.T = torch.randint(0, 8, (n_samples,))
        # 10 subjects (sub_00 .. sub_09): 2 samples per subject
        # Subjects 0-4 are HC (label 0), subjects 5-9 are MCI (label 1)
        sub_indices = [i // 2 for i in range(n_samples)]
        self.subject_ids = [f"sub_{s:02d}" for s in sub_indices]
        self.y = torch.tensor([0 if s < 5 else 1 for s in sub_indices], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.y[idx]


class TestLearnableVoting(unittest.TestCase):
    def setUp(self):
        self.dataset = DummyDataset()
        self.validator_manual = RepetitiveGroupValidator(
            dataset=self.dataset,
            n_splits=2,
            max_epochs=1,
            vote_mode="manual",
            task_weights={0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 0.0, 5: 1.0, 6: 2.0, 7: 3.0},
        )
        self.validator_auroc = RepetitiveGroupValidator(
            dataset=self.dataset,
            n_splits=2,
            max_epochs=1,
            vote_mode="auto_auroc",
        )
        self.validator_stacking = RepetitiveGroupValidator(
            dataset=self.dataset,
            n_splits=2,
            max_epochs=1,
            vote_mode="auto_stacking",
        )
        self.validator_true_stacking = RepetitiveGroupValidator(
            dataset=self.dataset,
            n_splits=2,
            max_epochs=1,
            vote_mode="auto_true_stacking",
        )
        self.validator_nnls = RepetitiveGroupValidator(
            dataset=self.dataset,
            n_splits=2,
            max_epochs=1,
            vote_mode="auto_nnls",
        )

    def test_init_invalid_vote_mode(self):
        with self.assertRaises(ValueError):
            RepetitiveGroupValidator(dataset=self.dataset, vote_mode="invalid_mode")

    def test_aggregate_subject_prob_manual_backward_compatibility(self):
        pairs = [(0.2, 1), (0.8, 3)]
        # task 1 weight = 0.5, task 3 weight = 1.5
        # expected: (0.5 * 0.2 + 1.5 * 0.8) / (0.5 + 1.5) = (0.1 + 1.2) / 2.0 = 0.65
        res = self.validator_manual._aggregate_subject_prob(pairs)
        self.assertAlmostEqual(res, 0.65, places=5)

        # Passing explicit weights should override self.task_weights
        custom_weights = {1: 1.0, 3: 0.0}
        res_custom = self.validator_manual._aggregate_subject_prob(pairs, weights=custom_weights)
        self.assertAlmostEqual(res_custom, 0.2, places=5)

    def test_auto_auroc_weights_basic(self):
        # Create synthetic train_records for 8 tasks
        train_records = []
        for t in range(8):
            # 5 negative samples (label 0) with fixed probs [0.40, 0.45, 0.50, 0.55, 0.60]
            for i in range(5):
                train_records.append({
                    "sid": f"s{i}",
                    "prob": 0.40 + 0.05 * i,
                    "task": t,
                    "label": 0,
                })
            # 5 positive samples (label 1) whose probs shift up with t
            for i in range(5):
                train_records.append({
                    "sid": f"s{i+5}",
                    "prob": 0.10 * t + 0.05 * i,
                    "task": t,
                    "label": 1,
                })

        weights = self.validator_auroc._compute_auto_auroc_weights(train_records, tau=0.1)
        self.assertEqual(len(weights), 8)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        # All weights should be non-negative
        self.assertTrue(all(w >= 0.0 for w in weights.values()))
        # Task 7 has highest AUROC so its weight should be strictly greatest
        self.assertEqual(max(weights, key=weights.get), 7)
        # Task 0 has lowest AUROC so its weight should be lowest
        self.assertEqual(min(weights, key=weights.get), 0)

    def test_auto_auroc_edge_case_single_class(self):
        # All samples have label=1 for task 2
        train_records = [
            {"sid": "s1", "prob": 0.8, "task": 2, "label": 1},
            {"sid": "s2", "prob": 0.6, "task": 2, "label": 1},
        ]
        weights = self.validator_auroc._compute_auto_auroc_weights(train_records, tau=0.1)
        # All tasks fall back to AUROC 0.5, so all weights should be equal (1/8 = 0.125)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        for t in range(8):
            self.assertAlmostEqual(weights[t], 0.125, places=5)

    def test_auto_stacking_weights_basic(self):
        # 10 subjects, 8 tasks
        train_records = []
        train_subjs = set()
        subj_label_map = {}
        for s_idx in range(10):
            sid = f"sub_{s_idx:02d}"
            train_subjs.add(sid)
            label = 0 if s_idx < 5 else 1
            subj_label_map[sid] = label
            for t in range(8):
                # task 6 and 7 strongly correlated with label
                if t in (6, 7):
                    prob = 0.1 if label == 0 else 0.9
                else:
                    prob = 0.5
                train_records.append({
                    "sid": sid,
                    "prob": float(prob),
                    "task": t,
                    "label": label,
                })

        weights = self.validator_stacking._compute_auto_stacking_weights(
            train_records, train_subjs, subj_label_map=subj_label_map
        )
        self.assertEqual(len(weights), 8)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        self.assertTrue(all(w >= 0.0 for w in weights.values()))
        # Tasks 6 and 7 should have higher weights than other tasks
        self.assertGreater(weights[7], weights[0])
        self.assertGreater(weights[6], weights[0])

    def test_auto_stacking_missing_task_imputation(self):
        # Subject s0 is missing task 3 (artifact rejection)
        train_records = []
        train_subjs = {"s0", "s1", "s2", "s3"}
        subj_label_map = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        for sid, label in subj_label_map.items():
            for t in range(8):
                if sid == "s0" and t == 3:
                    continue  # Missing!
                train_records.append({
                    "sid": sid,
                    "prob": 0.2 if label == 0 else 0.8,
                    "task": t,
                    "label": label,
                })

        weights = self.validator_stacking._compute_auto_stacking_weights(
            train_records, train_subjs, subj_label_map=subj_label_map
        )
        self.assertEqual(len(weights), 8)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        self.assertTrue(all(w >= 0.0 for w in weights.values()))

    def test_auto_stacking_single_class_edge_case(self):
        # Only label 0 exists
        train_records = []
        train_subjs = {"s0", "s1"}
        subj_label_map = {"s0": 0, "s1": 0}
        for sid in train_subjs:
            for t in range(8):
                train_records.append({
                    "sid": sid,
                    "prob": 0.3,
                    "task": t,
                    "label": 0,
                })

        weights = self.validator_stacking._compute_auto_stacking_weights(
            train_records, train_subjs, subj_label_map=subj_label_map
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        for t in range(8):
            self.assertAlmostEqual(weights[t], 0.125, places=5)

    def test_auto_auroc_empty_records(self):
        weights = self.validator_auroc._compute_auto_auroc_weights([], tau=0.1)
        self.assertEqual(len(weights), 8)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        for t in range(8):
            self.assertAlmostEqual(weights[t], 0.125, places=5)

    def test_auto_auroc_low_auroc_penalization(self):
        train_records = []
        # Task 0: inverted (AUROC 0.0) -> negative samples high, positive low
        for i in range(5):
            train_records.append({"sid": f"s{i}", "prob": 0.9, "task": 0, "label": 0})
            train_records.append({"sid": f"s{i+5}", "prob": 0.1, "task": 0, "label": 1})
        # Task 1: chance (AUROC 0.5)
        for i in range(5):
            train_records.append({"sid": f"s{i}", "prob": 0.5, "task": 1, "label": 0})
            train_records.append({"sid": f"s{i+5}", "prob": 0.5, "task": 1, "label": 1})
        # Task 2: separating (AUROC 1.0)
        for i in range(5):
            train_records.append({"sid": f"s{i}", "prob": 0.1, "task": 2, "label": 0})
            train_records.append({"sid": f"s{i+5}", "prob": 0.9, "task": 2, "label": 1})
        weights = self.validator_auroc._compute_auto_auroc_weights(train_records, tau=0.1)
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[1], weights[2])

    def test_auto_stacking_empty_records(self):
        weights = self.validator_stacking._compute_auto_stacking_weights([], set())
        self.assertEqual(len(weights), 8)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        for t in range(8):
            self.assertAlmostEqual(weights[t], 0.125, places=5)

    def test_auto_stacking_all_nan_task_column(self):
        train_records = []
        train_subjs = {"s0", "s1", "s2", "s3"}
        subj_label_map = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        for sid, label in subj_label_map.items():
            for t in range(7):  # Task 7 omitted completely
                train_records.append({
                    "sid": sid,
                    "prob": 0.1 if label == 0 else 0.9,
                    "task": t,
                    "label": label,
                })
        weights = self.validator_stacking._compute_auto_stacking_weights(
            train_records, train_subjs, subj_label_map=subj_label_map
        )
        self.assertEqual(len(weights), 8)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        self.assertLess(weights[7], weights[0])

    def test_aggregate_subject_prob_empty_and_zero_weight(self):
        self.assertEqual(self.validator_manual._aggregate_subject_prob([]), 0.5)
        pairs = [(0.75, 0)]  # task 0 has weight 0.0 in validator_manual
        res = self.validator_manual._aggregate_subject_prob(pairs)
        self.assertAlmostEqual(res, 0.75, places=5)

    @unittest.mock.patch("four_error_using.evaluators.repetitive_validator.ModelTrainer")
    def test_repetitive_validator_run_auto_auroc_mock(self, mock_trainer_cls):
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        def forward_mock(inputs, tasks):
            batch_size = inputs.shape[0]
            logits = torch.zeros(batch_size, 2)
            logits[:, 1] = 1.0
            return logits
        mock_model.side_effect = forward_mock
        mock_trainer.train_model.return_value = mock_model
        mock_trainer_cls.return_value = mock_trainer

        validator = RepetitiveGroupValidator(
            dataset=self.dataset,
            n_splits=1,
            max_epochs=1,
            vote_mode="auto_auroc",
            eval_batch_size=8,
        )
        res = validator.run()
        self.assertIn("learned_task_weights", res)
        self.assertEqual(len(res["learned_task_weights"]), 1)
        self.assertEqual(len(res["learned_task_weights"][0]), 8)
        self.assertIn("auroc", res)

    @unittest.mock.patch("four_error_using.evaluators.repetitive_validator.ModelTrainer")
    def test_repetitive_validator_run_auto_stacking_mock(self, mock_trainer_cls):
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        def forward_mock(inputs, tasks):
            batch_size = inputs.shape[0]
            logits = torch.zeros(batch_size, 2)
            logits[:, 1] = 1.0
            return logits
        mock_model.side_effect = forward_mock
        mock_trainer.train_model.return_value = mock_model
        mock_trainer_cls.return_value = mock_trainer

        validator = RepetitiveGroupValidator(
            dataset=self.dataset,
            n_splits=1,
            max_epochs=1,
            vote_mode="auto_stacking",
            eval_batch_size=8,
        )
        res = validator.run()
        self.assertIn("learned_task_weights", res)
        self.assertEqual(len(res["learned_task_weights"]), 1)
        self.assertEqual(len(res["learned_task_weights"][0]), 8)
        self.assertIn("auroc", res)

    @unittest.mock.patch("four_error_using.evaluators.repetitive_validator.ModelTrainer")
    def test_repetitive_validator_run_auto_true_stacking_mock(self, mock_trainer_cls):
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        def forward_mock(inputs, tasks):
            batch_size = inputs.shape[0]
            logits = torch.zeros(batch_size, 2)
            logits[:, 1] = 1.0
            return logits
        mock_model.side_effect = forward_mock
        mock_trainer.train_model.return_value = mock_model
        mock_trainer_cls.return_value = mock_trainer

        validator = RepetitiveGroupValidator(
            dataset=self.dataset,
            n_splits=1,
            max_epochs=1,
            vote_mode="auto_true_stacking",
            eval_batch_size=8,
        )
        res = validator.run()
        self.assertIn("learned_task_weights", res)
        self.assertEqual(len(res["learned_task_weights"]), 1)
        # Should record intercept and 8 task coefficients
        self.assertIn("intercept", res["learned_task_weights"][0])
        self.assertEqual(len(res["learned_task_weights"][0]), 9)
        self.assertIn("auroc", res)

    @unittest.mock.patch("four_error_using.evaluators.repetitive_validator.ModelTrainer")
    def test_repetitive_validator_run_auto_nnls_mock(self, mock_trainer_cls):
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        def forward_mock(inputs, tasks):
            batch_size = inputs.shape[0]
            logits = torch.zeros(batch_size, 2)
            logits[:, 1] = 1.0
            return logits
        mock_model.side_effect = forward_mock
        mock_trainer.train_model.return_value = mock_model
        mock_trainer_cls.return_value = mock_trainer

        validator = RepetitiveGroupValidator(
            dataset=self.dataset,
            n_splits=1,
            max_epochs=1,
            vote_mode="auto_nnls",
            eval_batch_size=8,
        )
        res = validator.run()
        self.assertIn("learned_task_weights", res)
        self.assertEqual(len(res["learned_task_weights"]), 1)
        self.assertEqual(len(res["learned_task_weights"][0]), 8)
        self.assertIn("auroc", res)

    def test_auto_nnls_sparse_weights(self):
        # 10 subjects, 8 tasks
        # Task 7 is strongly predictive: 0.1 for label 0, 0.9 for label 1
        # Task 0 is harmful/inverted: 0.9 for label 0, 0.1 for label 1
        # Other tasks are uninformative: 0.5 for all
        train_records = []
        train_subjs = set()
        subj_label_map = {}
        for s_idx in range(10):
            sid = f"sub_{s_idx:02d}"
            train_subjs.add(sid)
            label = 0 if s_idx < 5 else 1
            subj_label_map[sid] = label
            for t in range(8):
                if t == 7:
                    prob = 0.1 if label == 0 else 0.9
                elif t == 0:
                    prob = 0.9 if label == 0 else 0.1
                else:
                    prob = 0.5
                train_records.append({
                    "sid": sid,
                    "prob": float(prob),
                    "task": t,
                    "label": label,
                })

        weights = self.validator_nnls._compute_auto_nnls_weights(
            train_records, train_subjs, subj_label_map=subj_label_map
        )
        self.assertEqual(len(weights), 8)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        # Non-negative constraint
        self.assertTrue(all(w >= 0.0 for w in weights.values()))
        # Task 7 should have significant positive weight
        self.assertGreater(weights[7], 0.0)
        # Harmful Task 0 should receive an EXACT 0.0 weight from NNLS!
        self.assertEqual(weights[0], 0.0)

    def test_auto_nnls_empty_and_single_class(self):
        # Empty records fallback
        weights_empty = self.validator_nnls._compute_auto_nnls_weights([], set())
        self.assertAlmostEqual(sum(weights_empty.values()), 1.0, places=5)
        for t in range(8):
            self.assertAlmostEqual(weights_empty[t], 0.125, places=5)

        # Single class fallback
        single_class_records = [
            {"sid": "s0", "prob": 0.8, "task": 0, "label": 1},
            {"sid": "s1", "prob": 0.9, "task": 0, "label": 1},
        ]
        weights_sc = self.validator_nnls._compute_auto_nnls_weights(
            single_class_records, {"s0", "s1"}, subj_label_map={"s0": 1, "s1": 1}
        )
        self.assertAlmostEqual(sum(weights_sc.values()), 1.0, places=5)
        for t in range(8):
            self.assertAlmostEqual(weights_sc[t], 0.125, places=5)

    def test_auto_true_stacking_fit_and_inference(self):
        train_records = []
        train_subjs = set()
        subj_label_map = {}
        for s_idx in range(10):
            sid = f"sub_{s_idx:02d}"
            train_subjs.add(sid)
            label = 0 if s_idx < 5 else 1
            subj_label_map[sid] = label
            for t in range(8):
                prob = 0.1 if label == 0 else 0.9 if t == 7 else 0.5
                train_records.append({
                    "sid": sid,
                    "prob": float(prob),
                    "task": t,
                    "label": label,
                })

        clf, col_means = self.validator_true_stacking._fit_true_stacking(
            train_records, train_subjs, subj_label_map=subj_label_map
        )
        self.assertIsNotNone(clf)
        self.assertEqual(len(col_means), 8)
        self.assertEqual(len(clf.coef_[0]), 8)

        # Test inference with mock model
        mock_model = MagicMock()
        def mock_forward(x, t):
            # return high prob for label 1
            out = torch.zeros(x.shape[0], 2)
            out[:, 1] = 2.0
            return out
        mock_model.side_effect = mock_forward
        mock_model.eval.return_value = None

        test_idx = np.array([0, 1])
        subj_ids = np.array(["test_sub1", "test_sub1"])
        preds = self.validator_true_stacking._infer_subjects(
            mock_model,
            test_idx,
            subj_ids,
            fold_idx=0,
            stacking_model=clf,
            train_col_means=col_means,
        )
        self.assertIn("test_sub1", preds)
        pred_label, pred_prob = preds["test_sub1"]
        self.assertTrue(0.0 <= pred_prob <= 1.0)

    def test_auto_true_stacking_missing_task_imputation(self):
        train_records = []
        train_subjs = {"s0", "s1", "s2", "s3"}
        subj_label_map = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        for sid, label in subj_label_map.items():
            for t in range(8):
                train_records.append({
                    "sid": sid,
                    "prob": 0.2 if label == 0 else 0.8,
                    "task": t,
                    "label": label,
                })

        clf, col_means = self.validator_true_stacking._fit_true_stacking(
            train_records, train_subjs, subj_label_map=subj_label_map
        )
        self.assertIsNotNone(clf)

        # Simulate test subject with only task 0 (tasks 1..7 missing)
        mock_model = MagicMock()
        mock_model.side_effect = lambda x, t: torch.tensor([[0.0, 2.0]])
        mock_model.eval.return_value = None

        test_idx = np.array([0])
        subj_ids = np.array(["missing_sub"])
        preds = self.validator_true_stacking._infer_subjects(
            mock_model,
            test_idx,
            subj_ids,
            fold_idx=0,
            stacking_model=clf,
            train_col_means=col_means,
        )
        self.assertIn("missing_sub", preds)
        _, prob = preds["missing_sub"]
        self.assertTrue(0.0 <= prob <= 1.0)


class TestModelTrainerWeightsRestoration(unittest.TestCase):
    def test_model_trainer_restores_best_checkpoint(self):
        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(2, 2)
            def forward(self, x, t):
                return self.fc(x)

        model = SimpleNet()
        init_weight = model.fc.weight.clone()

        class TinyDS(torch.utils.data.Dataset):
            def __init__(self):
                self.y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
            def __len__(self): return 4
            def __getitem__(self, idx):
                return torch.randn(2), torch.tensor(0), self.y[idx]

        train_ds = TinyDS()
        val_ds = TinyDS()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = ModelTrainer(model, device="cpu", checkpoint_dir=tmp_dir, fold_idx=0)
            trained = trainer.train_model(train_ds, val_ds, max_epochs=3, batch_size=2, early_stop_patience=5)
            self.assertIsNotNone(trained)

            # Checkpoint file must exist
            ckpt_path = Path(tmp_dir) / "fold_00_best.pth"
            self.assertTrue(ckpt_path.exists())

            # Verify that returned model weights strictly match the saved best checkpoint weights
            saved_weights = torch.load(ckpt_path, map_location="cpu")
            for k in saved_weights:
                self.assertTrue(torch.allclose(trained.state_dict()[k], saved_weights[k]))

    def test_model_trainer_restores_in_memory_fallback(self):
        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(2, 2)
            def forward(self, x, t):
                return self.fc(x)

        model = SimpleNet()

        class TinyDS(torch.utils.data.Dataset):
            def __init__(self):
                self.y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
            def __len__(self): return 4
            def __getitem__(self, idx):
                return torch.randn(2), torch.tensor(0), self.y[idx]

        train_ds = TinyDS()
        val_ds = TinyDS()

        trainer = ModelTrainer(model, device="cpu", checkpoint_dir=None)
        trained = trainer.train_model(train_ds, val_ds, max_epochs=2, batch_size=2, early_stop_patience=2)
        self.assertIsNotNone(trained)
        self.assertEqual(trained.fc.weight.shape, torch.Size([2, 2]))


class TestHybridCWTPipeline(unittest.TestCase):
    def test_cwt_baseline_modes(self):
        # 1. Subtraction mode: all tasks subtract baseline
        p_sub = EventLockedCWTPipeline(cwt_baseline_mode="subtraction")
        self.assertEqual(p_sub.cwt_baseline_mode, "subtraction")
        self.assertTrue(p_sub.cwt_baseline_subtraction)
        for t in range(8):
            self.assertTrue(p_sub._should_subtract_baseline(t))

        # 2. Bypass mode: all tasks bypass baseline subtraction
        p_byp = EventLockedCWTPipeline(cwt_baseline_mode="bypass")
        self.assertEqual(p_byp.cwt_baseline_mode, "bypass")
        self.assertFalse(p_byp.cwt_baseline_subtraction)
        for t in range(8):
            self.assertFalse(p_byp._should_subtract_baseline(t))

        # 3. Hybrid mode: Task 0, 1 subtract baseline; Task 2..7 bypass
        p_hyb = EventLockedCWTPipeline(cwt_baseline_mode="hybrid")
        self.assertEqual(p_hyb.cwt_baseline_mode, "hybrid")
        self.assertTrue(p_hyb._should_subtract_baseline(0))
        self.assertTrue(p_hyb._should_subtract_baseline(1))
        self.assertFalse(p_hyb._should_subtract_baseline(2))
        self.assertFalse(p_hyb._should_subtract_baseline(3))
        self.assertFalse(p_hyb._should_subtract_baseline(4))
        self.assertFalse(p_hyb._should_subtract_baseline(5))
        self.assertFalse(p_hyb._should_subtract_baseline(6))
        self.assertFalse(p_hyb._should_subtract_baseline(7))

        # 4. Backward compatibility: cwt_baseline_subtraction=False maps to bypass
        p_compat = EventLockedCWTPipeline(cwt_baseline_subtraction=False)
        self.assertEqual(p_compat.cwt_baseline_mode, "bypass")
        self.assertFalse(p_compat._should_subtract_baseline(0))

        # 5. Invalid mode raises ValueError
        with self.assertRaises(ValueError):
            EventLockedCWTPipeline(cwt_baseline_mode="invalid_bl_mode")

        # 6. Config signature contains cwt_baseline_mode
        sig = p_hyb._config_signature()
        self.assertIn("cwt_baseline_mode", sig)
        self.assertEqual(sig["cwt_baseline_mode"], "hybrid")


class TestTrackRunsParsing(unittest.TestCase):
    def test_parse_vote_mode_from_log_text(self):
        log_content = (
            "2026-09-26 12:00:00 | [FULL    ] | INFO | __main__ | Run ID: 20260926_120000_full_vauroc | "
            "signal_mode=four_error | vote_mode=auto_auroc | weighted_vote=off | cwt_baseline=subtraction\n"
            "MC-CV Results [stratified-group sampling] (30 folds)\n"
            "Accuracy    : 0.850 ± 0.050\n"
            "Sensitivity : 0.820 ± 0.060\n"
            "Specificity : 0.880 ± 0.040\n"
            "AUROC       : 0.890 ± 0.030\n"
        )
        mock_path = MagicMock()
        mock_path.read_text.return_value = log_content
        mock_path.stem = "run_20260926_120000_full_vauroc"

        parsed = parse_log_file(mock_path)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["vote_mode"], "auto_auroc")
        self.assertEqual(parsed["run_id"], "20260926_120000_full_vauroc")
        self.assertEqual(parsed["cwt_baseline"], "subtraction")

    def test_parse_auto_true_stacking_and_nnls(self):
        # auto_true_stacking
        log_ts = (
            "2026-09-27 12:00:00 | [FULL    ] | INFO | __main__ | Run ID: 20260927_120000_full_vtruestack | "
            "signal_mode=four_error | vote_mode=auto_true_stacking | weighted_vote=off | cwt_baseline=hybrid\n"
            "MC-CV Results [stratified-group sampling] (15 folds)\n"
            "AUROC       : 0.895 ± 0.030\n"
        )
        mock_path = MagicMock()
        mock_path.read_text.return_value = log_ts
        mock_path.stem = "run_20260927_120000_full_vtruestack"
        parsed = parse_log_file(mock_path)
        self.assertEqual(parsed["vote_mode"], "auto_true_stacking")
        self.assertEqual(parsed["cwt_baseline"], "hybrid")

        # auto_nnls fallback from run_id
        log_nnls = (
            "2026-09-27 12:00:00 | [FULL    ] | INFO | __main__ | Run ID: 20260927_120000_full_vnnls_nocwtbl | "
            "signal_mode=four_error | weighted_vote=off\n"
            "MC-CV Results [stratified-group sampling] (15 folds)\n"
            "AUROC       : 0.880 ± 0.030\n"
        )
        mock_path.read_text.return_value = log_nnls
        mock_path.stem = "run_20260927_120000_full_vnnls_nocwtbl"
        parsed_nnls = parse_log_file(mock_path)
        self.assertEqual(parsed_nnls["vote_mode"], "auto_nnls")
        self.assertEqual(parsed_nnls["cwt_baseline"], "bypass")


class TestDetectionCallerCLI(unittest.TestCase):
    def test_default_vote_mode(self):
        from four_error_using.detection_caller.detection_caller import _parse_args
        with unittest.mock.patch("sys.argv", ["detection_caller"]):
            args = _parse_args()
            self.assertEqual(args.vote_mode, "manual")
            self.assertEqual(args.cwt_baseline_mode, "subtraction")

    def test_cli_vote_mode_auto_auroc(self):
        from four_error_using.detection_caller.detection_caller import _parse_args
        with unittest.mock.patch("sys.argv", ["detection_caller", "--vote-mode", "auto_auroc"]):
            args = _parse_args()
            self.assertEqual(args.vote_mode, "auto_auroc")

    def test_cli_vote_mode_auto_stacking(self):
        from four_error_using.detection_caller.detection_caller import _parse_args
        with unittest.mock.patch("sys.argv", ["detection_caller", "--vote-mode", "auto_stacking"]):
            args = _parse_args()
            self.assertEqual(args.vote_mode, "auto_stacking")

    def test_cli_vote_mode_auto_true_stacking(self):
        from four_error_using.detection_caller.detection_caller import _parse_args
        with unittest.mock.patch("sys.argv", ["detection_caller", "--vote-mode", "auto_true_stacking"]):
            args = _parse_args()
            self.assertEqual(args.vote_mode, "auto_true_stacking")

    def test_cli_vote_mode_auto_nnls(self):
        from four_error_using.detection_caller.detection_caller import _parse_args
        with unittest.mock.patch("sys.argv", ["detection_caller", "--vote-mode", "auto_nnls"]):
            args = _parse_args()
            self.assertEqual(args.vote_mode, "auto_nnls")

    def test_cli_vote_mode_invalid(self):
        from four_error_using.detection_caller.detection_caller import _parse_args
        with unittest.mock.patch("sys.argv", ["detection_caller", "--vote-mode", "unsupported_mode"]):
            with self.assertRaises(SystemExit):
                _parse_args()

    def test_cli_run4b_vote_weights(self):
        from four_error_using.detection_caller.detection_caller import _parse_args
        with unittest.mock.patch(
            "sys.argv",
            ["detection_caller", "--vote-weights", "0.0 0.0 0.0 1.5 0.5 1.5 1.0 3.5"]
        ):
            args = _parse_args()
            self.assertEqual(
                args.vote_weights,
                {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.5, 4: 0.5, 5: 1.5, 6: 1.0, 7: 3.5}
            )

    def test_cli_cwt_baseline_mode_hybrid(self):
        from four_error_using.detection_caller.detection_caller import _parse_args
        with unittest.mock.patch("sys.argv", ["detection_caller", "--cwt-baseline-mode", "hybrid"]):
            args = _parse_args()
            self.assertEqual(args.cwt_baseline_mode, "hybrid")

    def test_cli_hybrid_cwt_baseline_flag(self):
        from four_error_using.detection_caller.detection_caller import _parse_args
        with unittest.mock.patch("sys.argv", ["detection_caller", "--hybrid-cwt-baseline"]):
            args = _parse_args()
            self.assertTrue(args.hybrid_cwt_baseline)


class TestExperimentStages(unittest.TestCase):
    def test_stages_definitions(self):
        from scripts.run_experiments import STAGES
        self.assertIn("2", STAGES)
        self.assertIn("3a", STAGES)
        self.assertIn("3b", STAGES)
        self.assertIn("4", STAGES)

        # Stage 2: Run 4b benchmark
        s2 = STAGES["2"]
        self.assertIn("--vote-mode", s2)
        self.assertEqual(s2[s2.index("--vote-mode") + 1], "manual")
        self.assertIn("--vote-weights", s2)
        self.assertEqual(s2[s2.index("--vote-weights") + 1], "0.0 0.0 0.0 1.5 0.5 1.5 1.0 3.5")
        self.assertIn("--no-cwt-baseline", s2)

        # Stage 3a: True Stacking
        s3a = STAGES["3a"]
        self.assertIn("--vote-mode", s3a)
        self.assertEqual(s3a[s3a.index("--vote-mode") + 1], "auto_true_stacking")
        self.assertIn("--no-cwt-baseline", s3a)

        # Stage 3b: NNLS
        s3b = STAGES["3b"]
        self.assertIn("--vote-mode", s3b)
        self.assertEqual(s3b[s3b.index("--vote-mode") + 1], "auto_nnls")
        self.assertIn("--no-cwt-baseline", s3b)

        # Stage 4: Hybrid CWT Baseline
        s4 = STAGES["4"]
        self.assertIn("--cwt-baseline-mode", s4)
        self.assertEqual(s4[s4.index("--cwt-baseline-mode") + 1], "hybrid")
        self.assertIn("--vote-mode", s4)
        self.assertEqual(s4[s4.index("--vote-mode") + 1], "auto_true_stacking")

        # Stage 5a: Diff Sparsemax (Alternative A)
        self.assertIn("5a", STAGES)
        s5a = STAGES["5a"]
        self.assertIn("--vote-mode", s5a)
        self.assertEqual(s5a[s5a.index("--vote-mode") + 1], "auto_diff_sparsemax")
        self.assertIn("--no-cwt-baseline", s5a)

        # Stage 5b: Attention MIL (Alternative B)
        self.assertIn("5b", STAGES)
        s5b = STAGES["5b"]
        self.assertIn("--vote-mode", s5b)
        self.assertEqual(s5b[s5b.index("--vote-mode") + 1], "auto_attention_mil")
        self.assertIn("--no-cwt-baseline", s5b)

        # Stage 5c: Diff Softmax
        self.assertIn("5c", STAGES)
        s5c = STAGES["5c"]
        self.assertIn("--vote-mode", s5c)
        self.assertEqual(s5c[s5c.index("--vote-mode") + 1], "auto_diff_softmax")


class TestTrueStackingFallback(unittest.TestCase):
    @unittest.mock.patch("four_error_using.evaluators.repetitive_validator.ModelTrainer")
    def test_auto_true_stacking_fallback_when_fit_fails(self, mock_trainer_cls):
        # Create dataset where all subjects have the same label (single class)
        class SingleClassDS(torch.utils.data.Dataset):
            def __len__(self): return 10
            def __getitem__(self, idx):
                return torch.randn(4, 32, 32), torch.tensor(idx % 8), torch.tensor(0)
            @property
            def subject_ids(self):
                return [f"sub_{i % 3:02d}" for i in range(10)]
            @property
            def y(self):
                return torch.zeros(10, dtype=torch.long)

        ds = SingleClassDS()
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        mock_model.side_effect = lambda x, t: torch.zeros(x.shape[0], 2)
        mock_trainer.train_model.return_value = mock_model
        mock_trainer_cls.return_value = mock_trainer

        validator = RepetitiveGroupValidator(
            dataset=ds,
            n_splits=1,
            max_epochs=1,
            vote_mode="auto_true_stacking",
            eval_batch_size=4,
        )
        res = validator.run()
        # Fallback weights must be present even when stacking fit fails
        self.assertIn("learned_task_weights", res)
        self.assertEqual(len(res["learned_task_weights"]), 1)
        self.assertEqual(res["learned_task_weights"][0]["intercept"], 0.0)


class TestDatasetItemBackwardCompatibility(unittest.TestCase):
    def test_dataset_item_3_unpack(self):
        x = torch.randn(4, 32, 32)
        t = torch.tensor(2)
        y = torch.tensor(1)
        sid = "sub_01"
        item = DatasetItem(x, t, y, sid)

        # Unpacking into 3 variables should work seamlessly
        out_x, out_t, out_y = item
        self.assertTrue(torch.equal(out_x, x))
        self.assertTrue(torch.equal(out_t, t))
        self.assertTrue(torch.equal(out_y, y))

    def test_dataset_item_4_unpack(self):
        x = torch.randn(4, 32, 32)
        t = torch.tensor(3)
        y = torch.tensor(0)
        sid = "sub_05"
        item = DatasetItem(x, t, y, sid)

        # Unpacking into 4 variables should capture sid
        out_x, out_t, out_y, out_sid = item
        self.assertTrue(torch.equal(out_x, x))
        self.assertTrue(torch.equal(out_t, t))
        self.assertTrue(torch.equal(out_y, y))
        self.assertEqual(out_sid, sid)

    def test_dataset_item_properties_and_slice(self):
        x = torch.randn(2, 2)
        t = torch.tensor(1)
        y = torch.tensor(0)
        sid = "sub_09"
        item = DatasetItem(x, t, y, sid)

        self.assertTrue(torch.equal(item.X, x))
        self.assertTrue(torch.equal(item.T, t))
        self.assertTrue(torch.equal(item.y, y))
        self.assertEqual(item.sid, sid)
        self.assertEqual(len(item[:3]), 3)

    def test_augmented_subset_compatibility(self):
        class Mock4ItemDS(torch.utils.data.Dataset):
            def __len__(self): return 4
            def __getitem__(self, idx):
                return DatasetItem(torch.randn(4, 16, 16), torch.tensor(idx), torch.tensor(idx % 2), f"sub_{idx}")

        ds = Mock4ItemDS()
        aug = AugmentedSubset(ds, indices=[0, 1])

        # Test 3-unpack from AugmentedSubset
        x3, t3, y3 = aug[0]
        self.assertEqual(x3.shape, torch.Size([4, 16, 16]))
        self.assertEqual(int(t3), 0)

        # Test 4-unpack from AugmentedSubset
        x4, t4, y4, sid4 = aug[0]
        self.assertEqual(x4.shape, torch.Size([4, 16, 16]))
        self.assertEqual(sid4, "sub_0")

    def test_dataset_item_pickle_roundtrip(self):
        import pickle
        x = torch.randn(4, 16, 16)
        t = torch.tensor(1)
        y = torch.tensor(0)
        sid = "sub_test"
        item = DatasetItem(x, t, y, sid)

        data = pickle.dumps(item)
        loaded = pickle.loads(data)
        self.assertIsInstance(loaded, DatasetItem)
        self.assertTrue(torch.equal(loaded.X, x))
        self.assertTrue(torch.equal(loaded.T, t))
        self.assertTrue(torch.equal(loaded.y, y))
        self.assertEqual(loaded.sid, sid)

    def test_dataset_item_iterable_and_invalid_constructor(self):
        x = torch.randn(2, 2)
        t = torch.tensor(0)
        y = torch.tensor(1)
        sid = "sub_42"
        # From tuple
        item_from_tuple = DatasetItem((x, t, y, sid))
        self.assertEqual(item_from_tuple.sid, "sub_42")
        # From list
        item_from_list = DatasetItem([x, t, y, sid])
        self.assertEqual(item_from_list.sid, "sub_42")
        # Invalid arg count
        with self.assertRaises(TypeError):
            DatasetItem(x, t)


class TestSparsemax(unittest.TestCase):
    def test_sparsemax_simplex_constraints(self):
        z = torch.tensor([1.0, 2.0, 3.0, 0.5, -1.0, 2.5, 0.0, 1.2])
        w = sparsemax(z, dim=-1)
        self.assertEqual(w.shape, z.shape)
        # All weights non-negative
        self.assertTrue(torch.all(w >= 0.0))
        # Sum of weights strictly 1.0
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)

    def test_sparsemax_exact_zeros(self):
        # A strongly separated logit vector should produce EXACT 0.0 for low entries
        z = torch.tensor([10.0, -10.0, -20.0, -30.0])
        w = sparsemax(z, dim=-1)
        self.assertAlmostEqual(float(w[0]), 1.0, places=5)
        self.assertEqual(float(w[1]), 0.0)
        self.assertEqual(float(w[2]), 0.0)
        self.assertEqual(float(w[3]), 0.0)

    def test_sparsemax_uniform_identical_inputs(self):
        z = torch.full((8,), 2.0)
        w = sparsemax(z, dim=-1)
        for i in range(8):
            self.assertAlmostEqual(float(w[i]), 0.125, places=5)

    def test_sparsemax_batch_2d(self):
        z = torch.randn(5, 8)
        w = sparsemax(z, dim=-1)
        self.assertEqual(w.shape, torch.Size([5, 8]))
        self.assertTrue(torch.all(w >= 0.0))
        sums = w.sum(dim=-1)
        for s in sums:
            self.assertAlmostEqual(float(s), 1.0, places=5)

    def test_sparsemax_autograd_backward(self):
        z = torch.tensor([1.5, 2.0, 0.5, -0.5], requires_grad=True)
        w = sparsemax(z, dim=-1)
        loss = (w ** 2).sum()
        loss.backward()
        self.assertIsNotNone(z.grad)
        # Sum of gradients for sparsemax with respect to simplex shift is zero
        self.assertAlmostEqual(float(z.grad.sum()), 0.0, places=5)


class TestDifferentiableTaskWeighter(unittest.TestCase):
    def test_init_and_mode_selection(self):
        weighter_sparse = DifferentiableTaskWeighter(num_tasks=8, mode="sparsemax")
        self.assertEqual(weighter_sparse.mode, "sparsemax")
        weighter_soft = DifferentiableTaskWeighter(num_tasks=8, mode="softmax")
        self.assertEqual(weighter_soft.mode, "softmax")
        with self.assertRaises(ValueError):
            DifferentiableTaskWeighter(mode="invalid_weighter_mode")

    def test_forward_weights(self):
        weighter = DifferentiableTaskWeighter(num_tasks=8, mode="sparsemax")
        w = weighter.forward_weights()
        self.assertEqual(w.shape, torch.Size([8]))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)
        self.assertTrue(torch.all(w >= 0.0))

        # Dict export
        weights_dict = weighter.get_task_weights()
        self.assertEqual(len(weights_dict), 8)
        self.assertAlmostEqual(sum(weights_dict.values()), 1.0, places=5)

    def test_joint_loss_and_gradient_flow_to_theta(self):
        weighter = DifferentiableTaskWeighter(num_tasks=8, mode="sparsemax", lambda_subj=1.0, lambda_ent=0.01)
        win_logits = torch.randn(6, 2, requires_grad=True)
        tasks = torch.tensor([0, 1, 2, 7, 7, 0], dtype=torch.long)
        labels = torch.tensor([0, 0, 1, 1, 1, 0], dtype=torch.long)
        sids = ["sub_0", "sub_0", "sub_1", "sub_1", "sub_1", "sub_0"]
        win_loss = torch.tensor(0.5, requires_grad=True)

        total_loss, loss_dict = weighter(win_logits, tasks, labels, sids, win_loss)
        self.assertIn("total_loss", loss_dict)
        self.assertIn("subj_loss", loss_dict)
        self.assertIn("ent_loss", loss_dict)

        # Backward pass
        total_loss.backward()

        # Check gradients flow into theta!
        self.assertIsNotNone(weighter.theta.grad)
        self.assertTrue(torch.any(weighter.theta.grad != 0.0))
        # Check gradients flow into win_logits!
        self.assertIsNotNone(win_logits.grad)
        self.assertTrue(torch.any(win_logits.grad != 0.0))

    def test_joint_loss_softmax_mode(self):
        weighter = DifferentiableTaskWeighter(num_tasks=8, mode="softmax", lambda_subj=1.0, lambda_ent=0.01)
        win_logits = torch.randn(4, 2)
        tasks = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        sids = ["s0", "s0", "s1", "s1"]
        win_loss = torch.tensor(0.4)

        total_loss, loss_dict = weighter(win_logits, tasks, labels, sids, win_loss)
        self.assertGreater(total_loss.item(), 0.0)

    def test_zero_weight_fallback_and_no_gradient_explosion(self):
        # Construct weighter where Sparsemax forces task 0 to have exact weight 0.0
        weighter = DifferentiableTaskWeighter(num_tasks=8, mode="sparsemax", lambda_subj=1.0, lambda_ent=0.01)
        # Assign huge logit to task 1, and negative to task 0 -> task 0 weight = 0.0
        with torch.no_grad():
            weighter.theta.copy_(torch.tensor([-10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        weights = weighter.forward_weights()
        self.assertEqual(float(weights[0]), 0.0)

        # Subject s0 ONLY has windows in task 0 (which has weight 0.0)
        win_logits = torch.randn(4, 2, requires_grad=True)
        tasks = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        labels = torch.tensor([1, 1, 0, 0], dtype=torch.long)
        sids = ["s0", "s0", "s1", "s1"]
        win_loss = torch.tensor(0.5, requires_grad=True)

        total_loss, loss_dict = weighter(win_logits, tasks, labels, sids, win_loss)
        self.assertFalse(torch.isnan(total_loss))
        self.assertFalse(torch.isinf(total_loss))

        total_loss.backward()
        # Ensure theta grad is finite and does not have an explosive 10^7 gradient
        self.assertIsNotNone(weighter.theta.grad)
        self.assertFalse(torch.isnan(weighter.theta.grad).any())
        self.assertFalse(torch.isinf(weighter.theta.grad).any())
        self.assertLess(float(weighter.theta.grad.norm()), 100.0)


class TestGatedAttentionMILPooler(unittest.TestCase):
    def test_mil_pooler_shapes_and_attention(self):
        pooler = GatedAttentionMILPooler(in_features=672, hidden_dim=64)
        h_task = torch.randn(4, 8, 672)  # 4 subjects, 8 tasks
        logits, attn = pooler(h_task)
        self.assertEqual(logits.shape, torch.Size([4, 1]))
        self.assertEqual(attn.shape, torch.Size([4, 8]))

        # Attention weights must sum to 1.0 for each subject
        sums = attn.sum(dim=-1)
        for s in sums:
            self.assertAlmostEqual(float(s), 1.0, places=5)
        self.assertTrue(torch.all(attn >= 0.0))

    def test_mil_pooler_masking(self):
        pooler = GatedAttentionMILPooler(in_features=672, hidden_dim=64)
        h_task = torch.randn(2, 8, 672)
        mask = torch.ones(2, 8, dtype=torch.bool)
        # Subject 0 is missing task 3 and 5
        mask[0, 3] = False
        mask[0, 5] = False

        logits, attn = pooler(h_task, mask=mask)
        self.assertAlmostEqual(float(attn[0, 3]), 0.0, places=5)
        self.assertAlmostEqual(float(attn[0, 5]), 0.0, places=5)
        self.assertAlmostEqual(float(attn[0].sum()), 1.0, places=5)

    def test_mil_pooler_gradient_flow(self):
        pooler = GatedAttentionMILPooler(in_features=64, hidden_dim=16)
        h_task = torch.randn(2, 4, 64, requires_grad=True)
        logits, attn = pooler(h_task)
        loss = logits.sum()
        loss.backward()

        self.assertIsNotNone(h_task.grad)
        self.assertIsNotNone(pooler.attention_V.weight.grad)
        self.assertIsNotNone(pooler.attention_U.weight.grad)
        self.assertIsNotNone(pooler.attention_w.weight.grad)
        self.assertIsNotNone(pooler.classifier.weight.grad)

    def test_pool_window_features(self):
        features = torch.randn(10, 32)
        tasks = torch.tensor([0, 0, 1, 1, 2, 0, 1, 2, 3, 7])
        sids = ["s1", "s1", "s1", "s2", "s2", "s3", "s3", "s3", "s3", "s3"]

        h_task, mask, subjs = pool_window_features(features, tasks, sids, num_tasks=8)
        self.assertEqual(len(subjs), 3)  # s1, s2, s3
        self.assertEqual(h_task.shape, torch.Size([3, 8, 32]))
        self.assertEqual(mask.shape, torch.Size([3, 8]))

        # s1 has tasks 0 and 1
        s1_idx = subjs.index("s1")
        self.assertTrue(mask[s1_idx, 0])
        self.assertTrue(mask[s1_idx, 1])
        self.assertFalse(mask[s1_idx, 2])

    def test_mil_pooler_all_masked_and_single_input(self):
        pooler = GatedAttentionMILPooler(in_features=672, hidden_dim=64)
        # All tasks masked out for subject 0
        h_task = torch.randn(2, 8, 672)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[1, 0] = True  # Subject 1 has task 0

        logits, attn = pooler(h_task, mask=mask)
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isnan(attn).any())
        # All-masked subject 0 receives uniform attention 1/8
        for t in range(8):
            self.assertAlmostEqual(float(attn[0, t]), 0.125, places=4)

        # Single subject 2D input [8, 672]
        h_single = torch.randn(8, 672)
        logits_s, attn_s = pooler(h_single)
        self.assertEqual(logits_s.shape, torch.Size([1]))
        self.assertEqual(attn_s.shape, torch.Size([8]))
        self.assertAlmostEqual(float(attn_s.sum()), 1.0, places=5)

        # predict_proba
        prob = pooler.predict_proba(h_single)
        self.assertGreaterEqual(float(prob), 0.0)
        self.assertLessEqual(float(prob), 1.0)


class TestRepetitiveValidatorDiffAndMIL(unittest.TestCase):
    @unittest.mock.patch("four_error_using.evaluators.repetitive_validator.ModelTrainer")
    def test_repetitive_validator_run_auto_diff_sparsemax(self, mock_trainer_cls):
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        mock_model.side_effect = lambda x, t: torch.zeros(x.shape[0], 2)
        mock_trainer.train_model.return_value = mock_model
        mock_trainer_cls.return_value = mock_trainer

        validator = RepetitiveGroupValidator(
            dataset=DummyDataset(),
            n_splits=1,
            max_epochs=1,
            vote_mode="auto_diff_sparsemax",
            eval_batch_size=8,
        )
        res = validator.run()
        self.assertIn("learned_task_weights", res)
        self.assertEqual(len(res["learned_task_weights"]), 1)
        self.assertEqual(len(res["learned_task_weights"][0]), 8)
        self.assertAlmostEqual(sum(res["learned_task_weights"][0].values()), 1.0, places=5)

    @unittest.mock.patch("four_error_using.evaluators.repetitive_validator.ModelTrainer")
    def test_repetitive_validator_run_auto_diff_softmax(self, mock_trainer_cls):
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        mock_model.side_effect = lambda x, t: torch.zeros(x.shape[0], 2)
        mock_trainer.train_model.return_value = mock_model
        mock_trainer_cls.return_value = mock_trainer

        validator = RepetitiveGroupValidator(
            dataset=DummyDataset(),
            n_splits=1,
            max_epochs=1,
            vote_mode="auto_diff_softmax",
            eval_batch_size=8,
        )
        res = validator.run()
        self.assertIn("learned_task_weights", res)
        self.assertEqual(len(res["learned_task_weights"]), 1)
        self.assertEqual(len(res["learned_task_weights"][0]), 8)
        self.assertAlmostEqual(sum(res["learned_task_weights"][0].values()), 1.0, places=5)

    @unittest.mock.patch("four_error_using.evaluators.repetitive_validator.ModelTrainer")
    def test_repetitive_validator_run_auto_attention_mil(self, mock_trainer_cls):
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        mock_model.feature_dim = 672
        mock_model.side_effect = lambda x, t: torch.zeros(x.shape[0], 2)
        mock_model.extract_features = MagicMock(return_value=torch.zeros(8, 672))
        mock_trainer.train_model.return_value = mock_model
        mock_trainer_cls.return_value = mock_trainer

        validator = RepetitiveGroupValidator(
            dataset=DummyDataset(),
            n_splits=1,
            max_epochs=1,
            vote_mode="auto_attention_mil",
            eval_batch_size=8,
        )
        res = validator.run()
        self.assertIn("learned_task_weights", res)
        self.assertEqual(len(res["learned_task_weights"]), 1)
        self.assertEqual(len(res["learned_task_weights"][0]), 8)
        self.assertAlmostEqual(sum(res["learned_task_weights"][0].values()), 1.0, places=5)


class TestTrackRunsParsingDiffAndMIL(unittest.TestCase):
    def test_parse_diff_sparsemax_log(self):
        log_content = (
            "2026-09-28 10:00:00 | [FULL    ] | INFO | __main__ | Run ID: 20260928_100000_full_vdiffsparse | "
            "signal_mode=four_error | vote_mode=auto_diff_sparsemax | weighted_vote=off | cwt_baseline=bypass\n"
            "MC-CV Results [stratified-group sampling] (15 folds)\n"
            "AUROC       : 0.910 ± 0.025\n"
        )
        mock_path = MagicMock()
        mock_path.read_text.return_value = log_content
        mock_path.stem = "run_20260928_100000_full_vdiffsparse"

        parsed = parse_log_file(mock_path)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["vote_mode"], "auto_diff_sparsemax")

    def test_parse_diff_softmax_log(self):
        log_content = (
            "2026-09-28 10:00:00 | [FULL    ] | INFO | __main__ | Run ID: 20260928_100000_full_vdiffsoft | "
            "signal_mode=four_error | vote_mode=auto_diff_softmax | weighted_vote=off | cwt_baseline=bypass\n"
            "MC-CV Results [stratified-group sampling] (15 folds)\n"
            "AUROC       : 0.905 ± 0.025\n"
        )
        mock_path = MagicMock()
        mock_path.read_text.return_value = log_content
        mock_path.stem = "run_20260928_100000_full_vdiffsoft"

        parsed = parse_log_file(mock_path)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["vote_mode"], "auto_diff_softmax")

    def test_parse_attention_mil_log(self):
        log_content = (
            "2026-09-28 10:00:00 | [FULL    ] | INFO | __main__ | Run ID: 20260928_100000_full_vattnmil | "
            "signal_mode=four_error | vote_mode=auto_attention_mil | weighted_vote=off | cwt_baseline=bypass\n"
            "MC-CV Results [stratified-group sampling] (15 folds)\n"
            "AUROC       : 0.915 ± 0.020\n"
        )
        mock_path = MagicMock()
        mock_path.read_text.return_value = log_content
        mock_path.stem = "run_20260928_100000_full_vattnmil"

        parsed = parse_log_file(mock_path)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["vote_mode"], "auto_attention_mil")


class TestDetectionCallerCLIDiffAndMIL(unittest.TestCase):
    def test_cli_vote_mode_auto_diff_sparsemax(self):
        from four_error_using.detection_caller.detection_caller import _parse_args
        with unittest.mock.patch("sys.argv", ["detection_caller", "--vote-mode", "auto_diff_sparsemax"]):
            args = _parse_args()
            self.assertEqual(args.vote_mode, "auto_diff_sparsemax")

    def test_cli_vote_mode_auto_diff_softmax(self):
        from four_error_using.detection_caller.detection_caller import _parse_args
        with unittest.mock.patch("sys.argv", ["detection_caller", "--vote-mode", "auto_diff_softmax"]):
            args = _parse_args()
            self.assertEqual(args.vote_mode, "auto_diff_softmax")

    def test_cli_vote_mode_auto_attention_mil(self):
        from four_error_using.detection_caller.detection_caller import _parse_args
        with unittest.mock.patch("sys.argv", ["detection_caller", "--vote-mode", "auto_attention_mil"]):
            args = _parse_args()
            self.assertEqual(args.vote_mode, "auto_attention_mil")


class Dummy4ItemDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=20):
        self.X = torch.randn(n_samples, 4, 32, 32)
        self.T = torch.randint(0, 8, (n_samples,))
        sub_indices = [i // 2 for i in range(n_samples)]
        self.subject_ids = [f"sub_{s:02d}" for s in sub_indices]
        self.y = torch.tensor([0 if s < 5 else 1 for s in sub_indices], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return DatasetItem(self.X[idx], self.T[idx], self.y[idx], self.subject_ids[idx])


class TestRepetitiveValidatorWith4ItemDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = Dummy4ItemDataset()

    def test_predict_train_windows_4item_unpack_success(self):
        # This was previously broken by unpacking (inputs, tasks, labels) from a 4-item batch
        validator = RepetitiveGroupValidator(
            dataset=self.dataset,
            n_splits=1,
            max_epochs=1,
            vote_mode="auto_auroc",
            eval_batch_size=4,
        )
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.side_effect = lambda x, t: torch.zeros(x.shape[0], 2)

        train_idx = np.arange(10)
        subj_ids = np.array(self.dataset.subject_ids)
        records = validator._predict_train_windows(mock_model, train_idx, subj_ids)
        self.assertEqual(len(records), 10)
        self.assertIn("sid", records[0])
        self.assertIn("prob", records[0])
        self.assertIn("task", records[0])
        self.assertIn("label", records[0])

    @unittest.mock.patch("four_error_using.evaluators.repetitive_validator.ModelTrainer")
    def test_validator_4item_run_auto_diff_sparsemax(self, mock_trainer_cls):
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        mock_model.side_effect = lambda x, t: torch.zeros(x.shape[0], 2)
        mock_trainer.train_model.return_value = mock_model
        mock_trainer_cls.return_value = mock_trainer

        validator = RepetitiveGroupValidator(
            dataset=self.dataset,
            n_splits=1,
            max_epochs=1,
            vote_mode="auto_diff_sparsemax",
            eval_batch_size=4,
        )
        res = validator.run()
        self.assertIn("learned_task_weights", res)
        self.assertEqual(len(res["learned_task_weights"]), 1)

    @unittest.mock.patch("four_error_using.evaluators.repetitive_validator.ModelTrainer")
    def test_validator_4item_run_auto_attention_mil(self, mock_trainer_cls):
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        mock_model.feature_dim = 672
        mock_model.side_effect = lambda x, t: torch.zeros(x.shape[0], 2)
        mock_model.extract_features = MagicMock(side_effect=lambda x, t: torch.zeros(x.shape[0], 672))
        mock_trainer.train_model.return_value = mock_model
        mock_trainer_cls.return_value = mock_trainer

        validator = RepetitiveGroupValidator(
            dataset=self.dataset,
            n_splits=1,
            max_epochs=1,
            vote_mode="auto_attention_mil",
            eval_batch_size=4,
        )
        res = validator.run()
        self.assertIn("learned_task_weights", res)
        self.assertEqual(len(res["learned_task_weights"]), 1)

    @unittest.mock.patch("four_error_using.evaluators.repetitive_validator.ModelTrainer")
    def test_validator_4item_run_auto_true_stacking(self, mock_trainer_cls):
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        mock_model.side_effect = lambda x, t: torch.zeros(x.shape[0], 2)
        mock_trainer.train_model.return_value = mock_model
        mock_trainer_cls.return_value = mock_trainer

        validator = RepetitiveGroupValidator(
            dataset=self.dataset,
            n_splits=1,
            max_epochs=1,
            vote_mode="auto_true_stacking",
            eval_batch_size=4,
        )
        res = validator.run()
        self.assertIn("learned_task_weights", res)
        self.assertEqual(len(res["learned_task_weights"]), 1)


if __name__ == "__main__":
    unittest.main()



