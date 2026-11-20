"""Verify A2 against the teacher CSV and the actual submitted artifacts.

Run the A2 training script first, then from the repository root run:
    python3 -m unittest discover -s part_a/a2/tests -v
"""

import hashlib
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier

from part_a.a2.wine_quality import (
    FEATURES,
    clean_dataset,
    load_dataset,
    split_by_color,
    train_models,
)


A2_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = A2_DIR / "data" / "raw" / "winequality_with_missing.csv"
RAW_SHA256 = "3bccfeab023bee1a43b1b697a387eab12b145b5aa3c36adbcc31cf0649925361"
EXPECTED_FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


class WineQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(RAW_PATH, sep=";")
        cls.data = load_dataset(RAW_PATH)
        cls.cleaned, cls.audit = clean_dataset(cls.data)
        cls.splits = split_by_color(cls.cleaned)

    def test_original_teacher_file_and_normalized_headers(self):
        self.assertEqual(hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(), RAW_SHA256)
        self.assertEqual(self.raw.shape, (6697, 13))
        self.assertEqual(list(FEATURES), EXPECTED_FEATURES)
        expected = self.raw.copy()
        expected.columns = [name.replace(" ", "_") for name in expected.columns]
        pd.testing.assert_frame_equal(self.data, expected)

    def test_teacher_data_cleaning_preserves_rows_and_original_order(self):
        # These counts and boundaries were independently checked on the upload.
        self.assertEqual(int(self.raw.duplicated().sum()), 1177)
        self.assertEqual(int(self.raw.isna().sum().sum()), 407)
        self.assertEqual(int(self.raw.isna().any(axis=1).sum()), 200)
        self.assertEqual(len(self.cleaned), 5320)
        expected_audit = {
            "raw_rows": 6697,
            "duplicate_rows": 1177,
            "missing_cells": 407,
            "rows_with_missing": 200,
            "after_dedup_rows": 5520,
            "missing_rows_removed_after_dedup": 200,
            "clean_rows": 5320,
        }
        for name, value in expected_audit.items():
            self.assertEqual(self.audit[name], value, name)
        self.assertEqual(
            self.audit["missing_by_column"], self.data.isna().sum().to_dict()
        )
        self.assertFalse(self.cleaned.duplicated().any())
        self.assertFalse(self.cleaned.isna().any().any())
        expected_positions = self.raw.index[
            ~self.raw.duplicated() & ~self.raw.isna().any(axis=1)
        ]
        self.assertEqual(self.cleaned.index.tolist(), expected_positions.tolist())
        pd.testing.assert_frame_equal(self.cleaned, self.data.loc[expected_positions])

    def test_cleaning_duplicate_missing_overlap_and_input_immutability(self):
        # A missing record duplicated twice must count as two raw missing rows,
        # but deduplication removes one of those rows before dropna runs.
        complete = {feature: 1.0 for feature in EXPECTED_FEATURES}
        complete.update(quality=6, color="red")
        missing = dict(complete, fixed_acidity=np.nan)
        other = dict(complete, quality=7, color="white")
        frame = pd.DataFrame(
            [complete, complete, missing, missing, other, other],
            index=[40, 10, 30, 50, 20, 90],
        )
        before = frame.copy(deep=True)
        cleaned, audit = clean_dataset(frame)
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(cleaned, before.loc[[40, 20]])
        expected_audit = {
            "raw_rows": 6,
            "duplicate_rows": 3,
            "missing_cells": 2,
            "rows_with_missing": 2,
            "after_dedup_rows": 3,
            "missing_rows_removed_after_dedup": 1,
            "clean_rows": 2,
        }
        for name, value in expected_audit.items():
            self.assertEqual(audit[name], value, name)
        self.assertEqual(audit["missing_by_column"]["fixed_acidity"], 2)

    def test_teacher_split_uses_exact_first_80_percent_per_color(self):
        expected_counts = {"red": (1087, 272), "white": (3168, 793)}
        expected_boundaries = {"red": (1313, 1314), "white": (5678, 5680)}
        expected_positives = {"red": (161, 23), "white": (694, 131)}
        self.assertEqual(set(self.splits), {"red", "white"})
        all_rows = []
        for color in ["red", "white"]:
            with self.subTest(color=color):
                train, test = self.splits[color]
                color_rows = self.cleaned.loc[self.cleaned["color"] == color]
                count_train, count_test = expected_counts[color]
                self.assertEqual((len(train), len(test)), (count_train, count_test))
                self.assertEqual(len(train), int(0.8 * len(color_rows)))
                self.assertEqual(
                    (train.index[-1], test.index[0]), expected_boundaries[color]
                )
                self.assertEqual(
                    train.index.tolist() + test.index.tolist(), color_rows.index.tolist()
                )
                self.assertTrue(set(train.index).isdisjoint(test.index))
                pd.testing.assert_frame_equal(train, color_rows.iloc[:count_train])
                pd.testing.assert_frame_equal(test, color_rows.iloc[count_train:])
                self.assertEqual(
                    (int((train.quality >= 7).sum()), int((test.quality >= 7).sum())),
                    expected_positives[color],
                )
                all_rows.extend(train.index.tolist() + test.index.tolist())
        self.assertEqual(sorted(all_rows), sorted(self.cleaned.index.tolist()))

    def test_quality_six_is_not_good_and_seven_is_good_without_target_leakage(self):
        rows = []
        for quality in [6, 7]:
            for _ in range(12):
                row = {feature: 1.0 for feature in EXPECTED_FEATURES}
                row.update(fixed_acidity=float(quality), quality=quality, color="red")
                rows.append(row)
        training = pd.DataFrame(rows)
        models = train_models(training)
        self.assertEqual(set(models), {"regression", "classification"})
        self.assertIsInstance(models["regression"], LinearRegression)
        self.assertIsInstance(models["classification"], DecisionTreeClassifier)
        for model in models.values():
            self.assertEqual(model.n_features_in_, 11)
            self.assertEqual(model.feature_names_in_.tolist(), EXPECTED_FEATURES)
        query = training.loc[[0, 12], EXPECTED_FEATURES]
        np.testing.assert_array_equal(models["classification"].predict(query), [0, 1])
        np.testing.assert_allclose(models["regression"].predict(query), [6.0, 7.0])

    def assert_export_matches(self, exported, expected):
        self.assertEqual(exported["source_row"].tolist(), expected.index.tolist())
        self.assertEqual(set(exported.columns), set(expected.columns) | {"source_row", "is_good"})
        pd.testing.assert_frame_equal(
            exported.loc[:, expected.columns].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_array_equal(exported["is_good"], (expected["quality"] >= 7).astype(int))

    def test_saved_cleaned_dataset_matches_retained_source_records(self):
        path = A2_DIR / "data" / "cleaned_winequality.csv"
        self.assertTrue(path.is_file(), "Run the A2 training script to generate artifacts first.")
        self.assert_export_matches(pd.read_csv(path), self.cleaned)

    def test_saved_splits_match_row_identities_and_training_order(self):
        for color in ["red", "white"]:
            for name, expected in zip(["train", "test"], self.splits[color]):
                with self.subTest(color=color, split=name):
                    path = A2_DIR / "data" / "splits" / f"{color}_{name}.csv"
                    self.assertTrue(path.is_file(), "Run the A2 training script first.")
                    self.assert_export_matches(pd.read_csv(path), expected)

    def test_reloaded_models_reproduce_predictions_and_reported_test_metrics(self):
        metrics_path = A2_DIR / "results" / "metrics.json"
        self.assertTrue(metrics_path.is_file(), "Run the A2 training script first.")
        metrics = json.loads(metrics_path.read_text())
        for color in ["red", "white"]:
            with self.subTest(color=color):
                _, test = self.splits[color]
                models = {
                    name: joblib.load(A2_DIR / "models" / f"{color}_{name}.joblib")
                    for name in ["regression", "classification"]
                }
                for model in models.values():
                    self.assertEqual(model.n_features_in_, 11)
                    self.assertEqual(model.feature_names_in_.tolist(), EXPECTED_FEATURES)
                quality_pred = models["regression"].predict(test[EXPECTED_FEATURES])
                good_pred = models["classification"].predict(test[EXPECTED_FEATURES])
                quality_true = test["quality"]
                good_true = (quality_true >= 7).astype(int)
                saved = pd.read_csv(A2_DIR / "results" / f"predictions_{color}.csv")
                self.assertEqual(saved["source_row"].tolist(), test.index.tolist())
                np.testing.assert_array_equal(saved["actual_quality"], quality_true)
                np.testing.assert_allclose(saved["predicted_quality"], quality_pred, rtol=1e-10)
                np.testing.assert_array_equal(saved["actual_good"], good_true)
                np.testing.assert_array_equal(saved["predicted_good"], good_pred)
                regression = {
                    "mae": mean_absolute_error(quality_true, quality_pred),
                    "mse": mean_squared_error(quality_true, quality_pred),
                    "r2": r2_score(quality_true, quality_pred),
                }
                classification = {
                    "accuracy": accuracy_score(good_true, good_pred),
                    "precision": precision_score(good_true, good_pred, zero_division=0),
                    "recall": recall_score(good_true, good_pred, zero_division=0),
                    "f1": f1_score(good_true, good_pred, zero_division=0),
                }
                for name, expected in regression.items():
                    self.assertAlmostEqual(metrics[color]["regression"][name], expected, places=10)
                for name, expected in classification.items():
                    self.assertAlmostEqual(metrics[color]["classification"][name], expected, places=10)
                self.assertEqual(
                    metrics[color]["classification"]["confusion_matrix"],
                    confusion_matrix(good_true, good_pred, labels=[0, 1]).tolist(),
                )


if __name__ == "__main__":
    unittest.main()
