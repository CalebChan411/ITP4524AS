"""A2: textbook linear regression and decision-tree classification.

Run from the repository root:
    python part_a/a2/wine_quality.py
    python part_a/a2/wine_quality.py --verify
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

# Plotting is non-interactive and its cache stays in a writable directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "itp4524-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "itp4524-cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import sklearn
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score,
)
from sklearn.tree import DecisionTreeClassifier, export_text

HERE = Path(__file__).resolve().parent
FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol",
]
CLASSIFIER_PARAMETERS = {"max_depth": 5, "min_samples_leaf": 5, "random_state": 42}


def load_dataset(path):
    # The teacher's actual CSV uses semicolons and spaces in column names.
    frame = pd.read_csv(path, sep=";")
    frame.columns = ["_".join(str(name).strip().split()) for name in frame.columns]
    expected = FEATURES + ["quality", "color"]
    if len(frame.columns) != len(expected) or set(frame.columns) != set(expected):
        raise ValueError("Expected the teacher's 13-column, semicolon-separated CSV.")
    frame = frame[expected].copy()
    for column in FEATURES + ["quality"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if np.isinf(frame[FEATURES + ["quality"]].to_numpy()).any():
        raise ValueError("Infinite numeric values are not supported.")
    if not frame["color"].dropna().isin(["red", "white"]).all():
        raise ValueError("Unexpected wine colour; expected red or white.")
    quality = frame["quality"].dropna()
    if not ((quality % 1 == 0) & quality.between(0, 10)).all():
        raise ValueError("Quality must be an integer score between 0 and 10.")
    return frame


def clean_dataset(frame):
    """Count on raw data first; retain first duplicates, then drop missing rows.

    The original index/order is preserved for auditing the required split.
    No source-row identifier or classification target is added before dedup.
    """
    unique = frame.drop_duplicates(keep="first")
    clean = unique.dropna().copy()
    audit = {
        "raw_rows": len(frame),
        "duplicate_rows": int(frame.duplicated(keep="first").sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "rows_with_missing": int(frame.isna().any(axis=1).sum()),
        "missing_by_column": {name: int(value) for name, value in frame.isna().sum().items()},
        "after_dedup_rows": len(unique),
        "missing_rows_removed_after_dedup": len(unique) - len(clean),
        "clean_rows": len(clean),
    }
    return clean, audit


def split_by_color(clean):
    splits = {}
    for color in ["red", "white"]:
        subset = clean.loc[clean["color"] == color].copy()
        boundary = int(len(subset) * 0.8)
        if boundary == 0 or len(subset) - boundary < 2:
            raise ValueError(f"Not enough {color} rows for training and evaluation.")
        splits[color] = (subset.iloc[:boundary].copy(), subset.iloc[boundary:].copy())
    return splits


def train_models(train):
    # Both fits use ONLY the eleven chemistry features, never quality/is_good.
    regression = LinearRegression()
    regression.fit(train[FEATURES], train["quality"])
    classification = DecisionTreeClassifier(**CLASSIFIER_PARAMETERS)
    classification.fit(train[FEATURES], (train["quality"] >= 7).astype(int))
    return {"regression": regression, "classification": classification}


def evaluate_models(models, test):
    quality_prediction = models["regression"].predict(test[FEATURES])
    good_prediction = models["classification"].predict(test[FEATURES])
    good = (test["quality"] >= 7).astype(int)
    metrics = {
        "regression": {
            "mae": float(mean_absolute_error(test["quality"], quality_prediction)),
            "mse": float(mean_squared_error(test["quality"], quality_prediction)),
            "r2": float(r2_score(test["quality"], quality_prediction)),
        },
        "classification": {
            "accuracy": float(accuracy_score(good, good_prediction)),
            "precision": float(precision_score(good, good_prediction, zero_division=0)),
            "recall": float(recall_score(good, good_prediction, zero_division=0)),
            "f1": float(f1_score(good, good_prediction, zero_division=0)),
            "confusion_matrix": confusion_matrix(good, good_prediction, labels=[0, 1]).tolist(),
        },
    }
    predictions = pd.DataFrame({
        "source_row": test.index,
        "actual_quality": test["quality"].to_numpy(),
        "predicted_quality": quality_prediction,
        "actual_good": good.to_numpy(),
        "predicted_good": good_prediction,
    })
    return metrics, predictions


def export_dataset(frame, path):
    exported = frame.copy()
    exported.insert(0, "source_row", frame.index)
    exported["is_good"] = (exported["quality"] >= 7).astype(int)
    exported.to_csv(path, index=False, lineterminator="\n")


def save_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def plot_results(color, predictions, metrics, plot_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(predictions["actual_quality"], predictions["predicted_quality"],
               alpha=0.35, s=18)
    ax.plot([2, 10], [2, 10], "k--", label="Perfect prediction")
    ax.set(xlabel="Actual quality", ylabel="Predicted quality",
           title=f"{color.capitalize()} wine: Linear Regression (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"regression_{color}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(np.array(metrics["classification"]["confusion_matrix"]),
                           display_labels=["Not good (0)", "Good (1)"]).plot(
                               ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{color.capitalize()} wine: Decision Tree (test set)")
    fig.tight_layout()
    fig.savefig(plot_dir / f"confusion_{color}.png", dpi=160)
    plt.close(fig)


def run_pipeline(input_path, output_dir):
    frame = load_dataset(input_path)
    clean, audit = clean_dataset(frame)
    audit["raw_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    for relative in ["data/raw", "data/splits", "models", "results/plots"]:
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    raw_copy = output_dir / "data/raw/winequality_with_missing.csv"
    if input_path.resolve() != raw_copy.resolve():
        shutil.copyfile(input_path, raw_copy)
    export_dataset(clean, output_dir / "data/cleaned_winequality.csv")
    save_json(output_dir / "results/cleaning_summary.json", audit)

    fig, ax = plt.subplots(figsize=(7, 4))
    quality_counts = clean.groupby(["quality", "color"]).size().unstack(fill_value=0)
    quality_counts.plot.bar(ax=ax, color={"red": "firebrick", "white": "goldenrod"})
    ax.set(xlabel="Quality score", ylabel="Number of cleaned records",
           title="Cleaned dataset: quality distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "results/plots/quality_distribution.png", dpi=160)
    plt.close(fig)

    all_metrics, split_rows, metric_rows, coefficients = {}, [], [], []
    for color, (train, test) in split_by_color(clean).items():
        export_dataset(train, output_dir / f"data/splits/{color}_train.csv")
        export_dataset(test, output_dir / f"data/splits/{color}_test.csv")
        models = train_models(train)
        for task, model in models.items():
            joblib.dump(model, output_dir / f"models/{color}_{task}.joblib")
        # Evaluate saved/reloaded models, verifying submission artifacts can run.
        models = {task: joblib.load(output_dir / f"models/{color}_{task}.joblib")
                  for task in models}
        metrics, predictions = evaluate_models(models, test)
        mean_prediction = np.full(len(test), train["quality"].mean())
        metrics["regression"]["mean_baseline_mae"] = float(mean_absolute_error(test["quality"], mean_prediction))
        metrics["regression"]["mean_baseline_mse"] = float(mean_squared_error(test["quality"], mean_prediction))
        majority_class = int((train["quality"] >= 7).mean() >= 0.5)
        metrics["classification"]["majority_baseline_accuracy"] = float(
            accuracy_score((test["quality"] >= 7).astype(int), np.full(len(test), majority_class)))
        metrics["train_rows"], metrics["test_rows"] = len(train), len(test)
        all_metrics[color] = metrics
        predictions.to_csv(output_dir / f"results/predictions_{color}.csv", index=False)
        pd.DataFrame(metrics["classification"]["confusion_matrix"],
                     index=["actual_not_good", "actual_good"],
                     columns=["predicted_not_good", "predicted_good"]).to_csv(
                         output_dir / f"results/confusion_{color}.csv", index_label="actual_label")
        summary = {
            "color": color, "clean_rows": len(train) + len(test),
            "train_rows": len(train), "test_rows": len(test),
            "train_good": int((train["quality"] >= 7).sum()),
            "test_good": int((test["quality"] >= 7).sum()),
        }
        split_rows.append(summary)
        row = {"color": color, "train_rows": len(train), "test_rows": len(test)}
        for task in ["regression", "classification"]:
            for name, value in metrics[task].items():
                if name != "confusion_matrix":
                    row[f"{task}_{name}"] = value
        metric_rows.append(row)
        for feature, coefficient in zip(FEATURES, models["regression"].coef_):
            coefficients.append({"color": color, "feature": feature, "coefficient": float(coefficient)})
        coefficients.append({"color": color, "feature": "intercept", "coefficient": float(models["regression"].intercept_)})
        (output_dir / f"results/decision_tree_rules_{color}.txt").write_text(
            export_text(models["classification"], feature_names=FEATURES), encoding="utf-8")
        plot_results(color, predictions, metrics, output_dir / "results/plots")
    save_json(output_dir / "results/metrics.json", all_metrics)
    pd.DataFrame(metric_rows).to_csv(output_dir / "results/metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "results/split_summary.csv", index=False)
    pd.DataFrame(coefficients).to_csv(output_dir / "results/linear_coefficients.csv", index=False)
    save_json(output_dir / "models/metadata.json", {
        "features": FEATURES, "positive_class": "quality >= 7", "raw_sha256": audit["raw_sha256"],
        "regression": "LinearRegression (default ordinary least squares)",
        "classifier_parameters": CLASSIFIER_PARAMETERS,
        "split": "For each colour, first floor(0.8*n) cleaned rows train; remainder test; no shuffle",
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__, "joblib": joblib.__version__,
                     "scipy": scipy.__version__},
    })
    print(json.dumps({"cleaning": audit, "metrics": all_metrics}, indent=2))
    return all_metrics


def verify_outputs(output_dir):
    raw_path = output_dir / "data/raw/winequality_with_missing.csv"
    clean, _ = clean_dataset(load_dataset(raw_path))
    expected_metrics = json.loads((output_dir / "results/metrics.json").read_text())
    metadata = json.loads((output_dir / "models/metadata.json").read_text())
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != metadata["raw_sha256"]:
        raise AssertionError("Raw data differs from the model's recorded source.")
    for color, (train, test) in split_by_color(clean).items():
        for name, expected in [("train", train), ("test", test)]:
            saved = pd.read_csv(output_dir / f"data/splits/{color}_{name}.csv")
            np.testing.assert_array_equal(saved["source_row"], expected.index)
            np.testing.assert_allclose(saved[FEATURES + ["quality"]], expected[FEATURES + ["quality"]])
            np.testing.assert_array_equal(saved["is_good"], (expected["quality"] >= 7).astype(int))
        models = {task: joblib.load(output_dir / f"models/{color}_{task}.joblib")
                  for task in ["regression", "classification"]}
        for model in models.values():
            if list(model.feature_names_in_) != FEATURES:
                raise AssertionError("Unexpected model inputs.")
        metrics, predictions = evaluate_models(models, test)
        saved = pd.read_csv(output_dir / f"results/predictions_{color}.csv")
        np.testing.assert_allclose(saved, predictions, rtol=1e-9, atol=1e-9)
        for task, values in metrics.items():
            for name, value in values.items():
                np.testing.assert_allclose(value, expected_metrics[color][task][name], rtol=1e-9, atol=1e-9)
        print(f"{color}: saved split order, models, predictions and metrics verified")
    print("Verification passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "data/raw/winequality_with_missing.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--verify", action="store_true", help="Check saved artifacts without retraining")
    args = parser.parse_args()
    if args.verify:
        verify_outputs(args.output_dir)
    else:
        run_pipeline(args.input, args.output_dir)
        subprocess.run([sys.executable, str(HERE / "build_report.py"), str(args.output_dir)], check=True)
        verify_outputs(args.output_dir)


if __name__ == "__main__":
    main()
