"""Clean the supplied wine data and train one regression and one classifier."""

from pathlib import Path

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save figures without opening extra windows.
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay,
)

BASE = Path(__file__).resolve().parent
FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "color_code",
]


def split_by_color(clean):
    parts = {}
    for color in ["red", "white"]:
        wine = clean[clean["color"] == color]
        # Apply the required sequential split separately within each colour.
        boundary = int(len(wine) * 0.8)
        train = wine.iloc[:boundary].copy()
        test = wine.iloc[boundary:].copy()
        train.to_csv(BASE / f"data/splits/{color}_train.csv", index=False)
        test.to_csv(BASE / f"data/splits/{color}_test.csv", index=False)
        parts[color] = (train, test)
    return parts


def evaluate_wine(test, regression, classification, color, training_rows):
    # Evaluate the same two fitted models; no training takes place here.
    quality_prediction = regression.predict(test[FEATURES])
    good_prediction = classification.predict(test[FEATURES])
    scores = {
        "color": color,
        "train_rows": training_rows,
        "test_rows": len(test),
        "MAE": mean_absolute_error(test["quality"], quality_prediction),
        "MSE": mean_squared_error(test["quality"], quality_prediction),
        "R2": r2_score(test["quality"], quality_prediction),
        "Accuracy": accuracy_score(test["is_good"], good_prediction),
        "Precision": precision_score(test["is_good"], good_prediction, zero_division=0),
        "Recall": recall_score(test["is_good"], good_prediction, zero_division=0),
        "F1": f1_score(test["is_good"], good_prediction, zero_division=0),
    }
    matrix = confusion_matrix(test["is_good"], good_prediction, labels=[0, 1])
    if color in ["red", "white"]:
        pd.DataFrame(matrix, index=[0, 1], columns=[0, 1]).to_csv(
            BASE / f"results/confusion_{color}.csv", index_label="actual/predicted"
        )
        fig, ax = plt.subplots(figsize=(4, 3.3))
        ConfusionMatrixDisplay(
            matrix, display_labels=["Not good (0)", "Good (1)"]
        ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"Shared classifier: {color} wine (test set)")
        fig.tight_layout()
        fig.savefig(BASE / f"results/plots/confusion_{color}.png", dpi=160)
        plt.close(fig)
    print(f"\n{color}: shared models fitted on {training_rows} rows; {len(test)} test rows")
    print("Confusion matrix (actual rows, predicted columns; order: 0, 1):")
    print(matrix)
    return scores


def main():
    for folder in ["data/splits", "models", "results/plots"]:
        (BASE / folder).mkdir(parents=True, exist_ok=True)

    # Keep the teacher-supplied source file unchanged.
    data = pd.read_csv(BASE / "data/raw/winequality_with_missing.csv", sep=";")
    data.columns = data.columns.str.replace(" ", "_", regex=False)
    duplicate_rows = int(data.duplicated().sum())
    missing_cells = int(data.isna().sum().sum())
    missing_rows = int(data.isna().any(axis=1).sum())
    unique = data.drop_duplicates(keep="first")
    clean = unique.dropna().copy()
    cleaning = pd.DataFrame([
        {"item": "raw_rows", "count": len(data)},
        {"item": "duplicate_rows", "count": duplicate_rows},
        {"item": "missing_cells", "count": missing_cells},
        {"item": "rows_with_missing", "count": missing_rows},
        {"item": "after_dedup_rows", "count": len(unique)},
        {"item": "missing_rows_removed_after_dedup", "count": len(unique) - len(clean)},
        {"item": "clean_rows", "count": len(clean)},
    ])
    cleaning.to_csv(BASE / "results/cleaning.csv", index=False)
    print(cleaning.to_string(index=False))

    # Colour is a binary input; quality and is_good are targets, never features.
    clean["color_code"] = clean["color"].map({"red": 0, "white": 1})
    clean["is_good"] = (clean["quality"] >= 7).astype(int)
    parts = split_by_color(clean)

    # Pool only training records after each colour's split. Test records stay out.
    train = pd.concat([parts["red"][0], parts["white"][0]], ignore_index=True)
    test = pd.concat([parts["red"][1], parts["white"][1]], ignore_index=True)
    regression = LinearRegression()
    regression.fit(train[FEATURES], train["quality"])
    # Fixed settings control complexity; they are not selected using test results.
    classification = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=5, random_state=42
    )
    classification.fit(train[FEATURES], train["is_good"])
    joblib.dump(regression, BASE / "models/regression.joblib")
    joblib.dump(classification, BASE / "models/classification.joblib")

    distribution = clean.groupby(["quality", "color"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    distribution.plot.bar(ax=ax, color={"red": "firebrick", "white": "goldenrod"}, rot=0)
    ax.set(title="Cleaned wine data: quality distribution",
           xlabel="Quality score", ylabel="Number of records")
    fig.tight_layout()
    fig.savefig(BASE / "results/plots/quality_distribution.png", dpi=160)
    plt.close(fig)

    results = [evaluate_wine(test, regression, classification, "all", len(train))]
    for color in ["red", "white"]:
        results.append(evaluate_wine(parts[color][1], regression, classification, color, len(train)))
    metrics = pd.DataFrame(results)
    metrics.to_csv(BASE / "results/metrics.csv", index=False)
    print("\nTest-set results (the same two models for all three rows):")
    print(metrics.to_string(index=False))
    print("\nSaved two models: models/regression.joblib and models/classification.joblib")
    print("Three charts saved to:", BASE / "results/plots")


if __name__ == "__main__":
    main()
