"""A2: clean wine data, train models and save the required results.

Run: python wine_quality.py
"""

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
    "pH", "sulphates", "alcohol",
]


def analyse_wine(wine, color):
    # The classification label is created after cleaning.
    wine = wine.copy()
    wine["is_good"] = (wine["quality"] >= 7).astype(int)

    # Keep the original order: first 80% train, remaining 20% test.
    boundary = int(len(wine) * 0.8)
    train = wine.iloc[:boundary]
    test = wine.iloc[boundary:]
    train.to_csv(BASE / f"data/splits/{color}_train.csv", index=False)
    test.to_csv(BASE / f"data/splits/{color}_test.csv", index=False)

    # Use only the eleven chemistry features, not quality or is_good.
    X_train = train[FEATURES]
    X_test = test[FEATURES]

    regression = LinearRegression()
    regression.fit(X_train, train["quality"])
    quality_prediction = regression.predict(X_test)

    # Fixed limits reduce tree complexity; they are not tuned on the test set.
    classification = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=5, random_state=42
    )
    classification.fit(X_train, train["is_good"])
    good_prediction = classification.predict(X_test)

    joblib.dump(regression, BASE / f"models/{color}_regression.joblib")
    joblib.dump(classification, BASE / f"models/{color}_classification.joblib")

    # Good wine (1) is the positive class for precision, recall and F1.
    scores = {
        "color": color,
        "train_rows": len(train),
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
    pd.DataFrame(matrix, index=[0, 1], columns=[0, 1]).to_csv(
        BASE / f"results/confusion_{color}.csv", index_label="actual/predicted"
    )
    fig, ax = plt.subplots(figsize=(4, 3.3))
    ConfusionMatrixDisplay(
        matrix, display_labels=["Not good (0)", "Good (1)"]
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{color.capitalize()} wine (test set)")
    fig.tight_layout()
    fig.savefig(BASE / f"results/plots/confusion_{color}.png", dpi=160)
    plt.close(fig)
    print(f"\n{color}: {len(train)} training rows, {len(test)} testing rows")
    print("Confusion matrix (actual rows, predicted columns; order: 0, 1):")
    print(matrix)
    return scores


def main():
    for folder in ["data/splits", "models", "results/plots"]:
        (BASE / folder).mkdir(parents=True, exist_ok=True)

    # The teacher's file uses semicolons and spaces in its column names.
    data = pd.read_csv(BASE / "data/raw/winequality_with_missing.csv", sep=";")
    data.columns = data.columns.str.replace(" ", "_", regex=False)

    duplicate_rows = int(data.duplicated().sum())
    missing_cells = int(data.isna().sum().sum())
    missing_rows = int(data.isna().any(axis=1).sum())
    unique = data.drop_duplicates(keep="first")
    clean = unique.dropna()
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

    # This distribution includes all cleaned records, before the train/test split.
    distribution = clean.groupby(["quality", "color"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    distribution.plot.bar(ax=ax, color={"red": "firebrick", "white": "goldenrod"}, rot=0)
    ax.set(title="Cleaned wine data: quality distribution",
           xlabel="Quality score", ylabel="Number of records")
    fig.tight_layout()
    fig.savefig(BASE / "results/plots/quality_distribution.png", dpi=160)
    plt.close(fig)

    results = []
    for color in ["red", "white"]:
        wine = clean[clean["color"] == color]
        results.append(analyse_wine(wine, color))

    metrics = pd.DataFrame(results)
    metrics.to_csv(BASE / "results/metrics.csv", index=False)
    print("\nTest-set results:")
    print(metrics.to_string(index=False))
    print("\nThree charts saved to:", BASE / "results/plots")


if __name__ == "__main__":
    main()
