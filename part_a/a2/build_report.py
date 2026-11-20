"""Build the A2 Markdown and Word reports from the current experiment files.

Run: python3 part_a/a2/build_report.py [OUTPUT_DIR]
Report generation uses python-docx; the prediction models do not depend on it.
"""

import csv
import json
import re
import sys
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def number(value):
    return f"{float(value):.6f}"


def percent(value):
    return f"{100 * float(value):.2f}%"


def build_markdown(output):
    results = output / "results"
    cleaning = json.loads((results / "cleaning_summary.json").read_text())
    metrics = json.loads((results / "metrics.json").read_text())
    metadata = json.loads((output / "models" / "metadata.json").read_text())
    split_rows = read_csv(results / "split_summary.csv")
    coefficients = read_csv(results / "linear_coefficients.csv")
    if not split_rows or not coefficients:
        raise ValueError("The split summary and regression coefficients must not be empty.")
    split_lookup = {row["color"]: row for row in split_rows}
    features = metadata["features"]
    if len(features) != 11:
        raise ValueError("The A2 report expects the 11 numerical chemistry features.")
    parameters = metadata["classifier_parameters"]
    red, white = metrics["red"], metrics["white"]
    split_table = []
    regression_table = []
    classification_table = []
    missing_table = []
    confusion_tables = []
    observations = []
    for column, count in cleaning["missing_by_column"].items():
        missing_table.append(f"| {column} | {int(count)} |")
    for colour in ("red", "white"):
        item = metrics[colour]
        regression = item["regression"]
        classification = item["classification"]
        matrix = classification["confusion_matrix"]
        tn, fp = matrix[0]
        fn, tp = matrix[1]
        total = int(item["train_rows"]) + int(item["test_rows"])
        split = split_lookup[colour]
        if (int(split["clean_rows"]) != total or
                int(split["test_good"]) != fn + tp):
            raise ValueError("The split summary does not match the current metrics.")
        split_table.append(
            f"| {colour.title()} | {total} | {item['train_rows']} | "
            f"{item['test_rows']} | {split['train_good']} | {fn + tp} |"
        )
        regression_table.append(
            f"| {colour.title()} | {number(regression['mae'])} | "
            f"{number(regression['mse'])} | {number(regression['r2'])} | "
            f"{number(regression['mean_baseline_mae'])} | "
            f"{number(regression['mean_baseline_mse'])} |"
        )
        classification_table.append(
            f"| {colour.title()} | {number(classification['accuracy'])} | "
            f"{number(classification['precision'])} | "
            f"{number(classification['recall'])} | {number(classification['f1'])} | "
            f"{number(classification['majority_baseline_accuracy'])} |"
        )
        confusion_tables.append(
            f"**{colour.title()} wine**\n\n"
            "| Actual / predicted | Not good (0) | Good (1) |\n"
            "| --- | ---: | ---: |\n"
            f"| Not good (0) | {tn} (TN) | {fp} (FP) |\n"
            f"| Good (1) | {fn} (FN) | {tp} (TP) |"
        )
        difference = (classification["accuracy"] -
                      classification["majority_baseline_accuracy"]) * 100
        relation = "below" if difference < 0 else "above"
        observations.append(
            f"For {colour} wine, the classifier finds {tp} of {fn + tp} good wines "
            f"(recall {percent(classification['recall'])}), misses {fn}, and labels "
            f"{fp} not-good wines as good. Its accuracy is {abs(difference):.2f} "
            f"percentage points {relation} the always-not-good reference. "
            f"Only {percent((fn + tp) / item['test_rows'])} of this test set is good, "
            "so accuracy alone gives an incomplete assessment."
        )
    versions = "; ".join(f"{name} {version}"
                         for name, version in metadata["versions"].items())
    return f"""# Part A2: Machine Learning on Wine Quality

## 1. Objectives and approach

This experiment uses the teacher-supplied winequality_with_missing.csv to predict wine quality in two ways. Regression predicts the numerical quality score. Classification predicts whether a wine is good (quality >= 7, label 1) or not good (quality < 7, label 0). Red and white wines are processed separately after cleaning, and each colour has its own regression and classification model.

The methods follow the course material: multiple linear regression from Lecture 5, decision trees and classification evaluation from Lecture 6, and Pandas/NumPy processing from Python Foundations 2. Experiments use the teacher's modified dataset.

## 2. Dataset inspection and cleaning

The input is semicolon-separated. Its chemistry headers contain spaces, whereas the assignment lists underscore-separated names. Header spaces are replaced by underscores consistently; pH retains its original case. This changes names, not observations. The original file is retained. Its SHA-256 fingerprint is recorded in the cleaning summary for reproducibility.

Counts are measured on the raw dataset before deletion. A duplicate means an extra identical complete record, keeping the first occurrence; it does not mean every row belonging to a repeated group. Missing values are counted both as individual cells and as affected rows.

| Cleaning measurement | Count |
| --- | ---: |
| Original records | {cleaning['raw_rows']} |
| Duplicate extra records removed | {cleaning['duplicate_rows']} |
| Missing cells in original data | {cleaning['missing_cells']} |
| Original rows containing missing values | {cleaning['rows_with_missing']} |
| Records after removing duplicates | {cleaning['after_dedup_rows']} |
| Missing-value rows removed after deduplication | {cleaning['missing_rows_removed_after_dedup']} |
| Final clean records | {cleaning['clean_rows']} |

Missing values by column are:

| Column | Missing cells |
| --- | ---: |
{chr(10).join(missing_table)}

Cleaning first removes duplicate records with the first occurrence retained, then removes rows containing any missing value. No imputation is used. Surviving records retain their input order. Source-row identifiers support auditing but are excluded from model inputs.

## 3. Ordered training and testing split

After cleaning, records are separated by colour. For each colour, the boundary is floor(0.8 x number of clean records): the first portion is training data and all remaining rows are testing data. There is no shuffle, random split or stratification. This follows the assignment even though ordered subsets may have different class distributions.

| Colour | Clean rows | Training | Testing | Train good | Test good |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(split_table)}

![Quality-score distributions in the cleaned red and white datasets](results/plots/quality_distribution.png)

The same 11 chemistry features are used for both tasks:

{', '.join(features)}.

The regression target is quality. The binary target is_good is derived from quality solely as a label. Neither quality nor is_good is included in the feature matrix. Colour is constant within each separate experiment, so it is also excluded, along with source_row. Thus the classifier cannot read the score that defines its answer.

## 4. Model choices and training

Multiple linear regression is a clear, interpretable starting point. It estimates a quality score from a weighted sum of the chemistry variables and an intercept. Lecture 5 introduces this extension of single-feature regression. A linear model may miss nonlinear relationships, so its suitability is assessed through held-out errors rather than assumed.

The decision-tree classifier follows Lecture 6. It recursively splits feature ranges to separate the two labels and can express nonlinear decision rules. To limit model complexity, the fixed settings are max_depth={parameters['max_depth']}, min_samples_leaf={parameters['min_samples_leaf']} and random_state={parameters['random_state']}. The same settings are used for both colours; they are not selected or revised using test performance. These are conservative starting settings, not claimed optimal hyperparameters.

Each model is fitted using only its colour's training features and labels. Test labels are used only for final evaluation. No test-driven model selection, resampling or feature selection is performed. Four fitted models are saved. Keeping one regression algorithm and one classification algorithm makes the experiment consistent with the task and allows the colour-specific findings to be explained.

## 5. Regression evaluation

MAE is the mean absolute prediction error, expressed in quality-score points. MSE is the mean squared error and penalizes larger errors more strongly. R-squared measures performance relative to predicting the test-set mean: 1 is a perfect fit, 0 is that mean reference, and negative values are possible.

For an additional practical comparison, the mean-baseline predictor always outputs the corresponding training-set mean quality. It uses no test-set statistics. Reported regression predictions remain continuous: they are not rounded or clipped before computing metrics.

| Colour | MAE | MSE | R-squared | Training-mean baseline MAE | Baseline MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(regression_table)}

The positive R-squared values indicate some predictive information, but values around {red['regression']['r2']:.2f} and {white['regression']['r2']:.2f} show that substantial variation remains unexplained. Typical absolute errors are approximately half a score point. The mean-baseline errors help assess whether the fitted relationships improve on a simple prediction rule; these observed results do not establish reliability on a different wine population.

![Red-wine regression predictions and actual quality](results/plots/regression_red.png)

![White-wine regression predictions and actual quality](results/plots/regression_white.png)

Complete coefficients are exported to results/linear_coefficients.csv. Coefficients describe the fitted linear association while holding other inputs fixed. Their raw magnitudes are not directly comparable feature-importance scores because the variables have different units and may be correlated. They also do not demonstrate causal effects.

## 6. Classification evaluation

Good wine is the positive class. Accuracy measures overall correctness; precision is TP/(TP+FP); recall is TP/(TP+FN); and F1 is the harmonic mean of precision and recall. Confusion matrices below have actual labels as rows and predicted labels as columns, in the order [0, 1].

| Colour | Accuracy | Precision | Recall | F1 | Always-not-good accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(classification_table)}

{(chr(10) * 2).join(confusion_tables)}

{observations[0]}

{observations[1]}

The majority reference has zero good-wine recall. The fitted trees identify some good wines, but introduce false positives and still miss many positives. Their modest F1 scores do not support a strong-classifier claim. A use case prioritizing finding good wines would need to consider recall and the cost of incorrect recommendations, rather than choosing by accuracy alone.

![Red-wine confusion matrix](results/plots/confusion_red.png)

![White-wine confusion matrix](results/plots/confusion_white.png)

## 7. Limitations and reproducibility

Removing incomplete rows follows the task but can change the represented population. The ordered split can expose distribution differences, especially in the proportion of good wines. Quality is a discrete rating with limited range, while linear regression provides continuous estimates. Fixed tree depth and leaf size reduce complexity but do not guarantee generalization. Improving either model would require training-only validation or additional data; the final test set should remain reserved for evaluation.

Saved outputs include the four ordered train/test CSV files, four fitted models, model metadata, cleaning and split summaries, metrics, predictions, coefficients and plots. The model metadata records the feature order and package versions. The experiment versions are: {versions}.

The report is generated from the current result files, and its figures show the same run. CSV handling, model persistence with joblib, plotting and Word-report generation are supporting implementation details; they do not introduce another learning algorithm. Saved model files should be used with the recorded compatible library versions.

## References

1. ITP4524_Assignment_2627 v2 .docx, Section A2: tasks, cleaning, ordered split and required metrics.
2. Teacher-supplied winequality_with_missing.csv: experiment input. Original dataset source identified by the assignment: UCI Wine Quality, https://archive.ics.uci.edu/dataset/186/wine+quality.
3. ITP4524_Lecture_05_ML_Regression (3).pptx: multiple regression and MAE, MSE and R-squared evaluation.
4. ITP4524_Lecture_06_ML_Classification (3).pptx: decision trees and classification metrics.
5. Python Foundations 2 - Flow Control, Structured Data Types and NumPyPandas (3).pptx: arrays and tabular data processing.
"""


def plain(text):
    return text.replace("**", "").replace("`", "")


def write_docx(markdown, output):
    from docx import Document
    from docx.shared import Inches, Pt

    document = Document()
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    for section in document.sections:
        section.top_margin = Inches(0.65)
        section.bottom_margin = Inches(0.65)
        section.left_margin = Inches(0.65)
        section.right_margin = Inches(0.65)
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            document.add_heading(plain(line.lstrip("# ")), level=min(level, 3))
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append(cells)
                index += 1
            table = document.add_table(rows=1, cols=len(rows[0]))
            table.style = "Light Shading Accent 1"
            for column, cell in enumerate(rows[0]):
                table.rows[0].cells[column].text = plain(cell)
            for row in rows[1:]:
                cells = table.add_row().cells
                for column, cell in enumerate(row):
                    cells[column].text = plain(cell)
            continue
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
            if match is None:
                raise ValueError(f"Malformed figure reference: {line}")
            caption, relative = match.groups()
            document.add_picture(str(output / relative), width=Inches(6.3))
            document.add_paragraph(caption, style="Caption")
        elif re.match(r"\d+\. ", line):
            document.add_paragraph(plain(line))
        else:
            paragraph = [line]
            while index + 1 < len(lines) and lines[index + 1].strip():
                if (lines[index + 1].startswith(("#", "|", "![")) or
                        re.match(r"\d+\. ", lines[index + 1])):
                    break
                index += 1
                paragraph.append(lines[index].strip())
            document.add_paragraph(plain(" ".join(paragraph)))
        index += 1
    document.save(output / "report.docx")


def main():
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python3 build_report.py [OUTPUT_DIR]")
    output = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else Path(__file__).resolve().parent
    markdown = build_markdown(output)
    (output / "report.md").write_text(markdown, encoding="utf-8")
    write_docx(markdown, output)
    print("Reports written to", output)


if __name__ == "__main__":
    main()
