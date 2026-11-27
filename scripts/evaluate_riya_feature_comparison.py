"""Independently evaluate the assigned LightGBM/XGBoost feature-set runs.

This script reads Naman's saved validation/test predictions. It does not fit
models, select thresholds, or rescore a test matrix.
"""
from __future__ import annotations

import csv
import gzip
import html
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "reports" / "feature_set_results" / "metrics"
OUTPUT_DIR = ROOT / "reports" / "riya_evaluation"
FIGURES_DIR = OUTPUT_DIR / "confusion_matrices"
DATASETS = ("paysim", "ieee_cis")
MODELS = ("lightgbm", "xgboost")
FEATURE_SETS = ("full", "auth")
DISPLAY_MODEL = {"lightgbm": "LightGBM", "xgboost": "XGBoost"}
DISPLAY_FEATURE = {"full": "Model A (Full)", "auth": "Model B (Authorization-time)"}
FLOAT_TOLERANCE = 1e-6


def read_test_predictions(dataset: str, model: str, feature_set: str) -> tuple[dict, list[dict]]:
    stem = f"{dataset}_{feature_set}_{model}"
    metadata_path = METRICS_DIR / f"{stem}_metrics.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metrics metadata: {metadata_path.relative_to(ROOT)}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {"dataset": dataset, "model": model, "feature_set": feature_set}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{metadata_path.name}: expected {key}={value!r}, got {metadata.get(key)!r}")
    if metadata.get("threshold_selected_on") != "validation":
        raise ValueError(f"{metadata_path.name}: threshold was not recorded as validation-selected")
    if metadata.get("final_metrics_split") != "test":
        raise ValueError(f"{metadata_path.name}: final metrics split is not test")
    if metadata.get("test_scored_once_for_this_export") is not True:
        raise ValueError(f"{metadata_path.name}: test-once provenance flag is missing or false")

    relative_prediction = str(metadata["prediction_file"]).replace("\\", "/")
    prediction_path = ROOT / Path(relative_prediction)
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Missing saved predictions: {relative_prediction}")

    threshold = float(metadata["threshold"])
    rows: list[dict] = []
    with gzip.open(prediction_path, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"row_id", "true_label", "predicted_probability", "predicted_label", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{relative_prediction}: missing required prediction columns")
        for row in reader:
            if row["split"] != "test":
                continue
            y = int(row["true_label"])
            predicted = int(row["predicted_label"])
            score = float(row["predicted_probability"])
            if y not in (0, 1) or predicted not in (0, 1):
                raise ValueError(f"{relative_prediction}: non-binary label")
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"{relative_prediction}: invalid probability")
            # The exported score has fewer digits than the original model output.
            # Allow a tiny rounding margin for a score exactly at the threshold.
            if abs(score - threshold) > 1e-9 and predicted != int(score >= threshold):
                raise ValueError(f"{relative_prediction}: saved label disagrees with validation threshold")
            rows.append({
                "row_id": row["row_id"],
                "y": y,
                "predicted": predicted,
                "score": score,
            })
    if not rows:
        raise ValueError(f"{relative_prediction}: no test rows")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError(f"{relative_prediction}: duplicate test row IDs")
    if len(rows) != int(metadata["test_rows"]):
        raise ValueError(f"{relative_prediction}: test row count differs from metadata")
    if {row["y"] for row in rows} != {0, 1}:
        raise ValueError(f"{relative_prediction}: test split does not contain both classes")
    return metadata, rows


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        raise ValueError("Average Precision requires at least one positive label")
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positive = false_positive = 0
    previous_recall = 0.0
    ap = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        while index < len(ordered) and ordered[index][0] == score:
            if ordered[index][1] == 1:
                true_positive += 1
            else:
                false_positive += 1
            index += 1
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap


def evaluate(metadata: dict, rows: list[dict]) -> dict:
    labels = [row["y"] for row in rows]
    predictions = [row["predicted"] for row in rows]
    scores = [row["score"] for row in rows]
    counts = Counter()
    for actual, predicted in zip(labels, predictions):
        counts[("tn" if actual == 0 else "fn") if predicted == 0
               else ("fp" if actual == 0 else "tp")] += 1
    tn, fp, fn, tp = (counts[key] for key in ("tn", "fp", "fn", "tp"))
    result = {
        "ap": average_precision(labels, scores),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "test_rows": len(rows),
        "threshold": float(metadata["threshold"]),
    }
    reference = metadata["test_metrics"]
    for output_key, reference_key in (
        ("ap", "pr_auc"), ("precision", "precision"), ("recall", "recall"), ("fpr", "fpr")
    ):
        if abs(result[output_key] - float(reference[reference_key])) > FLOAT_TOLERANCE:
            raise ValueError(
                f"{metadata['dataset']}/{metadata['model']}/{metadata['feature_set']}: "
                f"recomputed {output_key}={result[output_key]:.10g} differs from saved "
                f"value {reference[reference_key]}"
            )
    for key in ("tn", "fp", "fn", "tp"):
        if result[key] != int(reference[key]):
            raise ValueError(
                f"{metadata['dataset']}/{metadata['model']}/{metadata['feature_set']}: "
                f"recomputed {key}={result[key]} differs from saved value {reference[key]}"
            )
    return result


def write_confusion_svg(path: Path, title: str, result: dict) -> None:
    matrix = ((result["tn"], result["fp"]), (result["fn"], result["tp"]))
    left, top, cell = 155, 108, 145
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="390" viewBox="0 0 600 390">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="300" y="34" text-anchor="middle" font-family="Arial" font-size="19" font-weight="bold">{html.escape(title)}</text>',
        '<text x="372" y="76" text-anchor="middle" font-family="Arial" font-size="14">Predicted label</text>',
        '<text x="72" y="245" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 72 245)">Actual label</text>',
    ]
    for col, label in enumerate(("0", "1")):
        parts.append(f'<text x="{left + col * cell + cell / 2}" y="101" text-anchor="middle" font-family="Arial" font-size="14">{label}</text>')
    for row, label in enumerate(("0", "1")):
        parts.append(f'<text x="137" y="{top + row * cell + cell / 2 + 5}" text-anchor="middle" font-family="Arial" font-size="14">{label}</text>')
        for col in range(2):
            x, y = left + col * cell, top + row * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="#eaf1f8" stroke="#526579" stroke-width="2"/>')
            parts.append(f'<text x="{x + cell / 2}" y="{y + cell / 2}" text-anchor="middle" dominant-baseline="middle" font-family="Arial" font-size="25" font-weight="bold">{matrix[row][col]:,}</text>')
    parts.append(f'<text x="300" y="388" text-anchor="middle" font-family="Arial" font-size="12">TN={result["tn"]:,}  FP={result["fp"]:,}  FN={result["fn"]:,}  TP={result["tp"]:,}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def fmt(value: float) -> str:
    return f"{value:.10g}"


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    evaluated: dict[tuple[str, str, str], dict] = {}
    metric_rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for feature_set in FEATURE_SETS:
                metadata, predictions = read_test_predictions(dataset, model, feature_set)
                result = evaluate(metadata, predictions)
                key = (dataset, model, feature_set)
                evaluated[key] = result
                metric_rows.append({
                    "Dataset": dataset,
                    "Model": DISPLAY_MODEL[model],
                    "Feature set": DISPLAY_FEATURE[feature_set],
                    "Threshold (validation)": fmt(result["threshold"]),
                    "Test AP": fmt(result["ap"]),
                    "Test Precision": fmt(result["precision"]),
                    "Test Recall": fmt(result["recall"]),
                    "Test FPR": fmt(result["fpr"]),
                    "TN": result["tn"], "FP": result["fp"],
                    "FN": result["fn"], "TP": result["tp"],
                    "Test rows": result["test_rows"],
                    "Verified against Naman metrics": "yes",
                })
                figure_name = f"{dataset}_{feature_set}_{model}_confusion.svg"
                write_confusion_svg(
                    FIGURES_DIR / figure_name,
                    f"{dataset} | {DISPLAY_MODEL[model]} | {DISPLAY_FEATURE[feature_set]}",
                    result,
                )

    metric_fields = [
        "Dataset", "Model", "Feature set", "Threshold (validation)", "Test AP",
        "Test Precision", "Test Recall", "Test FPR", "TN", "FP", "FN", "TP",
        "Test rows", "Verified against Naman metrics",
    ]
    write_csv(OUTPUT_DIR / "riya_evaluation_metrics.csv", metric_fields, metric_rows)

    paired_rows = []
    for dataset in DATASETS:
        for model in MODELS:
            full = evaluated[(dataset, model, "full")]
            auth = evaluated[(dataset, model, "auth")]
            row = {"Dataset": dataset, "Model": DISPLAY_MODEL[model]}
            for metric, label in (("ap", "AP"), ("precision", "Precision"), ("recall", "Recall"), ("fpr", "FPR")):
                row[f"Model A {label}"] = fmt(full[metric])
                row[f"Model B {label}"] = fmt(auth[metric])
                row[f"Model B minus Model A {label}"] = fmt(auth[metric] - full[metric])
            row["Model A threshold"] = fmt(full["threshold"])
            row["Model B threshold"] = fmt(auth["threshold"])
            paired_rows.append(row)
    paired_fields = ["Dataset", "Model"]
    for label in ("AP", "Precision", "Recall", "FPR"):
        paired_fields.extend((f"Model A {label}", f"Model B {label}", f"Model B minus Model A {label}"))
    paired_fields.extend(("Model A threshold", "Model B threshold"))
    write_csv(OUTPUT_DIR / "riya_model_a_vs_b_comparison.csv", paired_fields, paired_rows)

    notes = [
        "# Riyanshu: held-out feature-set evaluation",
        "",
        "This evaluation reads Naman's saved prediction files for the eight assigned LightGBM/XGBoost configurations. It does not train models, change thresholds, or score test matrices again. Thresholded metrics use the saved predicted labels; AP is recalculated from the saved test scores. Each result is checked against its per-run metrics JSON.",
        "",
        "## Model A vs Model B",
        "",
        "| Dataset | Model | A AP | B AP | A Precision | B Precision | A Recall | B Recall | A FPR | B FPR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in paired_rows:
        notes.append(
            f"| {row['Dataset']} | {row['Model']} | {float(row['Model A AP']):.4f} | {float(row['Model B AP']):.4f} "
            f"| {float(row['Model A Precision']):.4f} | {float(row['Model B Precision']):.4f} "
            f"| {float(row['Model A Recall']):.4f} | {float(row['Model B Recall']):.4f} "
            f"| {float(row['Model A FPR'])*100:.3f}% | {float(row['Model B FPR'])*100:.3f}% |"
        )
    notes.extend(["", "## Observations by dataset", ""])
    for dataset in DATASETS:
        notes.extend([f"### {dataset}", ""])
        for model in MODELS:
            a = evaluated[(dataset, model, "full")]
            b = evaluated[(dataset, model, "auth")]
            relative_ap = (b["ap"] - a["ap"]) / a["ap"] * 100 if a["ap"] else 0.0
            notes.append(
                f"- **{DISPLAY_MODEL[model]}:** Model B AP changed by {relative_ap:.1f}% "
                f"({a['ap']:.4f} to {b['ap']:.4f}); precision {a['precision']:.4f} to {b['precision']:.4f}; "
                f"recall {a['recall']:.4f} to {b['recall']:.4f}; FPR {a['fpr']*100:.3f}% to {b['fpr']*100:.3f}%."
            )
        notes.append("")
    notes.extend([
        "## Interpretation and limitations",
        "",
        "For PaySim, authorization-time features retain high AP and recall, but precision is much lower and false positives increase compared with the full set. The full set contains post-transaction balance information, so its stronger scores are retrospective and are not a fair authorization-time expectation.",
        "",
        "For IEEE-CIS, restricting the feature set produces a larger reduction in AP and recall for both models. The anonymized feature groups remain excluded from Model B because their operational timing is not established; the comparison reflects the reviewed feature lists, not proof that every Model A field is available in practice.",
        "",
        "The repository records a stratified 70/15/15 split with seed 42. This is the implemented team decision; it is not a future-time evaluation. Validation FPR was constrained to at most 1% when choosing thresholds, but that does not guarantee test FPR at or below 1%. Test values above 1% are reported as observed and must not be used to retune thresholds.",
        "",
        "The eight confusion-matrix SVGs are in the confusion_matrices folder. The full metric rows, including confusion counts, are in riya_evaluation_metrics.csv. All eight rows passed comparison against Naman's saved per-run metrics.",
        "",
    ])
    (OUTPUT_DIR / "riya_result_analysis.md").write_text("\n".join(notes), encoding="utf-8")
    print(f"Verified {len(metric_rows)} assigned configurations against saved metrics.")
    print(f"Created {len(metric_rows)} confusion matrices and the paired comparison.")
    print(f"Outputs: {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()