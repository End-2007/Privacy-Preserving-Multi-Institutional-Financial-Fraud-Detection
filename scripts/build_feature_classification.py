"""Build Anvi's dataset feature classifications and feature-set handoff files.

The source archives are read only for their CSV headers. Raw data and model
artifacts are intentionally not copied into the repository.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DOWNLOADS = Path(r"C:\Users\DELL\Downloads")
PAY_ZIP = DOWNLOADS / "stage1_transfer_paysim.zip"
IEEE_ZIP = DOWNLOADS / "stage1_transfer_ieee_cis.zip"
OUT_DIR = REPO / "configs" / "features"
DOCS_DIR = REPO / "docs"


def header_from_zip(archive: Path, member: str) -> list[str]:
    with zipfile.ZipFile(archive) as zf:
        with zf.open(member) as raw:
            return next(csv.reader(io.TextIOWrapper(raw, encoding="utf-8")))


def add(rows, feature, classification, model_a, model_b, identifier, leakage, rationale):
    rows.append(
        {
            "feature": feature,
            "classification": classification,
            "model_a": str(model_a).lower(),
            "model_b": str(model_b).lower(),
            "identifier": str(identifier).lower(),
            "leakage_risk": leakage,
            "rationale": rationale,
        }
    )


def classify_paysim(columns):
    rows = []
    for feature in columns:
        if feature == "isFraud":
            add(rows, feature, "target", False, False, False, "target", "Ground-truth label; never a predictor.")
        elif feature in {"nameOrig", "nameDest"}:
            add(rows, feature, "identifier", False, False, True, "medium", "Account identifier; retain only for traceability, not prediction.")
        elif feature in {"newbalanceOrig", "newbalanceDest"}:
            add(rows, feature, "post_transaction", True, False, False, "high", "Balance after the transaction; unavailable at authorization time and can leak transaction outcome.")
        elif feature == "isFlaggedFraud":
            add(rows, feature, "uncertain", False, False, False, "high", "Rule-based screening output; the active Stage 1 decision keeps it out of model inputs until availability and label dependence are verified.")
        elif feature in {"step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"}:
            add(rows, feature, "authorization_time", True, True, False, "low", "Transaction details or pre-transaction state available when authorization is decided.")
        else:
            add(rows, feature, "uncertain", False, False, False, "medium", "Not covered by the approved PaySim schema rules; requires team review.")
    return rows


def classify_ieee(columns):
    rows = []
    for feature in columns:
        if feature == "isFraud":
            add(rows, feature, "target", False, False, False, "target", "Ground-truth label; never a predictor.")
        elif feature == "TransactionID":
            add(rows, feature, "identifier", False, False, True, "medium", "Technical transaction key used for joining and row traceability; not a predictor.")
        elif feature in {"TransactionDT", "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain", "has_identity"}:
            add(rows, feature, "authorization_time", True, True, False, "low", "Transaction, payment, address, device-presence, or timing information plausibly available during authorization.")
        elif re.fullmatch(r"[CDMV]\d+", feature):
            add(rows, feature, "uncertain", True, False, False, "medium", "Anonymized engineered feature; the public schema does not establish whether its value is available before authorization.")
        elif feature.startswith("id_") or feature in {"DeviceType", "DeviceInfo"}:
            add(rows, feature, "uncertain", True, False, False, "medium", "Identity-table or device feature; availability is conditional and its operational timing is not documented in the supplied materials.")
        else:
            add(rows, feature, "uncertain", True, False, False, "medium", "Feature semantics are masked or undocumented; keep in Model A for the full eligible comparison but exclude from Model B pending review.")
    return rows


def write_dataset(dataset, rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DOCS_DIR / f"feature_classification_{dataset}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "classification", "model_a", "model_b", "identifier", "leakage_risk", "rationale"])
        writer.writeheader()
        writer.writerows(rows)

    stem = "IEEECIS" if dataset == "ieee_cis" else "PaySim"
    for key, suffix in (("model_a", "full"), ("model_b", "auth")):
        path = OUT_DIR / f"{stem}_{suffix}.txt"
        path.write_text("".join(f"{row['feature']}\n" for row in rows if row[key] == "true"), encoding="utf-8")

    counts = {}
    for row in rows:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    return csv_path, counts


def main():
    pay_columns = header_from_zip(PAY_ZIP, "stage1_transfer_paysim/data/raw/PS_20174392719_1491204439457_log.csv")
    ieee_tx = header_from_zip(IEEE_ZIP, "stage1_transfer_ieee_cis/data/raw/train_transaction.csv")
    ieee_id = header_from_zip(IEEE_ZIP, "stage1_transfer_ieee_cis/data/raw/train_identity.csv")[1:]
    pay_rows = classify_paysim(pay_columns)
    ieee_rows = classify_ieee(ieee_tx + ieee_id)
    pay_csv, pay_counts = write_dataset("paysim", pay_rows)
    ieee_csv, ieee_counts = write_dataset("ieee_cis", ieee_rows)

    report = DOCS_DIR / "anvi_feature_analysis.md"
    report.write_text(
        "# Anvi Feature Classification\n\n"
        "This handoff defines the two feature configurations for the FedGuard\n"
        "authorization-time comparison. Model A is the full eligible predictor\n"
        "set after removing the target and technical identifiers. Model B keeps\n"
        "only features classified as available at authorization time.\n\n"
        "## Rules used\n\n"
        "- `isFraud` is the target and is excluded from both models.\n"
        "- `TransactionID`, `row_id`, and account identifiers are excluded as\n"
        "  technical or entity identifiers.\n"
        "- Post-transaction balances are excluded from Model B because they are\n"
        "  not known when authorization is decided.\n"
        "- Anonymized IEEE-CIS groups (`C*`, `D*`, `M*`, and `V*`) are retained\n"
        "  in Model A but excluded from Model B until their operational timing\n"
        "  is documented.\n"
        "- IEEE-CIS identity and device fields are treated conservatively as\n"
        "  uncertain because the supplied data does not prove when they become\n"
        "  available.\n"
        "- PaySim `isFlaggedFraud` is excluded from both models and marked high\n"
        "  leakage risk, following the active Stage 1 decision for this\n"
        "  rule-based screening output.\n\n"
        "## Files\n\n"
        "- `configs/features/PaySim_full.txt` and `PaySim_auth.txt`\n"
        "- `configs/features/IEEECIS_full.txt` and `IEEECIS_auth.txt`\n"
        "- `docs/feature_classification_paysim.csv`\n"
        "- `docs/feature_classification_ieee_cis.csv`\n\n"
        "The CSV files preserve the classification, model inclusion, leakage\n"
        "risk, and rationale for every source column. These are reviewable\n"
        "assumptions, not claims that masked feature semantics are known.\n",
        encoding="utf-8",
    )
    summary = {"paysim": pay_counts, "ieee_cis": ieee_counts, "pay_columns": len(pay_rows), "ieee_columns": len(ieee_rows)}
    (DOCS_DIR / "anvi_feature_analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
