"""Generate a chunked, label-aware exploratory report for Stage 1 datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CHUNK_SIZE = 50_000
RNG_SEED = 42


def fmt(value: float) -> str:
    return f"{value:,.4f}" if abs(value) < 1000 else f"{value:,.2f}"


def markdown_counter_table(counter: Counter, label: str, *, min_total: int = 0, limit: int = 30) -> str:
    totals: Counter = Counter()
    positives: Counter = Counter()
    for (category, fraud), count in counter.items():
        totals[category] += count
        if int(fraud) == 1:
            positives[category] += count
    eligible = [key for key in totals if totals[key] >= min_total]
    eligible.sort(key=lambda key: totals[key], reverse=True)
    lines = [f"| {label} | Rows | Fraud | Fraud rate |", "| --- | ---: | ---: | ---: |"]
    for key in eligible[:limit]:
        fraud = positives[key]
        lines.append(f"| `{key}` | {totals[key]:,} | {fraud:,} | {fraud / totals[key]:.3%} |")
    if len(eligible) > limit:
        lines.append(f"| *Remaining {len(eligible) - limit} categories omitted* | | | |")
    return "\n".join(lines)


def scan_paysim() -> dict:
    path = RAW / "PS_20174392719_1491204439457_log.csv"
    target_counts: Counter = Counter()
    type_fraud: Counter = Counter()
    flagged_fraud: Counter = Counter()
    zero_orig: Counter = Counter()
    zero_dest: Counter = Counter()
    sample_parts: dict[int, list[pd.DataFrame]] = defaultdict(list)
    step_min, step_max = float("inf"), float("-inf")
    columns = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
               "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud", "nameDest"]

    for chunk in pd.read_csv(path, usecols=columns, chunksize=CHUNK_SIZE, low_memory=False):
        target_counts.update(chunk["isFraud"].value_counts().to_dict())
        type_fraud.update(Counter(zip(chunk["type"], chunk["isFraud"])))
        flagged_fraud.update(Counter(zip(chunk["isFlaggedFraud"], chunk["isFraud"])))
        for label, group in chunk.groupby("isFraud", sort=False):
            zero_orig[int(label)] += int((group["newbalanceOrig"] == 0).sum())
            zero_dest[int(label)] += int((group["newbalanceDest"] == 0).sum())
            sample_parts[int(label)].append(group.sample(
                n=min(2_000, len(group)), random_state=RNG_SEED
            ))
        step_min = min(step_min, float(chunk["step"].min()))
        step_max = max(step_max, float(chunk["step"].max()))

    samples = {}
    class_stats = {}
    for label, parts in sample_parts.items():
        sample = pd.concat(parts, ignore_index=True)
        if len(sample) > 30_000:
            sample = sample.sample(n=30_000, random_state=RNG_SEED)
        sample["error_balance_orig"] = (
            sample["newbalanceOrig"] + sample["amount"] - sample["oldbalanceOrg"]
        )
        sample["error_balance_dest"] = (
            sample["oldbalanceDest"] + sample["amount"] - sample["newbalanceDest"]
        )
        stats = {}
        for column in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
                       "error_balance_orig", "error_balance_dest"]:
            series = sample[column].replace([np.inf, -np.inf], np.nan).dropna()
            stats[column] = {
                "sample_n": int(series.size),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "p95": float(series.quantile(0.95)),
                "max": float(series.max()),
            }
        class_stats[str(label)] = stats
        samples[str(label)] = len(sample)

    fraud_total = int(target_counts[1])
    return {
        "rows": int(sum(target_counts.values())),
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "type_fraud": type_fraud,
        "flagged_fraud": flagged_fraud,
        "zero_orig": dict(zero_orig),
        "zero_dest": dict(zero_dest),
        "zero_orig_rates": {str(k): zero_orig[k] / int(target_counts[k]) for k in target_counts},
        "zero_dest_rates": {str(k): zero_dest[k] / int(target_counts[k]) for k in target_counts},
        "step_min": step_min,
        "step_max": step_max,
        "sample_n": samples,
        "sample_stats": class_stats,
        "fraud_rate": fraud_total / int(sum(target_counts.values())),
    }


def scan_ieee() -> dict:
    tx_path = RAW / "train_transaction.csv"
    id_path = RAW / "train_identity.csv"
    usecols = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
               "card4", "card6", "P_emaildomain"]
    id_frame = pd.read_csv(id_path, usecols=["TransactionID", "DeviceType"],
                           dtype={"TransactionID": "string"})
    device_by_id = id_frame.set_index("TransactionID")["DeviceType"].to_dict()

    target_counts: Counter = Counter()
    categoricals: dict[str, Counter] = {name: Counter() for name in ["ProductCD", "card4", "card6", "P_emaildomain", "DeviceType"]}
    sample_parts: dict[int, list[pd.DataFrame]] = defaultdict(list)
    dt_min, dt_max = float("inf"), float("-inf")
    txid_count = 0

    for chunk in pd.read_csv(tx_path, usecols=usecols, chunksize=CHUNK_SIZE,
                             low_memory=False, dtype={"TransactionID": "string"}):
        txid_count += len(chunk)
        target_counts.update(chunk["isFraud"].value_counts().to_dict())
        chunk["DeviceType"] = chunk["TransactionID"].map(device_by_id)
        for column in categoricals:
            values = chunk[column].fillna("<missing>").astype(str)
            categoricals[column].update(Counter(zip(values, chunk["isFraud"])))
        for label, group in chunk.groupby("isFraud", sort=False):
            sample_parts[int(label)].append(group[["TransactionAmt", "TransactionDT"]].sample(
                n=min(5_000, len(group)), random_state=RNG_SEED
            ))
        dt_min = min(dt_min, float(chunk["TransactionDT"].min()))
        dt_max = max(dt_max, float(chunk["TransactionDT"].max()))

    class_stats = {}
    for label, parts in sample_parts.items():
        sample = pd.concat(parts, ignore_index=True)
        if len(sample) > 30_000:
            sample = sample.sample(n=30_000, random_state=RNG_SEED)
        class_stats[str(label)] = {}
        for column in ["TransactionAmt", "TransactionDT"]:
            values = sample[column].dropna()
            class_stats[str(label)][column] = {
                "sample_n": int(values.size),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p95": float(values.quantile(0.95)),
                "max": float(values.max()),
            }

    all_ids = set(pd.read_csv(tx_path, usecols=["TransactionID"],
                              dtype={"TransactionID": "string"})["TransactionID"].astype(str))
    identity_ids = set(id_frame["TransactionID"].astype(str))
    return {
        "rows": txid_count,
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "fraud_rate": int(target_counts[1]) / txid_count,
        "transaction_dt_min": dt_min,
        "transaction_dt_max": dt_max,
        "identity_rows": len(identity_ids),
        "identity_coverage": len(identity_ids & all_ids) / txid_count,
        "identity_unmatched": len(identity_ids - all_ids),
        "categoricals": categoricals,
        "sample_stats": class_stats,
    }


def fmt_stats(stats: dict, column: str) -> str:
    lines = [f"| Label | Sample n | Mean | Median | P95 | Max |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label in sorted(stats, key=int):
        s = stats[label][column]
        lines.append(f"| {label} | {s['sample_n']:,} | {fmt(s['mean'])} | {fmt(s['median'])} | {fmt(s['p95'])} | {fmt(s['max'])} |")
    return "\n".join(lines)


def main() -> None:
    audit_path = ROOT / "reports" / "data_audit_summary.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    paysim = scan_paysim()
    ieee = scan_ieee()
    missing = audit["ieee_cis_train_transaction"]["left_joined_missing_values_by_column"]
    high_missing = audit["ieee_cis_train_transaction"]["left_joined_columns_over_90_percent_missing"]

    pay_type_table = markdown_counter_table(paysim["type_fraud"], "PaySim type", min_total=1)
    ieee_tables = {
        column: markdown_counter_table(ieee["categoricals"][column], column, min_total=500, limit=20)
        for column in ["ProductCD", "card4", "card6", "P_emaildomain", "DeviceType"]
    }
    top_ieee_missing = sorted(missing.items(), key=lambda item: item[1], reverse=True)[:15]
    missing_table = "\n".join(
        ["| Column | Missing rows | Missing |", "| --- | ---: | ---: |"]
        + [f"| `{col}` | {count:,} | {count / ieee['rows']:.2%} |" for col, count in top_ieee_missing]
    )

    report = f"""# Stage 1 Exploratory Data Analysis

**Prepared by:** Naman Dugar  
**Date:** 2026-10-21  
**Scope:** PaySim and labeled IEEE-CIS training data only. No models were trained.  
**Method:** Chunked scans; amount and balance quantiles below use deterministic class-stratified samples (seed {RNG_SEED}), while row counts, category counts, and label rates use all rows.

## PaySim

- Rows and columns: {paysim['rows']:,} x 11.
- Fraud: {paysim['target_counts']['1']:,} / {paysim['rows']:,} ({paysim['fraud_rate']:.4%}); no nulls or duplicate full rows in the source audit.
- `step` range: {paysim['step_min']:,.0f} to {paysim['step_max']:,.0f}.
- All {paysim['target_counts']['1']:,} fraud labels occur in `TRANSFER` and `CASH_OUT`.
- `isFlaggedFraud` is 1 for 16 rows, and all 16 are labeled fraud. Exclude this pre-existing rule flag from model inputs because it directly discloses a subset of the target.

### Fraud rate by transaction type (all rows)

{pay_type_table}

### Balance and amount distributions

Values are approximate sample summaries, split by target. Engineered errors follow the active plan's formulas: `newbalanceOrig + amount - oldbalanceOrg` and `oldbalanceDest + amount - newbalanceDest`.

#### Amount

{fmt_stats(paysim['sample_stats'], 'amount')}

#### Origin balance error

{fmt_stats(paysim['sample_stats'], 'error_balance_orig')}

#### Destination balance error

{fmt_stats(paysim['sample_stats'], 'error_balance_dest')}

### Zero ending-balance rates (all rows)

| Label | `newbalanceOrig == 0` | `newbalanceDest == 0` |
| --- | ---: | ---: |
| Non-fraud | {paysim['zero_orig_rates']['0']:.3%} | {paysim['zero_dest_rates']['0']:.3%} |
| Fraud | {paysim['zero_orig_rates']['1']:.3%} | {paysim['zero_dest_rates']['1']:.3%} |

### PaySim interpretation

Transaction type and balance-related features are worth testing in the Stage 1 baselines, but the strong synthetic balance behavior and the fact that fraud occurs only in two transaction types limit real-world generalization. Keep account IDs and `isFlaggedFraud` out of model inputs under the active plan. Confirm the exact feature list and record it in run metadata.

## IEEE-CIS

- Labeled transaction rows: {ieee['rows']:,}; fraud: {ieee['target_counts']['1']:,} ({ieee['fraud_rate']:.4%}).
- `TransactionDT` range: {ieee['transaction_dt_min']:,.0f} to {ieee['transaction_dt_max']:,.0f} (dataset-relative time units).
- Identity rows: {ieee['identity_rows']:,}; left-join coverage: {ieee['identity_coverage']:.2%}; unmatched identity keys: {ieee['identity_unmatched']}.
- Full-file joined columns over 90% missing: {', '.join(f'`{c}`' for c in high_missing)}. Recalculate this rule on the training split only before fitting or dropping columns.

### Most-missing joined columns

{missing_table}

### Transaction amount

The summary uses deterministic stratified samples, not full-file quantiles.

{fmt_stats(ieee['sample_stats'], 'TransactionAmt')}

### Categorical fraud rates (all rows; categories with at least 500 records)

{ieee_tables['ProductCD']}

{ieee_tables['card4']}

{ieee_tables['card6']}

{ieee_tables['P_emaildomain']}

{ieee_tables['DeviceType']}

### IEEE-CIS interpretation

The fraud rate is materially higher than PaySim's, and identity attributes cover roughly one quarter of transactions. Preserve a `has_identity` indicator so “no identity row” remains distinguishable from null values inside a matched identity row. The high-cardinality and sparse fields need train-only encoders and imputers. Use the internal stratified split for labeled metrics; the competition test files are unlabeled.

## Stage 1 actions from this EDA

1. Compare Logistic Regression, Random Forest, XGBoost, and LightGBM on each dataset. The updated sprint plan assigns 3, 6, 15, and 15 tuning sets respectively to LR, RF, XGBoost, and LightGBM.
2. Use the planned 70/15/15 stratified split with seed 42. Derive IEEE-CIS missingness filtering and all preprocessing from train only.
3. Keep validation and test at the natural class distribution; resample only train.
4. Select models using validation PR-AUC; choose the operating threshold on validation and report test metrics once for the final selected model. Report FPR with the other metrics.
5. Treat the distribution patterns as diagnostic findings, not evidence of production performance. SHAP and Streamlit remain post-Stage-1 work.

## Reproduction

Run `python scripts\\eda_stage1.py` from the repository root. The script reads raw files without modifying them and writes this report to `reports/stage1_eda.md`.
"""

    output = ROOT / "reports" / "stage1_eda.md"
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"PaySim rows={paysim['rows']:,}, fraud={paysim['target_counts']['1']:,} ({paysim['fraud_rate']:.4%})")
    print(f"IEEE-CIS rows={ieee['rows']:,}, fraud={ieee['target_counts']['1']:,} ({ieee['fraud_rate']:.4%}), identity coverage={ieee['identity_coverage']:.2%}")


if __name__ == "__main__":
    main()
