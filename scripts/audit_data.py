"""Audit the local Stage 1 source datasets without loading them all into memory."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CHUNK_SIZE = 25_000

FILES = {
    "paysim": RAW / "PS_20174392719_1491204439457_log.csv",
    "ieee_transaction": RAW / "train_transaction.csv",
    "ieee_identity": RAW / "train_identity.csv",
}


def scan_csv(path: Path, *, target: str | None = None, check_full_row_dupes: bool = False) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Expected dataset file was not found: {path}")

    columns = list(pd.read_csv(path, nrows=0).columns)
    row_count = 0
    missing = pd.Series(0, index=columns, dtype="int64")
    target_counts: dict[str, int] = {}
    transaction_ids: set[str] = set()
    duplicate_transaction_ids = 0
    duplicate_rows = 0
    seen_db_path: str | None = None
    connection = None

    if check_full_row_dupes:
        with tempfile.NamedTemporaryFile(prefix="fedguard_row_hashes_", suffix=".sqlite", delete=False) as tmp:
            seen_db_path = tmp.name
        connection = sqlite3.connect(seen_db_path)
        connection.execute("CREATE TABLE seen_hashes (row_hash INTEGER PRIMARY KEY)")

    try:
        for chunk in pd.read_csv(
            path,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            dtype={"TransactionID": "string"},
        ):
            row_count += len(chunk)
            missing = missing.add(chunk.isna().sum(), fill_value=0).astype("int64")

            if target and target in chunk.columns:
                counts = chunk[target].value_counts(dropna=False)
                for label, count in counts.items():
                    key = "<missing>" if pd.isna(label) else str(label)
                    target_counts[key] = target_counts.get(key, 0) + int(count)

            if "TransactionID" in chunk.columns:
                ids = chunk["TransactionID"].dropna().astype(str)
                for transaction_id in ids:
                    if transaction_id in transaction_ids:
                        duplicate_transaction_ids += 1
                    else:
                        transaction_ids.add(transaction_id)

            if connection is not None:
                row_hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype="int64")
                before = connection.total_changes
                connection.executemany(
                    "INSERT OR IGNORE INTO seen_hashes(row_hash) VALUES (?)",
                    ((int(value),) for value in row_hashes),
                )
                duplicate_rows += len(chunk) - (connection.total_changes - before)
                connection.commit()
    finally:
        if connection is not None:
            connection.close()
        if seen_db_path:
            Path(seen_db_path).unlink(missing_ok=True)

    result = {
        "file": str(path.relative_to(ROOT)),
        "rows": row_count,
        "columns": len(columns),
        "column_names": columns,
        "missing_values_by_column": {str(col): int(value) for col, value in missing.items()},
        "target_counts": target_counts,
        "duplicate_transaction_ids": duplicate_transaction_ids,
        "unique_transaction_ids": len(transaction_ids),
    }
    if check_full_row_dupes:
        result["duplicate_rows_hash_count"] = duplicate_rows
        result["duplicate_check"] = "64-bit pandas row hashes stored in temporary SQLite index"
    return result


def main() -> None:
    paysim = scan_csv(FILES["paysim"], target="isFraud", check_full_row_dupes=True)
    ieee_tx = scan_csv(FILES["ieee_transaction"], target="isFraud")
    ieee_identity = scan_csv(FILES["ieee_identity"])

    identity_ids = pd.read_csv(
        FILES["ieee_identity"],
        usecols=["TransactionID"],
        dtype={"TransactionID": "string"},
    )["TransactionID"].dropna().astype(str)
    identity_id_set = set(identity_ids)
    transaction_id_set = set()

    for chunk in pd.read_csv(
        FILES["ieee_transaction"],
        usecols=["TransactionID"],
        chunksize=CHUNK_SIZE,
        dtype={"TransactionID": "string"},
    ):
        transaction_id_set.update(chunk["TransactionID"].dropna().astype(str))

    ieee_tx["identity_match_count"] = len(transaction_id_set & identity_id_set)
    ieee_tx["identity_unmatched_count"] = len(identity_id_set - transaction_id_set)
    ieee_tx["identity_match_rate"] = (
        ieee_tx["identity_match_count"] / len(transaction_id_set) if transaction_id_set else 0.0
    )

    # Calculate missingness on the actual left-joined training table. This is the
    # relevant view for the planned >90% column-drop rule.
    identity_frame = pd.read_csv(
        FILES["ieee_identity"], dtype={"TransactionID": "string"}, low_memory=False
    )
    matched_identity = identity_frame[
        identity_frame["TransactionID"].astype(str).isin(transaction_id_set)
    ]
    merged_missing = dict(ieee_tx["missing_values_by_column"])
    unmatched_transactions = ieee_tx["rows"] - ieee_tx["identity_match_count"]
    for column in identity_frame.columns:
        if column == "TransactionID":
            continue
        missing_on_matched_rows = int(matched_identity[column].isna().sum())
        merged_missing[column] = unmatched_transactions + missing_on_matched_rows
    ieee_tx["left_joined_rows"] = ieee_tx["rows"]
    ieee_tx["left_joined_columns"] = ieee_tx["columns"] + len(identity_frame.columns) - 1
    ieee_tx["left_joined_missing_values_by_column"] = merged_missing
    ieee_tx["left_joined_columns_over_90_percent_missing"] = [
        column
        for column, count in merged_missing.items()
        if count / ieee_tx["rows"] > 0.9
    ]
    ieee_identity["unique_transaction_ids"] = len(identity_id_set)
    ieee_identity["duplicate_transaction_ids"] = len(identity_ids) - len(identity_id_set)

    results = {
        "paysim": paysim,
        "ieee_cis_train_transaction": ieee_tx,
        "ieee_cis_train_identity": ieee_identity,
    }
    output = ROOT / "reports" / "data_audit_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for name, result in results.items():
        missing_columns = {
            column: count
            for column, count in result["missing_values_by_column"].items()
            if count
        }
        print(f"\n{name}: {result['rows']:,} rows x {result['columns']} columns")
        print(f"  target counts: {result['target_counts']}")
        print(f"  duplicate TransactionIDs: {result['duplicate_transaction_ids']:,}")
        if "duplicate_rows_hash_count" in result:
            print(f"  duplicate full rows (hash count): {result['duplicate_rows_hash_count']:,}")
        if "identity_match_rate" in result:
            print(
                "  identity join coverage: "
                f"{result['identity_match_count']:,} IDs "
                f"({result['identity_match_rate']:.2%})"
            )
            joined_missing = result["left_joined_missing_values_by_column"]
            joined_high = result["left_joined_columns_over_90_percent_missing"]
            print(f"  left-joined columns: {result['left_joined_columns']}")
            print(f"  left-joined columns over 90% missing: {joined_high}")
            print(f"  left-joined columns with any missing values: {sum(v > 0 for v in joined_missing.values())}")
        else:
            print(f"  columns with missing values: {missing_columns}")

    print(f"\nSaved complete column-level results to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
