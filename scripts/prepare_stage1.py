"""Split and preprocess PaySim and IEEE-CIS without fitting on held-out rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from scipy.sparse import save_npz
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.frequency_encoder import FrequencyEncoder  # noqa: E402
from src.common.splitting import stratified_split_codes  # noqa: E402

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
CHUNK_SIZE = 50_000
SEED = 42
SPLITS = {0: "train", 1: "validation", 2: "test"}
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}

IEEE_HIGH_CARD = {"card1", "card2", "addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo"}
IEEE_LOW_CARD = {
    "ProductCD", "card3", "card4", "card5", "card6", "addr2", "DeviceType",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30", "id_31", "id_33",
    "id_34", "id_35", "id_36", "id_37", "id_38",
    *(f"M{i}" for i in range(1, 10)),
}
IEEE_IDENTITY_CATEGORICAL = {
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30", "id_31", "id_33",
    "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo",
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


class SplitParquetWriters:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.writers = {}
        self.schemas = {}

    def write(self, split_name: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        path = self.output_dir / f"{split_name}.parquet"
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if split_name not in self.writers:
            self.schemas[split_name] = table.schema
            self.writers[split_name] = pq.ParquetWriter(
                path, table.schema, compression="zstd", use_dictionary=True
            )
        else:
            table = table.cast(self.schemas[split_name], safe=False)
        self.writers[split_name].write_table(table, row_group_size=len(frame))

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()


def make_manifest(labels: np.ndarray, codes: np.ndarray, source_files: list[str]) -> dict:
    result = {
        "seed": SEED,
        "requested_split_ratios": SPLIT_RATIOS,
        "source_files": source_files,
        "row_order": "Original source row order; row_id identifies the source row.",
        "splits": {},
    }
    for code, name in SPLITS.items():
        mask = codes == code
        positive = int(np.count_nonzero(labels[mask] == 1))
        result["splits"][name] = {
            "rows": int(mask.sum()),
            "fraud_rows": positive,
            "fraud_rate": positive / int(mask.sum()),
            "actual_ratio": float(mask.mean()),
        }
    return result


def prepare_paysim() -> None:
    source = RAW / "PS_20174392719_1491204439457_log.csv"
    output_dir = PROCESSED / "paysim"
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(source, usecols=["isFraud"], dtype={"isFraud": "int8"})["isFraud"].to_numpy()
    codes = stratified_split_codes(labels, seed=SEED)
    manifest = make_manifest(labels, codes, [str(source.relative_to(ROOT))])
    write_json(output_dir / "split_manifest.json", manifest)

    dtype = {
        "step": "int16", "type": "string", "amount": "float32", "nameOrig": "string",
        "oldbalanceOrg": "float32", "newbalanceOrig": "float32", "nameDest": "string",
        "oldbalanceDest": "float32", "newbalanceDest": "float32", "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }
    writers = SplitParquetWriters(output_dir)
    offset = 0
    try:
        for chunk in pd.read_csv(source, dtype=dtype, chunksize=CHUNK_SIZE, low_memory=False):
            chunk = chunk.reset_index(drop=True)
            n_rows = len(chunk)
            chunk_codes = codes[offset:offset + n_rows]
            if len(chunk_codes) != n_rows:
                raise RuntimeError("PaySim row count changed between split and feature scans.")
            if (chunk["amount"] < 0).any():
                raise ValueError("PaySim contains negative transaction amounts; inspect before log transform.")

            frame = pd.DataFrame()
            frame["row_id"] = np.arange(offset, offset + n_rows, dtype=np.int64)
            frame["type"] = chunk["type"].astype("string")
            frame["amount"] = chunk["amount"].astype(np.float32)
            frame["oldbalanceOrg"] = chunk["oldbalanceOrg"].astype(np.float32)
            frame["newbalanceOrig"] = chunk["newbalanceOrig"].astype(np.float32)
            frame["oldbalanceDest"] = chunk["oldbalanceDest"].astype(np.float32)
            frame["newbalanceDest"] = chunk["newbalanceDest"].astype(np.float32)
            frame["dest_is_merchant"] = chunk["nameDest"].str.startswith("M", na=False).astype("int8")
            frame["orig_balance_delta"] = (chunk["newbalanceOrig"] - chunk["oldbalanceOrg"]).astype(np.float32)
            frame["dest_balance_delta"] = (chunk["newbalanceDest"] - chunk["oldbalanceDest"]).astype(np.float32)
            frame["error_balance_orig"] = (
                chunk["newbalanceOrig"] + chunk["amount"] - chunk["oldbalanceOrg"]
            ).astype(np.float32)
            frame["error_balance_dest"] = (
                chunk["oldbalanceDest"] + chunk["amount"] - chunk["newbalanceDest"]
            ).astype(np.float32)
            frame["orig_zero_after"] = (chunk["newbalanceOrig"] == 0).astype("int8")
            frame["hour"] = (chunk["step"] % 24).astype("int8")
            frame["log_amount"] = np.log1p(chunk["amount"].to_numpy(dtype=np.float32)).astype(np.float32)
            frame["isFraud"] = chunk["isFraud"].astype("int8").to_numpy()

            for code, split_name in SPLITS.items():
                selection = chunk_codes == code
                if selection.any():
                    writers.write(split_name, frame.loc[selection].reset_index(drop=True))
            offset += n_rows
            if offset % 500_000 == 0:
                print(f"PaySim split preparation: {offset:,}/{len(labels):,} rows")
    finally:
        writers.close()

    if offset != len(labels):
        raise RuntimeError(f"PaySim scan produced {offset:,} rows; expected {len(labels):,}.")
    print(f"Prepared PaySim splits in {output_dir.relative_to(ROOT)}")
    fit_and_transform("paysim", categorical_high=[])


def ieee_dtype_map(columns: list[str]) -> dict[str, str]:
    dtype = {}
    for column in columns:
        if column == "TransactionID":
            dtype[column] = "int64"
        elif column == "isFraud":
            dtype[column] = "int8"
        elif column == "TransactionDT":
            dtype[column] = "int32"
        elif column in IEEE_HIGH_CARD or column in IEEE_LOW_CARD:
            dtype[column] = "string"
        else:
            dtype[column] = "float32"
    return dtype


def prepare_ieee() -> None:
    tx_path = RAW / "train_transaction.csv"
    identity_path = RAW / "train_identity.csv"
    output_dir = PROCESSED / "ieee_cis"
    output_dir.mkdir(parents=True, exist_ok=True)

    tx_columns = list(pd.read_csv(tx_path, nrows=0).columns)
    id_columns = list(pd.read_csv(identity_path, nrows=0).columns)
    labels = pd.read_csv(tx_path, usecols=["isFraud"], dtype={"isFraud": "int8"})["isFraud"].to_numpy()
    codes = stratified_split_codes(labels, seed=SEED)
    manifest = make_manifest(
        labels,
        codes,
        [str(tx_path.relative_to(ROOT)), str(identity_path.relative_to(ROOT))],
    )
    manifest["join"] = {
        "type": "left",
        "key": "TransactionID",
        "identity_columns": len(id_columns) - 1,
        "has_identity_feature": True,
    }
    write_json(output_dir / "split_manifest.json", manifest)

    identity_dtype = ieee_dtype_map(id_columns)
    identity = pd.read_csv(identity_path, dtype=identity_dtype, low_memory=False)
    if identity["TransactionID"].duplicated().any():
        raise ValueError("Duplicate TransactionID values found in IEEE-CIS identity file.")
    for column in IEEE_IDENTITY_CATEGORICAL & set(identity.columns):
        identity[column] = identity[column].astype("string")

    tx_dtype = ieee_dtype_map(tx_columns)
    writers = SplitParquetWriters(output_dir)
    offset = 0
    try:
        for chunk in pd.read_csv(tx_path, dtype=tx_dtype, chunksize=CHUNK_SIZE, low_memory=False):
            chunk = chunk.reset_index(drop=True)
            n_rows = len(chunk)
            chunk_codes = codes[offset:offset + n_rows]
            if len(chunk_codes) != n_rows:
                raise RuntimeError("IEEE-CIS row count changed between split and merge scans.")
            if (chunk["TransactionAmt"] < 0).any():
                raise ValueError("IEEE-CIS contains negative TransactionAmt; inspect before log transform.")

            chunk["has_identity"] = chunk["TransactionID"].isin(identity["TransactionID"]).astype("int8")
            frame = chunk.merge(
                identity, on="TransactionID", how="left", sort=False, validate="many_to_one"
            )
            frame["row_id"] = np.arange(offset, offset + n_rows, dtype=np.int64)
            frame["transaction_hour"] = ((frame["TransactionDT"] // 3600) % 24).astype("int8")
            frame["transaction_day"] = (frame["TransactionDT"] // 86400).astype("int16")
            frame["log_amt"] = np.log1p(frame["TransactionAmt"].to_numpy(dtype=np.float32)).astype(np.float32)
            for column in (IEEE_HIGH_CARD | IEEE_LOW_CARD) & set(frame.columns):
                frame[column] = frame[column].astype("string")
            frame["has_identity"] = frame["has_identity"].fillna(0).astype("int8")

            for code, split_name in SPLITS.items():
                selection = chunk_codes == code
                if selection.any():
                    writers.write(split_name, frame.loc[selection].reset_index(drop=True))
            offset += n_rows
            if offset % 100_000 == 0:
                print(f"IEEE-CIS split preparation: {offset:,}/{len(labels):,} rows")
    finally:
        writers.close()

    if offset != len(labels):
        raise RuntimeError(f"IEEE-CIS scan produced {offset:,} rows; expected {len(labels):,}.")
    print(f"Prepared IEEE-CIS splits in {output_dir.relative_to(ROOT)}")
    fit_and_transform("ieee_cis", categorical_high=sorted(IEEE_HIGH_CARD))


def fit_and_transform(dataset: str, *, categorical_high: list[str]) -> None:
    data_dir = PROCESSED / dataset
    model_dir = MODELS / dataset
    model_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(data_dir / "train.parquet")
    excluded = {"isFraud", "row_id", "TransactionID"}
    candidate_features = [column for column in train.columns if column not in excluded]
    train_missing = train[candidate_features].isna().mean()
    dropped = train_missing[train_missing > 0.90].index.tolist()
    feature_columns = [column for column in candidate_features if column not in dropped]
    numeric_columns = [
        column for column in feature_columns
        if pd.api.types.is_numeric_dtype(train[column].dtype)
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    frequency_columns = [column for column in categorical_high if column in categorical_columns]
    onehot_columns = [column for column in categorical_columns if column not in frequency_columns]
    missing_indicator_columns = [
        column for column in numeric_columns
        if column.startswith("D") or column.startswith("id_")
    ]

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric_columns))
    if missing_indicator_columns:
        transformers.append(("missing", MissingIndicator(features="all"), missing_indicator_columns))
    if frequency_columns:
        transformers.append(("frequency", FrequencyEncoder(), frequency_columns))
    if onehot_columns:
        transformers.append((
            "categorical",
            __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
            ]),
            onehot_columns,
        ))
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )

    # Normalize pandas string columns to object + np.nan for sklearn 1.3 compatibility.
    for column in categorical_columns:
        train[column] = train[column].astype(object).where(train[column].notna(), np.nan)
    X_train = preprocessor.fit_transform(train[feature_columns])
    y_train = train["isFraud"].to_numpy(dtype=np.int8)
    if not sparse.issparse(X_train):
        X_train = sparse.csr_matrix(X_train, dtype=np.float32)
    else:
        X_train = X_train.tocsr().astype(np.float32)
    save_npz(data_dir / "X_train.npz", X_train, compressed=True)
    np.save(data_dir / "y_train.npy", y_train)
    lr_scaler = StandardScaler(with_mean=False)
    X_train_lr = lr_scaler.fit_transform(X_train).tocsr().astype(np.float32)
    save_npz(data_dir / "X_lr_train.npz", X_train_lr, compressed=True)
    joblib.dump(lr_scaler, model_dir / "lr_scaler.joblib", compress=3)

    feature_names = preprocessor.get_feature_names_out().tolist()
    split_shapes = {"train": list(X_train.shape)}
    for split_name in ("validation", "test"):
        frame = pd.read_parquet(data_dir / f"{split_name}.parquet")
        y = frame["isFraud"].to_numpy(dtype=np.int8)
        for column in categorical_columns:
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
        X = preprocessor.transform(frame[feature_columns])
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X, dtype=np.float32)
        else:
            X = X.tocsr().astype(np.float32)
        save_npz(data_dir / f"X_{split_name}.npz", X, compressed=True)
        X_lr = lr_scaler.transform(X).tocsr().astype(np.float32)
        save_npz(data_dir / f"X_lr_{split_name}.npz", X_lr, compressed=True)
        np.save(data_dir / f"y_{split_name}.npy", y)
        split_shapes[split_name] = list(X.shape)
        del frame, X, y

    joblib.dump(preprocessor, model_dir / "preprocessor.joblib", compress=3)
    write_json(data_dir / "feature_metadata.json", {
        "dataset": dataset,
        "target": "isFraud",
        "excluded_from_predictors": sorted(excluded),
        "dropped_over_90_percent_missing_on_train": dropped,
        "numeric_features": numeric_columns,
        "frequency_encoded_features": frequency_columns,
        "one_hot_features": onehot_columns,
        "missing_indicator_features": missing_indicator_columns,
        "transformed_feature_names": feature_names,
        "preprocessing_fit_split": "train",
        "unknown_frequency_category_value": 0.0,
        "transform_output": "float32 scipy CSR; X_<split>.npz is for tree models and X_lr_<split>.npz is scaled for Logistic Regression; y_<split>.npy holds labels",
        "matrix_shapes": split_shapes,
        "lr_scaler": str((model_dir / "lr_scaler.joblib").relative_to(ROOT)),
    })
    train_manifest = json.loads((data_dir / "split_manifest.json").read_text(encoding="utf-8"))
    train_manifest["preprocessing"] = {
        "fit_split": "train",
        "dropped_over_90_percent_missing_on_train": dropped,
        "numeric_feature_count": len(numeric_columns),
        "frequency_feature_count": len(frequency_columns),
        "one_hot_input_feature_count": len(onehot_columns),
        "transformed_feature_count": len(feature_names),
        "matrix_shapes": split_shapes,
        "transformer_path": str((model_dir / "preprocessor.joblib").relative_to(ROOT)),
        "lr_scaler_path": str((model_dir / "lr_scaler.joblib").relative_to(ROOT)),
    }
    write_json(data_dir / "split_manifest.json", train_manifest)
    del train, X_train, y_train
    print(
        f"{dataset}: fitted on train only; dropped {len(dropped)} >90% missing columns; "
        f"matrix shapes {split_shapes}"
    )


def main() -> None:
    prepare_paysim()
    prepare_ieee()


if __name__ == "__main__":
    main()
