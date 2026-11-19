"""
data_loader.py

Loads and cleans all three datasets, then maps them into the canonical schema
(feature_schema.py). Returns (X: DataFrame of canonical features, y: Series of raw labels).

Usage:
    from data_loader import load_dataset
    X, y, spec = load_dataset("ertam_kaya", path="data/raw/ertam_kaya_firewall.csv")
"""

import glob
import os
import numpy as np
import pandas as pd

from feature_schema import ALL_SPECS, CANONICAL_FEATURES, derived_bytes_total


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """CIC-IDS2017 often has leading spaces in column names (' Destination Port'); always strip."""
    df.columns = [c.strip() for c in df.columns]
    return df


def report_data_quality(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """Summarise missing / inf / dtype per column before deciding how to drop/impute.
    Always inspect first, especially for CIC-IDS2017 which often has inf from Flow Bytes/s when duration=0.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    report = pd.DataFrame({
        "dtype": df.dtypes,
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().sum() / len(df) * 100).round(3),
    })
    inf_counts = {}
    for col in numeric_df.columns:
        inf_counts[col] = int(np.isinf(numeric_df[col]).sum())
    report["n_inf"] = pd.Series(inf_counts)
    report["n_inf"] = report["n_inf"].fillna(0).astype(int)
    if name:
        print(f"--- data quality report: {name} ({len(df)} rows) ---")
    return report.sort_values("pct_missing", ascending=False)


def _check_path_exists(path: str, dataset_key: str):
    if not os.path.exists(path):
        abs_path = os.path.abspath(path)
        raise FileNotFoundError(
            f"[{dataset_key}] No file or folder found at this path: '{path}'\n"
            f"  (the absolute path Python looked for is: {abs_path})\n"
            f"  To check in PowerShell: Test-Path '{path}'\n"
            f"  To list the parent folder: dir '{os.path.dirname(abs_path)}'"
        )


def _load_ertam_kaya(path: str) -> pd.DataFrame:
    _check_path_exists(path, "ertam_kaya")
    df = pd.read_csv(path)
    df = _strip_columns(df)
    return df


def _load_cicids2017(path: str) -> pd.DataFrame:
    """path points to a folder of sub-CSV files (from MachineLearningCSV.zip),
    or to a single file if already merged."""
    _check_path_exists(path, "cicids2017")
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            actual_files = os.listdir(path)
            raise FileNotFoundError(
                f"Folder '{os.path.abspath(path)}' exists but contains no .csv files.\n"
                f"  Items actually found: {actual_files[:20]}"
            )
        frames = []
        for f in files:
            d = pd.read_csv(f, low_memory=False)
            d = _strip_columns(d)
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path, low_memory=False)
        df = _strip_columns(df)
    return df


def _load_unsw_nb15(path: str, feature_names_csv: str = None) -> pd.DataFrame:
    """path points to a folder containing UNSW-NB15_1.csv ... _4.csv (no header in the files).
    feature_names_csv: path to UNSW-NB15_features.csv (the Name column holds feature names in order).
    """
    # Always check the given path exists (file or folder) first, so the error
    # message is clear here rather than surfacing as a vague pandas error later
    abs_path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No file or folder found at this path: '{path}'\n"
            f"  (the absolute path Python looked for is: {abs_path})\n"
            f"  To check (PowerShell): Test-Path '{path}'  and  dir '{os.path.dirname(abs_path)}'\n"
            f"  Most common cause: a wrong folder/path name, or unzipping produced a nested folder "
            f"(e.g. .../UNSW-NB15/UNSW-NB15/*.csv instead of .../UNSW-NB15/*.csv directly)."
        )

    if feature_names_csv and os.path.exists(feature_names_csv):
        feat_df = pd.read_csv(feature_names_csv, encoding="latin1")
        # Handle the column name flexibly (some versions use "Name", others "name")
        name_col = "Name" if "Name" in feat_df.columns else feat_df.columns[1]
        col_names = feat_df[name_col].str.strip().tolist()
    elif feature_names_csv:
        print(f"[warn] feature_names_csv='{feature_names_csv}' was given but not found "
              f"- loading without column names (using index numbers instead); check this path too.")
        col_names = None
    else:
        col_names = None  # let pandas use index numbers as temporary column names

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "UNSW-NB15_*.csv")))
        if not files:
            # List the actual files found so it is clear whether the names simply differ from the expected pattern
            actual_files = os.listdir(path)
            raise FileNotFoundError(
                f"Folder '{abs_path}' exists but contains no files matching 'UNSW-NB15_*.csv'.\n"
                f"  Files actually in this folder (first {len(actual_files)}): {actual_files[:20]}\n"
                f"  If the real names differ (e.g. different capitalisation), adjust the glob pattern "
                f"in the code to match."
            )
        frames = []
        for f in files:
            d = pd.read_csv(f, header=None, low_memory=False)
            if col_names and len(col_names) == d.shape[1]:
                d.columns = col_names
            else:
                print(f"[warn] {os.path.basename(f)}: column count mismatch - "
                      f"feature_names_csv has {len(col_names) if col_names else 0} names "
                      f"but the data file has {d.shape[1]} columns "
                      f"(falling back to index 0..{d.shape[1]-1} as temporary names, which will make "
                      f"the next step fail to find 'attack_cat'/'sport'/etc. - fix the features_csv "
                      f"or column_map to match this actual column count first).")
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path, low_memory=False)
        df = _strip_columns(df)

    # empty attack_cat = normal traffic; label it explicitly as "Normal"
    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].fillna("Normal").str.strip()
    return df


def load_dataset(dataset_key: str, path: str, **kwargs):
    """
    dataset_key: one of "ertam_kaya", "cicids2017", "unsw_nb15"
    path: path to the raw data file / folder
    Returns: (X canonical DataFrame, y label Series, DatasetSpec)
    """
    if dataset_key not in ALL_SPECS:
        raise ValueError(f"unknown dataset_key={dataset_key} (must be one of {list(ALL_SPECS)})")
    spec = ALL_SPECS[dataset_key]

    if dataset_key == "ertam_kaya":
        raw = _load_ertam_kaya(path)
    elif dataset_key == "cicids2017":
        raw = _load_cicids2017(path)
    elif dataset_key == "unsw_nb15":
        raw = _load_unsw_nb15(path, feature_names_csv=kwargs.get("feature_names_csv"))
    else:
        raise NotImplementedError(dataset_key)

    missing_cols = [c for c in spec.column_map if c not in raw.columns]
    if missing_cols:
        print(f"[warn] {dataset_key}: the following columns were not found (skipping for now, "
              f"verify the schema): {missing_cols}")

    rename_map = {k: v for k, v in spec.column_map.items() if k in raw.columns}
    df = raw.rename(columns=rename_map)

    if spec.label_column not in raw.columns:
        raise KeyError(f"{dataset_key}: label column '{spec.label_column}' not found in the loaded file "
                        f"- check the actual column names with print(raw.columns.tolist())")
    y = raw[spec.label_column].astype(str).str.strip()

    # UNSW-NB15 has a well-known label inconsistency: some rows across the 4 files use "Backdoor"
    # and others use "Backdoors" for the same attack category. Without merging them first,
    # both names become needlessly even rarer classes (fragmentation).
    if dataset_key == "unsw_nb15":
        n_backdoor_variant = (y == "Backdoor").sum()
        if n_backdoor_variant > 0:
            print(f"[info] unsw_nb15: found label 'Backdoor' (singular) in {n_backdoor_variant} rows "
                  f"- merging into 'Backdoors' (plural) as one class, since this is a known label "
                  f"inconsistency in this dataset (not a genuinely different attack category).")
            y = y.replace({"Backdoor": "Backdoors"})

    # Convert units only for CIC-IDS2017 (microseconds -> seconds)
    if "duration_microsec" in df.columns:
        df["duration"] = df["duration_microsec"] / 1_000_000.0
        df = df.drop(columns=["duration_microsec"])

    df = derived_bytes_total(df, spec)

    canonical_present = [c for c in CANONICAL_FEATURES if c in df.columns]
    X = df[canonical_present].copy()

    # "protocol" is always categorical across datasets (e.g. "tcp"/"udp"/"unas" in UNSW-NB15,
    # or a protocol number in CIC-IDS2017). Using pd.to_numeric directly like other columns would
    # turn string values into all-NaN (as actually happens with UNSW-NB15), so encode with factorize
    # instead to preserve this feature's signal rather than losing it entirely.
    if "protocol" in X.columns and not pd.api.types.is_numeric_dtype(X["protocol"]):
        codes, uniques = pd.factorize(X["protocol"].astype(str).fillna("unknown"))
        print(f"[info] {dataset_key}: 'protocol' column is textual (e.g. {list(uniques[:5])}) "
              f"- encoding to integer codes with factorize ({len(uniques)} distinct values) instead of dropping to NaN")
        X["protocol"] = codes

    for col in X.columns:
        if col == "protocol":
            continue  # already encoded above; skip the to_numeric coercion
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y, spec


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test dataset loading and print a data-quality report")
    parser.add_argument("--dataset", required=True, choices=list(ALL_SPECS.keys()))
    parser.add_argument("--path", required=True)
    parser.add_argument("--feature_names_csv", default=None)
    args = parser.parse_args()

    X, y, spec = load_dataset(args.dataset, args.path, feature_names_csv=args.feature_names_csv)
    print(f"Loaded successfully: {spec.name} -> X shape={X.shape}, y unique classes={y.nunique()}")
    print(y.value_counts())
    print(report_data_quality(X, name=spec.name))
