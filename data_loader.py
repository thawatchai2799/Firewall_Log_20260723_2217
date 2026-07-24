"""
data_loader.py

โหลดและทำความสะอาด dataset ทั้ง 3 แหล่ง แล้ว map เข้าสู่ canonical schema
(feature_schema.py) — คืนค่าเป็น (X: DataFrame ของ canonical features, y: Series ของ label ดิบ)

ใช้งาน:
    from data_loader import load_dataset
    X, y, spec = load_dataset("ertam_kaya", path="data/raw/ertam_kaya_firewall.csv")
"""

import glob
import os
import numpy as np
import pandas as pd

from feature_schema import ALL_SPECS, CANONICAL_FEATURES, derived_bytes_total


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """CIC-IDS2017 มีช่องว่างนำหน้าชื่อคอลัมน์บ่อยมาก (' Destination Port') ต้อง strip เสมอ"""
    df.columns = [c.strip() for c in df.columns]
    return df


def report_data_quality(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """สรุป missing / inf / dtype ของแต่ละคอลัมน์ ก่อนตัดสินใจว่าจะ drop/impute อย่างไร
    เรียกดูก่อนเสมอ โดยเฉพาะกับ CIC-IDS2017 ที่มักมีค่า inf จาก Flow Bytes/s เมื่อ duration=0
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
            f"[{dataset_key}] ไม่พบไฟล์หรือโฟลเดอร์ที่ path นี้เลย: '{path}'\n"
            f"  (path แบบเต็มที่ Python มองหาคือ: {abs_path})\n"
            f"  วิธีตรวจสอบใน PowerShell: Test-Path '{path}'\n"
            f"  วิธีดูว่ามีอะไรอยู่ในโฟลเดอร์แม่: dir '{os.path.dirname(abs_path)}'"
        )


def _load_ertam_kaya(path: str) -> pd.DataFrame:
    _check_path_exists(path, "ertam_kaya")
    df = pd.read_csv(path)
    df = _strip_columns(df)
    return df


def _load_cicids2017(path: str) -> pd.DataFrame:
    """path ชี้ไปที่โฟลเดอร์ที่มีไฟล์ CSV ย่อยหลายไฟล์ (จาก MachineLearningCSV.zip)
    หรือชี้ไฟล์เดียวก็ได้ถ้ารวมมาแล้ว"""
    _check_path_exists(path, "cicids2017")
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            actual_files = os.listdir(path)
            raise FileNotFoundError(
                f"เจอโฟลเดอร์ '{os.path.abspath(path)}' จริง แต่ไม่พบไฟล์ .csv ข้างในเลย\n"
                f"  รายการที่เจอจริง: {actual_files[:20]}"
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
    """path ชี้ไปที่โฟลเดอร์ที่มี UNSW-NB15_1.csv ... _4.csv (ไม่มี header ในตัวไฟล์)
    feature_names_csv: path ของ UNSW-NB15_features.csv (คอลัมน์ Name เก็บชื่อ feature ตามลำดับ)
    """
    # เช็คก่อนเสมอว่า path ที่ได้รับมามีอยู่จริงหรือไม่ (ไฟล์ก็ได้ โฟลเดอร์ก็ได้) เพื่อให้ error
    # message ชัดเจนตั้งแต่จุดนี้ แทนที่จะปล่อยให้ตกไปเป็น error คลุมเครือจาก pandas ทีหลัง
    abs_path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ไม่พบไฟล์หรือโฟลเดอร์ที่ path นี้เลย: '{path}'\n"
            f"  (path แบบเต็มที่ Python มองหาคือ: {abs_path})\n"
            f"  วิธีตรวจสอบ (PowerShell): Test-Path '{path}'  และ  dir '{os.path.dirname(abs_path)}'\n"
            f"  สาเหตุที่พบบ่อยที่สุด: ตั้งชื่อโฟลเดอร์/path ผิด, หรือแตกไฟล์ zip แล้วได้โฟลเดอร์ซ้อนกัน "
            f"อีกชั้น (เช่น .../UNSW-NB15/UNSW-NB15/*.csv แทนที่จะเป็น .../UNSW-NB15/*.csv โดยตรง)"
        )

    if feature_names_csv and os.path.exists(feature_names_csv):
        feat_df = pd.read_csv(feature_names_csv, encoding="latin1")
        # ปรับชื่อคอลัมน์ให้ยืดหยุ่นตามไฟล์จริง (บางเวอร์ชันใช้ "Name" บางเวอร์ชันใช้ "name")
        name_col = "Name" if "Name" in feat_df.columns else feat_df.columns[1]
        col_names = feat_df[name_col].str.strip().tolist()
    elif feature_names_csv:
        print(f"[warn] ระบุ feature_names_csv='{feature_names_csv}' แต่ไม่พบไฟล์นี้ "
              f"— จะโหลดต่อโดยไม่มีชื่อคอลัมน์ (ใช้เลข index แทน) ตรวจสอบ path นี้ด้วย")
        col_names = None
    else:
        col_names = None  # จะปล่อยให้ pandas ใช้ index เป็นชื่อคอลัมน์ชั่วคราว

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "UNSW-NB15_*.csv")))
        if not files:
            # แสดงรายชื่อไฟล์จริงที่เจอในโฟลเดอร์ จะได้เห็นว่าปัญหาคือชื่อไฟล์ไม่ตรงที่คาดหรือไม่
            actual_files = os.listdir(path)
            raise FileNotFoundError(
                f"เจอโฟลเดอร์ '{abs_path}' จริง แต่ไม่พบไฟล์ที่ตรงกับรูปแบบ 'UNSW-NB15_*.csv' ข้างใน\n"
                f"  ไฟล์ที่เจอจริงในโฟลเดอร์นี้ ({len(actual_files)} รายการแรก): {actual_files[:20]}\n"
                f"  ถ้าชื่อไฟล์จริงต่างจากรูปแบบนี้ (เช่นตัวพิมพ์เล็ก/ใหญ่ต่างกัน) แจ้งชื่อไฟล์จริงมาได้เลย "
                f"จะปรับ pattern ในโค้ดให้ตรง"
            )
        frames = []
        for f in files:
            d = pd.read_csv(f, header=None, low_memory=False)
            if col_names and len(col_names) == d.shape[1]:
                d.columns = col_names
            else:
                print(f"[warn] {os.path.basename(f)}: จำนวนคอลัมน์ไม่ตรงกัน — "
                      f"feature_names_csv มี {len(col_names) if col_names else 0} ชื่อคอลัมน์ "
                      f"แต่ไฟล์ข้อมูลจริงมี {d.shape[1]} คอลัมน์ "
                      f"(จะใช้เลข index 0..{d.shape[1]-1} เป็นชื่อคอลัมน์ชั่วคราวแทน ซึ่งจะทำให้ "
                      f"ขั้นตอนถัดไปหา 'attack_cat'/'sport'/ฯลฯ ไม่เจอ — ต้องแก้ไฟล์ features_csv "
                      f"หรือ column_map ให้ตรงกับจำนวนคอลัมน์จริงนี้ก่อน)")
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path, low_memory=False)
        df = _strip_columns(df)

    # attack_cat ว่าง = traffic ปกติ ให้ label เป็น "Normal" อย่างชัดเจน
    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].fillna("Normal").str.strip()
    return df


def load_dataset(dataset_key: str, path: str, **kwargs):
    """
    dataset_key: หนึ่งใน "ertam_kaya", "cicids2017", "unsw_nb15"
    path: path ของไฟล์ / โฟลเดอร์ raw data
    คืนค่า: (X canonical DataFrame, y label Series, DatasetSpec)
    """
    if dataset_key not in ALL_SPECS:
        raise ValueError(f"ไม่รู้จัก dataset_key={dataset_key} (ต้องเป็นหนึ่งใน {list(ALL_SPECS)})")
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
        print(f"[warn] {dataset_key}: ไม่พบคอลัมน์ต่อไปนี้ในไฟล์จริง (ข้ามไปก่อน, "
              f"ต้อง verify schema): {missing_cols}")

    rename_map = {k: v for k, v in spec.column_map.items() if k in raw.columns}
    df = raw.rename(columns=rename_map)

    if spec.label_column not in raw.columns:
        raise KeyError(f"{dataset_key}: ไม่พบ label column '{spec.label_column}' ในไฟล์ที่โหลดมา "
                        f"— ตรวจสอบชื่อคอลัมน์จริงด้วย print(raw.columns.tolist())")
    y = raw[spec.label_column].astype(str).str.strip()

    # UNSW-NB15 มีปัญหา label ไม่สม่ำเสมอที่รู้กันดีในวงการ: บางแถวใน 4 ไฟล์ใช้ "Backdoor"
    # บางแถวใช้ "Backdoors" ทั้งที่เป็น attack category เดียวกัน ถ้าไม่รวมให้เป็นชื่อเดียวกันก่อน
    # จะกลายเป็น class ที่หายากยิ่งขึ้นไปอีกทั้งสองชื่อโดยไม่จำเป็น (fragmentation)
    if dataset_key == "unsw_nb15":
        n_backdoor_variant = (y == "Backdoor").sum()
        if n_backdoor_variant > 0:
            print(f"[info] unsw_nb15: พบ label 'Backdoor' (เอกพจน์) {n_backdoor_variant} แถว "
                  f"— รวมเข้ากับ 'Backdoors' (พหูพจน์) ให้เป็น class เดียวกัน เพราะเป็นปัญหา label "
                  f"ไม่สม่ำเสมอที่ทราบกันในชุดข้อมูลนี้ (ไม่ใช่ attack category ที่ต่างกันจริง)")
            y = y.replace({"Backdoor": "Backdoors"})

    # แปลงหน่วยเฉพาะกรณี CIC-IDS2017 (microsecond -> second)
    if "duration_microsec" in df.columns:
        df["duration"] = df["duration_microsec"] / 1_000_000.0
        df = df.drop(columns=["duration_microsec"])

    df = derived_bytes_total(df, spec)

    canonical_present = [c for c in CANONICAL_FEATURES if c in df.columns]
    X = df[canonical_present].copy()

    # "protocol" เป็นค่าประเภท category เสมอในทุก dataset (เช่น "tcp"/"udp"/"unas" ใน UNSW-NB15,
    # หรือเลข protocol number ใน CIC-IDS2017) ถ้าใช้ pd.to_numeric ตรงๆ แบบคอลัมน์อื่น ค่า string
    # จะกลายเป็น NaN ทั้งคอลัมน์ทันที (แบบที่เจอใน UNSW-NB15 จริง) ต้อง encode แยกด้วย factorize
    # แทน เพื่อรักษาสัญญาณของ feature นี้ไว้ ไม่ปล่อยให้กลายเป็นคอลัมน์ที่ไม่มีข้อมูลเลย
    if "protocol" in X.columns and not pd.api.types.is_numeric_dtype(X["protocol"]):
        codes, uniques = pd.factorize(X["protocol"].astype(str).fillna("unknown"))
        print(f"[info] {dataset_key}: คอลัมน์ 'protocol' เป็นข้อความ (เช่น {list(uniques[:5])}) "
              f"— แปลงเป็นรหัสตัวเลขด้วย factorize ({len(uniques)} ค่าที่ต่างกัน) แทนการทิ้งเป็น NaN")
        X["protocol"] = codes

    for col in X.columns:
        if col == "protocol":
            continue  # encode ไปแล้วด้านบน ไม่ต้องผ่าน to_numeric coercion ซ้ำ
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y, spec


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ทดสอบโหลด dataset และแสดง data-quality report")
    parser.add_argument("--dataset", required=True, choices=list(ALL_SPECS.keys()))
    parser.add_argument("--path", required=True)
    parser.add_argument("--feature_names_csv", default=None)
    args = parser.parse_args()

    X, y, spec = load_dataset(args.dataset, args.path, feature_names_csv=args.feature_names_csv)
    print(f"โหลดสำเร็จ: {spec.name} -> X shape={X.shape}, y unique classes={y.nunique()}")
    print(y.value_counts())
    print(report_data_quality(X, name=spec.name))
