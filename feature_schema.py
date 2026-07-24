"""
feature_schema.py

Canonical (unified) feature schema สำหรับงาน firewall / network-flow classification
พร้อมตาราง mapping คอลัมน์ดิบของแต่ละ dataset ไปยัง schema กลาง

หลักการ: ใช้ "flow-level" feature ที่ทั้ง 3 dataset มีร่วมกันในทางแนวคิด (ต่อให้ชื่อคอลัมน์
และหน่วยต่างกัน) ได้แก่ duration, byte count (ทั้งสองทิศทาง), packet count (ทั้งสองทิศทาง),
source/destination port, protocol
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Canonical schema: ชื่อ feature กลางที่ pipeline ทั้งหมดจะใช้ต่อจากนี้
# ---------------------------------------------------------------------------
CANONICAL_FEATURES: List[str] = [
    "src_port",
    "dst_port",
    "protocol",          # ถ้า dataset ไม่มีคอลัมน์ protocol แยก ให้ infer จาก port/flag
    "duration",           # หน่วย: วินาที
    "bytes_total",
    "bytes_sent",
    "bytes_received",
    "packets_total",
    "packets_sent",
    "packets_received",
]

# feature ที่มีเฉพาะใน Ertam & Kaya (ไม่มีใน dataset อื่น) — เก็บแยกไว้ไม่บังคับรวม
ERTAM_ONLY_FEATURES: List[str] = ["nat_src_port", "nat_dst_port"]


@dataclass
class DatasetSpec:
    """อธิบาย mapping คอลัมน์ดิบ -> canonical schema ของ 1 dataset"""
    name: str
    label_column: str
    column_map: Dict[str, str]           # raw_column_name -> canonical_name
    label_classes: List[str]
    is_firewall_action_label: bool        # True เฉพาะ Ertam & Kaya (label space เดิม)
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# 1) Ertam & Kaya — Internet Firewall Data Set (Kaggle, Firat University)
#    คอลัมน์ต้นฉบับ (ตามที่ paper/ Kaggle ระบุ):
#    Source Port, Destination Port, NAT Source Port, NAT Destination Port,
#    Action, Bytes, Bytes Sent, Bytes Received, Packets, Elapsed Time (sec),
#    pkts_sent, pkts_received
# ---------------------------------------------------------------------------
ERTAM_KAYA_SPEC = DatasetSpec(
    name="ertam_kaya_firewall",
    label_column="Action",
    is_firewall_action_label=True,
    label_classes=["allow", "deny", "drop", "reset-both"],
    column_map={
        "Source Port": "src_port",
        "Destination Port": "dst_port",
        "NAT Source Port": "nat_src_port",
        "NAT Destination Port": "nat_dst_port",
        "Bytes": "bytes_total",
        "Bytes Sent": "bytes_sent",
        "Bytes Received": "bytes_received",
        "Packets": "packets_total",
        "Elapsed Time (sec)": "duration",
        "pkts_sent": "packets_sent",
        "pkts_received": "packets_received",
    },
    notes="ไม่มีคอลัมน์ protocol แยกในเวอร์ชันที่พบทั่วไปบน Kaggle — ต้อง verify ก่อนใช้ "
          "ถ้าไม่มี ให้ตั้ง protocol='unknown' หรือ infer จาก dst_port (heuristic เท่านั้น)",
)


# ---------------------------------------------------------------------------
# 2) CIC-IDS2017 — MachineLearningCSV.zip (CICFlowMeter output, 78-79 columns)
#    หมายเหตุ: ชื่อคอลัมน์ต้นฉบับมีช่องว่างนำหน้า (" Destination Port") ต้อง strip() ก่อนใช้
# ---------------------------------------------------------------------------
CICIDS2017_SPEC = DatasetSpec(
    name="cicids2017",
    label_column="Label",
    is_firewall_action_label=False,
    label_classes=[
        "BENIGN", "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye",
        "FTP-Patator", "SSH-Patator", "DoS slowloris", "DoS Slowhttptest",
        "Bot", "Web Attack – Brute Force", "Web Attack – XSS",
        "Web Attack – Sql Injection", "Infiltration", "Heartbleed",
    ],
    column_map={
        "Destination Port": "dst_port",
        "Flow Duration": "duration_microsec",   # หน่วยเป็น microsecond ต้องแปลง /1e6 -> วินาที
        "Total Fwd Packets": "packets_sent",
        "Total Backward Packets": "packets_received",
        "Total Length of Fwd Packets": "bytes_sent",
        "Total Length of Bwd Packets": "bytes_received",
        "Protocol": "protocol",
    },
    notes="ไม่มี src_port แยกอย่างชัดเจนในบางเวอร์ชัน (บางไฟล์ไม่มีคอลัมน์นี้เลยเพราะเป็น "
          "flow-based ไม่ใช่ packet-based) — ตั้งเป็น NaN แล้ว impute/ตัดออกตอน feature "
          "selection หากไม่มีจริง ต้อง verify คอลัมน์ที่ดาวน์โหลดมาก่อนเสมอ",
)


# ---------------------------------------------------------------------------
# 3) UNSW-NB15 — UNSW-NB15_1.csv ... _4.csv (49 features ตาม UNSW-NB15_features.csv)
# ---------------------------------------------------------------------------
UNSW_NB15_SPEC = DatasetSpec(
    name="unsw_nb15",
    label_column="attack_cat",     # ใช้ attack_cat (multi-class) แทน label (binary) เพื่อให้
                                     # เทียบ severe multi-class imbalance ได้ตรงตามโจทย์
    is_firewall_action_label=False,
    label_classes=[
        "Normal", "Fuzzers", "Analysis", "Backdoors", "DoS", "Exploits",
        "Generic", "Reconnaissance", "Shellcode", "Worms",
    ],
    column_map={
        "sport": "src_port",
        "dsport": "dst_port",
        "proto": "protocol",
        "dur": "duration",
        "sbytes": "bytes_sent",
        "dbytes": "bytes_received",
        "Spkts": "packets_sent",
        "Dpkts": "packets_received",
    },
    notes="คอลัมน์ raw ไฟล์ CSV (UNSW-NB15_1.csv..._4.csv) ไม่มี header ในตัวไฟล์เอง "
          "ต้องโหลด header จากไฟล์ features แยกต่างหาก ซึ่งในการดาวน์โหลดจริงมักชื่อ "
          "'NUSW-NB15_features.csv' (สะกด N-U-S-W ไม่ใช่ U-N-S-W — เป็นจุดพิมพ์ผิดในชุดข้อมูล "
          "ต้นฉบับที่มีมานาน) ไม่ใช่ 'UNSW-NB15_features.csv' ตามชื่อ dataset โดยรวม "
          "นอกจากนี้คอลัมน์ชื่อ Spkts/Dpkts ใช้ตัวพิมพ์ใหญ่นำหน้า ต่างจากคอลัมน์อื่นๆ ส่วนใหญ่ที่เป็น "
          "ตัวพิมพ์เล็กทั้งหมด (sport, dsport, proto, dur, sbytes, dbytes) ต้อง verify ตาม features "
          "file จริงเสมอเพราะรายละเอียดนี้ต่างกันในแต่ละ mirror",
)


ALL_SPECS: Dict[str, DatasetSpec] = {
    "ertam_kaya": ERTAM_KAYA_SPEC,
    "cicids2017": CICIDS2017_SPEC,
    "unsw_nb15": UNSW_NB15_SPEC,
}


def derived_bytes_total(df, spec: DatasetSpec):
    """dataset บางตัวไม่มี bytes_total ตรงๆ ต้องคำนวณจาก sent+received"""
    if "bytes_total" not in df.columns and {"bytes_sent", "bytes_received"} <= set(df.columns):
        df["bytes_total"] = df["bytes_sent"].fillna(0) + df["bytes_received"].fillna(0)
    if "packets_total" not in df.columns and {"packets_sent", "packets_received"} <= set(df.columns):
        df["packets_total"] = df["packets_sent"].fillna(0) + df["packets_received"].fillna(0)
    return df
