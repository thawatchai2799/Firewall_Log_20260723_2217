"""
feature_schema.py

Canonical (unified) feature schema for firewall / network-flow classification,
with a mapping table from each dataset's raw columns to the shared schema.

Principle: use "flow-level" features that all three datasets share conceptually (even if column
names and units differ): duration, byte count (both directions), packet count (both directions),
source/destination port, and protocol.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Canonical schema: the shared feature names the whole pipeline uses from here on
# ---------------------------------------------------------------------------
CANONICAL_FEATURES: List[str] = [
    "src_port",
    "dst_port",
    "protocol",          # if a dataset lacks a protocol column, infer from port/flag
    "duration",           # unit: seconds
    "bytes_total",
    "bytes_sent",
    "bytes_received",
    "packets_total",
    "packets_sent",
    "packets_received",
]

# Features unique to Ertam & Kaya (absent from the others) - kept separate, not required
ERTAM_ONLY_FEATURES: List[str] = ["nat_src_port", "nat_dst_port"]


@dataclass
class DatasetSpec:
    """Describes the raw-column -> canonical-schema mapping for one dataset."""
    name: str
    label_column: str
    column_map: Dict[str, str]           # raw_column_name -> canonical_name
    label_classes: List[str]
    is_firewall_action_label: bool        # True only for Ertam & Kaya (original label space)
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# 1) Ertam & Kaya - Internet Firewall Data Set (Kaggle, Firat University)
#    Original columns (per this study / Kaggle):
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
    notes="No separate protocol column in the common Kaggle version - verify before use. "
          "If absent, set protocol='unknown' or infer from dst_port (heuristic only).",
)


# ---------------------------------------------------------------------------
# 2) CIC-IDS2017 - MachineLearningCSV.zip (CICFlowMeter output, 78-79 columns)
#    Note: original column names have leading spaces (" Destination Port"); strip() before use
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
        "Flow Duration": "duration_microsec",   # in microseconds; convert /1e6 -> seconds
        "Total Fwd Packets": "packets_sent",
        "Total Backward Packets": "packets_received",
        "Total Length of Fwd Packets": "bytes_sent",
        "Total Length of Bwd Packets": "bytes_received",
        "Protocol": "protocol",
    },
    notes="Some versions lack a clear src_port column (flow-based files may omit it entirely "
          "since they are flow-based, not packet-based) - set to NaN then impute/drop during "
          "feature selection if truly absent. Always verify the downloaded columns first.",
)


# ---------------------------------------------------------------------------
# 3) UNSW-NB15 - UNSW-NB15_1.csv ... _4.csv (49 features per UNSW-NB15_features.csv)
# ---------------------------------------------------------------------------
UNSW_NB15_SPEC = DatasetSpec(
    name="unsw_nb15",
    label_column="attack_cat",     # use attack_cat (multi-class) rather than label (binary) to
                                     # match the severe multi-class imbalance the study targets
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
    notes="The raw CSV files (UNSW-NB15_1.csv..._4.csv) have no header row themselves. "
          "The header must be loaded from a separate features file, which is usually named "
          "'NUSW-NB15_features.csv' (spelled N-U-S-W, not U-N-S-W - a long-standing typo in the "
          "original dataset), not 'UNSW-NB15_features.csv' as the dataset name would suggest. "
          "Also, Spkts/Dpkts are capitalised unlike most other columns, which are all "
          "lowercase (sport, dsport, proto, dur, sbytes, dbytes). Always verify against the features "
          "file, because these details differ between mirrors.",
)


ALL_SPECS: Dict[str, DatasetSpec] = {
    "ertam_kaya": ERTAM_KAYA_SPEC,
    "cicids2017": CICIDS2017_SPEC,
    "unsw_nb15": UNSW_NB15_SPEC,
}


def derived_bytes_total(df, spec: DatasetSpec):
    """Some datasets lack bytes_total directly; compute it from sent+received."""
    if "bytes_total" not in df.columns and {"bytes_sent", "bytes_received"} <= set(df.columns):
        df["bytes_total"] = df["bytes_sent"].fillna(0) + df["bytes_received"].fillna(0)
    if "packets_total" not in df.columns and {"packets_sent", "packets_received"} <= set(df.columns):
        df["packets_total"] = df["packets_sent"].fillna(0) + df["packets_received"].fillna(0)
    return df
