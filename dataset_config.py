"""
dataset_config.py

ไฟล์ config แยกต่างหากจากตัวโค้ดหลัก (run_full_experiment.py) โดยเจตนา

**ไฟล์นี้จะไม่ถูกแก้ไขโดยผู้พัฒนาอีกเลยไม่ว่าจะส่ง run_full_experiment.py เวอร์ชันใหม่กี่ครั้ง
เพื่อแก้บั๊กก็ตาม** — แก้ path/ค่าตั้งต้นทั้งหมดที่ไฟล์นี้ไฟล์เดียว ปลอดภัยจากการถูก overwrite
เวลาที่ต้องอัปเดตไฟล์โค้ดอื่นๆ (run_full_experiment.py, data_loader.py, ฯลฯ)
"""

# =============================================================================
# Path ของข้อมูลจริง — แก้ตรงนี้ให้ตรงกับเครื่องของคุณ
# =============================================================================
DATASET_PATHS = {
    "ertam_kaya": {
        "path": "../data/Internet_Firewall_DataSet/log2.csv",
        "feature_names_csv": None,
    },
    "cicids2017": {
        "path": "../data/CIC_IDS2017/MachineLearningCVE/",
        "feature_names_csv": None,
    },
    "unsw_nb15": {
        "path": "../data/UNSW_NB15/",
        "feature_names_csv": "../data/UNSW_NB15/NUSW-NB15_features.csv",
    },
}

# =============================================================================
# ค่าตั้งต้นอื่นๆ ของการทดลอง (ปกติไม่ต้องแก้ เว้นแต่ต้องการปรับ)
# =============================================================================
TECHNIQUE_NAMES = ["baseline_smote", "cost_sensitive", "focal_loss", "hybrid_resample"]
N_FOLDS = 5
EPOCHS = 100
HIDDEN_SIZE = 200
BATCH_SIZE = 512          # mini-batch — สำคัญมากสำหรับ dataset ขนาดล้านแถวขึ้นไป
LEARNING_RATE = 1e-3
HEARTBEAT_SECONDS = 60    # รายงานความคืบหน้าทุกกี่วินาที

SHAP_DATASETS = {         # (dataset_key -> hardest_class_name) สำหรับ SHAP analysis
    "ertam_kaya": "deny",
    "unsw_nb15": "Analysis",
}
ADVERSARIAL_DATASET = "ertam_kaya"

OUTPUT_DIR = "results_final"
PROGRESS_LOG_PATH = "progress.log"
