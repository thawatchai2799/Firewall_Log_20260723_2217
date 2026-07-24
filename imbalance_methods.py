"""
imbalance_methods.py

รวม 5 เทคนิคจัดการ extreme class imbalance ตาม proposal ข้อ 4.2 (RQ1):
  1. baseline_smote        - SMOTE + standard cross-entropy (ของเดิม)
  2. cost_sensitive        - class-weighted cross-entropy
  3. focal_loss            - focal loss (gamma, alpha ปรับได้)
  4. hybrid_resample       - SMOTE-Tomek หรือ ADASYN + cost-sensitive loss
  5. gan_oversample        - Conditional GAN/VAE oversampling (stub, ต้องพัฒนาเพิ่ม)

ทุกฟังก์ชันคืนค่าเป็น (X_resampled, y_resampled, loss_fn) หรือ (X, y, loss_fn) แล้วแต่กรณี
เพื่อให้ pipeline.py เรียกใช้แบบเดียวกันทุกเทคนิค (uniform interface)
"""

from typing import Tuple, Callable
import numpy as np
import torch
import torch.nn as nn

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

from model import FocalLoss, compute_class_weights


def cap_majority_classes(X: np.ndarray, y: np.ndarray, max_class_size: int = 100_000,
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """ลดขนาดเฉพาะ class ที่ใหญ่เกิน max_class_size ลงมาที่ max_class_size ก่อนทำ Tomek-link
    cleaning (ไม่แตะ class ที่เล็กกว่านี้อยู่แล้ว)

    เหตุผลที่ต้องมีฟังก์ชันนี้: ขั้นตอน Tomek-link ใน SMOTETomek ต้องค้นหาคู่ nearest-neighbor
    ที่ทับซ้อนกันข้ามทั้ง dataset รวมถึง majority class ด้วย (ไม่ใช่แค่ minority class ที่ SMOTE
    สร้างเพิ่ม) สำหรับข้อมูลจริงขนาดหลักล้านแถว (เช่น CIC-IDS2017 ที่ BENIGN ~2 ล้านแถวต่อ train
    fold) ขั้นตอนนี้จะช้ามากจนแทบไม่ขยับ (สังเกตได้จาก progress log ที่ค้างอยู่ unit เดียวนานผิด
    ปกติ) การ undersample majority class ลงมาก่อน ทำให้ Tomek-link ทำงานบนข้อมูลขนาดที่จัดการได้
    โดยยังคงแนวคิด "hybrid" (oversample minority + clean boundary) ไว้ครบถ้วน
    """
    classes, counts = np.unique(y, return_counts=True)
    strategy = {int(c): max_class_size for c, cnt in zip(classes, counts) if cnt > max_class_size}
    if not strategy:
        return X, y  # ไม่มี class ไหนใหญ่เกินเพดาน ไม่ต้องทำอะไร
    rus = RandomUnderSampler(sampling_strategy=strategy, random_state=random_state)
    return rus.fit_resample(X, y)


def compute_capped_sampling_strategy(y_train: np.ndarray, ratio_cap: float = 20.0,
                                      absolute_cap: int = 50_000) -> dict:
    """คำนวณ sampling_strategy แบบจำกัดเพดาน แทนการใช้ SMOTE ค่า default ('auto') ที่จะพยายาม
    oversample ทุก minority class ให้เท่ากับ majority class เป๊ะ

    ปัญหาของ default 'auto' บนข้อมูลจริงที่ imbalance รุนแรงมาก (เช่น CIC-IDS2017 ที่ BENIGN
    ~2.27 ล้านแถว เทียบกับ Heartbleed แค่ 11 แถว = อัตราส่วน ~206,000:1) คือ SMOTE จะพยายามสร้าง
    ตัวอย่างสังเคราะห์ของ Heartbleed ขึ้นมาถึง ~2.27 ล้านแถวจากต้นฉบับแค่ 11 แถว ซึ่ง:
      1. ไม่สมเหตุสมผลทางสถิติ (สร้างจาก neighbor แค่ไม่กี่จุดซ้ำๆ นับแสนรอบ)
      2. ทำให้ dataset หลัง resample บวมขึ้นมหาศาล (รวมทุก class อาจแตะหลักสิบล้านแถว)
         กิน RAM/VRAM และเวลาฝึกจนไม่สามารถรันจบได้จริงในทางปฏิบัติ

    แนวทางนี้จำกัดไว้ 2 ชั้น: (1) oversample แต่ละ minority class ได้ไม่เกิน ratio_cap เท่า
    ของจำนวนตัวอย่างเดิมของ class นั้น (ป้องกันการขยายจากฐานที่เล็กเกินไป) และ (2) ไม่เกิน
    absolute_cap แถวต่อ class ไม่ว่ากรณีใด (ป้องกัน dataset บวมเกินจน RAM/VRAM รับไม่ไหว)
    class ที่มีจำนวนมากกว่าเพดานอยู่แล้วจะไม่ถูกแตะต้อง (SMOTE oversample อย่างเดียว ไม่ลด)
    """
    classes, counts = np.unique(y_train, return_counts=True)
    majority_count = counts.max()
    strategy = {}
    for cls, count in zip(classes, counts):
        target = min(count * ratio_cap, absolute_cap, majority_count)
        if target > count:  # ระบุใน strategy เฉพาะ class ที่จะ oversample จริง (target > ของเดิม)
            strategy[int(cls)] = int(target)
    return strategy


def _safe_smote(X_train, y_train, random_state=42, **smote_kwargs):
    strategy = compute_capped_sampling_strategy(y_train)
    if not strategy:
        # ไม่มี class ไหนต้อง oversample เลย (เช่นข้อมูลสมดุลอยู่แล้ว หรือทุก class เกิน cap แล้ว)
        return X_train, y_train
    # k_neighbors ต้องไม่เกิน (จำนวนตัวอย่างของ class ที่เล็กที่สุดที่จะ oversample - 1)
    min_class_count = min(np.sum(y_train == cls) for cls in strategy)
    k_neighbors = max(1, min(5, min_class_count - 1))
    sm = SMOTE(sampling_strategy=strategy, random_state=random_state, k_neighbors=k_neighbors, **smote_kwargs)
    return sm.fit_resample(X_train, y_train)


def baseline_smote(X_train: np.ndarray, y_train: np.ndarray, n_classes: int,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Callable]:
    """เทคนิคเดิม (ของ paper ต้นฉบับ) — ใช้เป็นจุดอ้างอิงหลักที่ต้องเอาชนะให้ได้
    ใช้ capped sampling strategy แทน 'auto' ตรงๆ เพื่อป้องกันปัญหา oversample เกินจริงบนข้อมูล
    ที่ imbalance รุนแรงมาก (ดู docstring ของ compute_capped_sampling_strategy)"""
    X_res, y_res = _safe_smote(X_train, y_train, random_state=random_state)
    loss_fn = nn.CrossEntropyLoss()
    return X_res, y_res, loss_fn


def cost_sensitive(X_train: np.ndarray, y_train: np.ndarray, n_classes: int,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Callable]:
    """ไม่ resample ข้อมูล แต่ปรับ loss ให้ให้น้ำหนัก minority class มากขึ้น"""
    y_tensor = torch.as_tensor(y_train, dtype=torch.long)
    weights = compute_class_weights(y_tensor, n_classes)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    return X_train, y_train, loss_fn


def focal_loss_method(X_train: np.ndarray, y_train: np.ndarray, n_classes: int,
                       gamma: float = 2.0, use_alpha: bool = True,
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Callable]:
    y_tensor = torch.as_tensor(y_train, dtype=torch.long)
    alpha = compute_class_weights(y_tensor, n_classes) if use_alpha else None
    loss_fn = FocalLoss(gamma=gamma, alpha=alpha)
    return X_train, y_train, loss_fn


def hybrid_resample(X_train: np.ndarray, y_train: np.ndarray, n_classes: int,
                     method: str = "smote_tomek",
                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Callable]:
    """SMOTE-Tomek (oversample + clean overlap ด้วย Tomek links) หรือ ADASYN
    (สร้างตัวอย่างมากขึ้นตรงบริเวณที่ classifier แยกยาก) ร่วมกับ cost-sensitive loss
    ใช้ capped sampling strategy เดียวกับ baseline_smote เพื่อป้องกัน dataset บวมเกินจริง
    และ cap majority class ก่อนเสมอ เพื่อป้องกัน Tomek-link/ADASYN ค้างนานผิดปกติบนข้อมูลจริง
    ขนาดหลักล้านแถว (ดู cap_majority_classes)"""
    # cap majority class ก่อนเสมอ ไม่ว่าจะเลือก method ไหน (ป้องกันปัญหาความเร็วที่ต้นตอ)
    X_train, y_train = cap_majority_classes(X_train, y_train, max_class_size=100_000,
                                             random_state=random_state)

    strategy = compute_capped_sampling_strategy(y_train)
    if not strategy:
        X_res, y_res = X_train, y_train
    elif method == "smote_tomek":
        min_class_count = min(np.sum(y_train == cls) for cls in strategy)
        k_neighbors = max(1, min(5, min_class_count - 1))
        sampler = SMOTETomek(sampling_strategy=strategy, random_state=random_state,
                              smote=SMOTE(sampling_strategy=strategy, random_state=random_state,
                                          k_neighbors=k_neighbors))
        X_res, y_res = sampler.fit_resample(X_train, y_train)
    elif method == "adasyn":
        min_class_count = min(np.sum(y_train == cls) for cls in strategy)
        n_neighbors = max(1, min(5, min_class_count - 1))
        sampler = ADASYN(sampling_strategy=strategy, random_state=random_state, n_neighbors=n_neighbors)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
    else:
        raise ValueError(f"ไม่รู้จัก method={method}")

    y_tensor = torch.as_tensor(y_res, dtype=torch.long)
    weights = compute_class_weights(y_tensor, n_classes)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    return X_res, y_res, loss_fn


class GANOversampler:
    """STUB — โครงสร้างเตรียมไว้สำหรับพัฒนาเพิ่มตาม timeline เดือน 5-6

    แนวทางที่แนะนำ: ใช้ library `ctgan` หรือ `sdv` (Synthetic Data Vault) ฝึก Conditional
    GAN แยกเฉพาะ minority-class rows แล้ว sample สร้างตัวอย่างสังเคราะห์เพิ่ม
    ข้อดีกว่า SMOTE เชิงเรขาคณิต: จับ non-linear joint distribution ของ feature ได้ดีกว่า
    ตัวอย่างการต่อยอด (pseudo-code):

        from ctgan import CTGAN
        minority_df = train_df[train_df[label_col] == minority_class]
        ctgan = CTGAN(epochs=300)
        ctgan.fit(minority_df, discrete_columns=[])
        synthetic = ctgan.sample(n_samples_needed)
    """

    def __init__(self, epochs: int = 300):
        self.epochs = epochs
        raise NotImplementedError(
            "GANOversampler ยังไม่ implement เต็มรูปแบบ — ติดตั้ง `ctgan` แล้วพัฒนาต่อ "
            "ตาม docstring ด้านบน ก่อนใช้งานจริงในเดือน 5-6 ของแผนงาน"
        )


TECHNIQUES = {
    "baseline_smote": baseline_smote,
    "cost_sensitive": cost_sensitive,
    "focal_loss": focal_loss_method,
    "hybrid_resample": hybrid_resample,
    # "gan_oversample": GANOversampler  # เปิดใช้เมื่อพัฒนาเสร็จ
}
