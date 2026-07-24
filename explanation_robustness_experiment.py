"""
explanation_robustness_experiment.py

ทดสอบสิ่งที่หัวข้อ 7.2 ของบทความระบุไว้ว่า "ยังไม่ได้ทำ": ตรวจสอบว่าคำอธิบาย SHAP
เสถียรแค่ไหนภายใต้ (1) การโจมตีแบบปรปักษ์ และ (2) การทำ adversarial training

เปรียบเทียบ 3 เงื่อนไข บนตัวอย่าง background/explain ชุดเดียวกันทุกเงื่อนไข (เพื่อให้เทียบกันได้
อย่างยุติธรรม ไม่ปนกับความแปรปรวนจากการสุ่มตัวอย่างต่างกัน):

  1. standard_clean      : โมเดลมาตรฐาน, input ปกติ (ไม่ถูกรบกวน)
  2. standard_perturbed  : โมเดลมาตรฐาน (ตัวเดิม), input ที่ถูกโจมตีด้วย FGSM ที่ ε=0.3
                           (เดียวกับที่ใช้ตอน adversarial training ใน Section 4.6/6.4)
  3. adversarial_clean   : โมเดลที่ผ่าน adversarial training, input ปกติ (ไม่ถูกรบกวน)

การเปรียบเทียบ (1)-(2) ตอบคำถาม "explanation เปลี่ยนไปหรือไม่เมื่อโดนโจมตี" (explainability
drift ตาม Rajhans & Khawarey [25])
การเปรียบเทียบ (1)-(3) ตอบคำถาม "การทำ robust training เปลี่ยนสิ่งที่โมเดลใช้ตัดสินใจหรือไม่"

วัด drift ด้วย 3 ตัวชี้วัด: Spearman rank correlation ของอันดับความสำคัญคุณลักษณะ,
cosine similarity ของ importance vector, และ top-1/top-3 feature overlap

ใช้งาน:
    python explanation_robustness_experiment.py --dataset ertam_kaya --path /path/to/log2.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from data_loader import load_dataset
from model import FirewallMLP
from imbalance_methods import baseline_smote
from adversarial_robustness import fgsm_attack, PERTURBABLE_FEATURES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_SIZE = 200
LEARNING_RATE = 1e-3
EPOCHS = 150
EPSILON = 0.3           # ระดับ perturbation เดียวกับที่ใช้ตอน adversarial training ใน Section 4.6
N_BACKGROUND = 50
N_EXPLAIN = 100


def train_model(X_train, y_train, n_features, n_classes, adversarial, perturb_mask):
    """ฝึกโมเดลมาตรฐาน (adversarial=False) หรือโมเดลที่ผ่าน adversarial training
    (adversarial=True) — โครงสร้างเหมือนกับที่ใช้ใน run_adversarial_unit ของ
    run_full_experiment.py ทุกประการ เพื่อให้โมเดลที่ได้เทียบเคียงกับผลใน Table 8 ได้ตรงๆ"""
    model = FirewallMLP(n_features=n_features, n_classes=n_classes, hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    X_t = torch.as_tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.as_tensor(y_train, dtype=torch.long).to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        if adversarial:
            X_adv = fgsm_attack(model, loss_fn, X_train, y_train, EPSILON, perturb_mask)
            X_adv_t = torch.as_tensor(X_adv, dtype=torch.float32).to(DEVICE)
            loss = 0.5 * loss + 0.5 * loss_fn(model(X_adv_t), y_t)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def make_predict_proba(model):
    def predict_proba(x_numpy):
        with torch.no_grad():
            logits = model(torch.as_tensor(x_numpy, dtype=torch.float32).to(DEVICE))
            return torch.softmax(logits, dim=1).cpu().numpy()
    return predict_proba


def global_importance_vector(shap_values, n_classes):
    """รวม mean(|SHAP value|) ข้ามทุก class ให้เหลือ vector เดียวความยาว n_features
    (เหมือนความยาวรวมของแท่งใน summary_plot type='bar') ใช้เป็นตัวแทน 'อันดับความสำคัญ
    ของคุณลักษณะในภาพรวม' สำหรับเปรียบเทียบ 3 เงื่อนไข"""
    if isinstance(shap_values, list):
        stacked = np.stack(shap_values, axis=0)  # (n_classes, n_samples, n_features)
    else:
        stacked = np.transpose(shap_values, (2, 0, 1))  # -> (n_classes, n_samples, n_features)
    per_class_mean_abs = np.mean(np.abs(stacked), axis=1)  # (n_classes, n_features)
    return per_class_mean_abs.sum(axis=0)  # (n_features,) รวมทุก class


def compare_importance(vec_a, vec_b, feature_names, label_a, label_b):
    rho, pval = spearmanr(vec_a, vec_b)
    cos_sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-12))
    top3_a = set(np.array(feature_names)[np.argsort(vec_a)[::-1][:3]])
    top3_b = set(np.array(feature_names)[np.argsort(vec_b)[::-1][:3]])
    top1_a = feature_names[int(np.argmax(vec_a))]
    top1_b = feature_names[int(np.argmax(vec_b))]
    return {
        "comparison": f"{label_a}_vs_{label_b}",
        "spearman_rho": float(rho), "spearman_p": float(pval),
        "cosine_similarity": cos_sim,
        "top1_feature_a": top1_a, "top1_feature_b": top1_b,
        "top1_same": bool(top1_a == top1_b),
        "top3_overlap_count": len(top3_a & top3_b),
        "top3_features_a": sorted(top3_a), "top3_features_b": sorted(top3_b),
    }


def main(dataset_key, path, feature_names_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[1/6] โหลดข้อมูล {dataset_key} จาก {path}")
    X_df, y_raw, spec = load_dataset(dataset_key, path, feature_names_csv=feature_names_csv)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
    feature_names = X_df.columns.tolist()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_.tolist()
    n_classes = len(class_names)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_df.values, y, test_size=0.25, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    perturb_mask = np.array([1.0 if f in PERTURBABLE_FEATURES else 0.0 for f in feature_names])

    print("[2/6] ฝึกโมเดลมาตรฐาน (standard) และโมเดลที่ผ่าน adversarial training")
    X_res, y_res, _ = baseline_smote(X_train, y_train, n_classes)
    model_standard = train_model(X_res, y_res, X_train.shape[1], n_classes, adversarial=False, perturb_mask=perturb_mask)
    model_adversarial = train_model(X_res, y_res, X_train.shape[1], n_classes, adversarial=True, perturb_mask=perturb_mask)

    acc_standard = (make_predict_proba(model_standard)(X_test).argmax(axis=1) == y_test).mean()
    acc_adversarial = (make_predict_proba(model_adversarial)(X_test).argmax(axis=1) == y_test).mean()
    print(f"      clean accuracy: standard={acc_standard:.4f}, adversarial-trained={acc_adversarial:.4f}")

    print("[3/6] เตรียมตัวอย่าง background/explain ชุดเดียวกันสำหรับทุกเงื่อนไข")
    background = shap.sample(X_train, N_BACKGROUND, random_state=42)
    explain_clean = X_test[:N_EXPLAIN].copy()

    print("[4/6] สร้างตัวอย่างปรปักษ์จาก explain set เดียวกัน (ε=0.3, ใช้ gradient ของโมเดลมาตรฐาน)")
    loss_fn = nn.CrossEntropyLoss()
    explain_perturbed = fgsm_attack(model_standard, loss_fn, explain_clean,
                                     y_test[:N_EXPLAIN], EPSILON, perturb_mask)

    print("[5/6] คำนวณ SHAP ทั้ง 3 เงื่อนไข (อาจใช้เวลาหลายนาทีต่อเงื่อนไข)")
    results = {}
    conditions = [
        ("standard_clean", model_standard, explain_clean),
        ("standard_perturbed", model_standard, explain_perturbed),
        ("adversarial_clean", model_adversarial, explain_clean),
    ]
    importance_vectors = {}
    for name, model, sample in conditions:
        print(f"      กำลังคำนวณ SHAP: {name}")
        explainer = shap.KernelExplainer(make_predict_proba(model), background)
        raw_shap = explainer.shap_values(sample, nsamples=100)
        shap_values = raw_shap if isinstance(raw_shap, list) else \
            [raw_shap[:, :, c] for c in range(n_classes)]
        importance_vectors[name] = global_importance_vector(shap_values, n_classes)

        plt.figure()
        shap.summary_plot(shap_values, sample, feature_names=feature_names,
                           class_names=class_names, plot_type="bar", show=False)
        plt.title(f"Global Feature Importance \u2014 {name} ({dataset_key})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_explanation_robustness_{name}.png"), dpi=150)
        plt.close()

    print("[6/6] เปรียบเทียบ drift ระหว่างเงื่อนไข")
    comparisons = [
        compare_importance(importance_vectors["standard_clean"], importance_vectors["standard_perturbed"],
                            feature_names, "standard_clean", "standard_perturbed"),
        compare_importance(importance_vectors["standard_clean"], importance_vectors["adversarial_clean"],
                            feature_names, "standard_clean", "adversarial_clean"),
        compare_importance(importance_vectors["standard_perturbed"], importance_vectors["adversarial_clean"],
                            feature_names, "standard_perturbed", "adversarial_clean"),
    ]
    for c in comparisons:
        print(f"  {c['comparison']}: Spearman rho={c['spearman_rho']:.3f} (p={c['spearman_p']:.4f}), "
              f"cosine={c['cosine_similarity']:.3f}, top1 same={c['top1_same']}, "
              f"top3 overlap={c['top3_overlap_count']}/3")

    # กราฟเปรียบเทียบ importance ranking แบบเห็นภาพรวมทีเดียว (grouped bar ของ 3 เงื่อนไข)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(feature_names))
    width = 0.27
    order = np.argsort(importance_vectors["standard_clean"])[::-1]
    sorted_features = [feature_names[i] for i in order]
    for i, name in enumerate(["standard_clean", "standard_perturbed", "adversarial_clean"]):
        vals = importance_vectors[name][order]
        ax.bar(x + i * width, vals, width, label=name)
    ax.set_xticks(x + width)
    ax.set_xticklabels(sorted_features, rotation=45, ha="right")
    ax.set_ylabel("Total mean(|SHAP value|) across classes")
    ax.set_title(f"Explanation-Robustness Comparison ({dataset_key})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_explanation_robustness_comparison.png"), dpi=150)
    plt.close()

    output = {
        "dataset_key": dataset_key,
        "epsilon": EPSILON,
        "clean_accuracy": {"standard": float(acc_standard), "adversarial_trained": float(acc_adversarial)},
        "feature_names": feature_names,
        "importance_vectors": {k: v.tolist() for k, v in importance_vectors.items()},
        "comparisons": comparisons,
    }
    out_path = os.path.join(output_dir, "explanation_robustness_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nบันทึกผลลัพธ์ที่ {out_path}")
    print(f"บันทึกกราฟที่ {output_dir}/fig_explanation_robustness_comparison.png และ shap_explanation_robustness_*.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ertam_kaya")
    parser.add_argument("--path", required=True)
    parser.add_argument("--feature_names_csv", default=None)
    parser.add_argument("--output", default="explanation_robustness_results")
    args = parser.parse_args()
    main(args.dataset, args.path, args.feature_names_csv, args.output)
