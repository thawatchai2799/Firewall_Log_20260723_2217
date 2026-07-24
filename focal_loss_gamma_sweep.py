"""
focal_loss_gamma_sweep.py

การทดลองต่อยอด (ข้อ 3 ที่เคยเสนอไว้): ค้นหาว่าค่า focusing parameter gamma ของ focal loss
ที่เหมาะสมกับข้อมูล firewall log ที่ไม่สมดุลรุนแรง ควรเป็นเท่าไหร่ แทนที่จะใช้ค่า default
gamma=2.0 (ซึ่งออกแบบมาสำหรับ object detection) ที่ผลการทดลองหลักพบว่าให้ผลแย่ที่สุดในทุก
dataset โดยไม่มีหลักฐานรองรับว่าเป็นเพราะค่า gamma ไม่เหมาะ หรือเป็นข้อจำกัดของ focal loss เอง

ทดสอบ gamma = 0.5, 1, 2 (baseline เดิม), 5 บน Dataset A (Ertam & Kaya) ด้วย stratified
5-fold cross-validation แบบเดียวกับการทดลองหลัก (Section 6.1 ของบทความ) เพื่อให้เทียบผลกันได้
ตรงๆ พร้อมทดสอบนัยสำคัญทางสถิติ (Wilcoxon) และ effect size (Cohen's d) ระหว่าง gamma ที่ดี
ที่สุดกับ gamma=2.0 เดิม

ใช้งาน:
    python focal_loss_gamma_sweep.py --dataset ertam_kaya --path /path/to/log2.csv
"""

import argparse
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, precision_recall_fscore_support

from data_loader import load_dataset
from imbalance_methods import focal_loss_method
from run_full_experiment import train_and_eval_fold, HIDDEN_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS

GAMMA_VALUES = [0.5, 1.0, 2.0, 5.0]   # 2.0 = ค่า default เดิมที่ใช้ในผลการทดลองหลัก (Table 5)
N_FOLDS = 5


def cohend(m1, s1, n1, m2, s2, n2):
    pooled = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return 0.0 if pooled == 0 else (m1 - m2) / pooled


def main(dataset_key, path, feature_names_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[1/3] โหลดข้อมูล {dataset_key} จาก {path}")
    X_df, y_raw, spec = load_dataset(dataset_key, path, feature_names_csv=feature_names_csv)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_.tolist()
    n_classes = len(class_names)
    X = X_df.values

    # ใช้ fold เดียวกับการทดลองหลัก (random_state=42) เพื่อให้เทียบกับ Table 5 ได้ตรงๆ
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = list(skf.split(X, y))

    print(f"[2/3] ทดสอบ focal loss ที่ gamma = {GAMMA_VALUES} บน {N_FOLDS}-fold CV")
    results_by_gamma = {}
    for gamma in GAMMA_VALUES:
        print(f"  กำลังทดสอบ gamma={gamma}")
        fold_macro_f1 = []
        fold_per_class_f1 = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train_raw, X_val_raw = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)

            X_res, y_res, loss_fn = focal_loss_method(X_train, y_train, n_classes, gamma=gamma)
            val_pred = train_and_eval_fold(
                X_res, y_res, X_val, y_val,
                n_features=X_train.shape[1], n_classes=n_classes, loss_fn=loss_fn,
                epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, hidden_size=HIDDEN_SIZE,
            )
            macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
            _, _, per_class_f1, _ = precision_recall_fscore_support(
                y_val, val_pred, labels=range(n_classes), zero_division=0)
            fold_macro_f1.append(float(macro_f1))
            fold_per_class_f1.append(per_class_f1.tolist())
            print(f"    fold {fold_idx+1}/{N_FOLDS}: macro-F1={macro_f1:.4f}")

        results_by_gamma[gamma] = {
            "fold_macro_f1": fold_macro_f1,
            "mean_macro_f1": float(np.mean(fold_macro_f1)),
            "std_macro_f1": float(np.std(fold_macro_f1)),
            "mean_per_class_f1": np.mean(fold_per_class_f1, axis=0).tolist(),
            "class_names": class_names,
        }

    print("[3/3] เปรียบเทียบ gamma ที่ดีที่สุดกับ gamma=2.0 (ค่า default เดิม) ทางสถิติ")
    best_gamma = max(GAMMA_VALUES, key=lambda g: results_by_gamma[g]["mean_macro_f1"])
    baseline_gamma = 2.0
    comparisons = {}
    if best_gamma != baseline_gamma:
        a = results_by_gamma[best_gamma]["fold_macro_f1"]
        b = results_by_gamma[baseline_gamma]["fold_macro_f1"]
        try:
            stat, pval = wilcoxon(a, b)
        except ValueError:
            stat, pval = float("nan"), float("nan")
        d = cohend(np.mean(a), np.std(a), len(a), np.mean(b), np.std(b), len(b))
        comparisons[f"gamma_{best_gamma}_vs_gamma_{baseline_gamma}"] = {
            "wilcoxon_statistic": float(stat), "wilcoxon_p": float(pval), "cohens_d": float(d),
        }
        print(f"  gamma={best_gamma} (ดีที่สุด) vs gamma={baseline_gamma} (default เดิม): "
              f"Wilcoxon p={pval:.4f}, Cohen's d={d:.3f}")
    else:
        print(f"  gamma=2.0 (ค่า default เดิม) ยังคงเป็นค่าที่ดีที่สุดในบรรดาที่ทดสอบ ไม่มีอะไรต้องเทียบเพิ่ม")

    # กราฟเปรียบเทียบ macro-F1 และ per-class F1 ตามค่า gamma
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    means = [results_by_gamma[g]["mean_macro_f1"] for g in GAMMA_VALUES]
    stds = [results_by_gamma[g]["std_macro_f1"] for g in GAMMA_VALUES]
    colors = ["#4C72B0" if g != 2.0 else "#DD8452" for g in GAMMA_VALUES]
    axes[0].bar([str(g) for g in GAMMA_VALUES], means, yerr=stds, capsize=5, color=colors)
    axes[0].set_xlabel("Focal loss gamma")
    axes[0].set_ylabel("Macro-F1 (5-fold CV mean \u00b1 SD)")
    axes[0].set_title(f"(a) Macro-F1 vs. gamma ({dataset_key})")
    axes[0].grid(axis="y", alpha=0.3)

    x = np.arange(len(class_names))
    width = 0.2
    for i, g in enumerate(GAMMA_VALUES):
        axes[1].bar(x + i * width, results_by_gamma[g]["mean_per_class_f1"], width, label=f"\u03b3={g}")
    axes[1].set_xticks(x + width * 1.5)
    axes[1].set_xticklabels(class_names)
    axes[1].set_ylabel("F1-score")
    axes[1].set_title(f"(b) Per-Class F1 vs. gamma ({dataset_key})")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "fig_focal_loss_gamma_sweep.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    output = {
        "dataset_key": dataset_key,
        "gamma_values_tested": GAMMA_VALUES,
        "results_by_gamma": {str(g): v for g, v in results_by_gamma.items()},
        "best_gamma": best_gamma,
        "comparisons_vs_default_gamma_2": comparisons,
    }
    out_path = os.path.join(output_dir, "focal_loss_gamma_sweep_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nบันทึกผลลัพธ์ที่ {out_path}")
    print(f"บันทึกกราฟที่ {fig_path}")
    print(f"\nสรุป: gamma ที่ดีที่สุดคือ {best_gamma} "
          f"(macro-F1={results_by_gamma[best_gamma]['mean_macro_f1']:.4f}) "
          f"เทียบกับ gamma=2.0 เดิม (macro-F1={results_by_gamma[2.0]['mean_macro_f1']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ertam_kaya")
    parser.add_argument("--path", required=True)
    parser.add_argument("--feature_names_csv", default=None)
    parser.add_argument("--output", default="focal_loss_sweep_results")
    args = parser.parse_args()
    main(args.dataset, args.path, args.feature_names_csv, args.output)
