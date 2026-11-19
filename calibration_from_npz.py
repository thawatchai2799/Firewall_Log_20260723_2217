"""
calibration_from_npz.py

Calibration analysis computed from the per-unit prediction files written by
rerun_with_predictions.py. Nothing is retrained: every number comes from the saved
predicted probabilities (proba) and true labels (y_true) in

    <input>/<dataset>/<variant>/seed<S>_fold<F>.npz

Metrics per unit (one dataset x variant x seed x fold):
  * ece_top        Top-label expected calibration error (Guo et al., 2017): equal-width
                   bins on the confidence of the predicted class, weighted mean of
                   |accuracy - mean confidence| over bins.
  * ece_classwise  Class-wise ECE (Kull et al., 2019): the same binned error computed
                   one-vs-rest on p_k for every class k, then averaged over classes. More
                   informative than ece_top under heavy imbalance, because ece_top is
                   dominated by the majority class.
  * brier          Multiclass Brier score: mean over rows of sum_k (p_k - 1[y = k])^2.
  * nll            Negative log-likelihood: mean of -log p_y, with p_y clipped at 1e-12.
  * mean_conf, accuracy, overconfidence (= mean_conf - accuracy).

Per class, per unit: one-vs-rest ECE on p_k, mean predicted p_k, and the observed class
frequency in the validation fold (a class whose mean p_k far exceeds its frequency is
being over-predicted, as expected after SMOTE or inverse-frequency weighting).

Reliability diagrams: for the headline variants, the top-label confidence bins are
pooled over all seeds and folds of a dataset (bin sums are accumulated, so memory use
does not grow with dataset size). The bin table is saved as CSV so the figure can be
redrawn without re-reading the .npz files.

Robustness details:
  * Probabilities are stored as float16; each row is cast to float64 and renormalised to
    sum to 1 before any metric is computed.
  * Non-finite or all-zero rows (possible if a model diverged) are replaced by a uniform
    row and counted in the n_bad_rows column, so they are visible rather than silent.
  * Units without a proba array (run with --no_proba) are skipped and listed.

Outputs (in --output):
  calibration_per_unit.csv        one row per unit
  calibration_per_class.csv       one row per unit x class
  calibration_summary.csv         mean and 95% CI (mean +/- 1.96 SD / sqrt(n)) per
                                  dataset x variant
  calibration_reliability_bins.csv  pooled bins for the headline variants
  fig_reliability.png / .pdf      reliability diagrams (600 DPI)

Usage:
  python calibration_from_npz.py --input rerun_results --output calibration_results
  python calibration_from_npz.py --selftest          # synthetic check, writes nothing
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HEADLINE_DEFAULT = ["xgb_classweight", "xgb_smote", "mlp_baseline_smote"]
DATASET_LABEL = {"ertam_kaya": "Dataset A", "cicids2017": "Dataset B",
                 "unsw_nb15": "Dataset C"}
UNIT_RE = re.compile(r"seed(\d+)_fold(\d+)\.npz$")


# ---------------------------------------------------------------------------
# Core metric functions (pure numpy; tested by --selftest)

def clean_proba(proba):
    """float64, rows renormalised; returns (p, n_bad_rows)."""
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"proba must be 2-D, got shape {p.shape}")
    finite = np.isfinite(p).all(axis=1)
    p = np.where(np.isfinite(p), p, 0.0)
    p = np.clip(p, 0.0, None)
    s = p.sum(axis=1)
    bad = (~finite) | (s <= 0)
    if bad.any():
        p[bad] = 1.0 / p.shape[1]
        s = p.sum(axis=1)
    p = p / s[:, None]
    return p, int(bad.sum())


def bin_index(values, n_bins):
    """Equal-width bins on [0, 1]; value 1.0 falls in the last bin."""
    idx = np.floor(values * n_bins).astype(np.int64)
    return np.clip(idx, 0, n_bins - 1)


def binned_sums(conf, hit, n_bins):
    """Per-bin count, sum of confidence, sum of hits."""
    b = bin_index(conf, n_bins)
    cnt = np.bincount(b, minlength=n_bins).astype(np.float64)
    s_conf = np.bincount(b, weights=conf, minlength=n_bins)
    s_hit = np.bincount(b, weights=hit.astype(np.float64), minlength=n_bins)
    return cnt, s_conf, s_hit


def ece_from_sums(cnt, s_conf, s_hit):
    n = cnt.sum()
    if n == 0:
        return float("nan")
    nz = cnt > 0
    return float(np.sum(np.abs(s_hit[nz] - s_conf[nz])) / n)


def top_label_ece(p, y, n_bins):
    conf = p.max(axis=1)
    hit = (p.argmax(axis=1) == y)
    return ece_from_sums(*binned_sums(conf, hit, n_bins))


def one_vs_rest_ece(p, y, k, n_bins):
    return ece_from_sums(*binned_sums(p[:, k], (y == k), n_bins))


def unit_metrics(proba, y_true, n_bins):
    p, n_bad = clean_proba(proba)
    y = np.asarray(y_true, dtype=np.int64)
    if len(y) != p.shape[0]:
        raise ValueError(f"y_true has {len(y)} rows, proba has {p.shape[0]}")
    if y.min() < 0 or y.max() >= p.shape[1]:
        raise ValueError("y_true contains a label outside 0..n_classes-1")
    n, k = p.shape
    rows = np.arange(n)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    acc = float(np.mean(pred == y))

    per_class = []
    for c in range(k):
        per_class.append({
            "class_index": c,
            "ece_ovr": one_vs_rest_ece(p, y, c, n_bins),
            "mean_p": float(p[:, c].mean()),
            "observed_freq": float(np.mean(y == c)),
            "support": int(np.sum(y == c)),
        })

    # Brier, computed without building a one-hot matrix
    sq = np.sum(p * p, axis=1) - 2.0 * p[rows, y] + 1.0
    m = {
        "n_rows": int(n), "n_classes": int(k), "n_bad_rows": n_bad,
        "accuracy": acc,
        "mean_conf": float(conf.mean()),
        "overconfidence": float(conf.mean() - acc),
        "ece_top": top_label_ece(p, y, n_bins),
        "ece_classwise": float(np.mean([r["ece_ovr"] for r in per_class])),
        "brier": float(np.mean(sq)),
        "nll": float(np.mean(-np.log(np.clip(p[rows, y], 1e-12, None)))),
    }
    sums = binned_sums(conf, pred == y, n_bins)
    return m, per_class, sums


def ci95(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mu = float(x.mean())
    if n == 1:
        return mu, float("nan"), float("nan"), 1
    h = 1.96 * float(x.std(ddof=1)) / math.sqrt(n)
    return mu, mu - h, mu + h, n


# ---------------------------------------------------------------------------
# Driver

def find_units(input_dir):
    units = []
    for path in glob.glob(os.path.join(input_dir, "*", "*", "seed*_fold*.npz")):
        m = UNIT_RE.search(os.path.basename(path))
        if not m:
            continue
        variant_dir = os.path.dirname(path)
        dataset = os.path.basename(os.path.dirname(variant_dir))
        variant = os.path.basename(variant_dir)
        units.append((dataset, variant, int(m.group(1)), int(m.group(2)), path))
    units.sort()
    return units


def load_class_names(input_dir, dataset, k):
    f = os.path.join(input_dir, f"{dataset}_class_names.json")
    try:
        with open(f, encoding="utf-8") as fh:
            names = json.load(fh)["class_names"]
        if len(names) == k:
            return names
    except (OSError, KeyError, ValueError):
        pass
    return [f"class_{i}" for i in range(k)]


def run(args):
    t0 = time.time()
    units = find_units(args.input)
    if not units:
        print(f"No seed*_fold*.npz files found under {os.path.abspath(args.input)}")
        print("Expected layout: <input>/<dataset>/<variant>/seed<S>_fold<F>.npz")
        return 1
    os.makedirs(args.output, exist_ok=True)
    print(f"Found {len(units)} units under {os.path.abspath(args.input)}")

    unit_rows, class_rows, skipped, failed = [], [], [], []
    pooled = {}  # (dataset, variant) -> [cnt, s_conf, s_hit, n_units]
    names_cache = {}
    for i, (ds, var, seed, fold, path) in enumerate(units, 1):
        tag = f"{ds}/{var}/seed{seed}_fold{fold}"
        try:
            with np.load(path) as z:
                if "proba" not in z.files:
                    skipped.append(tag)
                    continue
                proba = z["proba"]
                y_true = z["y_true"]
            m, per_class, sums = unit_metrics(proba, y_true, args.bins)
        except Exception as e:  # keep going; report at the end
            failed.append((tag, repr(e)))
            print(f"  [{i}/{len(units)}] FAILED {tag}: {e!r}")
            continue

        unit_rows.append({"dataset": ds, "variant": var, "seed": seed, "fold": fold, **m})
        key = (ds, m["n_classes"])
        if key not in names_cache:
            names_cache[key] = load_class_names(args.input, ds, m["n_classes"])
        names = names_cache[key]
        for r in per_class:
            class_rows.append({"dataset": ds, "variant": var, "seed": seed, "fold": fold,
                               "class_name": names[r["class_index"]], **r})
        if var in args.headline:
            acc = pooled.setdefault((ds, var), [np.zeros(args.bins), np.zeros(args.bins),
                                                np.zeros(args.bins), 0])
            for j in range(3):
                acc[j] += sums[j]
            acc[3] += 1
        if i % 10 == 0 or i == len(units):
            el = time.time() - t0
            print(f"  [{i}/{len(units)}] {100*i/len(units):5.1f}%  "
                  f"elapsed {el/60:.1f} min  last: {tag}  ECE={m['ece_top']:.4f}")

    if not unit_rows:
        print("No unit could be processed; nothing written.")
        if skipped:
            print(f"  {len(skipped)} units have no 'proba' array (they were run with "
                  "--no_proba), so calibration cannot be computed for them.")
        for tag, err in failed:
            print(f"  FAILED {tag}: {err}")
        return 1

    per_unit = pd.DataFrame(unit_rows)
    per_unit.to_csv(os.path.join(args.output, "calibration_per_unit.csv"), index=False)
    pd.DataFrame(class_rows).to_csv(
        os.path.join(args.output, "calibration_per_class.csv"), index=False)

    summ = []
    for (ds, var), g in per_unit.groupby(["dataset", "variant"], sort=True):
        row = {"dataset": ds, "variant": var, "n_units": len(g),
               "n_bad_rows_total": int(g.n_bad_rows.sum())}
        for col in ["ece_top", "ece_classwise", "brier", "nll", "accuracy",
                    "mean_conf", "overconfidence"]:
            mu, lo, hi, _ = ci95(g[col])
            row[col] = mu
            row[col + "_lo"] = lo
            row[col + "_hi"] = hi
        summ.append(row)
    summary = pd.DataFrame(summ)
    summary.to_csv(os.path.join(args.output, "calibration_summary.csv"), index=False)

    bin_rows = []
    edges = np.linspace(0, 1, args.bins + 1)
    for (ds, var), (cnt, s_conf, s_hit, nu) in sorted(pooled.items()):
        for b in range(args.bins):
            bin_rows.append({
                "dataset": ds, "variant": var, "n_units_pooled": nu, "bin": b,
                "bin_lo": edges[b], "bin_hi": edges[b + 1], "count": int(cnt[b]),
                "mean_conf": s_conf[b] / cnt[b] if cnt[b] else np.nan,
                "accuracy": s_hit[b] / cnt[b] if cnt[b] else np.nan,
            })
    bins_df = pd.DataFrame(bin_rows)
    bins_df.to_csv(os.path.join(args.output, "calibration_reliability_bins.csv"),
                   index=False)

    if not args.no_figure and len(bins_df):
        try:
            draw_reliability(bins_df, summary, args.headline, args.output,
                             args.min_bin_count)
        except Exception as e:
            print(f"Figure not drawn ({e!r}); the bin table was saved, so it can be "
                  "redrawn later.")

    print()
    show = summary[summary.variant.isin(args.headline)]
    if len(show):
        print("Headline variants (mean [95% CI]):")
        for _, r in show.iterrows():
            print(f"  {r.dataset:11s} {r.variant:20s} "
                  f"ECE_top {r.ece_top:.4f} [{r.ece_top_lo:.4f}, {r.ece_top_hi:.4f}]  "
                  f"ECE_cw {r.ece_classwise:.4f}  Brier {r.brier:.4f}  "
                  f"overconf {r.overconfidence:+.4f}  (n={r.n_units})")
    print(f"\n{len(unit_rows)} units processed in {(time.time()-t0)/60:.1f} min "
          f"-> {os.path.abspath(args.output)}")
    if per_unit.n_bad_rows.sum():
        print(f"NOTE: {int(per_unit.n_bad_rows.sum())} non-finite/zero probability rows "
              "were replaced by uniform rows (see n_bad_rows).")
    if skipped:
        print(f"Skipped {len(skipped)} units without probabilities, e.g. {skipped[0]}")
    if failed:
        print(f"FAILED {len(failed)} units:")
        for tag, err in failed:
            print(f"  {tag}: {err}")
        return 2
    return 0


def draw_reliability(bins_df, summary, headline, out_dir, min_bin_count=30):
    """Bins with fewer than min_bin_count rows are not plotted (they stay in the CSV)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [d for d in ["ertam_kaya", "cicids2017", "unsw_nb15"]
                if d in set(bins_df.dataset)]
    datasets += sorted(set(bins_df.dataset) - set(datasets))
    variants = [v for v in headline if v in set(bins_df.variant)]
    colors = ["#1f5fa8", "#d1762b", "#3a8f4b", "#8a3fa0", "#777777"]
    markers = ["o", "s", "^", "D", "v"]

    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.7),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--", zorder=1)
        for j, var in enumerate(variants):
            g = bins_df[(bins_df.dataset == ds) & (bins_df.variant == var) &
                        (bins_df["count"] >= min_bin_count)]
            if not len(g):
                continue
            s = summary[(summary.dataset == ds) & (summary.variant == var)]
            ece = float(s.ece_top.iloc[0]) if len(s) else float("nan")
            ax.plot(g.mean_conf, g.accuracy, marker=markers[j % 5], ms=3.5, lw=1.1,
                    color=colors[j % 5], label=f"{var} (ECE {ece:.3f})", zorder=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(DATASET_LABEL.get(ds, ds), fontsize=10)
        ax.set_xlabel("Mean predicted confidence", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=6.5, loc="lower right", frameon=False)
        ax.text(0.02, 0.98, f"bins with n < {min_bin_count} hidden", fontsize=6,
                color="#777777", va="top", transform=ax.transAxes)
    axes[0][0].set_ylabel("Observed accuracy", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_reliability.{ext}"), dpi=600,
                    bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Self-test

def _naive_ece(conf, hit, n_bins):
    """Loop implementation used only to cross-check the vectorised one."""
    total, n = 0.0, len(conf)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        if b == n_bins - 1:
            sel = (conf >= lo) & (conf <= hi)
        else:
            sel = (conf >= lo) & (conf < hi)
        if sel.sum():
            total += sel.sum() / n * abs(hit[sel].mean() - conf[sel].mean())
    return total


def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
        ok &= bool(cond)

    n, k = 200_000, 6
    logits = rng.normal(0, 2.0, size=(n, k))
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    # labels drawn from p -> perfectly calibrated in expectation
    cum = p.cumsum(1)
    y = (rng.random(n)[:, None] > cum).sum(1)
    y = np.minimum(y, k - 1)
    m, pc, _ = unit_metrics(p.astype(np.float16), y, 15)
    check("calibrated model has small top-label ECE", m["ece_top"] < 0.01,
          f"({m['ece_top']:.4f})")
    check("calibrated model has small class-wise ECE", m["ece_classwise"] < 0.01,
          f"({m['ece_classwise']:.4f})")
    check("mean p_k matches observed frequency",
          max(abs(r["mean_p"] - r["observed_freq"]) for r in pc) < 0.01)

    # sharpen -> overconfident
    p2 = p ** 3
    p2 /= p2.sum(1, keepdims=True)
    m2, _, _ = unit_metrics(p2, y, 15)
    check("overconfident model has larger ECE", m2["ece_top"] > m["ece_top"] + 0.05,
          f"({m2['ece_top']:.4f} vs {m['ece_top']:.4f})")
    check("overconfidence is positive", m2["overconfidence"] > 0.05,
          f"({m2['overconfidence']:+.4f})")

    # vectorised vs naive ECE
    pc64, _ = clean_proba(p2)
    conf = pc64.max(1)
    hit = (pc64.argmax(1) == y).astype(float)
    a, b = top_label_ece(pc64, y, 15), _naive_ece(conf, hit, 15)
    check("vectorised ECE equals loop ECE", abs(a - b) < 1e-12, f"({a:.10f} / {b:.10f})")

    # Brier and NLL against direct formulas
    onehot = np.eye(k)[y]
    check("Brier equals direct formula",
          abs(m2["brier"] - np.mean(((pc64 - onehot) ** 2).sum(1))) < 1e-10)
    check("NLL equals direct formula",
          abs(m2["nll"] - np.mean(-np.log(np.clip(pc64[np.arange(n), y], 1e-12, None))))
          < 1e-10)

    # confidence exactly 1.0 lands in the last bin, not out of range
    pe = np.zeros((4, 3)); pe[:, 0] = 1.0
    me, _, sums = unit_metrics(pe, np.array([0, 0, 0, 1]), 15)
    check("confidence 1.0 handled", sums[0][-1] == 4 and abs(me["ece_top"] - 0.25) < 1e-12)

    # bad rows
    pb = p[:10].copy(); pb[3] = np.nan; pb[5] = 0.0
    mb, _, _ = unit_metrics(pb, y[:10], 15)
    check("non-finite and zero rows counted", mb["n_bad_rows"] == 2)

    # float16 rows that do not sum to 1
    p16 = (p * 1.003).astype(np.float16)
    pcl, _ = clean_proba(p16)
    check("rows renormalised to 1", np.allclose(pcl.sum(1), 1.0, atol=1e-12))

    # absent class (support 0) does not crash
    y_abs = np.where(y == 5, 0, y)
    ma, pca, _ = unit_metrics(p, y_abs, 15)
    check("absent class handled", pca[5]["support"] == 0 and np.isfinite(ma["ece_classwise"]))

    # end-to-end on a fake results tree, including a --no_proba unit
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "rr"), os.path.join(tmp, "out")
        for ds in ["ertam_kaya", "unsw_nb15"]:
            os.makedirs(inp, exist_ok=True)
            with open(os.path.join(inp, f"{ds}_class_names.json"), "w") as fh:
                json.dump({"class_names": [f"c{i}" for i in range(k)]}, fh)
            for var in HEADLINE_DEFAULT + ["lgbm_plain"]:
                d = os.path.join(inp, ds, var)
                os.makedirs(d, exist_ok=True)
                for fold in range(3):
                    sl = slice(fold * 1000, fold * 1000 + 1000)
                    arrs = {"y_true": y[sl].astype(np.int16),
                            "y_pred": p[sl].argmax(1).astype(np.int16)}
                    if not (var == "lgbm_plain" and fold == 2):
                        arrs["proba"] = p[sl].astype(np.float16)
                    np.savez_compressed(os.path.join(d, f"seed42_fold{fold}.npz"), **arrs)
        ns = argparse.Namespace(input=inp, output=out, bins=15,
                                headline=HEADLINE_DEFAULT, no_figure=False,
                                min_bin_count=30)
        rc = run(ns)
        pu = pd.read_csv(os.path.join(out, "calibration_per_unit.csv"))
        sm = pd.read_csv(os.path.join(out, "calibration_summary.csv"))
        bn = pd.read_csv(os.path.join(out, "calibration_reliability_bins.csv"))
        pcf = pd.read_csv(os.path.join(out, "calibration_per_class.csv"))
        check("end-to-end return code 0", rc == 0)
        check("no_proba unit skipped", len(pu) == 2 * 4 * 3 - 2, f"({len(pu)} units)")
        check("summary rows", len(sm) == 8)
        check("per-class rows", len(pcf) == len(pu) * k)
        check("class names read", set(pcf.class_name) == {f"c{i}" for i in range(k)})
        check("pooled bins only for headline", set(bn.variant) == set(HEADLINE_DEFAULT))
        check("pooled counts equal rows",
              all(g["count"].sum() == 3000 for _, g in bn.groupby(["dataset", "variant"])))
        check("figure written", os.path.exists(os.path.join(out, "fig_reliability.png")))

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", default="rerun_results",
                    help="folder written by rerun_with_predictions.py --output")
    ap.add_argument("--output", default="calibration_results")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--headline", nargs="+", default=HEADLINE_DEFAULT,
                    help="variants pooled into reliability diagrams")
    ap.add_argument("--min_bin_count", type=int, default=30,
                    help="figure only: hide bins with fewer rows (CSV keeps all)")
    ap.add_argument("--no_figure", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.bins < 2:
        ap.error("--bins must be at least 2")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
