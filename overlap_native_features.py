"""
overlap_native_features.py

Re-measures class overlap on each dataset's native feature set and compares it, class by
class, with the overlap measured on the common feature schema. The two measurements use
the same five measures (N1, N2, N3, R, F1inv from overlap_analysis.py), the same
stratified subsampling protocol and, within each draw, the SAME rows, so every
difference is due to the feature set alone.

Question answered: when a class is hard under the common schema, is that because the
class genuinely overlaps others (overlap stays high with every native feature available)
or because the common schema discards the features that separate it (overlap falls)?

Native features are built exactly as in evaluation_extensions.py (identifiers, labels and
absolute time stamps removed; nominal columns integer-encoded; infinities and missing
values median-imputed; constant columns dropped), so they are the features on which the
full-feature models were trained.

Outputs (in <output>/run_YYYYMMDD_HHMMSS/, never overwriting an earlier run):
  <dataset>_overlap_native_vs_common.csv   one row per class: class count, each measure
                                           under both schemas (mean and SD over draws),
                                           the paired difference, and two composites
  <dataset>_overlap_per_draw.json          per-draw values for both schemas
  all_datasets_overlap_native_vs_common.csv  the per-class rows of every dataset
  run_info.json                            protocol and timings

Composites: 'composite_within' min-max scales each measure within its own schema (how the
main analysis defines the composite); 'composite_common_scale' scales the native values
with the common schema's min and max, so the two composites are directly comparable.

Requirements: this file, overlap_analysis.py, evaluation_extensions.py and
rerun_with_predictions.py in the folder that holds the pipeline modules.

Usage:
  python overlap_native_features.py --selftest
  python overlap_native_features.py                 # all three datasets, about one hour
  python overlap_native_features.py --datasets unsw_nb15
"""

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import overlap_analysis as OA
    import evaluation_extensions as E
    import rerun_with_predictions as R
except ModuleNotFoundError as e:
    raise SystemExit(
        f"\n[setup error] Could not import '{e.name}'.\n"
        "overlap_native_features.py needs overlap_analysis.py, evaluation_extensions.py\n"
        "and rerun_with_predictions.py next to it, in the folder that holds the pipeline\n"
        "modules (data_loader.py, dataset_config.py, feature_schema.py), and must be run\n"
        "from that folder.\n")

ALL_DATASETS = ["ertam_kaya", "cicids2017", "unsw_nb15"]
MEASURES = list(OA.MEASURES)          # same five measures, same orientation
SHORT = {"N1_boundary_fraction": "N1", "N2_intra_inter_ratio": "N2",
         "N3_1nn_loo_error": "N3", "R_knn_majority_other": "R", "F1inv_fisher": "F1inv"}


def hms(s):
    return str(timedelta(seconds=int(max(0, s))))


class Log:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def __call__(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with self.lock:
            print(line, flush=True)
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Measurement

def draw_indices(y, class_cap, total_cap, rng):
    """Row indices of one stratified draw (overlap_analysis.stratified_draw applied to
    the row numbers, so both feature matrices get exactly the same rows)."""
    idx, _ = OA.stratified_draw(np.arange(len(y)).reshape(-1, 1), y, class_cap,
                                total_cap, rng)
    return idx[:, 0]


def measure_all(Xs, ys, classes, k):
    vals = {}
    for name, fn in OA.MEASURES.items():
        out = fn(Xs, ys, classes, k) if name == "R_knn_majority_other" else fn(Xs, ys, classes)
        vals[name] = {int(c): float(v) for c, v in out.items()
                      if v is not None and not (isinstance(v, float) and np.isnan(v))}
    return vals


def scale(X):
    return StandardScaler().fit_transform(np.asarray(X, dtype=np.float64)).astype(np.float32)


def minmax(series, lo=None, hi=None):
    lo = series.min() if lo is None else lo
    hi = series.max() if hi is None else hi
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def build_table(ds, class_names, counts, per_draw):
    """per_draw[schema][measure][class] -> list of values."""
    rows = []
    for c, name in enumerate(class_names):
        row = {"dataset": ds, "class_index": c, "class_name": name,
               "class_count": int(counts[c])}
        for m in MEASURES:
            s = SHORT[m]
            a = per_draw["common"][m].get(c, [])
            b = per_draw["native"][m].get(c, [])
            row[f"{s}_common"] = float(np.mean(a)) if a else np.nan
            row[f"{s}_common_sd"] = float(np.std(a)) if len(a) > 1 else np.nan
            row[f"{s}_native"] = float(np.mean(b)) if b else np.nan
            row[f"{s}_native_sd"] = float(np.std(b)) if len(b) > 1 else np.nan
            paired = [bb - aa for aa, bb in zip(a, b)]
            row[f"{s}_diff_native_minus_common"] = float(np.mean(paired)) if paired else np.nan
            row[f"{s}_n_draws"] = min(len(a), len(b))
        rows.append(row)
    df = pd.DataFrame(rows)
    comp_c = pd.concat([minmax(df[f"{SHORT[m]}_common"]) for m in MEASURES], axis=1).mean(axis=1)
    comp_n_within = pd.concat([minmax(df[f"{SHORT[m]}_native"]) for m in MEASURES],
                              axis=1).mean(axis=1)
    comp_n_cs = pd.concat([minmax(df[f"{SHORT[m]}_native"], df[f"{SHORT[m]}_common"].min(),
                                  df[f"{SHORT[m]}_common"].max()).clip(0, 1)
                           for m in MEASURES], axis=1).mean(axis=1)
    df["composite_common"] = comp_c
    df["composite_native_within"] = comp_n_within
    df["composite_native_common_scale"] = comp_n_cs
    df["composite_diff_common_scale"] = comp_n_cs - comp_c
    return df


def run_dataset(ds, args, out_dir, log, progress, synthetic=False):
    t_start = time.time()
    log(f"=== {ds} ===")
    if synthetic:
        X_df, y_raw, raw, _ = E.make_synthetic_raw()
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
        enc = LabelEncoder()
        y = enc.fit_transform(y_raw)
        class_names = enc.classes_.tolist()
        X_common = X_df.values.astype(np.float64)
        X_native, native_names, finfo = E.native_features_synthetic(raw)
    else:
        X_common, y, class_names, _ = R.load_prepared(ds, False)
        raw, _, _ = E.load_raw(ds)
        E.check_alignment(ds, np.asarray(class_names)[y], raw)
        X_native, native_names, finfo = E.native_features(ds, raw)
        del raw
    if len(X_native) != len(y) or len(X_common) != len(y):
        raise RuntimeError(f"{ds}: feature matrices and labels have different lengths")
    n_classes = len(class_names)
    counts = np.bincount(y, minlength=n_classes)
    log(f"  {len(y):,} rows, {n_classes} classes; common features {X_common.shape[1]}, "
        f"native features {X_native.shape[1]}")

    Xc = scale(X_common)
    Xn = scale(X_native)
    del X_common, X_native

    per_draw = {s: {m: {} for m in MEASURES} for s in ("common", "native")}
    draw_sizes = []
    for d in range(args.n_draws):
        t0 = time.time()
        rng = np.random.default_rng(OA.GLOBAL_SEED + d)
        idx = draw_indices(y, args.class_cap, args.total_cap, rng)
        ys = y[idx]
        classes = np.unique(ys)
        draw_sizes.append(int(len(idx)))
        for schema, X in (("common", Xc), ("native", Xn)):
            vals = measure_all(X[idx], ys, classes, args.k)
            for m, cv in vals.items():
                for c, v in cv.items():
                    per_draw[schema][m].setdefault(c, []).append(v)
            progress["done"] += 1
        log(f"  draw {d + 1}/{args.n_draws}: {len(idx):,} rows, {time.time() - t0:.1f} s")

    df = build_table(ds, class_names, counts, per_draw)
    df.to_csv(os.path.join(out_dir, f"{ds}_overlap_native_vs_common.csv"), index=False)
    with open(os.path.join(out_dir, f"{ds}_overlap_per_draw.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": ds, "class_names": class_names,
                   "class_counts": counts.tolist(), "draw_sizes": draw_sizes,
                   "native_feature_info": finfo,
                   "protocol": {"class_cap": args.class_cap, "total_cap": args.total_cap,
                                "n_draws": args.n_draws, "k_for_R": args.k,
                                "global_seed": OA.GLOBAL_SEED,
                                "same_rows_for_both_schemas": True},
                   "per_draw_values": {s: {m: {str(c): v for c, v in cv.items()}
                                           for m, cv in pm.items()}
                                       for s, pm in per_draw.items()}}, f, indent=2)
    show = df[["class_name", "class_count", "N3_common", "N3_native", "R_common", "R_native",
               "composite_common", "composite_native_common_scale"]].copy()
    log("  N3 and R under the common and native schemas:\n"
        + show.round(3).to_string(index=False))
    log(f"{ds}: done in {hms(time.time() - t_start)}")
    return df


def heartbeat(progress, log, every, stop):
    while not stop.wait(every):
        done, total, t0 = progress["done"], progress["total"], progress["t0"]
        el = time.time() - t0
        if done:
            rem = el / done * (total - done)
            fin = (datetime.now() + timedelta(seconds=rem)).strftime("%a %Y-%m-%d %H:%M")
            log(f"PROGRESS {100 * done / total:5.1f}%  steps {done}/{total}  elapsed {hms(el)}"
                f"  remaining ~{hms(rem)}  expected finish {fin}")
        else:
            log(f"PROGRESS   0.0%  steps 0/{total}  elapsed {hms(el)}  (first draw running)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--output", default="overlap_native_results")
    ap.add_argument("--class_cap", type=int, default=OA.DEFAULT_CLASS_CAP)
    ap.add_argument("--total_cap", type=int, default=OA.DEFAULT_TOTAL_CAP)
    ap.add_argument("--n_draws", type=int, default=OA.DEFAULT_N_DRAWS)
    ap.add_argument("--k", type=int, default=OA.DEFAULT_K)
    ap.add_argument("--heartbeat", type=float, default=30.0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    synthetic = args.selftest
    if synthetic:
        import tempfile
        args.output = tempfile.mkdtemp(prefix="overlap_native_selftest_")
        args.datasets = ["synthetic"]
        args.n_draws = 3
        args.total_cap = 3000
        args.heartbeat = 5.0
    for ds in args.datasets:
        if not synthetic and ds not in ALL_DATASETS:
            raise SystemExit(f"unknown dataset: {ds}")
    if args.n_draws < 1 or args.class_cap < 2 or args.total_cap < 10:
        raise SystemExit("--n_draws >= 1, --class_cap >= 2, --total_cap >= 10 required")

    os.makedirs(args.output, exist_ok=True)
    run_dir = os.path.join(args.output, "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    while os.path.exists(run_dir):
        time.sleep(1)
        run_dir = os.path.join(args.output, "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir)
    log = Log(os.path.join(run_dir, "progress.log"))
    log(f"new run: {os.path.abspath(run_dir)}")
    log(f"datasets={args.datasets} draws={args.n_draws} class_cap={args.class_cap} "
        f"total_cap={args.total_cap} k={args.k}")

    progress = {"done": 0, "total": 2 * args.n_draws * len(args.datasets), "t0": time.time()}
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(progress, log, args.heartbeat, stop),
                     daemon=True).start()

    frames, failed = [], []
    try:
        for ds in args.datasets:
            try:
                frames.append(run_dataset(ds, args, run_dir, log, progress, synthetic))
            except KeyboardInterrupt:
                raise
            except Exception as e:
                import traceback
                log(f"FAILED {ds}: {e!r}\n{traceback.format_exc()}")
                failed.append(ds)
    except KeyboardInterrupt:
        log("interrupted by user; datasets finished so far are saved")
    finally:
        stop.set()

    if frames:
        alldf = pd.concat(frames, ignore_index=True)
        alldf.to_csv(os.path.join(run_dir, "all_datasets_overlap_native_vs_common.csv"),
                     index=False)
    with open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({"finished": datetime.now().isoformat(timespec="seconds"),
                   "args": vars(args), "failed": failed,
                   "elapsed_seconds": time.time() - progress["t0"]}, f, indent=2)
    log(f"done in {hms(time.time() - progress['t0'])} -> {os.path.abspath(run_dir)}"
        + (f"  FAILED: {failed}" if failed else ""))

    if synthetic:
        ok = _selftest_check(run_dir, log)
        log("SELFTEST " + ("PASSED" if ok else "FAILED"))
        return 0 if ok else 1
    return 2 if failed else 0


def _selftest_check(run_dir, log):
    ok = True

    def chk(name, cond):
        nonlocal ok
        log(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok &= bool(cond)

    p = os.path.join(run_dir, "synthetic_overlap_native_vs_common.csv")
    chk("per-class table written", os.path.exists(p))
    if not os.path.exists(p):
        return False
    df = pd.read_csv(p)
    chk("one row per class", len(df) == 6)
    for s in SHORT.values():
        for sch in ("common", "native"):
            col = f"{s}_{sch}"
            chk(f"{col} in [0, 1]", df[col].between(0, 1).all())
    chk("all draws used", (df["N3_n_draws"] == 3).all())
    chk("composites in [0, 1]", df[["composite_common", "composite_native_within",
                                     "composite_native_common_scale"]].stack().between(0, 1).all())
    j = json.load(open(os.path.join(run_dir, "synthetic_overlap_per_draw.json")))
    chk("same rows for both schemas flagged", j["protocol"]["same_rows_for_both_schemas"])
    chk("native has more features than common",
        j["native_feature_info"]["n_features"] > 12)

    # Direct check of the logic: a native schema that adds separating features must
    # drive N3 to zero while the common (noise-only) schema stays overlapping.
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(4), 150)
    Xc = rng.normal(size=(600, 3))                         # pure noise: heavy overlap
    Xn = np.column_stack([Xc] + [y * 10.0 + rng.normal(0, 0.3, 600) for _ in range(3)])
    idx = draw_indices(y, 2000, 12000, np.random.default_rng(1))
    idx2 = draw_indices(y, 2000, 12000, np.random.default_rng(1))
    chk("draw is reproducible", np.array_equal(idx, idx2))
    vc = measure_all(scale(Xc)[idx], y[idx], np.unique(y[idx]), 5)
    vn = measure_all(scale(Xn)[idx], y[idx], np.unique(y[idx]), 5)
    n3c = np.mean(list(vc["N3_1nn_loo_error"].values()))
    n3n = np.mean(list(vn["N3_1nn_loo_error"].values()))
    chk("noise-only schema is overlapping (N3 > 0.5)", n3c > 0.5)
    chk("separating features remove overlap (N3 < 0.02)", n3n < 0.02)
    chk("F1inv falls with the separating feature",
        np.mean(list(vn["F1inv_fisher"].values())) < np.mean(list(vc["F1inv_fisher"].values())))
    # composite on the common scale: native must be lower than common here
    pdw = {"common": {m: {c: [v] for c, v in vc[m].items()} for m in MEASURES},
           "native": {m: {c: [v] for c, v in vn[m].items()} for m in MEASURES}}
    t = build_table("t", ["a", "b", "c", "d"], np.bincount(y), pdw)
    chk("common-scale composite drops for the separable schema",
        (t.composite_native_common_scale < t.composite_common).all())
    chk("paired difference column consistent",
        np.allclose(t.N3_diff_native_minus_common, t.N3_native - t.N3_common))
    return ok


if __name__ == "__main__":
    sys.exit(main())
