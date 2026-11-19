"""
run_extensions.py

One command that runs every extension experiment in sequence and reports progress:

  stage 1  shap           SHAP class profiles + background stability   (shap_overlap_profiles.py)
  stage 2  time_blocks    per-class contiguous time-block folds         (evaluation_extensions.py)
  stage 3  full_features  native feature sets instead of the common schema (evaluation_extensions.py)

Progress is printed every --heartbeat seconds (default 30): percentage done, elapsed time,
estimated remaining time and the expected finish date and time. The estimate is based on
the training times of the main re-run (rerun_results/rerun_metrics.csv) when available,
scaled for the number of features, and is replaced by measured times as units finish, so it
becomes accurate after the first unit of each kind.

Results never overwrite earlier results: every invocation creates a new folder
    <output>/run_YYYYMMDD_HHMMSS/{shap,time_blocks,full_features}/
To continue an interrupted run instead, pass --resume latest (or --resume <run folder>);
finished units are skipped and nothing already written is touched.

Requirements: this file, evaluation_extensions.py, shap_overlap_profiles.py and
rerun_with_predictions.py in the same folder as the pipeline modules (data_loader.py,
dataset_config.py, imbalance_methods.py, model.py, tree_baselines.py).

Usage:
  python run_extensions.py --selftest                 # synthetic data, about 2 minutes
  python run_extensions.py                            # everything, default order
  python run_extensions.py --stages time_blocks full_features --datasets unsw_nb15
  python run_extensions.py --resume latest
  python run_extensions.py --full_feature_folds 0 1 2 # cap the slowest stage
"""

import argparse
import glob
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import rerun_with_predictions as R
    import evaluation_extensions as E
    import shap_overlap_profiles as SH
except ModuleNotFoundError as e:
    raise SystemExit(
        f"\n[setup error] Could not import '{e.name}'.\n"
        "run_extensions.py needs evaluation_extensions.py, shap_overlap_profiles.py and\n"
        "rerun_with_predictions.py next to it, in the folder that holds the pipeline\n"
        "modules (data_loader.py, dataset_config.py, imbalance_methods.py, model.py,\n"
        "tree_baselines.py), and must be run from that folder.\n"
        "If the missing name is 'shap': pip install shap\n")

STAGES = ["shap", "time_blocks", "full_features"]
ALL_DATASETS = ["ertam_kaya", "cicids2017", "unsw_nb15"]
HEADLINE = ["xgb_classweight", "xgb_smote", "mlp_baseline_smote"]

# Fallback prior durations in seconds when rerun_metrics.csv is not available
# (median training seconds per fold measured in the main re-run on one laptop GPU).
PRIOR_SECONDS = {
    ("ertam_kaya", "xgb"): 4, ("ertam_kaya", "mlp"): 33,
    ("cicids2017", "xgb"): 550, ("cicids2017", "mlp"): 873,
    ("unsw_nb15", "xgb"): 380, ("unsw_nb15", "mlp"): 846,
}
PRIOR_SHAP_SECONDS = {"ertam_kaya": 180, "cicids2017": 3000, "unsw_nb15": 2100,
                      "synthetic": 30}
# Native feature counts used only for the prior; the real count is read from the data.
PRIOR_NATIVE_FEATURES = {"ertam_kaya": 11, "cicids2017": 78, "unsw_nb15": 42}
CANONICAL_FEATURES = {"ertam_kaya": 9, "cicids2017": 8, "unsw_nb15": 10}


def hms(seconds):
    return str(timedelta(seconds=int(max(0, seconds))))


# ---------------------------------------------------------------------------
# Progress state shared with the heartbeat thread

class Progress:
    def __init__(self, log):
        self.log = log
        self.lock = threading.Lock()
        self.units = []            # dicts: id, kind, done(bool), seconds(measured or None)
        self.kind_prior = {}       # kind -> prior seconds
        self.measured = {}         # kind -> list of measured seconds
        self.current = None        # (unit index, start time, fraction hint or None)
        self.t0 = time.time()
        self.stop = threading.Event()

    def est(self, kind):
        m = self.measured.get(kind)
        if m:
            return float(np.mean(m))
        return float(self.kind_prior.get(kind, 600.0))

    def add_units(self, units):
        with self.lock:
            self.units.extend(units)

    def start(self, idx):
        with self.lock:
            self.current = [idx, time.time(), None]

    def hint(self, fraction):
        with self.lock:
            if self.current is not None:
                self.current[2] = float(min(max(fraction, 0.0), 1.0))

    def finish(self, idx, seconds, measured=True):
        with self.lock:
            u = self.units[idx]
            u["done"] = True
            u["seconds"] = seconds
            if measured:
                self.measured.setdefault(u["kind"], []).append(seconds)
            self.current = None

    def snapshot(self):
        with self.lock:
            total = done = 0.0
            n_done = 0
            # The running unit refines the estimate of its own kind: if it has already
            # taken longer than the prior, or its epoch counter says it will, the
            # projected duration replaces the prior for every unit of that kind that
            # has not been measured yet.
            cur_kind, cur_proj, cur_frac = None, None, None
            if self.current is not None:
                ci, cstart, hint = self.current
                cur_kind = self.units[ci]["kind"]
                el = time.time() - cstart
                e0 = self.est(cur_kind)
                if hint is not None and hint > 0.05:
                    cur_proj = max(el / hint, el)
                    cur_frac = hint
                else:
                    cur_proj = max(e0, el / 0.9)
                    cur_frac = min(el / cur_proj, 0.95) if cur_proj > 0 else 0.0
            for i, u in enumerate(self.units):
                if (cur_kind is not None and u["kind"] == cur_kind
                        and not self.measured.get(cur_kind)):
                    e = cur_proj
                else:
                    e = self.est(u["kind"])
                total += e
                if u["done"]:
                    done += e
                    n_done += 1
                elif self.current is not None and self.current[0] == i:
                    done += cur_frac * e
            remaining = max(total - done, 0.0)
            pct = 100.0 * done / total if total > 0 else 0.0
            cur = None
            if self.current is not None:
                cur = (self.units[self.current[0]]["id"], time.time() - self.current[1])
            return pct, n_done, len(self.units), remaining, cur

    def line(self):
        pct, n_done, n, remaining, cur = self.snapshot()
        finish = datetime.now() + timedelta(seconds=remaining)
        s = (f"PROGRESS {pct:5.1f}%  units {n_done}/{n}  elapsed {hms(time.time() - self.t0)}"
             f"  remaining ~{hms(remaining)} ({remaining / 3600:.1f} h)"
             f"  expected finish {finish.strftime('%a %Y-%m-%d %H:%M')}")
        if cur:
            s += f"  | now: {cur[0]} ({hms(cur[1])})"
        return s


def heartbeat(progress, every):
    while not progress.stop.wait(every):
        progress.log(progress.line())


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
# Priors from the main re-run

def load_priors(rerun_dir, log):
    priors = dict(PRIOR_SECONDS)
    p = os.path.join(rerun_dir, "rerun_metrics.csv") if rerun_dir else None
    if p and os.path.exists(p):
        try:
            df = pd.read_csv(p)
            med = df.groupby(["dataset", "family"]).train_seconds.median()
            for (ds, fam), v in med.items():
                if fam in ("xgb", "mlp") and np.isfinite(v):
                    priors[(ds, fam)] = float(v)
            log(f"duration priors taken from {p}")
        except Exception as e:
            log(f"could not read {p} ({e}); using built-in priors")
    else:
        log("rerun_metrics.csv not found; using built-in duration priors")
    return priors


def family_of(variant):
    return "mlp" if variant.startswith("mlp") else "xgb"


# ---------------------------------------------------------------------------
# Run folder

def make_run_dir(output, resume):
    if resume:
        if resume == "latest":
            runs = sorted(glob.glob(os.path.join(output, "run_*")))
            if not runs:
                raise SystemExit(f"--resume latest: no run_* folder under {output}")
            return runs[-1], True
        if not os.path.isdir(resume):
            raise SystemExit(f"--resume: folder not found: {resume}")
        return resume, True
    os.makedirs(output, exist_ok=True)
    d = os.path.join(output, "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    while os.path.exists(d):          # two starts in the same second
        time.sleep(1)
        d = os.path.join(output, "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(d)
    return d, False


# ---------------------------------------------------------------------------
# Stage: SHAP

def shap_units(datasets, out_dir):
    return [{"id": f"shap/{ds}", "kind": ("shap", ds), "stage": "shap", "dataset": ds,
             "done": os.path.exists(os.path.join(out_dir, ds + "_shap_model_check.json")),
             "seconds": None} for ds in datasets]


def run_shap_unit(u, args, out_dir, synthetic):
    ns = argparse.Namespace(output=out_dir, fold=0, per_class=args.shap_per_class,
                            min_per_class=20, n_bg=args.shap_n_bg, bg_size=200,
                            stab_per_class=args.shap_stab_per_class,
                            rerun_dir=None if synthetic else args.rerun_dir, force=False)
    SH.run_dataset(u["dataset"], ns, synthetic=synthetic)


# ---------------------------------------------------------------------------
# Stage: evaluation extensions (time_blocks / full_features)

def eval_units(experiment, datasets, variants, seeds, folds, out_dir):
    units = []
    for ds in datasets:
        for k in folds:                 # fold-major, so each fold is scaled once
            for v in variants:
                for s in seeds:
                    jp = E.unit_paths(out_dir, ds, v, s, k)[1]
                    units.append({"id": f"{experiment}/{ds}/{v}/seed{s}/fold{k}",
                                  "kind": (experiment, ds, family_of(v)),
                                  "stage": experiment, "dataset": ds, "variant": v,
                                  "seed": s, "fold": k, "done": os.path.exists(jp),
                                  "seconds": None})
    return units


def prepare_dataset(experiment, ds, n_folds, out_dir, log, synthetic):
    """Load data once per (stage, dataset); returns dict with X, y, folds, names."""
    if synthetic:
        X_df, y_raw, raw, stime = E.make_synthetic_raw()
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
        enc = LabelEncoder()
        y = enc.fit_transform(y_raw)
        class_names = enc.classes_.tolist()
        X = X_df.values.astype(np.float64)
        feature_names = X_df.columns.tolist()
        fidx, ridx = np.zeros(len(y), int), np.arange(len(y))
    else:
        X, y, class_names, feature_names = R.load_prepared(ds, False)
        raw, fidx, ridx = E.load_raw(ds)
        E.check_alignment(ds, np.asarray(class_names)[y], raw)
    n_classes = len(class_names)
    log(f"  {X.shape[0]:,} rows, {n_classes} classes, canonical features: {X.shape[1]}")

    if experiment == "time_blocks":
        if synthetic:
            order = np.argsort(stime, kind="stable")
            desc = {"order": "synthetic stime", "chronology_known": True}
        else:
            order, desc = E.order_key(ds, raw, fidx, ridx)
        folds = E.time_block_folds(y, order, n_folds)
        feats = feature_names
        info = {"dataset": ds, "experiment": experiment, "n_folds": n_folds,
                "ordering": desc, "class_names": class_names,
                "val_sizes": [int(len(v)) for _, v in folds],
                "val_class_counts": [np.bincount(y[v], minlength=n_classes).tolist()
                                     for _, v in folds]}
        _write_json(os.path.join(out_dir, f"{ds}_split_info.json"), info)
        log(f"  ordering: {desc['order']}")
    else:
        Xn, feats, finfo = (E.native_features_synthetic(raw) if synthetic
                            else E.native_features(ds, raw))
        if len(Xn) != len(y):
            raise RuntimeError("native feature matrix is misaligned with labels")
        X = Xn
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=R.SEED_BASE)
        folds = list(skf.split(X, y))
        finfo.update({"dataset": ds, "experiment": experiment,
                      "canonical_n_features": len(feature_names)})
        _write_json(os.path.join(out_dir, f"{ds}_feature_info.json"), finfo)
        log(f"  native features: {len(feats)} (encoded: "
            f"{[e['column'] for e in finfo['integer_encoded']]}, constant dropped: "
            f"{finfo['dropped_constant_columns']})")
    del raw
    _write_json(os.path.join(out_dir, f"{ds}_class_names.json"),
                {"class_names": class_names, "feature_names": feats,
                 "class_counts": np.bincount(y, minlength=n_classes).tolist()})
    return {"X": X, "y": y, "class_names": class_names, "feats": feats, "folds": folds,
            "scaled": {}}


def fold_arrays(data, k):
    """Scale once per fold and cache; the cache holds one fold at a time."""
    if k not in data["scaled"]:
        data["scaled"].clear()
        tr, va = data["folds"][k]
        sc = StandardScaler()
        X_tr = sc.fit_transform(data["X"][tr])
        X_va = sc.transform(data["X"][va])
        data["scaled"][k] = (X_tr, X_va, data["y"][tr], data["y"][va])
    return data["scaled"][k]


def run_eval_unit(u, data, out_dir, log, progress):
    X_tr, X_va, y_tr, y_va = fold_arrays(data, u["fold"])
    class_names = data["class_names"]
    n_classes = len(class_names)
    if len(np.unique(y_tr)) != n_classes:
        log(f"  NOTE: {n_classes - len(np.unique(y_tr))} class(es) absent from the "
            f"training part of fold {u['fold']}")
    family, variant, spec = next(v for v in R.variant_list(["mlp", "xgb", "lgbm"])
                                 if v[1] == u["variant"])
    d, jp, npz = E.unit_paths(out_dir, u["dataset"], variant, u["seed"], u["fold"])
    os.makedirs(d, exist_ok=True)
    t0 = time.time()

    def hb(k, n, el):
        progress.hint(k / n if n else 0)

    if family == "mlp":
        y_pred, proba, info = R.run_mlp(spec, X_tr, y_tr, X_va, n_classes, u["seed"],
                                        heartbeat=hb)
    else:
        y_pred, proba, info = R.run_tree(family, spec, X_tr, y_tr, X_va, n_classes,
                                         u["seed"])
    secs = time.time() - t0
    m = R.compute_metrics(y_va, y_pred, n_classes, class_names)
    m.update({"dataset": u["dataset"], "variant": variant, "family": family,
              "technique_or_mode": spec, "seed": u["seed"], "fold": u["fold"],
              "experiment": u["stage"], "n_features": int(X_tr.shape[1]),
              "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
              "n_classes": n_classes, "class_names": class_names,
              "train_seconds": round(secs, 2), "device": R.DEVICE, **info})
    np.savez_compressed(npz, y_true=y_va.astype(np.int16), y_pred=y_pred.astype(np.int16),
                        proba=proba.astype(np.float16))
    _write_json(jp, m)                     # json last: it is the "done" marker
    return secs, m["macro_f1"]


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--variants", nargs="+", default=HEADLINE)
    ap.add_argument("--output", default="extension_results")
    ap.add_argument("--resume", default=None,
                    help="'latest' or a run folder; continue instead of starting a new run")
    ap.add_argument("--rerun_dir", default="rerun_results",
                    help="main re-run results (for duration priors and the SHAP model check)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--time_block_folds", type=int, nargs="+", default=None,
                    help="fold indices to run in the time_blocks stage (default: all)")
    ap.add_argument("--full_feature_folds", type=int, nargs="+", default=None,
                    help="fold indices to run in the full_features stage (default: all)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--heartbeat", type=float, default=30.0)
    ap.add_argument("--shap_n_bg", type=int, default=5)
    ap.add_argument("--shap_per_class", type=int, default=300)
    ap.add_argument("--shap_stab_per_class", type=int, default=100)
    ap.add_argument("--synthetic", action="store_true",
                    help="use a small synthetic dataset (for testing the resume logic)")
    ap.add_argument("--selftest", action="store_true",
                    help="run every stage on synthetic data into a temporary folder "
                         "and check the outputs")
    args = ap.parse_args()

    synthetic = args.selftest or args.synthetic
    if synthetic:
        if args.selftest:
            import tempfile
            args.output = tempfile.mkdtemp(prefix="run_extensions_selftest_")
            args.resume = None
        args.datasets = ["synthetic"]
        args.folds = 2
        args.shap_n_bg = 2
        args.shap_per_class = 60
        args.shap_stab_per_class = 20
        args.heartbeat = 5.0

    unknown = set(args.variants) - {v[1] for v in R.variant_list(["mlp", "xgb", "lgbm"])}
    if unknown:
        raise SystemExit(f"unknown variants: {sorted(unknown)}")
    for ds in args.datasets:
        if not synthetic and ds not in ALL_DATASETS:
            raise SystemExit(f"unknown dataset: {ds}")
    if not 1 <= args.folds <= 20:
        raise SystemExit("--folds must be between 1 and 20")
    for name in ("time_block_folds", "full_feature_folds"):
        v = getattr(args, name)
        if v and any(k < 0 or k >= args.folds for k in v):
            raise SystemExit(f"--{name} must be within 0..{args.folds - 1}")

    run_dir, resumed = make_run_dir(args.output, args.resume)
    log = Log(os.path.join(run_dir, "progress.log"))
    log(f"{'resuming' if resumed else 'new run'}: {os.path.abspath(run_dir)}")
    log(f"device={R.DEVICE} stages={args.stages} datasets={args.datasets} "
        f"variants={args.variants} seeds={args.seeds} folds={args.folds}")
    manifest = {"started": datetime.now().isoformat(timespec="seconds"),
                "resumed": resumed, "device": R.DEVICE, "args": vars(args)}
    mp = os.path.join(run_dir, "run_manifest.json")
    if os.path.exists(mp):      # never overwrite: a resume gets its own manifest
        mp = os.path.join(run_dir, "resume_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                          + ".json")
    _write_json(mp, manifest)
    SH.log = log                # SHAP stage messages go to the same progress.log

    seeds = [R.SEED_BASE + i for i in range(args.seeds)]
    all_folds = list(range(args.folds))
    stage_folds = {"time_blocks": args.time_block_folds or all_folds,
                   "full_features": args.full_feature_folds or all_folds}
    stage_dirs = {s: os.path.join(run_dir, s) for s in STAGES}
    for s in args.stages:
        os.makedirs(stage_dirs[s], exist_ok=True)

    # ---- plan every unit up front so the ETA covers the whole run
    priors = load_priors(args.rerun_dir if not synthetic else None, log)
    progress = Progress(log)
    for ds in args.datasets:
        progress.kind_prior[("shap", ds)] = PRIOR_SHAP_SECONDS.get(ds, 1800)
        for fam in ("xgb", "mlp"):
            base = priors.get((ds, fam), 600 if not synthetic else 5)
            progress.kind_prior[("time_blocks", ds, fam)] = base
            ratio = (PRIOR_NATIVE_FEATURES.get(ds, 10) / CANONICAL_FEATURES.get(ds, 10)
                     if fam == "xgb" else 1.1)
            progress.kind_prior[("full_features", ds, fam)] = base * ratio
    plan = []
    for stage in args.stages:
        if stage == "shap":
            plan += shap_units(args.datasets, stage_dirs["shap"])
        else:
            # cheapest dataset first, so the slowest one (cicids2017 with 78 native
            # features) is last and can be cut short without losing the others
            cost = {ds: progress.kind_prior.get((stage, ds, "xgb"), 0)
                    for ds in args.datasets}
            ds_order = sorted(args.datasets, key=lambda d: cost[d])
            plan += eval_units(stage, ds_order, args.variants, seeds,
                               stage_folds[stage], stage_dirs[stage])
    progress.add_units(plan)
    n_done = sum(u["done"] for u in plan)
    log(f"planned {len(plan)} units, {n_done} already finished"
        + (" (resumed)" if resumed else ""))
    log(progress.line())

    hb_thread = threading.Thread(target=heartbeat, args=(progress, args.heartbeat),
                                 daemon=True)
    hb_thread.start()

    failed = []
    interrupted = False
    try:
        for stage in args.stages:
            units = [(i, u) for i, u in enumerate(plan) if u["stage"] == stage]
            if all(u["done"] for _, u in units):
                log(f"=== stage {stage}: nothing to do ===")
                continue
            log(f"=== stage {stage} ===")
            if stage == "shap":
                for i, u in units:
                    if u["done"]:
                        continue
                    progress.start(i)
                    t0 = time.time()
                    try:
                        run_shap_unit(u, args, stage_dirs["shap"], synthetic)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        log(f"FAILED {u['id']}: {e!r}\n{traceback.format_exc()}")
                        failed.append(u["id"])
                        progress.finish(i, time.time() - t0, measured=False)
                        continue
                    progress.finish(i, time.time() - t0)
                    log(f"done {u['id']} in {hms(time.time() - t0)}")
                    log(progress.line())
                continue

            data, data_key = None, None
            for i, u in units:
                if u["done"]:
                    continue
                if data_key != u["dataset"]:
                    data = None                    # free the previous dataset first
                    log(f"loading {u['dataset']} for {stage}")
                    data = prepare_dataset(stage, u["dataset"], args.folds,
                                           stage_dirs[stage], log, synthetic)
                    data_key = u["dataset"]
                progress.start(i)
                log(f"start {u['id']}")
                t0 = time.time()
                try:
                    secs, mf1 = run_eval_unit(u, data, stage_dirs[stage], log, progress)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    log(f"FAILED {u['id']}: {e!r}\n{traceback.format_exc()}")
                    failed.append(u["id"])
                    progress.finish(i, time.time() - t0, measured=False)
                    continue
                progress.finish(i, secs)
                log(f"done {u['id']} in {hms(secs)}  macro-F1 {mf1:.4f}")
                log(progress.line())
            data = None
            E.summarize(stage_dirs[stage], log)
    except KeyboardInterrupt:
        interrupted = True
        log("interrupted by user")
    finally:
        progress.stop.set()
        for stage in args.stages:
            if stage != "shap" and os.path.isdir(stage_dirs[stage]):
                try:
                    E.summarize(stage_dirs[stage], lambda m: None)
                except Exception:
                    pass

    total = hms(time.time() - progress.t0)
    if interrupted:
        log(f"stopped after {total}. Finished units are saved. To continue:\n"
            f"    python run_extensions.py --resume \"{run_dir}\"")
        return 130
    if failed:
        log(f"finished with {len(failed)} FAILED unit(s) after {total}: {failed}")
        return 2
    log(f"ALL DONE in {total}. Results: {os.path.abspath(run_dir)}")
    if synthetic:
        ok = _selftest_check(run_dir, args, log)
        log("SELFTEST " + ("PASSED" if ok else "FAILED"))
        return 0 if ok else 1
    return 0


def _selftest_check(run_dir, args, log):
    ok = True

    def chk(name, cond):
        nonlocal ok
        log(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok &= bool(cond)

    n_eval = len(args.variants) * args.folds
    for stage in ("time_blocks", "full_features"):
        p = os.path.join(run_dir, stage, "metrics.csv")
        chk(f"{stage} metrics.csv exists", os.path.exists(p))
        if os.path.exists(p):
            df = pd.read_csv(p)
            chk(f"{stage} has {n_eval} units", len(df) == n_eval)
            chk(f"{stage} macro-F1 in range", df.macro_f1.between(0, 1).all())
            chk(f"{stage} experiment column", (df.experiment == stage).all())
    chk("time_blocks split_info", os.path.exists(
        os.path.join(run_dir, "time_blocks", "synthetic_split_info.json")))
    fi = os.path.join(run_dir, "full_features", "synthetic_feature_info.json")
    chk("full_features feature_info", os.path.exists(fi))
    if os.path.exists(fi):
        chk("full_features used more features than canonical",
            json.load(open(fi))["n_features"] > json.load(open(fi))["canonical_n_features"])
    chk("shap summary", os.path.exists(
        os.path.join(run_dir, "shap", "synthetic_shap_class_summary.csv")))
    chk("shap stability", os.path.exists(
        os.path.join(run_dir, "shap", "synthetic_shap_stability.csv")))
    chk("run_manifest", os.path.exists(os.path.join(run_dir, "run_manifest.json")))
    # resume must find nothing to do and must not create a new folder
    before = sorted(glob.glob(os.path.join(os.path.dirname(run_dir), "run_*")))
    rd, resumed = make_run_dir(os.path.dirname(run_dir), "latest")
    chk("resume latest points at this run", resumed and os.path.abspath(rd) == os.path.abspath(run_dir))
    after = sorted(glob.glob(os.path.join(os.path.dirname(run_dir), "run_*")))
    chk("resume does not create a folder", before == after)
    plan = eval_units("time_blocks", ["synthetic"], args.variants,
                      [R.SEED_BASE], list(range(args.folds)),
                      os.path.join(run_dir, "time_blocks"))
    chk("resume sees all units as done", all(u["done"] for u in plan))
    return ok


if __name__ == "__main__":
    sys.exit(main())
