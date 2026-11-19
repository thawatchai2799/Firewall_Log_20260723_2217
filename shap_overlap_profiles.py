"""
shap_overlap_profiles.py

SHAP attribution profiles of every class for the study's headline model (class-weighted
XGBoost), built to test one question: do classes that are easy despite being rare carry a
distinctive attribution signature, while classes that overlap others share theirs?

For each dataset:
  1. Rebuild fold 0 exactly as rerun_with_predictions.py does (same loader, same
     StratifiedKFold, same scaler fitted on the training part) and train class-weighted
     XGBoost with seed 42 through the same factory (tree_baselines._make_xgb) and the same
     balanced sample weights. Validation macro-F1 is printed and, if --rerun_dir is given,
     compared with the saved result of that unit, so the explained model is shown to be the
     model reported in the main results.
  2. Exact TreeSHAP (path-dependent, via XGBoost's own pred_contribs). It needs no
     background sample, so the result does not depend on a background size; additivity
     (contributions + bias = margin) is checked.
  3. Explanation set: up to --per_class validation records of every class. Classes with
     fewer than --min_per_class validation records are topped up with training records
     (the source of every row is recorded), so very rare classes such as Heartbleed are
     described by more than two points.
  4. Per class c, from the SHAP values of output c on the records of class c:
       * importance profile I_c[f] = mean |SHAP|, and the signed mean;
       * concentration = share of the total importance on the top feature;
       * nearest profile = the other class whose importance profile is most similar
         (cosine), and that similarity;
       * attribution margin = mean SHAP evidence for c on c's own records minus the
         largest mean SHAP evidence for c on any other class's records (logit units).
         A small margin means the features push another class towards c almost as hard
         as they push c itself.
  5. Background-sensitivity check (answers the request for a stability analysis): the
     same profiles recomputed with interventional TreeSHAP and a background of
     --bg_size training records, redrawn --n_bg times; per class, the Spearman rank
     correlation of the importance profile between draws and against the exact values,
     and the agreement of the top feature.

Outputs in --output (one set of files per dataset, prefixed by the dataset key):
  <ds>_shap_class_profiles.csv   class x feature: mean |SHAP|, signed mean
  <ds>_shap_class_summary.csv    one row per class: n explained, concentration, top
                                 features, nearest profile, attribution margin
  <ds>_shap_pair_similarity.csv  class x class cosine similarity of importance profiles
  <ds>_shap_stability.csv        one row per class: rank stability across backgrounds
  <ds>_shap_values.npz           per-record exact SHAP values, labels, row sources
  <ds>_shap_model_check.json     fold-0 macro-F1, comparison with the re-run, timings

Usage (from the folder that holds rerun_with_predictions.py and the pipeline modules):
  python shap_overlap_profiles.py --selftest
  python shap_overlap_profiles.py --datasets ertam_kaya cicids2017 unsw_nb15 \
      --rerun_dir rerun_results --output shap_results
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import f1_score
    from scipy.stats import spearmanr
    import xgboost as xgb
except ModuleNotFoundError as e:
    raise SystemExit(f"[setup error] missing package: {e.name}")

SEED = 42


def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, flush=True)


# ---------------------------------------------------------------------------
# Pure functions (exercised by --selftest)

def to_nkf(values, n, k, f):
    """Normalise SHAP output of any shap/xgboost version to shape (n, k, f)."""
    if isinstance(values, list):                      # old shap: list of k arrays (n, f)
        arr = np.stack([np.asarray(v) for v in values], axis=1)
    else:
        arr = np.asarray(values)
        if arr.ndim == 3 and arr.shape == (n, f, k):  # new shap: (n, f, k)
            arr = np.transpose(arr, (0, 2, 1))
        elif arr.ndim == 3 and arr.shape == (n, k, f + 1):   # pred_contribs incl. bias
            arr = arr[:, :, :f]
    if arr.shape != (n, k, f):
        raise ValueError(f"unexpected SHAP shape {getattr(arr, 'shape', None)}, "
                         f"expected {(n, k, f)}")
    return arr.astype(np.float64)


def class_profiles(S, y, classes):
    """Mean |SHAP| and signed mean of output c over the records of class c."""
    k, f = S.shape[1], S.shape[2]
    I = np.full((k, f), np.nan)
    M = np.full((k, f), np.nan)
    for c in classes:
        rows = y == c
        if rows.any():
            I[c] = np.abs(S[rows, c, :]).mean(axis=0)
            M[c] = S[rows, c, :].mean(axis=0)
    return I, M


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0 or not np.isfinite(na) or not np.isfinite(nb):
        return np.nan
    return float(a @ b / (na * nb))


def attribution_margin(S, y, c, classes):
    """Mean summed SHAP evidence for output c on class c minus the largest on another
    class; also returns which class that is."""
    own = S[y == c, c, :].sum(axis=1).mean()
    best, best_d = -np.inf, None
    for d in classes:
        if d == c or not (y == d).any():
            continue
        v = S[y == d, c, :].sum(axis=1).mean()
        if v > best:
            best, best_d = v, d
    return float(own - best), best_d


def spearman(a, b):
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return float(spearmanr(a, b).correlation)


def summarise(S, y, classes, class_names, feature_names, src):
    I, M = class_profiles(S, y, classes)
    rows, pairs = [], []
    for c in classes:
        if not np.isfinite(I[c]).all():
            continue
        tot = I[c].sum()
        order = np.argsort(-I[c])
        sims = {d: cosine(I[c], I[d]) for d in classes
                if d != c and np.isfinite(I[d]).all()}
        nearest = (max(sims, key=lambda d: (sims[d] if np.isfinite(sims[d]) else -9))
                   if sims else None)
        margin, rival = attribution_margin(S, y, c, classes)
        rows.append({
            "class": class_names[c],
            "n_explained": int((y == c).sum()),
            "n_from_validation": int(((y == c) & (src == 0)).sum()),
            "n_from_training": int(((y == c) & (src == 1)).sum()),
            "total_importance": float(tot),
            "concentration_top1": float(I[c, order[0]] / tot) if tot > 0 else np.nan,
            "top1_feature": feature_names[order[0]],
            "top2_feature": feature_names[order[1]] if len(order) > 1 else "",
            "nearest_profile_class": class_names[nearest] if nearest is not None else "",
            "nearest_profile_cosine": sims[nearest] if nearest is not None else np.nan,
            "attribution_margin": margin,
            "closest_rival_class": class_names[rival] if rival is not None else "",
        })
        for d in classes:
            if np.isfinite(I[d]).all():
                pairs.append({"class": class_names[c], "other": class_names[d],
                              "cosine": cosine(I[c], I[d])})
    prof = []
    for c in classes:
        if np.isfinite(I[c]).all():
            for j, fn in enumerate(feature_names):
                prof.append({"class": class_names[c], "feature": fn,
                             "mean_abs_shap": float(I[c, j]),
                             "mean_signed_shap": float(M[c, j])})
    return pd.DataFrame(rows), pd.DataFrame(pairs), pd.DataFrame(prof), I


def choose_explain_rows(y_va, y_tr, n_classes, per_class, min_per_class, rng):
    """Indices into validation (src 0) and training (src 1) parts."""
    va_idx, tr_idx = [], []
    for c in range(n_classes):
        v = np.where(y_va == c)[0]
        if len(v) > per_class:
            v = rng.choice(v, per_class, replace=False)
        va_idx.append(np.sort(v))
        need = max(0, min_per_class - len(v))
        if need:
            t = np.where(y_tr == c)[0]
            t = rng.choice(t, min(need, len(t)), replace=False) if len(t) else t
            tr_idx.append(np.sort(t))
    va = np.concatenate(va_idx) if va_idx else np.array([], int)
    tr = np.concatenate(tr_idx) if tr_idx else np.array([], int)
    return va.astype(int), tr.astype(int)


# ---------------------------------------------------------------------------
# SHAP computations

def exact_tree_shap(model, X, n_classes):
    booster = model.get_booster()
    dm = xgb.DMatrix(X)
    contrib = booster.predict(dm, pred_contribs=True)
    margin = booster.predict(dm, output_margin=True)
    n, f = X.shape
    contrib = np.asarray(contrib)
    if contrib.ndim == 2:      # binary case: (n, f+1) -> treat as one output
        raise ValueError("binary problems are not supported by this script")
    S = to_nkf(contrib, n, n_classes, f)
    bias = contrib[:, :, f]
    err = float(np.max(np.abs(S.sum(axis=2) + bias - margin)))
    return S, err


def interventional_shap(model, X_explain, background, n_classes):
    """Interventional TreeSHAP against an explicit background. Not every shap/xgboost
    version pair supports this for multiclass XGBoost models; callers must be prepared
    for an exception."""
    import shap
    masker = shap.maskers.Independent(background, max_samples=len(background))
    expl = shap.TreeExplainer(model, data=masker,
                              feature_perturbation="interventional",
                              model_output="raw")
    vals = expl.shap_values(X_explain, check_additivity=False)
    n, f = X_explain.shape
    S = to_nkf(vals, n, n_classes, f)
    # When the number of features equals the number of classes the (n, f, k) and
    # (n, k, f) layouts cannot be told apart by shape, so the orientation is settled
    # by additivity: contributions + expected value must reproduce the margin.
    margin = model.get_booster().predict(xgb.DMatrix(X_explain), output_margin=True)
    ev = np.asarray(expl.expected_value, dtype=np.float64).reshape(-1)
    if ev.shape[0] == n_classes and margin.shape == (n, n_classes):
        err = float(np.max(np.abs(S.sum(axis=2) + ev - margin)))
        if f == n_classes:
            S_t = np.transpose(S, (0, 2, 1))
            err_t = float(np.max(np.abs(S_t.sum(axis=2) + ev - margin)))
            if err_t < err:
                S, err = S_t, err_t
        if err > 1e-2:
            raise ValueError(f"interventional SHAP additivity error {err:.3g}; "
                             "output layout not understood for this shap version")
    return S


def _agreement_rows(profiles, I_ref, y_sub, classes, class_names, prefix):
    """Per-class rank agreement between repeated profiles and against a reference."""
    n = len(profiles)
    rows = {}
    for c in classes:
        if not np.isfinite(I_ref[c]).all():
            continue
        between = [spearman(profiles[a][c], profiles[b][c])
                   for a in range(n) for b in range(a + 1, n)
                   if np.isfinite(profiles[a][c]).all() and np.isfinite(profiles[b][c]).all()]
        vs_ref = [spearman(p[c], I_ref[c]) for p in profiles if np.isfinite(p[c]).all()]
        tops = [int(np.argmax(p[c])) for p in profiles if np.isfinite(p[c]).all()]
        ref_top = int(np.argmax(I_ref[c]))
        rows[class_names[c]] = {
            f"{prefix}_rho_between_draws_mean": float(np.nanmean(between)) if between else np.nan,
            f"{prefix}_rho_between_draws_min": float(np.nanmin(between)) if between else np.nan,
            f"{prefix}_rho_vs_exact_mean": float(np.nanmean(vs_ref)) if vs_ref else np.nan,
            f"{prefix}_rho_vs_exact_min": float(np.nanmin(vs_ref)) if vs_ref else np.nan,
            f"{prefix}_top1_agreement_between_draws":
                float(np.mean([t == tops[0] for t in tops])) if tops else np.nan,
            f"{prefix}_top1_matches_exact":
                float(np.mean([t == ref_top for t in tops])) if tops else np.nan,
        }
    return rows


def stability(model, X_tr, X_sub, y_sub, S_exact_sub, classes, class_names,
              bg_size, n_bg, seed, log_fn=None):
    """Two uncertainty checks of the per-class importance profiles.

    sample:      bootstrap over the explained records. Within each class the records are
                 resampled with replacement n_bg times and the exact profile is recomputed.
                 This measures how much the ranking of features depends on which records
                 were explained, and works with every shap/xgboost version because it
                 reuses the exact SHAP values already computed.
    background:  interventional TreeSHAP with a background of bg_size training records,
                 redrawn n_bg times. This measures sensitivity to the choice and size of
                 the background. Some shap/xgboost version pairs cannot run it for
                 multiclass XGBoost; the columns are then left empty and the reason is
                 recorded in the returned info dict.
    """
    log_fn = log_fn or log
    rng = np.random.default_rng(seed)
    I_ex, _ = class_profiles(S_exact_sub, y_sub, classes)
    info = {"n_draws": n_bg, "bg_size": bg_size}

    # 1. bootstrap over explained records
    boot = []
    for r in range(n_bg):
        idx = np.concatenate([rng.choice(np.where(y_sub == c)[0], (y_sub == c).sum(),
                                         replace=True)
                              for c in classes if (y_sub == c).any()])
        I_r, _ = class_profiles(S_exact_sub[idx], y_sub[idx], classes)
        boot.append(I_r)
    rows_s = _agreement_rows(boot, I_ex, y_sub, classes, class_names, "sample")
    info["sample_bootstrap"] = "ok"

    # 2. background sensitivity (optional)
    rows_b = {}
    t0 = time.time()
    try:
        profiles = []
        for r in range(n_bg):
            bg = X_tr[rng.choice(len(X_tr), min(bg_size, len(X_tr)), replace=False)]
            S_r = interventional_shap(model, X_sub, bg, len(class_names))
            profiles.append(class_profiles(S_r, y_sub, classes)[0])
            el = time.time() - t0
            log_fn(f"    background draw {r + 1}/{n_bg} done "
                   f"({el / 60:.1f} min, ~{el / (r + 1) * (n_bg - r - 1) / 60:.1f} min left)")
        rows_b = _agreement_rows(profiles, I_ex, y_sub, classes, class_names, "background")
        info["background_check"] = "ok"
    except Exception as e:  # version-dependent; recorded, not fatal
        info["background_check"] = f"unavailable: {e!r}"
        log_fn(f"    background check unavailable on this shap/xgboost version "
               f"({type(e).__name__}); sample bootstrap is reported alone")

    rows = []
    for c in classes:
        name = class_names[c]
        if name not in rows_s:
            continue
        row = {"class": name, "n_explained": int((y_sub == c).sum())}
        row.update(rows_s[name])
        row.update(rows_b.get(name, {}))
        rows.append(row)
    return pd.DataFrame(rows), info


# ---------------------------------------------------------------------------
# Driver

def get_data(dataset_key, synthetic):
    from rerun_with_predictions import load_prepared
    return load_prepared(dataset_key, synthetic)


def make_model(seed):
    from tree_baselines import _make_xgb
    return _make_xgb(0, random_state=seed)


def run_dataset(ds, args, synthetic=False):
    out = args.output
    pre = os.path.join(out, ds)
    if os.path.exists(pre + "_shap_model_check.json") and not args.force:
        log(f"{ds}: results exist, skipping (use --force to recompute)")
        return True
    t_start = time.time()
    log(f"{ds}: loading")
    X, y, class_names, feature_names = get_data(ds, synthetic)
    k = len(class_names)
    log(f"  {X.shape[0]:,} rows, {X.shape[1]} features, {k} classes")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = list(skf.split(X, y))[args.fold]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr])
    X_va = scaler.transform(X[va])
    y_tr, y_va = y[tr], y[va]

    log(f"  training class-weighted XGBoost on fold {args.fold} "
        f"({len(y_tr):,} rows)")
    t0 = time.time()
    try:
        from rerun_with_predictions import set_all_seeds
        set_all_seeds(SEED)
    except Exception:
        np.random.seed(SEED)
    model = make_model(SEED)
    sw = compute_sample_weight(class_weight="balanced", y=y_tr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr, sample_weight=sw)
    train_s = time.time() - t0
    if len(model.classes_) != k:
        raise RuntimeError(f"model saw {len(model.classes_)} classes, expected {k}")
    pred = model.predict(X_va)
    mf1 = float(f1_score(y_va, pred, average="macro", zero_division=0))
    log(f"  trained in {train_s / 60:.1f} min; fold-{args.fold} macro-F1 = {mf1:.6f}")

    check = {"dataset": ds, "fold": args.fold, "seed": SEED, "macro_f1": mf1,
             "train_seconds": train_s}
    if args.rerun_dir:
        jp = os.path.join(args.rerun_dir, ds, "xgb_classweight",
                          f"seed{SEED}_fold{args.fold}.json")
        try:
            with open(jp, encoding="utf-8") as fh:
                ref = float(json.load(fh)["macro_f1"])
            check["rerun_macro_f1"] = ref
            check["abs_difference"] = abs(ref - mf1)
            tag = "MATCH" if abs(ref - mf1) < 1e-9 else "DIFFERENT"
            log(f"  re-run unit macro-F1 = {ref:.6f} -> {tag} "
                f"(|diff| = {abs(ref - mf1):.2e})")
        except (OSError, KeyError, ValueError) as e:
            check["rerun_macro_f1"] = None
            log(f"  could not read {jp} ({e}); comparison skipped")

    rng = np.random.default_rng(SEED)
    vi, ti = choose_explain_rows(y_va, y_tr, k, args.per_class, args.min_per_class, rng)
    X_ex = np.vstack([X_va[vi], X_tr[ti]]) if len(ti) else X_va[vi]
    y_ex = np.concatenate([y_va[vi], y_tr[ti]]) if len(ti) else y_va[vi]
    src = np.concatenate([np.zeros(len(vi), int), np.ones(len(ti), int)])
    log(f"  explaining {len(y_ex):,} records ({len(ti)} topped up from training)")

    t0 = time.time()
    S, add_err = exact_tree_shap(model, X_ex, k)
    exact_s = time.time() - t0
    log(f"  exact TreeSHAP in {exact_s:.1f} s; max additivity error {add_err:.2e}")
    check["exact_shap_seconds"] = exact_s
    check["additivity_max_abs_error"] = add_err
    if add_err > 1e-2:
        log("  WARNING: additivity error above 1e-2; check the xgboost version")

    classes = list(range(k))
    summ, pairs, prof, _ = summarise(S, y_ex, classes, class_names, feature_names, src)
    os.makedirs(out, exist_ok=True)
    summ.to_csv(pre + "_shap_class_summary.csv", index=False)
    pairs.to_csv(pre + "_shap_pair_similarity.csv", index=False)
    prof.to_csv(pre + "_shap_class_profiles.csv", index=False)
    np.savez_compressed(pre + "_shap_values.npz", shap=S.astype(np.float32),
                        y=y_ex.astype(np.int16), source=src.astype(np.int8),
                        class_names=np.array(class_names),
                        feature_names=np.array(feature_names))

    if args.n_bg > 0:
        sub = np.concatenate([rng.permutation(np.where(y_ex == c)[0])[:args.stab_per_class]
                              for c in classes if (y_ex == c).any()])
        log(f"  stability check: {args.n_bg} bootstrap draws over {len(sub):,} explained "
            f"records, plus {args.n_bg} backgrounds of {args.bg_size} training records "
            f"where supported")
        t0 = time.time()
        try:
            stab, sinfo = stability(model, X_tr, X_ex[sub], y_ex[sub], S[sub], classes,
                                    class_names, args.bg_size, args.n_bg, SEED, log)
            stab.to_csv(pre + "_shap_stability.csv", index=False)
            check["stability_seconds"] = time.time() - t0
            check["stability_info"] = sinfo
            msg = (f"  stability (sample bootstrap): median rho between draws "
                   f"{stab.sample_rho_between_draws_mean.median():.3f}")
            if "background_rho_between_draws_mean" in stab.columns:
                msg += (f"; (background): median rho between draws "
                        f"{stab.background_rho_between_draws_mean.median():.3f}, vs exact "
                        f"{stab.background_rho_vs_exact_mean.median():.3f}")
            log(msg)
        except Exception as e:
            check["stability_error"] = repr(e)
            log(f"  stability check FAILED ({e!r}); exact results are still saved")

    check["total_seconds"] = time.time() - t_start
    with open(pre + "_shap_model_check.json", "w", encoding="utf-8") as fh:
        json.dump(check, fh, indent=2)
    log(f"{ds}: done in {check['total_seconds'] / 60:.1f} min")
    return True


def selftest():
    import tempfile
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
        ok &= bool(cond)

    n, k, f = 7, 3, 4
    base = np.arange(n * k * f, dtype=float).reshape(n, k, f)
    chk("shape list->nkf", np.allclose(to_nkf([base[:, c, :] for c in range(k)], n, k, f), base))
    chk("shape nfk->nkf", np.allclose(to_nkf(np.transpose(base, (0, 2, 1)), n, k, f), base))
    withbias = np.concatenate([base, np.ones((n, k, 1))], axis=2)
    chk("shape contribs->nkf", np.allclose(to_nkf(withbias, n, k, f), base))

    # attribution margin on a hand-built case
    S = np.zeros((4, 2, 2)); y = np.array([0, 0, 1, 1])
    S[0, 0] = [2, 0]; S[1, 0] = [2, 0]; S[2, 0] = [0.5, 0]; S[3, 0] = [0.5, 0]
    m, rival = attribution_margin(S, y, 0, [0, 1])
    chk("attribution margin", abs(m - 1.5) < 1e-12 and rival == 1, f"({m})")

    # end-to-end on a synthetic problem through the real code paths
    try:
        import rerun_with_predictions  # noqa: F401
        import tree_baselines          # noqa: F401
    except SystemExit as e:
        print(f"  cannot import pipeline modules: {e}")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        a = argparse.Namespace(output=tmp, fold=0, per_class=60, min_per_class=20,
                               n_bg=3, bg_size=100, stab_per_class=20, rerun_dir=None,
                               force=False)
        run_dataset("synthetic", a, synthetic=True)
        summ = pd.read_csv(os.path.join(tmp, "synthetic_shap_class_summary.csv"))
        stab = pd.read_csv(os.path.join(tmp, "synthetic_shap_stability.csv"))
        chkj = json.load(open(os.path.join(tmp, "synthetic_shap_model_check.json")))
        z = np.load(os.path.join(tmp, "synthetic_shap_values.npz"))
        chk("one summary row per class", len(summ) == 6)
        chk("rare class topped up to min_per_class",
            int(summ.n_explained.min()) >= 12 and (summ.n_from_training > 0).any(),
            f"(min n = {summ.n_explained.min()})")
        chk("additivity of exact TreeSHAP", chkj["additivity_max_abs_error"] < 1e-3,
            f"({chkj['additivity_max_abs_error']:.1e})")
        chk("stability rows", len(stab) == 6)
        chk("stability sample rho in [-1, 1]",
            stab.sample_rho_between_draws_mean.between(-1, 1).all())
        chk("stability has class column", "class" in stab.columns and len(stab))
        chk("npz shapes", z["shap"].shape[1] == 6 and z["shap"].shape[0] == len(z["y"]))
        chk("concentration in (0, 1]", summ.concentration_top1.between(0, 1).all())
        # skip-if-done
        t0 = time.time(); run_dataset("synthetic", a, synthetic=True)
        chk("existing results are skipped", time.time() - t0 < 1.0)
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+",
                    default=["ertam_kaya", "cicids2017", "unsw_nb15"])
    ap.add_argument("--output", default="shap_results")
    ap.add_argument("--rerun_dir", default="rerun_results",
                    help="folder of rerun_with_predictions.py results, used to confirm "
                         "that the explained model is the reported one")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--per_class", type=int, default=300)
    ap.add_argument("--min_per_class", type=int, default=20)
    ap.add_argument("--n_bg", type=int, default=5)
    ap.add_argument("--bg_size", type=int, default=200)
    ap.add_argument("--stab_per_class", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not 0 <= args.fold < 5:
        ap.error("--fold must be 0..4")
    failed = []
    t0 = time.time()
    for i, ds in enumerate(args.datasets, 1):
        log(f"=== dataset {i}/{len(args.datasets)}: {ds} "
            f"(elapsed {(time.time() - t0) / 60:.1f} min) ===")
        try:
            run_dataset(ds, args)
        except KeyboardInterrupt:
            log("interrupted; finished datasets are saved, rerun to continue")
            return 130
        except Exception as e:
            import traceback
            log(f"FAILED {ds}: {e!r}\n{traceback.format_exc()}")
            failed.append(ds)
    log(f"all done in {(time.time() - t0) / 60:.1f} min -> {os.path.abspath(args.output)}")
    if failed:
        log(f"FAILED datasets: {failed}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
