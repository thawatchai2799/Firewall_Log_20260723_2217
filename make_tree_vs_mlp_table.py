"""
make_tree_vs_mlp_table.py

Merge the MLP technique results (from pipeline.py -> {dataset}_comparison_results.json) with the
tree baseline results (from tree_baselines.py -> {dataset}_tree_baseline_results.json) into a single
ranked macro-F1 + per-class-F1 table, ranked for direct comparison.

This does NOT re-run anything; it only reads the two JSON files and prints/writes a combined table,
so the tree numbers sit directly beside the four MLP techniques computed on the same folds.

Usage:
    python make_tree_vs_mlp_table.py --dataset ertam_kaya --results results/
"""
import argparse
import json
import os


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build(dataset_key: str, results_dir: str):
    mlp = _load(os.path.join(results_dir, f"{dataset_key}_comparison_results.json"))
    tree = _load(os.path.join(results_dir, f"{dataset_key}_tree_baseline_results.json"))
    tabnet = _load(os.path.join(results_dir, f"{dataset_key}_tabnet_baseline_results.json"))

    rows = []  # (name, mean, std, per_class_f1, class_names, family)
    if mlp:
        for name, s in mlp["summary"].items():
            rows.append((name, s["mean_macro_f1"], s["std_macro_f1"],
                         s["mean_per_class_f1"], s["class_names"], "MLP"))
    if tree:
        for name, s in tree["summary"].items():
            rows.append((name, s["mean_macro_f1"], s["std_macro_f1"],
                         s["mean_per_class_f1"], s["class_names"], "Tree"))
    if tabnet:
        for name, s in tabnet["summary"].items():
            rows.append((name, s["mean_macro_f1"], s["std_macro_f1"],
                         s["mean_per_class_f1"], s["class_names"], "TabNet"))

    if not rows:
        raise SystemExit(f"no result JSONs found for '{dataset_key}' in {results_dir}")

    # rank by macro-F1 descending
    rows.sort(key=lambda r: r[1], reverse=True)

    # Reference class order = first (best) row's order. Every other row is REALIGNED BY NAME
    # against this reference before printing. Without this, two JSONs whose class_names happen
    # to be in a different order (e.g. produced by different dataset versions) would silently
    # place per-class F1 values under the wrong column headers - a silent-corruption bug.
    class_names = rows[0][4]
    ref_set = set(class_names)
    aligned_rows = []
    for (name, mean, std, pcf, cnames, fam) in rows:
        if set(cnames) != ref_set:
            raise SystemExit(
                f"class-name mismatch: '{name}' has classes {sorted(cnames)} but the reference "
                f"row has {sorted(class_names)}. The two result JSONs were produced from "
                f"different datasets/label sets and cannot be merged into one table.")
        by_name = dict(zip(cnames, pcf))
        aligned_pcf = [by_name[c] for c in class_names]
        aligned_rows.append((name, mean, std, aligned_pcf, class_names, fam))
    rows = aligned_rows

    # ---- plain-text table ----
    header = ["Rank", "Model", "Family", "Macro-F1 (mean +/- std)"] + class_names
    widths = [4, 18, 6, 24] + [max(10, len(c) + 1) for c in class_names]
    def fmt_row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    lines = [fmt_row(header), fmt_row(["-" * w for w in widths])]
    for i, (name, mean, std, pcf, cnames, fam) in enumerate(rows, 1):
        cells = [i, name, fam, f"{mean:.4f} +/- {std:.4f}"] + [f"{v:.3f}" for v in pcf]
        lines.append(fmt_row(cells))

    table_txt = "\n".join(lines)
    print(table_txt)

    out_txt = os.path.join(results_dir, f"{dataset_key}_tree_vs_mlp_table.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(table_txt + "\n")
    out_json = os.path.join(results_dir, f"{dataset_key}_tree_vs_mlp_combined.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([
            {"rank": i, "model": name, "family": fam,
             "mean_macro_f1": mean, "std_macro_f1": std,
             "per_class_f1": dict(zip(cnames, pcf))}
            for i, (name, mean, std, pcf, cnames, fam) in enumerate(rows, 1)
        ], f, ensure_ascii=False, indent=2)
    print(f"\nwritten: {out_txt}\nwritten: {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    build(args.dataset, args.results)
