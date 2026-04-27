"""
Evaluation script: compare ablation runs, print summary table.
"""
import argparse
import csv
import os
from pathlib import Path

import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HSM ablation runs")
    parser.add_argument("--run_dir", type=str, default=None, help="Single run directory")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory containing multiple run dirs")
    parser.add_argument("--config", type=str, default=None, help="Config for loading model checkpoint")
    return parser.parse_args()


def read_csv_metrics(csv_path: str) -> list:
    """Read all rows from a metrics CSV."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def summarize_run(run_dir: str) -> dict:
    """Extract summary statistics from a run directory."""
    summary = {"run_dir": run_dir, "mode": "unknown"}

    # Try summary.txt first
    summary_path = os.path.join(run_dir, "summary.txt")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    summary[k.strip()] = v.strip()

    # Read config to get mode if missing
    cfg_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            summary["config"] = cfg
        except Exception:
            pass

    # Read metrics CSV for detailed stats
    csv_path = os.path.join(run_dir, "metrics.csv")
    rows = read_csv_metrics(csv_path)
    if rows:
        # Find best val_acc
        val_accs = [float(r["val_acc"]) for r in rows if r.get("val_acc") not in ("", None)]
        if val_accs:
            summary["max_acc"] = max(val_accs)
            summary["final_acc"] = val_accs[-1]

        # Compute mean gate stats from last 20% of training steps
        n = len(rows)
        tail = rows[int(n * 0.8):]
        for key in ["mean_g_g", "mean_g_s", "gate_entropy", "util"]:
            vals = [float(r[key]) for r in tail if r.get(key) not in ("", None)]
            if vals:
                summary[key] = sum(vals) / len(vals)

    return summary


def discover_runs(results_dir: str) -> list:
    """Find all run directories under results_dir."""
    runs = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return runs
    for d in sorted(results_path.iterdir()):
        if d.is_dir() and (d / "metrics.csv").exists():
            runs.append(str(d))
    return runs


def print_table(summaries: list):
    """Print comparison table."""
    cols = ["mode", "final_acc", "max_acc", "mean_g_g", "mean_g_s", "gate_entropy", "util"]
    col_widths = {c: max(len(c), 10) for c in cols}

    # Header
    header = " | ".join(c.ljust(col_widths[c]) for c in cols)
    sep = "-+-".join("-" * col_widths[c] for c in cols)
    print("\n" + header)
    print(sep)

    for s in summaries:
        row_parts = []
        for c in cols:
            val = s.get(c, "N/A")
            if isinstance(val, float):
                val = f"{val:.4f}"
            row_parts.append(str(val).ljust(col_widths[c]))
        print(" | ".join(row_parts))

    print()


def check_hypotheses(summaries: list):
    """Print hypothesis checks."""
    by_mode = {s.get("mode", "unknown"): s for s in summaries}

    def get_acc(mode):
        s = by_mode.get(mode, {})
        v = s.get("max_acc") or s.get("final_acc")
        return float(v) if v is not None else None

    full_acc = get_acc("full")
    local_acc = get_acc("local_only")
    global_acc = get_acc("global_only")
    fixed_acc = get_acc("fixed_mix")

    print("=== Hypothesis Checks ===")
    if full_acc is not None and local_acc is not None and global_acc is not None:
        result = full_acc >= max(local_acc, global_acc)
        print(f"Full >= max(Local-only, Global-only): {result}  "
              f"(full={full_acc:.4f}, local={local_acc:.4f}, global={global_acc:.4f})")
    else:
        missing = [m for m, v in [("full", full_acc), ("local_only", local_acc), ("global_only", global_acc)] if v is None]
        print(f"Full >= max(Local-only, Global-only): N/A (missing runs: {missing})")

    if full_acc is not None and fixed_acc is not None:
        result = full_acc > fixed_acc
        print(f"Full > Fixed naive mix: {result}  (full={full_acc:.4f}, fixed={fixed_acc:.4f})")
    else:
        print(f"Full > Fixed naive mix: N/A (missing runs)")


def eval_checkpoint(run_dir: str):
    """Load checkpoint and run evaluation."""
    ckpt_path = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}")
        return

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        # Fall back if checkpoint contains non-tensor data (e.g. config dict)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # nosec
    print(f"Checkpoint: step={ckpt.get('step')}, val_acc={ckpt.get('val_acc', 'N/A'):.4f}")
    print(f"Mode: {ckpt.get('mode', 'unknown')}")


def main():
    args = parse_args()

    if args.run_dir:
        # Single run
        summary = summarize_run(args.run_dir)
        print_table([summary])
        eval_checkpoint(args.run_dir)
    elif args.results_dir:
        # Multiple runs
        run_dirs = discover_runs(args.results_dir)
        if not run_dirs:
            print(f"No runs found in {args.results_dir}")
            return
        summaries = [summarize_run(d) for d in run_dirs]
        print(f"Found {len(summaries)} run(s) in {args.results_dir}")
        print_table(summaries)
        check_hypotheses(summaries)
    else:
        print("Provide --run_dir or --results_dir")


if __name__ == "__main__":
    main()
