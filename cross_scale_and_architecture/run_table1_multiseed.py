"""Run Table 1 dataset experiments across multiple seeds and plot variation.

This wrapper runs, for the chosen ``--architecture``:

    python <harness>.py --mode dataset --output-dir <seed_output> --seed <seed>

for seven seeds by default. Each seed writes to its own directory so results are
not overwritten. After the runs finish, the script combines the Table 1 CSVs and
creates scatter plots for every numeric metric.

``--architecture`` selects which intervention harness to drive and which output
subdirectory to read back:

    gpt2  → ../common/intervention_analysis.py  (results/.../dataset_analysis/)
    opt   → opt/intervention_analysis_opt.py    (results/.../dataset_analysis_opt/)
    neo   -> neo/intervention_analysis_neo.py   (results/.../dataset_analysis_neo/)
    qwen  -> qwen/intervention_analysis_qwen.py (results/.../dataset_analysis_qwen/)

All harnesses share the same Table 1 CSV schema, so aggregation and plotting
are architecture-agnostic below.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SEEDS = [0, 1, 2]

# The harness scripts below are named relative to this file, and are resolved against
# _DRIVER_DIR before use. Seeds run from _REPO_ROOT rather than inheriting the caller's
# cwd, because HuggingFace resolves a bare model id as a *local directory* when one
# exists: running from cross_scale_and_architecture/ (which the README instructs) made
# `--model gpt2` pick up the documentation stub in ./gpt2/ and fail with
# "no file named model.safetensors ... found in directory gpt2". The repo root has no
# such shadowing directory. Only bare ids collide; `gpt2-medium`, `facebook/opt-125m`
# and friends were never affected.
_DRIVER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DRIVER_DIR.parent

# Per-architecture harness script and the dataset-analysis subdirectory it writes.
ARCHITECTURES = {
    "gpt2": {
        "script": "../common/intervention_analysis.py",
        "dataset_dir": "dataset_analysis",
        "default_model": "gpt2",
    },
    "opt": {
        "script": "opt/intervention_analysis_opt.py",
        "dataset_dir": "dataset_analysis_opt",
        "default_model": "facebook/opt-125m",
    },
    "neo": {
        "script": "neo/intervention_analysis_neo.py",
        "dataset_dir": "dataset_analysis_neo",
        "default_model": "EleutherAI/gpt-neo-125m",
    },
    "qwen": {
        "script": "qwen/intervention_analysis_qwen.py",
        "dataset_dir": "dataset_analysis_qwen",
        "default_model": "Qwen/Qwen2.5-0.5B",
    },
}
DEFAULT_ARCHITECTURE = "gpt2"


def parse_seeds(value: str) -> list[int]:
    seeds = []
    for part in value.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError("Provide at least one seed.")
    return seeds


def seed_dir(root: Path, seed: int) -> Path:
    return root / f"seed_{seed:03d}"


def safe_model_tag(model_name: str) -> str:
    return (
        model_name.replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def run_seed(args: argparse.Namespace, seed: int, out_dir: Path) -> None:
    script = str((_DRIVER_DIR / ARCHITECTURES[args.architecture]["script"]).resolve())
    cmd = [
        args.python,
        script,
        "--mode",
        "dataset",
        "--output-dir",
        str(Path(out_dir).resolve()),
        "--seed",
        str(seed),
        "--model-name",
        args.model_name,
    ]
    if args.sample_size is not None:
        cmd.extend(["--sample-size", str(args.sample_size)])
    if args.cut_length is not None:
        cmd.extend(["--cut-length", str(args.cut_length)])
    # All four harnesses now accept --layer-mode and --dtype.
    if args.layer_mode is not None:
        cmd.extend(["--layer-mode", args.layer_mode])
    if args.dtype is not None:
        # 'auto' is only supported by the Qwen harness; other harnesses take an explicit dtype.
        cmd.extend(["--dtype", args.dtype])
    # All four harnesses accept --engine; 'manual' is each harness's own default and is
    # what reproduces docs/E1-E2_Summary.md, so an unset flag forwards nothing.
    if args.engine is not None:
        cmd.extend(["--engine", args.engine])
    if args.remote:
        cmd.append("--remote")

    print("\n" + "=" * 80)
    print(f"Seed {seed}: {' '.join(cmd)}")
    print("=" * 80)
    # cwd=_REPO_ROOT: see the note beside ARCHITECTURES — a bare `--model gpt2` would
    # otherwise be shadowed by the ./gpt2 documentation stub.
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)


def read_seed_results(root: Path, seed: int, dataset_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_dir = seed_dir(root, seed) / dataset_dir
    overall_path = analysis_dir / "bos_attention_stats_overall.csv"
    by_dataset_path = analysis_dir / "bos_attention_stats_by_dataset.csv"

    if not overall_path.exists():
        raise FileNotFoundError(f"Missing {overall_path}")
    if not by_dataset_path.exists():
        raise FileNotFoundError(f"Missing {by_dataset_path}")

    overall = pd.read_csv(overall_path)
    by_dataset = pd.read_csv(by_dataset_path)
    overall.insert(0, "seed", seed)
    by_dataset.insert(0, "seed", seed)
    return overall, by_dataset


def add_relative_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Add percentage of baseline per seed and dataset group."""
    df = df.copy()
    group_cols = ["seed"]
    if "dataset" in df.columns:
        group_cols.append("dataset")

    df["mean_bos_attention"] = pd.to_numeric(df["mean_bos_attention"])
    df["stderr"] = pd.to_numeric(df["stderr"])

    baseline = (
        df[df["intervention"] == "int_a"][group_cols + ["mean_bos_attention"]]
        .rename(columns={"mean_bos_attention": "baseline_mean_bos_attention"})
    )
    df = df.merge(baseline, on=group_cols, how="left")
    df["relative_to_baseline_pct"] = (
        100.0 * df["mean_bos_attention"] / df["baseline_mean_bos_attention"]
    )
    return df


def write_combined_csvs(root: Path, seeds: list[int], dataset_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    aggregate_dir = root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    overall_frames = []
    by_dataset_frames = []
    for seed in seeds:
        overall, by_dataset = read_seed_results(root, seed, dataset_dir)
        overall_frames.append(overall)
        by_dataset_frames.append(by_dataset)

    overall_all = add_relative_metric(pd.concat(overall_frames, ignore_index=True))
    by_dataset_all = add_relative_metric(pd.concat(by_dataset_frames, ignore_index=True))

    overall_all.to_csv(aggregate_dir / "table1_multiseed_overall.csv", index=False)
    by_dataset_all.to_csv(aggregate_dir / "table1_multiseed_by_dataset.csv", index=False)

    summary_cols = ["dataset", "intervention", "description"]
    summary = (
        by_dataset_all.groupby(summary_cols, as_index=False)
        .agg(
            mean_bos_attention_mean=("mean_bos_attention", "mean"),
            mean_bos_attention_std=("mean_bos_attention", "std"),
            stderr_mean=("stderr", "mean"),
            relative_to_baseline_pct_mean=("relative_to_baseline_pct", "mean"),
            relative_to_baseline_pct_std=("relative_to_baseline_pct", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    summary.to_csv(aggregate_dir / "table1_multiseed_summary_by_dataset.csv", index=False)

    overall_summary = (
        overall_all.groupby(["intervention", "description"], as_index=False)
        .agg(
            mean_bos_attention_mean=("mean_bos_attention", "mean"),
            mean_bos_attention_std=("mean_bos_attention", "std"),
            stderr_mean=("stderr", "mean"),
            relative_to_baseline_pct_mean=("relative_to_baseline_pct", "mean"),
            relative_to_baseline_pct_std=("relative_to_baseline_pct", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    overall_summary.to_csv(aggregate_dir / "table1_multiseed_summary_overall.csv", index=False)

    return overall_all, by_dataset_all, aggregate_dir


def scatter_metric(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    dataset: str | None = None,
) -> None:
    plot_df = df.copy()
    if dataset is not None:
        plot_df = plot_df[plot_df["dataset"] == dataset]

    descriptions = (
        plot_df[["intervention", "description"]]
        .drop_duplicates()
        .sort_values("intervention")
    )

    fig, ax = plt.subplots(figsize=(13, 7))
    for _, row in descriptions.iterrows():
        intervention = row["intervention"]
        label = f"{intervention}: {row['description']}"
        sub = plot_df[plot_df["intervention"] == intervention].sort_values("seed")
        ax.scatter(sub["seed"], sub[metric], label=label, s=55, alpha=0.85)
        ax.plot(sub["seed"], sub[metric], linewidth=1, alpha=0.35)

    ax.set_title(title)
    ax.set_xlabel("Seed")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_plots(overall: pd.DataFrame, by_dataset: pd.DataFrame, aggregate_dir: Path) -> None:
    plot_dir = aggregate_dir / "scatter_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["mean_bos_attention", "stderr", "relative_to_baseline_pct"]
    for metric in metrics:
        scatter_metric(
            overall,
            metric,
            f"Table 1 multiseed overall: {metric}",
            plot_dir / f"overall_{metric}.png",
        )

        for dataset in sorted(by_dataset["dataset"].unique()):
            scatter_metric(
                by_dataset,
                metric,
                f"Table 1 multiseed {dataset}: {metric}",
                plot_dir / f"{dataset}_{metric}.png",
                dataset=dataset,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Table 1 dataset analysis seven times and scatter-plot metrics."
    )
    parser.add_argument(
        "--architecture",
        choices=sorted(ARCHITECTURES),
        default=DEFAULT_ARCHITECTURE,
        help="Which intervention harness to drive: 'gpt2' (../common/intervention_analysis.py), "
             "'opt' (intervention_analysis_opt.py), 'neo' (intervention_analysis_neo.py), "
             "or 'qwen' (intervention_analysis_qwen.py). Default: gpt2.",
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="Comma-separated seeds. Default: 0,1,2,3,4,5,6",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Root results directory. Default: results",
    )
    parser.add_argument(
        "--experiment-name",
        default="table1_multiseed",
        help="Subdirectory under --output-dir for multiseed runs.",
    )
    parser.add_argument(
        "--model-name",
        "--model",
        dest="model_name",
        default=None,
        help="Hugging Face model name or local path. Defaults to the chosen "
             "architecture's base model.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the intervention harness.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional override forwarded to the intervention harness.",
    )
    parser.add_argument(
        "--cut-length",
        type=int,
        default=None,
        help="Optional override forwarded to the intervention harness.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16", "auto"],
        default=None,
        help="Model dtype, forwarded to every harness (default: harness default float32). "
             "'auto' is Qwen-only. Larger checkpoints typically need float16/bfloat16.",
    )
    parser.add_argument(
        "--layer-mode",
        choices=["scaled", "fixed"],
        default=None,
        help="Mid-layer band, forwarded to every harness (default: harness default 'scaled', "
             "which excludes the first 3 and last layer and reduces to layers 4-11 for 12-layer "
             "models). Use 'fixed' to force layers 4-11 on every size.",
    )
    parser.add_argument(
        "--engine",
        choices=["manual", "nnsight"],
        default=None,
        help="Execution engine, forwarded to every harness (default: harness default "
             "'manual', which re-implements the forward pass by hand and reproduces the "
             "published Table-1 numbers). 'nnsight' runs the real HuggingFace forward under "
             "NNsight and reads attention from the model itself.",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Execute the NNsight engine remotely on NDIF (requires --engine nnsight).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a seed when its overall CSV already exists.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Do not run experiments; only aggregate and plot existing seed outputs.",
    )
    args = parser.parse_args()

    if args.remote and args.engine != "nnsight":
        parser.error("--remote requires --engine nnsight")

    # Seeds run from _REPO_ROOT, so a caller-relative --python (e.g. ../.venv/Scripts/
    # python.exe) must be resolved against the *caller's* cwd first. The default is
    # sys.executable, which is already absolute.
    _py = Path(args.python)
    if _py.exists():
        args.python = str(_py.resolve())

    architecture = ARCHITECTURES[args.architecture]
    dataset_dir = architecture["dataset_dir"]
    if args.model_name is None:
        args.model_name = architecture["default_model"]

    if args.experiment_name == "table1_multiseed":
        parts = ["table1_multiseed"]
        if args.architecture != "gpt2":
            parts.append(args.architecture)
        if args.model_name != "gpt2":
            parts.append(safe_model_tag(args.model_name))
        # Tag the engine so an NNsight run cannot silently overwrite the manual results
        # directory (which would also break the md5-distinctness audit in
        # docs/E1-E2_Summary.md). Default naming is unchanged when --engine is unset.
        if args.engine == "nnsight":
            parts.append("nnsight")
        args.experiment_name = "_".join(parts)

    root = args.output_dir / args.experiment_name
    root.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        for seed in args.seeds:
            out_dir = seed_dir(root, seed)
            overall_csv = out_dir / dataset_dir / "bos_attention_stats_overall.csv"
            if args.skip_existing and overall_csv.exists():
                print(f"Seed {seed}: found {overall_csv}; skipping.")
                continue
            run_seed(args, seed, out_dir)

    overall, by_dataset, aggregate_dir = write_combined_csvs(root, args.seeds, dataset_dir)
    make_plots(overall, by_dataset, aggregate_dir)

    print("\nDone.")
    print(f"Seed outputs: {root}")
    print(f"Combined CSVs and scatter plots: {aggregate_dir}")


if __name__ == "__main__":
    main()
