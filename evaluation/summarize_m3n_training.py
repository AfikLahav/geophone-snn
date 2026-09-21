"""Create report-ready tables and a figure from the M3N transfer audit database."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "report" / "m3n_training_analysis" / "m3n_positive_transfer.sqlite"
DEFAULT_OUT = ROOT / "report" / "m3n_training_analysis" / "m3n_focused_summary"
AUDIT_DB = ROOT / "report" / "m3n_gps_analysis" / "m3n_gps_audit.sqlite"
CONDITIONS = ["synthetic_only", "adapted_positive_replay"]
LABELS = {"synthetic_only": "Synthetic only", "adapted_positive_replay": "M3N adapted"}
FAMILY_ORDER = ["SNN-S", "SNN-M", "SNN-L"]


def weighted_by_seed(frame: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(group + ["seed", "condition"], sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = part["n_windows"].sum()
        rows.append(
            dict(
                zip(group + ["seed", "condition"], keys),
                n_windows=int(n),
                vehicle_recall=float(np.average(part["vehicle_recall"], weights=part["n_windows"])),
                human_head_rate=float(np.average(part["human_head_rate"], weights=part["n_windows"])),
            )
        )
    return pd.DataFrame(rows)


def across_seeds(per_seed: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    result = (
        per_seed.groupby(group + ["condition"], sort=False)
        .agg(
            unique_windows=("n_windows", "first"),
            vehicle_recall_mean=("vehicle_recall", "mean"),
            vehicle_recall_sd=("vehicle_recall", "std"),
            human_head_rate_mean=("human_head_rate", "mean"),
            human_head_rate_sd=("human_head_rate", "std"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    return result


def load_tables(database: Path):
    con = sqlite3.connect(database)
    metrics = pd.read_sql_query(
        """
        SELECT e.family,e.seed,e.parameter_count,e.heldout_scene,m.condition,m.subset,
               m.terrain,m.n_windows,m.vehicle_recall,m.human_head_rate,m.human_decision_rate
        FROM metrics m JOIN experiments e USING(experiment_id)
        WHERE e.status='complete'
        """,
        con,
    )
    distance = pd.read_sql_query(
        """
        SELECT e.family,e.seed,e.parameter_count,e.heldout_scene,d.condition,d.distance_bin,
               d.n_windows,d.vehicle_recall,d.human_head_rate
        FROM distance_metrics d JOIN experiments e USING(experiment_id)
        WHERE e.status='complete'
        """,
        con,
    )
    experiments = pd.read_sql_query(
        "SELECT * FROM experiments WHERE status='complete' ORDER BY family,seed,heldout_scene", con
    )
    con.close()
    return metrics, distance, experiments


def make_figure(overall, scenes, distance, output_dir):
    colors = {"synthetic_only": "#64748b", "adapted_positive_replay": "#0f766e"}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)

    def grouped_bars(ax, frame, category, value, ylabel, ylim=(0, 1.04)):
        categories = list(dict.fromkeys(frame[category]))
        x = np.arange(len(categories))
        width = 0.36
        for offset, condition in zip((-0.5, 0.5), CONDITIONS):
            part = frame.set_index([category, "condition"])
            values = [part.loc[(item, condition), value] for item in categories]
            ax.bar(
                x + offset * width,
                values,
                width,
                label=LABELS[condition],
                color=colors[condition],
            )
        ax.set_xticks(x, categories)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)

    grouped_bars(
        axes[0, 0], overall, "family", "vehicle_recall_mean", "Vehicle detection rate"
    )
    axes[0, 0].set_title("A  Held-out M3N detection")
    axes[0, 0].legend(frameon=False, loc="lower right")

    grouped_bars(
        axes[0, 1], overall, "family", "human_head_rate_mean",
        "Human-head activation rate", (0, 0.25),
    )
    axes[0, 1].set_title("B  Incorrect human-head activation")

    medium_scene = scenes[scenes.family == "SNN-M"].copy()
    grouped_bars(
        axes[1, 0], medium_scene, "heldout_scene", "vehicle_recall_mean",
        "Vehicle detection rate",
    )
    axes[1, 0].set_title("C  SNN-M by held-out scene")

    medium_distance = distance[distance.family == "SNN-M"].copy()
    bins = ["0-10 m", "10-20 m", "20-40 m", "40-80 m", "80+ m"]
    for condition in CONDITIONS:
        part = medium_distance[medium_distance.condition == condition].set_index("distance_bin")
        axes[1, 1].plot(
            bins,
            [part.loc[item, "vehicle_recall_mean"] for item in bins],
            marker="o",
            linewidth=2.2,
            color=colors[condition],
            label=LABELS[condition],
        )
    axes[1, 1].set_ylim(0, 1.04)
    axes[1, 1].set_ylabel("Vehicle detection rate")
    axes[1, 1].set_title("D  SNN-M by GPS distance")
    axes[1, 1].grid(alpha=0.22)
    axes[1, 1].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "M3N vehicle detection before and after positive-only adaptation",
        fontsize=15,
        fontweight="semibold",
    )
    fig.savefig(output_dir / "m3n_training_results.png", dpi=220)
    fig.savefig(output_dir / "m3n_training_results.svg")
    plt.close(fig)


def write_readme(output_dir, overall, scenes, terrain, distance, experiments):
    adapted = overall[overall.condition == "adapted_positive_replay"].set_index("family")
    baseline = overall[overall.condition == "synthetic_only"].set_index("family")
    lines = [
        "# M3N positive-only transfer experiment",
        "",
        "The 200 Hz SNN models were fine-tuned with M3N vehicle windows from four scenes and "
        "tested on the fifth scene. Each fold therefore measures transfer to a recording that "
        "was absent from training. A window was eligible when its timestamp sequence was continuous "
        "and its 5-25 Hz energy was at least 6 dB above the recording background. Every update batch "
        "contained equal numbers of eligible M3N vehicle windows and synthetic training windows.",
        "",
        "M3N does not provide a verified negative class. The valid measures are vehicle detection "
        "rate and incorrect activation of the human head on vehicle windows. Accuracy, precision, "
        "and false-positive rate cannot be measured from this dataset.",
        "",
        "| Model | Parameters | Synthetic-only detection | After M3N adaptation | Human-head activation after adaptation |",
        "|---|---:|---:|---:|---:|",
    ]
    params = experiments.groupby("family").parameter_count.first()
    for family in FAMILY_ORDER:
        lines.append(
            f"| {family} | {int(params[family]):,} | "
            f"{100 * baseline.loc[family, 'vehicle_recall_mean']:.1f}% | "
            f"{100 * adapted.loc[family, 'vehicle_recall_mean']:.1f}% | "
            f"{100 * adapted.loc[family, 'human_head_rate_mean']:.1f}% |"
        )
    lines += [
        "",
        "The medium model is the clearest result. Across 1,096 unique held-out windows and three "
        "seeds, its detection rate rises from 82.8% to 96.9%. Incorrect human-head activation falls "
        "from 19.0% to 2.1%. The large model reaches 97.1%, which is only 0.2 percentage points above "
        "the medium model despite having 3.8 times as many parameters.",
        "",
        "The improvement is concentrated where the synthetic-only models were weak. SNN-M rises "
        "from 56.6% to 92.5% on asphalt and from 56.2% to 90.0% on concrete. Gravel scenes were "
        "already near saturation. This supports the narrower conclusion that a small amount of real "
        "vehicle data can correct terrain-specific gaps in the synthetic training set.",
        "",
        "The result remains a detection result. Without verified quiet or non-vehicle intervals, it "
        "does not establish M3N accuracy or precision.",
        "",
        f"Completed fits: {len(experiments)}. Training steps per fit: "
        f"{int(experiments.steps.iloc[0])}. Batch size: {int(experiments.batch_size.iloc[0])}. "
        f"Unique eligible windows: {int(overall.unique_windows.iloc[0])}.",
        "",
        "The SQLite database contains experiment settings, train/test membership, loss history, "
        "thresholds, every held-out window score, aggregate metrics, distance metrics, and the saved "
        "model path for each fit.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    metrics, distance_raw, experiments = load_tables(args.database)
    base = metrics[(metrics.subset == "signal_ge6") & (metrics.terrain == "ALL")]
    overall_seed = weighted_by_seed(base, ["family"])
    overall = across_seeds(overall_seed, ["family"])
    overall["parameter_count"] = overall.family.map(
        experiments.groupby("family").parameter_count.first()
    )

    scene_seed = weighted_by_seed(base, ["family", "heldout_scene"])
    scenes = across_seeds(scene_seed, ["family", "heldout_scene"])

    terrain_base = metrics[(metrics.subset == "signal_ge6") & (metrics.terrain != "ALL")]
    terrain_seed = weighted_by_seed(terrain_base, ["family", "terrain"])
    terrain = across_seeds(terrain_seed, ["family", "terrain"])

    distance_seed = weighted_by_seed(distance_raw, ["family", "distance_bin"])
    distance = across_seeds(distance_seed, ["family", "distance_bin"])

    overall.to_csv(args.output / "overall_metrics.csv", index=False)
    scenes.to_csv(args.output / "scene_metrics.csv", index=False)
    terrain.to_csv(args.output / "terrain_metrics.csv", index=False)
    distance.to_csv(args.output / "distance_metrics.csv", index=False)
    experiments.to_csv(args.output / "experiments.csv", index=False)
    make_figure(overall, scenes, distance, args.output)
    write_readme(args.output, overall, scenes, terrain, distance, experiments)

    print(overall.to_string(index=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
