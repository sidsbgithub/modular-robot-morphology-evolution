"""
Final-experiment analysis for Assignment 1.

Reads the EA variants produced by A1_evolution_tree.py --final (one mutation
operator per variant, 10 independent seeds each) and the matched-budget
random-search baseline, and writes the report figures and statistics:

    fig_convergence.png     best fitness per generation, mean +/- std, all groups
    fig_tree_size.png       mean modules per individual over generations vs targets
    fig_final_boxplot.png   distribution of final best fitness per run
    final_stats.csv         n / mean / std / median / min / max / 95% CI per group
    pairwise_tests.csv      Mann-Whitney U (two-sided) + Cliff's delta per pair
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "__data__" / "A1_evolution_tree"

SERIES_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

TARGET_SIZES = [7, 11, 15, 19, 25]

RANDOM_LABEL = "random search"


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def variant_name(row: pd.Series) -> str:
    return f"{row['mutation_operator']} mutation"


def load_inputs(experiment: str, random_csv: Path):
    exp_dir = DATA_DIR / experiment
    raw = pd.read_csv(exp_dir / "raw_results.csv")
    gens = pd.read_csv(exp_dir / "generation_results.csv")
    random_df = pd.read_csv(random_csv)

    raw["variant"] = raw.apply(variant_name, axis=1)
    gens["variant"] = gens.apply(variant_name, axis=1)

    return exp_dir, raw, gens, random_df


def align_random_search(random_df: pd.DataFrame, pop_size: int, generations: int) -> pd.DataFrame:
    """Best-so-far of random search at the evaluation count each EA generation reaches."""

    checkpoints = pop_size * (np.arange(generations + 1) + 1)
    rows = []

    for seed, group in random_df.groupby("seed"):
        group = group.sort_values("evaluation")
        best = group["best_so_far"].to_numpy()
        for generation, evaluation in enumerate(checkpoints):
            index = min(int(evaluation), len(best)) - 1
            rows.append(
                {
                    "seed": seed,
                    "generation": generation,
                    "evaluations": int(evaluation),
                    "best_fitness": float(best[index]),
                }
            )

    return pd.DataFrame(rows)


def mean_std_curve(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return df.groupby("generation")[value].agg(mean="mean", std="std").reset_index()


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Probability(x > y) - Probability(x < y); negative = x tends to be lower."""
    greater = sum(np.sum(xi > y) for xi in x)
    less = sum(np.sum(xi < y) for xi in x)
    return float((greater - less) / (len(x) * len(y)))


def summarize(values: np.ndarray) -> dict:
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else float("nan")
    half = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def analyze(experiment: str, random_csv: Path) -> None:
    """Produce figures and statistics for one final experiment folder."""

    exp_dir, raw, gens, random_df = load_inputs(experiment, Path(random_csv))

    pop_size = int(raw["population"].iloc[0])
    generations = int(raw["generations"].iloc[0])
    budget = pop_size * (generations + 1)

    variants = list(dict.fromkeys(raw["variant"]))
    colors = {name: SERIES_COLORS[i] for i, name in enumerate(variants)}
    colors[RANDOM_LABEL] = SERIES_COLORS[len(variants)]

    random_aligned = align_random_search(random_df, pop_size, generations)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    style_axes(ax)

    curves = {}
    for name in variants:
        curves[name] = mean_std_curve(gens[gens["variant"] == name], "best_fitness")
    curves[RANDOM_LABEL] = mean_std_curve(random_aligned, "best_fitness")

    # Final values go into the legend: with six series ending within a narrow
    # band, direct end-labels collide.
    for name, curve in curves.items():
        style = "--" if name == RANDOM_LABEL else "-"
        ax.plot(
            curve["generation"],
            curve["mean"],
            color=colors[name],
            linewidth=2,
            linestyle=style,
            label=f"{name}  (final {curve['mean'].iloc[-1]:.2f})",
        )
        ax.fill_between(
            curve["generation"],
            curve["mean"] - curve["std"].fillna(0),
            curve["mean"] + curve["std"].fillna(0),
            color=colors[name],
            alpha=0.15,
            linewidth=0,
        )

    ax.set_xlim(0, generations)
    ax.set_xlabel(
        f"Generation  (random search aligned by evaluations: {pop_size} per generation, "
        f"{budget} total)"
    )
    ax.set_ylabel("Best fitness (lower is better)")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(exp_dir / "fig_convergence.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    style_axes(ax)

    for name in variants:
        curve = mean_std_curve(gens[gens["variant"] == name], "mean_modules")
        ax.plot(
            curve["generation"],
            curve["mean"],
            color=colors[name],
            linewidth=2,
            label=f"{name}  (final {curve['mean'].iloc[-1]:.1f})",
        )
        ax.fill_between(
            curve["generation"],
            curve["mean"] - curve["std"].fillna(0),
            curve["mean"] + curve["std"].fillna(0),
            color=colors[name],
            alpha=0.15,
            linewidth=0,
        )

    for size in TARGET_SIZES:
        ax.axhline(size, color=MUTED, linewidth=0.8, linestyle=":")
        ax.annotate(
            f"target body: {size} modules",
            xy=(generations, size),
            xytext=(-2, 2),
            textcoords="offset points",
            fontsize=7,
            color=MUTED,
            ha="right",
        )

    ax.set_xlim(0, generations)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean modules per individual (population)")
    # The only region tall enough for five legend rows is the empty band above
    # the curves; a white backing keeps the dotted target lines from cutting
    # through the text.
    ax.legend(
        fontsize=9,
        loc="upper center",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="none",
    )
    fig.tight_layout()
    fig.savefig(exp_dir / "fig_tree_size.png")
    plt.close(fig)

    groups = {name: raw.loc[raw["variant"] == name, "best_fitness"].to_numpy() for name in variants}
    groups[RANDOM_LABEL] = (
        random_df.sort_values("evaluation").groupby("seed")["best_so_far"].last().to_numpy()
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)
    style_axes(ax)
    ax.grid(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)

    names = list(groups)
    box = ax.boxplot(
        [groups[n] for n in names],
        widths=0.5,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": INK, "linewidth": 1.2},
        meanprops={"marker": "D", "markerfacecolor": INK, "markeredgecolor": INK, "markersize": 4},
        whiskerprops={"color": AXIS},
        capprops={"color": AXIS},
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markeredgecolor": MUTED,
            "markerfacecolor": "none",
        },
    )
    for patch, name in zip(box["boxes"], names):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.35)
        patch.set_edgecolor(colors[name])
    rng = np.random.default_rng(0)
    for i, name in enumerate(names, start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(groups[name]))
        ax.scatter(i + jitter, groups[name], s=14, color=colors[name], zorder=3)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([name.replace(" mutation", "") for name in names], fontsize=9, color=INK)
    ax.set_xlabel("Mutation operator (other settings identical); random search = same budget")
    ax.set_ylabel("Final best fitness per run (lower is better)")
    fig.tight_layout()
    fig.savefig(exp_dir / "fig_final_boxplot.png")
    plt.close(fig)

    stats_rows = []
    for name in names:
        row = {"group": name, **summarize(groups[name])}
        if name != RANDOM_LABEL:
            last = gens[(gens["variant"] == name) & (gens["generation"] == generations)]
            row["final_mean_modules"] = float(last["mean_modules"].mean())
            row["final_mean_modules_std"] = float(last["mean_modules"].std(ddof=1))
            row["final_best_modules"] = float(last["best_modules"].mean())
            row["final_best_depth"] = float(last["best_depth"].mean())
            random_final = groups[RANDOM_LABEL].mean()
            curve = curves[name]
            reached = curve.loc[curve["mean"] <= random_final, "generation"]
            row["generation_reaching_random_final_mean"] = (
                int(reached.iloc[0]) if not reached.empty else None
            )
        stats_rows.append(row)

    final_stats = pd.DataFrame(stats_rows)
    final_stats.to_csv(exp_dir / "final_stats.csv", index=False)

    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    test_rows = []
    for a, b in pairs:
        x, y = groups[a], groups[b]
        p = float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
        test_rows.append(
            {
                "group_a": a,
                "group_b": b,
                "mean_diff_a_minus_b": float(x.mean() - y.mean()),
                "mann_whitney_p": p,
                "cliffs_delta_a_vs_b": cliffs_delta(x, y),
            }
        )
    tests = pd.DataFrame(test_rows)
    tests.to_csv(exp_dir / "pairwise_tests.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\nExperiment: {exp_dir}")
    print(
        f"Population={pop_size}, generations={generations}, budget={budget} evaluations, "
        f"seeds per group: {[len(groups[n]) for n in names]}"
    )
    print("\nFinal statistics:")
    print(final_stats.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nPairwise tests (Mann-Whitney U, two-sided; Cliff's delta < 0 means A lower):")
    print(tests.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(
        f"\nFigures written to {exp_dir}: fig_convergence.png, fig_tree_size.png, fig_final_boxplot.png"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Final experiment analysis")
    parser.add_argument("--experiment", default="final_experiment")
    parser.add_argument(
        "--random-csv",
        type=Path,
        default=DATA_DIR / "random_search" / "random_search_budget6060.csv",
    )
    args = parser.parse_args()

    analyze(args.experiment, args.random_csv)


if __name__ == "__main__":
    main()
