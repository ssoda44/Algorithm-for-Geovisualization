# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pandas",
#     "geopandas",
#     "matplotlib",
#     "numpy",
#     "shapely",
#     "pyproj",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    import multisource_flow as msf

    IMG = Path("__marimo__")
    IMG.mkdir(exist_ok=True)
    return IMG, mo, msf, np, plt


@app.cell
def _(mo):
    mo.md(
        """
        # Multi-Source Flow Map Algorithms

        This notebook compares the four current strategies:

        - `raw_directed`
        - `force_adjusted_curved_od`
        - `quality_aware_force_adjusted_od`
        - `quality_aware_polyline_od`
        """
    )
    return


@app.cell
def _(msf):
    provinces, province_flows, pair_summary = msf.load_flow_dataset(period="2024JJ00")
    return pair_summary, province_flows, provinces


@app.cell
def _(msf):
    strategy_configs = msf.default_strategy_configs(period="2024JJ00")
    return (strategy_configs,)


@app.cell
def _(msf, pair_summary, province_flows, provinces, strategy_configs):
    comparison, strategy_results = msf.build_strategy_suite(
        provinces,
        province_flows,
        pair_summary,
        strategy_configs,
    )
    return comparison, strategy_results


@app.cell
def _(comparison):
    comparison
    return


@app.cell
def _(mo, comparison):
    best = comparison.iloc[0]
    mo.md(
        "## Best Current Trade-off\n"
        f"`{best['strategy']}` currently gives "
        f"coverage={best['coverage']:.3f}, crossings={best['crossings']:.0f}, "
        f"and clutter={best['clutter_score']:.0f}."
    )
    return


@app.cell
def _(np, plt, provinces):
    def plot_strategy_map(ax, carriers, title):
        provinces.plot(ax=ax, edgecolor="black", facecolor="#f5f5f5", linewidth=0.8)
        for _, row in carriers.iterrows():
            coords = np.asarray(row["geometry"].coords)
            width = 0.8 + 5.2 * float(row["width_score"])
            alpha = float(row["alpha_score"])
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                color="steelblue",
                linewidth=width,
                alpha=alpha,
                solid_capstyle="round",
            )
            if len(coords) >= 2:
                start = coords[-2]
                end = coords[-1]
                ax.annotate(
                    "",
                    xy=(end[0], end[1]),
                    xytext=(start[0], start[1]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="steelblue",
                        lw=max(0.8, width * 0.6),
                        alpha=alpha,
                        mutation_scale=8 + 10 * float(row["arrow_score"]),
                    ),
                )

        for _, province in provinces.iterrows():
            anchor = province["anchor"]
            ax.annotate(
                province["name"],
                xy=(anchor.x, anchor.y),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
            )

        ax.set_title(title)
        ax.set_axis_off()

    return (plot_strategy_map,)


@app.cell
def _(IMG, plot_strategy_map, plt, strategy_results):
    titles = {
        "raw_directed": "Algorithm 1: Raw Directed",
        "force_adjusted_curved_od": "Algorithm 2: Force-Adjusted Curved OD",
        "quality_aware_force_adjusted_od": "Algorithm 3: Quality-Aware Greedy OD",
        "quality_aware_polyline_od": "Algorithm 4: Quality-Aware Polyline OD",
    }

    for key, title in titles.items():
        result = strategy_results[key]
        metrics = result.metrics
        fig, ax = plt.subplots(figsize=(10, 12))
        plot_strategy_map(
            ax,
            result.carriers,
            f"{title}\n"
            f"cov={metrics['coverage']:.3f}, "
            f"cross={metrics['crossings']:.0f}, "
            f"detour={metrics['mean_detour_ratio']:.3f}",
        )
        fig.tight_layout()
        fig.savefig(IMG / f"{key}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    return


@app.cell
def _(IMG, plot_strategy_map, plt, strategy_results):
    order = [
        ("raw_directed", "Algorithm 1: Raw Directed"),
        ("force_adjusted_curved_od", "Algorithm 2: Force-Adjusted Curved OD"),
        ("quality_aware_force_adjusted_od", "Algorithm 3: Quality-Aware Greedy OD"),
        ("quality_aware_polyline_od", "Algorithm 4: Quality-Aware Polyline OD"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.ravel()
    for ax, (key, title) in zip(axes, order, strict=False):
        metrics = strategy_results[key].metrics
        plot_strategy_map(
            ax,
            strategy_results[key].carriers,
            f"{title}\n"
            f"cov={metrics['coverage']:.3f}, "
            f"cross={metrics['crossings']:.0f}, "
            f"detour={metrics['mean_detour_ratio']:.3f}",
        )
    fig.tight_layout()
    fig.savefig(IMG / "four_algorithms_comparison.png", dpi=150, bbox_inches="tight")
    fig
    return


@app.cell
def _(msf, pair_summary, province_flows, provinces):
    force_sweep = msf.compare_strategies(
        provinces,
        province_flows,
        pair_summary,
        msf.force_adjusted_sweep_configs(period="2024JJ00", top_k=40),
    )
    return (force_sweep,)


@app.cell
def _(force_sweep):
    force_sweep
    return


@app.cell
def _(msf, pair_summary, province_flows, provinces):
    quality_sweep = msf.compare_strategies(
        provinces,
        province_flows,
        pair_summary,
        msf.quality_aware_sweep_configs(period="2024JJ00"),
    )
    return (quality_sweep,)


@app.cell
def _(quality_sweep):
    quality_sweep
    return


@app.cell
def _(msf, pair_summary, province_flows, provinces):
    polyline_sweep = msf.compare_strategies(
        provinces,
        province_flows,
        pair_summary,
        msf.polyline_sweep_configs(period="2024JJ00", top_k=40),
    )
    return (polyline_sweep,)


@app.cell
def _(polyline_sweep):
    polyline_sweep
    return


if __name__ == "__main__":
    app.run()
