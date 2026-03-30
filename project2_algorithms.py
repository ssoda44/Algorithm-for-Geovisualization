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

    DATA = Path("data")
    IMG = Path("__marimo__")
    IMG.mkdir(exist_ok=True)
    return DATA, IMG, mo, msf, np, plt


@app.cell
def _(mo):
    mo.md(
        """
        # Multi-Source Flow Map Algorithms

        This notebook is separate from `exploration.py`.
        It focuses on algorithmic comparison for Project 2:

        - `raw_directed`: thresholded direct OD baseline
        - `greedy_spiral_tree`: simplified Greedy Spiral Tree
        - `enhanced_greedy_spiral_tree`: upgraded spiral-tree heuristic
        """
    )
    return


@app.cell
def _(msf):
    provinces, province_flows, pair_summary = msf.load_flow_dataset(period="2024JJ00")
    return pair_summary, province_flows, provinces


@app.cell
def _(mo, pair_summary, province_flows, provinces):
    mo.md(
        "## Dataset\n"
        f"- Provinces: **{len(provinces)}**\n"
        f"- Directed province flows in 2024: **{len(province_flows)}**\n"
        f"- Unordered province pairs: **{len(pair_summary)}**"
    )
    return


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
    _best = comparison.iloc[0]
    mo.md(
        "## Best Current Strategy\n"
        f"`{_best['strategy']}` currently has the best "
        f"`coverage_minus_clutter` score with "
        f"coverage={_best['coverage']:.3f} and "
        f"clutter={_best['clutter_score']:.0f}."
    )
    return


@app.cell
def _(np, plt, provinces):
    def _point_key(point):
        return (round(point.x, 6), round(point.y, 6))

    def _bundle_segments(carriers):
        bundles = {}
        terminals = []
        for _, row in carriers.iterrows():
            waypoints = row.get("waypoints")
            if not isinstance(waypoints, list) or len(waypoints) < 2:
                continue

            for start, end in zip(waypoints, waypoints[1:], strict=False):
                key = (_point_key(start), _point_key(end))
                if key not in bundles:
                    bundles[key] = {
                        "start": start,
                        "end": end,
                        "flow": 0.0,
                    }
                bundles[key]["flow"] += float(row["represented_flow"])

            terminals.append(
                {
                    "start": waypoints[-2],
                    "end": waypoints[-1],
                    "width_score": float(row["width_score"]),
                    "alpha_score": float(row["alpha_score"]),
                    "arrow_score": float(row["arrow_score"]),
                }
            )
        return bundles, terminals

    def plot_strategy_map(carriers, title, save_path):
        _fig, _ax = plt.subplots(figsize=(10, 12))
        provinces.plot(ax=_ax, edgecolor="black", facecolor="#f5f5f5", linewidth=0.8)

        _can_bundle = (
            "waypoints" in carriers.columns
            and carriers["waypoints"].apply(lambda w: isinstance(w, list) and len(w) > 2).any()
        )
        if _can_bundle:
            _bundles, _terminals = _bundle_segments(carriers)
            _max_bundle_flow = max(bundle["flow"] for bundle in _bundles.values())

            for _bundle in sorted(_bundles.values(), key=lambda item: item["flow"]):
                _norm = _bundle["flow"] / _max_bundle_flow if _max_bundle_flow else 0.0
                _width = 0.8 + 6.8 * _norm
                _alpha = 0.25 + 0.65 * _norm
                _ax.plot(
                    [_bundle["start"].x, _bundle["end"].x],
                    [_bundle["start"].y, _bundle["end"].y],
                    color="steelblue",
                    linewidth=_width,
                    alpha=_alpha,
                    solid_capstyle="round",
                )

            for _terminal in _terminals:
                _width = 0.8 + 4.0 * _terminal["width_score"]
                _alpha = max(0.45, _terminal["alpha_score"])
                _ax.annotate(
                    "",
                    xy=(_terminal["end"].x, _terminal["end"].y),
                    xytext=(_terminal["start"].x, _terminal["start"].y),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="steelblue",
                        lw=max(0.8, _width * 0.5),
                        alpha=_alpha,
                        mutation_scale=8 + 10 * _terminal["arrow_score"],
                    ),
                )
        else:
            for _, _row in carriers.iterrows():
                _coords = np.asarray(_row["geometry"].coords)
                _width = 0.8 + 5.2 * float(_row["width_score"])
                _alpha = float(_row["alpha_score"])

                _ax.plot(
                    _coords[:, 0],
                    _coords[:, 1],
                    color="steelblue",
                    linewidth=_width,
                    alpha=_alpha,
                    solid_capstyle="round",
                )

                if len(_coords) >= 2:
                    _start = _coords[-2]
                    _end = _coords[-1]
                    _ax.annotate(
                        "",
                        xy=(_end[0], _end[1]),
                        xytext=(_start[0], _start[1]),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color="steelblue",
                            lw=max(0.8, _width * 0.6),
                            alpha=_alpha,
                            mutation_scale=8 + 10 * float(_row["arrow_score"]),
                        ),
                    )

        for _, _province in provinces.iterrows():
            _anchor = _province["anchor"]
            _ax.annotate(
                _province["name"],
                xy=(_anchor.x, _anchor.y),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
            )

        _ax.set_title(title)
        _ax.set_axis_off()
        _fig.tight_layout()
        _fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return _fig

    return (plot_strategy_map,)


@app.cell
def _(IMG, plot_strategy_map, strategy_results):
    _result = strategy_results["raw_directed"]
    _fig = plot_strategy_map(
        _result.carriers,
        "Algorithm 1: thresholded direct OD lines",
        IMG / "07_algorithm1_raw_directed.png",
    )
    _fig
    return


@app.cell
def _(IMG, plot_strategy_map, strategy_results):
    _result = strategy_results["greedy_spiral_tree"]
    _fig = plot_strategy_map(
        _result.carriers,
        "Algorithm 2: Greedy Spiral Tree",
        IMG / "08_algorithm2_greedy_spiral_tree.png",
    )
    _fig
    return


@app.cell
def _(IMG, plot_strategy_map, strategy_results):
    _result = strategy_results["enhanced_greedy_spiral_tree"]
    _fig = plot_strategy_map(
        _result.carriers,
        "Algorithm 3: Enhanced Greedy Spiral Tree",
        IMG / "09_algorithm3_enhanced_spiral_tree.png",
    )
    _fig
    return


@app.cell
def _(mo, strategy_results):
    _lines = []
    for _name, _result in strategy_results.items():
        _m = _result.metrics
        _lines.append(
            f"- **{_name}**: coverage={_m['coverage']:.3f}, "
            f"crossings={_m['crossings']:.0f}, "
            f"intrusions={_m['node_intrusions']:.0f}, "
            f"mean detour={_m['mean_detour_ratio']:.3f}"
        )

    mo.md("## Metric Summary\n" + "\n".join(_lines))
    return


@app.cell
def _(msf, pair_summary, province_flows, provinces):
    sweep_configs = msf.enhanced_spiral_sweep_configs(period="2024JJ00")
    parameter_sweep = msf.compare_strategies(
        provinces,
        province_flows,
        pair_summary,
        sweep_configs,
    )
    return (parameter_sweep,)


@app.cell
def _(parameter_sweep):
    parameter_sweep.head(12)
    return


@app.cell
def _(mo, parameter_sweep):
    _best = parameter_sweep.iloc[0]
    mo.md(
        "## Parameter Sweep\n"
        f"Best enhanced spiral-tree setting in this sweep: "
        f"`top_k={int(_best['top_k'])}`, "
        f"`spiral_turns={_best['spiral_turns']:.2f}` "
        f"with coverage={_best['coverage']:.3f} and "
        f"clutter={_best['clutter_score']:.0f}."
    )
    return


if __name__ == "__main__":
    app.run()
