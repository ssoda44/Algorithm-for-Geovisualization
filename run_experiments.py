"""Run the cleaned experiment suite for the four current comparison algorithms."""

from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

import multisource_flow as msf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("experiments")
OUT.mkdir(exist_ok=True)

ALL_STRATEGIES = [
    "raw_directed",
    "force_adjusted_curved_od",
    "quality_aware_force_adjusted_od",
    "quality_aware_polyline_od",
]


def load_province_dataset(period: str = "2024JJ00"):
    return msf.load_flow_dataset(period=period)


def load_municipality_dataset(period: str = "2024JJ00"):
    """Municipality-level dataset with the same schema as the province data."""

    data_dir = Path("data")
    municipalities = gpd.read_file(data_dir / "municipalities.geojson")
    municipalities = municipalities.rename(columns={"statcode": "code", "statnaam": "name"})
    municipalities = municipalities[["code", "name", "geometry"]].copy()
    municipalities["anchor"] = municipalities.geometry.representative_point()

    obs = msf.load_observations(data_dir)
    geo_codes = set(municipalities["code"])
    muni_flows = obs[
        obs["RegioVanVestiging"].str.startswith("GM", na=False)
        & obs["RegioVanVertrek"].str.startswith("GM", na=False)
        & (obs["RegioVanVestiging"] != obs["RegioVanVertrek"])
        & obs["RegioVanVestiging"].isin(geo_codes)
        & obs["RegioVanVertrek"].isin(geo_codes)
        & (obs["Perioden"] == period)
    ][["RegioVanVertrek", "RegioVanVestiging", "Value"]].copy()

    muni_flows = muni_flows.rename(
        columns={
            "RegioVanVertrek": "origin",
            "RegioVanVestiging": "destination",
            "Value": "flow",
        }
    )
    muni_flows["flow"] = muni_flows["flow"].fillna(0.0)
    pair_summary = msf.summarize_bidirectional_flows(muni_flows)
    return municipalities, muni_flows, pair_summary


def _make_config(
    strategy: str,
    top_k: int,
    force_iterations: int = 5,
    greedy_candidate_limit: int = 60,
    polyline_lane_ratio: float = 0.010,
) -> msf.FlowMapConfig:
    base = dict(
        strategy=strategy,
        period="2024JJ00",
        top_k=top_k,
        min_flow=0.0,
        greedy_candidate_limit=greedy_candidate_limit,
        coverage_weight=1.0,
        crossing_weight=0.008,
        intrusion_weight=0.004,
        detour_weight=0.20,
        curvature_scale=0.10,
        force_iterations=force_iterations,
        selection_force_iterations=0,
        force_step_scale=0.010,
        force_proximity_scale=0.028,
        force_base_pull=0.18,
        polyline_lane_ratio=polyline_lane_ratio,
        polyline_proximity_ratio=0.020,
    )
    return msf.FlowMapConfig(**base)


def _run_config(regions, flows, pairs, config: msf.FlowMapConfig) -> dict:
    result = msf.build_solution(regions, flows, pairs, config)
    return {
        "strategy": config.strategy,
        "top_k": config.top_k,
        "greedy_candidate_limit": config.greedy_candidate_limit,
        "force_iterations": config.force_iterations,
        "polyline_lane_ratio": config.polyline_lane_ratio,
        **result.metrics,
    }


def _plot_strategy_map(ax, regions, carriers, title: str) -> None:
    regions.plot(ax=ax, edgecolor="black", facecolor="#f5f5f5", linewidth=0.8)
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

    for _, region in regions.iterrows():
        anchor = region["anchor"]
        ax.annotate(
            region["name"],
            xy=(anchor.x, anchor.y),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
        )

    ax.set_title(title)
    ax.set_axis_off()


def _export_strategy_figures(regions, strategy_results: dict[str, msf.FlowMapResult]) -> None:
    titles = {
        "raw_directed": "Algorithm 1: Raw Directed",
        "force_adjusted_curved_od": "Algorithm 2: Force-Adjusted Curved OD",
        "quality_aware_force_adjusted_od": "Algorithm 3: Quality-Aware Greedy OD",
        "quality_aware_polyline_od": "Algorithm 4: Quality-Aware Polyline OD",
    }

    file_names = {
        "raw_directed": "exp1_algorithm1_raw_directed.png",
        "force_adjusted_curved_od": "exp1_algorithm2_force_adjusted_curved_od.png",
        "quality_aware_force_adjusted_od": "exp1_algorithm3_quality_aware_force_adjusted_od.png",
        "quality_aware_polyline_od": "exp1_algorithm4_quality_aware_polyline_od.png",
    }

    for key, title in titles.items():
        result = strategy_results[key]
        metrics = result.metrics
        fig, ax = plt.subplots(figsize=(10, 12))
        _plot_strategy_map(
            ax,
            regions,
            result.carriers,
            f"{title}\n"
            f"cov={metrics['coverage']:.3f}, "
            f"cross={metrics['crossings']:.0f}, "
            f"detour={metrics['mean_detour_ratio']:.3f}",
        )
        fig.tight_layout()
        fig.savefig(OUT / file_names[key], dpi=150, bbox_inches="tight")
        plt.close(fig)

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
        _plot_strategy_map(
            ax,
            regions,
            strategy_results[key].carriers,
            f"{title}\n"
            f"cov={metrics['coverage']:.3f}, "
            f"cross={metrics['crossings']:.0f}, "
            f"detour={metrics['mean_detour_ratio']:.3f}",
        )
    fig.tight_layout()
    fig.savefig(OUT / "exp1_four_algorithms_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def experiment_1_strategy_comparison():
    """Main province-level comparison used in the report."""

    print("=== Experiment 1: Strategy Comparison ===")
    regions, flows, pairs = load_province_dataset()
    configs = msf.default_strategy_configs(period="2024JJ00")
    comparison, strategy_results = msf.build_strategy_suite(regions, flows, pairs, configs)
    comparison.to_csv(OUT / "exp1_strategy_comparison.csv", index=False)
    _export_strategy_figures(regions, strategy_results)
    return comparison


def experiment_2_flow_density():
    """Compare the four strategies under different top-k settings."""

    print("=== Experiment 2: Flow Density ===")
    regions, flows, pairs = load_province_dataset()
    rows = []

    for top_k in [10, 20, 30, 40, 50, 60]:
        for strategy in ALL_STRATEGIES:
            row = _run_config(regions, flows, pairs, _make_config(strategy, top_k=top_k))
            rows.append(row)
            print(
                f"  top_k={top_k}, {strategy}: "
                f"coverage={row['coverage']:.3f}, crossings={row['crossings']:.0f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp2_flow_density.csv", index=False)
    return df


def experiment_3_parameter_sensitivity():
    """Collect the three parameter sweeps into one compact table."""

    print("=== Experiment 3: Parameter Sensitivity ===")
    regions, flows, pairs = load_province_dataset()
    rows = []

    for config in msf.force_adjusted_sweep_configs(period="2024JJ00", top_k=40):
        row = _run_config(regions, flows, pairs, config)
        row["parameter_family"] = "force_adjusted_curved_od"
        row["parameter_name"] = "force_iterations"
        row["parameter_value"] = float(config.force_iterations)
        rows.append(row)
        print(
            f"  curved, iterations={config.force_iterations}: "
            f"crossings={row['crossings']:.0f}, detour={row['mean_detour_ratio']:.3f}"
        )

    for config in msf.quality_aware_sweep_configs(period="2024JJ00"):
        row = _run_config(regions, flows, pairs, config)
        row["parameter_family"] = "quality_aware_force_adjusted_od"
        row["parameter_name"] = "greedy_candidate_limit"
        row["parameter_value"] = float(config.greedy_candidate_limit or 0)
        rows.append(row)

    for config in msf.polyline_sweep_configs(period="2024JJ00", top_k=40):
        row = _run_config(regions, flows, pairs, config)
        row["parameter_family"] = "quality_aware_polyline_od"
        row["parameter_name"] = "polyline_lane_ratio"
        row["parameter_value"] = float(config.polyline_lane_ratio)
        rows.append(row)
        print(
            f"  polyline, lane_ratio={config.polyline_lane_ratio:.3f}: "
            f"crossings={row['crossings']:.0f}, detour={row['mean_detour_ratio']:.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp3_parameter_sensitivity.csv", index=False)
    return df


def experiment_4_granularity():
    """Compare the four strategies across two geographic granularities."""

    print("=== Experiment 4: Geographic Granularity ===")
    prov_regions, prov_flows, prov_pairs = load_province_dataset()
    muni_regions, muni_flows, muni_pairs = load_municipality_dataset()

    configs = [_make_config(strategy, top_k=40) for strategy in ALL_STRATEGIES]

    rows = []
    for level, data in [
        ("province", (prov_regions, prov_flows, prov_pairs)),
        ("municipality", (muni_regions, muni_flows, muni_pairs)),
    ]:
        regions, flows, pairs = data
        print(f"  {level}:")
        for config in configs:
            row = _run_config(regions, flows, pairs, config)
            row["level"] = level
            row["n_regions"] = len(regions)
            rows.append(row)
            print(
                f"    {config.strategy}: "
                f"coverage={row['coverage']:.3f}, crossings={row['crossings']:.0f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp4_granularity.csv", index=False)
    return df


if __name__ == "__main__":
    experiment_1_strategy_comparison()
    experiment_2_flow_density()
    experiment_3_parameter_sensitivity()
    experiment_4_granularity()
    print("All experiments complete.")
