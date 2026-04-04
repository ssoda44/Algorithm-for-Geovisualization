"""Run the three experiments for the report and save results to CSV."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

import multisource_flow as msf

OUT = Path("experiments")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_province_dataset(period: str = "2024JJ00"):
    """Province-level: 12 regions."""
    return msf.load_flow_dataset(period=period)


def load_municipality_dataset(period: str = "2024JJ00"):
    """Municipality-level: ~342 regions."""
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

    muni_flows = muni_flows.rename(columns={
        "RegioVanVertrek": "origin",
        "RegioVanVestiging": "destination",
        "Value": "flow",
    })
    muni_flows["flow"] = muni_flows["flow"].fillna(0.0)

    pair_summary = msf.summarize_bidirectional_flows(muni_flows)
    return municipalities, muni_flows, pair_summary


# ---------------------------------------------------------------------------
# Shared strategy builder
# ---------------------------------------------------------------------------

ALL_STRATEGIES = [
    "raw_directed",
    "greedy_spiral_tree",
    "enhanced_greedy_spiral_tree",
    "obstacle_aware_spiral_tree",
]


def _make_config(strategy: str, top_k: int, source_order: str = "descending_outflow", **kw) -> msf.FlowMapConfig:
    """Build a FlowMapConfig for the given strategy with sensible defaults."""
    base = dict(
        strategy=strategy,
        period="2024JJ00",
        top_k=top_k,
        min_total_flow=0.0,
        selection_mode="total_flow",
        spiral_turns=1.10,
        hub_radius_ratio=0.10,
        source_order=source_order,
    )
    if strategy == "raw_directed":
        base["min_flow"] = 0.0
    elif strategy == "greedy_spiral_tree":
        base["spiral_candidate_count"] = 6
        base["merge_reward"] = 0.60
        base["corridor_reward"] = 0.30
    elif strategy == "enhanced_greedy_spiral_tree":
        base["spiral_candidate_count"] = 6
        base["length_penalty"] = 0.25
        base["merge_reward"] = 0.80
        base["corridor_reward"] = 0.45
    # obstacle_aware uses defaults
    base.update(kw)
    return msf.FlowMapConfig(**base)


def _run_config(regions, flows, pairs, config: msf.FlowMapConfig) -> dict:
    """Run a single config and return a flat metrics dict."""
    result = msf.build_solution(regions, flows, pairs, config)
    return {
        "strategy": config.strategy,
        "top_k": config.top_k,
        "source_order": config.source_order,
        "spiral_turns": config.spiral_turns,
        **result.metrics,
    }


# ---------------------------------------------------------------------------
# Experiment 1: Source ordering
# ---------------------------------------------------------------------------

def experiment_1_source_ordering():
    """Test whether source ordering affects obstacle-aware quality."""
    print("=== Experiment 1: Source Ordering ===")
    provinces, flows, pairs = load_province_dataset()

    orderings = [
        "descending_outflow",
        "ascending_outflow",
        "west_to_east",
        "north_to_south",
    ]

    rows = []
    for order in orderings:
        config = _make_config("obstacle_aware_spiral_tree", top_k=40, source_order=order)
        row = _run_config(provinces, flows, pairs, config)
        rows.append(row)
        print(f"  {order}: crossings={row['crossings']:.0f}, "
              f"detour={row['mean_detour_ratio']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp1_source_ordering.csv", index=False)
    print(f"  Saved to {OUT / 'exp1_source_ordering.csv'}\n")
    return df


# ---------------------------------------------------------------------------
# Experiment 2: Flow density (top_k) across strategies
# ---------------------------------------------------------------------------

def experiment_2_flow_density():
    """Compare the four strategies under different top-k settings and export maps."""

    print("=== Experiment 2: Flow Density ===")
    regions, flows, pairs = load_province_dataset()
    rows = []

    # 保存每个 top_k 下四个算法的完整结果
    all_results: dict[int, dict[str, msf.FlowMapResult]] = {}

    for top_k in [20, 40, 60]:
        strategy_results = {}
        print(f"  top_k={top_k}")

        for strategy in ALL_STRATEGIES:
            config = _make_config(strategy, top_k=top_k)
            result = msf.build_solution(regions, flows, pairs, config)
            strategy_results[strategy] = result

            row = {
                "strategy": config.strategy,
                "top_k": config.top_k,
                "greedy_candidate_limit": config.greedy_candidate_limit,
                "force_iterations": config.force_iterations,
                "polyline_lane_ratio": config.polyline_lane_ratio,
                **result.metrics,
            }
            rows.append(row)

            print(
                f"    {strategy}: "
                f"coverage={row['coverage']:.3f}, crossings={row['crossings']:.0f}"
            )

        all_results[top_k] = strategy_results

        # 为每个 top_k 导出一张 2x2 比较图
        _export_topk_comparison_figure(regions, strategy_results, top_k)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp2_flow_density.csv", index=False)

    # 再导出单算法跨 top_k 的 panel 图
    for strategy in ALL_STRATEGIES:
        results_by_topk = {top_k: all_results[top_k][strategy] for top_k in all_results}
        _export_single_strategy_topk_panel(
            regions,
            results_by_topk,
            strategy,
            topks=[20, 40, 60],
        )

    return df

# ---------------------------------------------------------------------------
# Experiment 3: Geographic granularity
# ---------------------------------------------------------------------------

def experiment_3_granularity():
    """Compare strategies at province vs municipality level."""
    print("=== Experiment 3: Geographic Granularity ===")

    # Province level
    prov_regions, prov_flows, prov_pairs = load_province_dataset()
    # Municipality level
    muni_regions, muni_flows, muni_pairs = load_municipality_dataset()

    rows = []

    # Province: top_k=40
    print("  Province level (top_k=40):")
    for strategy in ALL_STRATEGIES:
        config = _make_config(strategy, top_k=40)
        row = _run_config(prov_regions, prov_flows, prov_pairs, config)
        row["level"] = "province"
        row["n_regions"] = len(prov_regions)
        rows.append(row)
        print(f"    {strategy}: crossings={row['crossings']:.0f}, "
              f"detour={row['mean_detour_ratio']:.3f}")

    # Municipality: top_k=100
    print("  Municipality level (top_k=100):")
    for strategy in ALL_STRATEGIES:
        config = _make_config(strategy, top_k=100)
        row = _run_config(muni_regions, muni_flows, muni_pairs, config)
        row["level"] = "municipality"
        row["n_regions"] = len(muni_regions)
        rows.append(row)
        print(f"    {strategy}: crossings={row['crossings']:.0f}, "
              f"detour={row['mean_detour_ratio']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp3_granularity.csv", index=False)
    print(f"  Saved to {OUT / 'exp3_granularity.csv'}\n")
    return df


# ---------------------------------------------------------------------------
# Experiment 4: Spiral turns (angle restriction α)
# ---------------------------------------------------------------------------

def experiment_4_spiral_turns():
    """Test how the angle restriction parameter affects quality across strategies."""
    print("=== Experiment 4: Spiral Turns (α) ===")
    provinces, flows, pairs = load_province_dataset()

    spiral_values = [0.50, 0.75, 1.00, 1.10, 1.25, 1.50]
    tree_strategies = [
        "greedy_spiral_tree",
        "enhanced_greedy_spiral_tree",
        "obstacle_aware_spiral_tree",
    ]

    rows = []
    for turns in spiral_values:
        for strategy in tree_strategies:
            config = _make_config(strategy, top_k=40, spiral_turns=turns)
            row = _run_config(provinces, flows, pairs, config)
            rows.append(row)
            print(f"  turns={turns:.2f}, {strategy}: crossings={row['crossings']:.0f}, "
                  f"detour={row['mean_detour_ratio']:.3f}, "
                  f"length={row['total_tree_length']:.0f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp4_spiral_turns.csv", index=False)
    print(f"  Saved to {OUT / 'exp4_spiral_turns.csv'}\n")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exp1 = experiment_1_source_ordering()
    exp2 = experiment_2_flow_density()
    exp3 = experiment_3_granularity()
    exp4 = experiment_4_spiral_turns()
    print("All experiments complete.")
