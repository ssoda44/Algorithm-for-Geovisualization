"""Run experiments for the enhanced greedy spiral tree and save results to CSV."""

from pathlib import Path
import multisource_flow as msf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
OUT = Path("experiments")
OUT.mkdir(exist_ok=True)
import pandas as pd
import geopandas as gpd

import multisource_flow as msf

OUT = Path("experiments_enhanced")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data loaders (shared with main experiments)
# ---------------------------------------------------------------------------

def load_province_dataset(period: str = "2024JJ00"):
    return msf.load_flow_dataset(period=period)


def load_municipality_dataset(period: str = "2024JJ00"):
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


ALL_STRATEGIES = [
    "raw_directed",
    "greedy_spiral_tree",
    "enhanced_greedy_spiral_tree",
]


def _make_config(strategy: str, top_k: int, spiral_turns: float = 1.10, **kw) -> msf.FlowMapConfig:
    base = dict(
        strategy=strategy,
        period="2024JJ00",
        top_k=top_k,
        min_total_flow=0.0,
        selection_mode="total_flow",
        spiral_turns=spiral_turns,
        hub_radius_ratio=0.10,
        source_order="descending_outflow",
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
    base.update(kw)
    return msf.FlowMapConfig(**base)


def _run_config(regions, flows, pairs, config: msf.FlowMapConfig) -> dict:
    result = msf.build_solution(regions, flows, pairs, config)
    return {
        "strategy": config.strategy,
        "top_k": config.top_k,
        "spiral_turns": config.spiral_turns,
        "source_order": config.source_order,
        **result.metrics,
    }


# ---------------------------------------------------------------------------
# Experiment 2: Flow density (top_k) — enhanced tree focus
# ---------------------------------------------------------------------------

def experiment_2_flow_density():
    """Test how crossing count scales with top_k for enhanced vs baselines."""
    print("=== Experiment 2 (Enhanced): Flow Density (top_k) ===")
    provinces, flows, pairs = load_province_dataset()

    top_ks = [20,40, 60]

    rows = []
    for top_k in top_ks:
        for strategy in ALL_STRATEGIES:
            config = _make_config(strategy, top_k=top_k)
            row = _run_config(provinces, flows, pairs, config)
            rows.append(row)
            print(f"  top_k={top_k}, {strategy}: crossings={row['crossings']:.0f}, "
                  f"detour={row['mean_detour_ratio']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp2_flow_density.csv", index=False)
    print(f"  Saved to {OUT / 'exp2_flow_density.csv'}\n")
    return df


# ---------------------------------------------------------------------------
# Experiment 3: Geographic granularity — enhanced tree focus
# ---------------------------------------------------------------------------

def experiment_3_granularity():
    """Compare enhanced tree vs baselines at province and municipality level."""
    print("=== Experiment 3 (Enhanced): Geographic Granularity ===")

    prov_regions, prov_flows, prov_pairs = load_province_dataset()
    muni_regions, muni_flows, muni_pairs = load_municipality_dataset()

    rows = []

    print("  Province level (top_k=40):")
    for strategy in ALL_STRATEGIES:
        config = _make_config(strategy, top_k=40)
        row = _run_config(prov_regions, prov_flows, prov_pairs, config)
        row["level"] = "province"
        row["n_regions"] = len(prov_regions)
        rows.append(row)
        print(f"    {strategy}: crossings={row['crossings']:.0f}, "
              f"detour={row['mean_detour_ratio']:.3f}")

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
    """Test how the angle restriction parameter affects quality for enhanced vs baselines."""
    print("=== Experiment 4 (Enhanced): Spiral Turns (α) ===")
    provinces, flows, pairs = load_province_dataset()

    spiral_values = [0.50, 0.75, 1.00, 1.10, 1.25, 1.50]
    tree_strategies = [
        "greedy_spiral_tree",
        "enhanced_greedy_spiral_tree",
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
    exp2 = experiment_2_flow_density()
    exp3 = experiment_3_granularity()
    exp4 = experiment_4_spiral_turns()
    print("All experiments complete.")
