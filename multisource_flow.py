"""Core utilities for the 2IMA20 multi-source flow-map project.

This module keeps the modeling and algorithmic parts separate from the
exploratory notebook in ``exploration.py``. The current focus is a province-
level prototype for Dutch migration flows:

- load and filter the CBS migration data
- aggregate bidirectional flows into unordered area pairs
- select salient pairs for display
- route selected pairs with baseline or spiral-tree carriers
- compute a few simple quality measures for experiments/reporting
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


@dataclass(frozen=True)
class FlowMapConfig:
    """Parameters for the summarized multi-source flow map."""

    strategy: str = "enhanced_greedy_spiral_tree"
    period: str = "2024JJ00"
    top_k: int | None = 40
    min_flow: float = 0.0
    min_total_flow: float = 0.0
    selection_mode: str = "total_flow"
    hybrid_alpha: float = 0.75
    length_penalty: float = 0.25
    samples_per_curve: int = 25
    node_buffer_ratio: float = 0.018
    spiral_turns: float = 1.10
    spiral_candidate_count: int = 6
    hub_radius_ratio: float = 0.10
    merge_reward: float = 0.60
    corridor_reward: float = 0.30
    attachment_samples: tuple[float, ...] = (0.35, 0.50, 0.65)
    source_order: str = "descending_outflow"


@dataclass
class FlowMapResult:
    """Bundle the output of one strategy run."""

    strategy: str
    selected: pd.DataFrame
    carriers: pd.DataFrame
    metrics: dict[str, float]


@dataclass
class SpiralTreeNode:
    """One node in a source-rooted spiral tree."""

    node_id: int
    point: Point
    radius: float
    angle: float
    angle_min: float
    angle_max: float
    flow_sum: float
    leaf_ids: list[int]
    parent_id: int | None = None


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_provinces(data_dir: str | Path = "data") -> gpd.GeoDataFrame:
    """Load Dutch provinces with a compact schema."""

    data_dir = Path(data_dir)
    provinces = gpd.read_file(data_dir / "provinces.geojson")
    provinces = provinces.rename(columns={"statcode": "code", "statnaam": "name"})
    provinces = provinces[["code", "name", "geometry"]].copy()
    provinces["anchor"] = provinces.geometry.representative_point()
    return provinces


def load_observations(data_dir: str | Path = "data") -> pd.DataFrame:
    """Load the CBS observations table."""

    data_dir = Path(data_dir)
    obs = pd.read_csv(data_dir / "Observations.csv", sep=";")
    obs["Value"] = pd.to_numeric(obs["Value"], errors="coerce")
    return obs


def build_province_flows(
    obs: pd.DataFrame,
    period: str = "2024JJ00",
) -> pd.DataFrame:
    """Return directed province-to-province flows for a single period."""

    province_flows = obs[
        obs["RegioVanVestiging"].str.startswith("PV")
        & obs["RegioVanVertrek"].str.startswith("PV")
        & (obs["RegioVanVestiging"] != obs["RegioVanVertrek"])
        & (obs["Perioden"] == period)
    ][["RegioVanVertrek", "RegioVanVestiging", "Value"]].copy()

    province_flows = province_flows.rename(
        columns={
            "RegioVanVertrek": "origin",
            "RegioVanVestiging": "destination",
            "Value": "flow",
        }
    )
    province_flows["flow"] = province_flows["flow"].fillna(0.0)
    return province_flows


def summarize_bidirectional_flows(province_flows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate directed flows into one record per unordered province pair."""

    grouped: dict[tuple[str, str], dict[str, float | str]] = {}

    for row in province_flows.itertuples(index=False):
        a, b = sorted((row.origin, row.destination))
        key = (a, b)
        if key not in grouped:
            grouped[key] = {
                "a": a,
                "b": b,
                "flow_ab": 0.0,
                "flow_ba": 0.0,
            }

        if row.origin == a:
            grouped[key]["flow_ab"] += float(row.flow)
        else:
            grouped[key]["flow_ba"] += float(row.flow)

    summary = pd.DataFrame(grouped.values())
    summary["total_flow"] = summary["flow_ab"] + summary["flow_ba"]
    summary["net_flow"] = (summary["flow_ab"] - summary["flow_ba"]).abs()
    summary["imbalance_ratio"] = np.where(
        summary["total_flow"] > 0,
        summary["net_flow"] / summary["total_flow"],
        0.0,
    )
    summary["dominant_origin"] = np.where(
        summary["flow_ab"] >= summary["flow_ba"],
        summary["a"],
        summary["b"],
    )
    summary["dominant_destination"] = np.where(
        summary["flow_ab"] >= summary["flow_ba"],
        summary["b"],
        summary["a"],
    )
    summary = summary.sort_values(
        ["total_flow", "net_flow", "a", "b"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return summary


def select_salient_directed_flows(
    province_flows: pd.DataFrame,
    top_k: int | None = None,
    min_flow: float = 0.0,
) -> pd.DataFrame:
    """Keep the most salient directed flows for the raw baseline."""

    selected = province_flows[province_flows["flow"] >= float(min_flow)].copy()
    if top_k is not None:
        selected = selected.nlargest(top_k, "flow").copy()

    selected["represented_flow"] = selected["flow"]
    selected["net_flow"] = selected["flow"]
    selected["imbalance_ratio"] = 1.0
    return selected.reset_index(drop=True)


def select_salient_pairs(
    pair_summary: pd.DataFrame,
    top_k: int | None = None,
    min_total_flow: float = 0.0,
    selection_mode: str = "total_flow",
    hybrid_alpha: float = 0.75,
) -> pd.DataFrame:
    """Keep only the pairs that are salient enough to visualize."""

    selected = pair_summary[pair_summary["total_flow"] >= float(min_total_flow)].copy()
    if selected.empty:
        return selected

    if selection_mode == "total_flow":
        selected["selection_score"] = selected["total_flow"]
    elif selection_mode == "net_flow":
        selected["selection_score"] = selected["net_flow"]
    elif selection_mode == "imbalance_ratio":
        selected["selection_score"] = selected["imbalance_ratio"]
    elif selection_mode == "hybrid":
        total_scale = selected["total_flow"].max()
        net_scale = selected["net_flow"].max()
        norm_total = selected["total_flow"] / total_scale if total_scale else 0.0
        norm_net = selected["net_flow"] / net_scale if net_scale else 0.0
        selected["selection_score"] = hybrid_alpha * norm_total + (1.0 - hybrid_alpha) * norm_net
    else:
        raise ValueError(
            "selection_mode must be one of "
            "'total_flow', 'net_flow', 'imbalance_ratio', or 'hybrid'"
        )

    if top_k is not None:
        selected = selected.nlargest(top_k, "selection_score").copy()

    selected["represented_flow"] = selected["total_flow"]
    return selected.reset_index(drop=True)


def load_flow_dataset(
    data_dir: str | Path = "data",
    period: str = "2024JJ00",
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Load provinces and build the derived province-flow tables for one period."""

    provinces = load_provinces(data_dir)
    obs = load_observations(data_dir)
    province_flows = build_province_flows(obs, period=period)
    pair_summary = summarize_bidirectional_flows(province_flows)
    return provinces, province_flows, pair_summary


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _map_center(provinces: gpd.GeoDataFrame) -> tuple[float, float]:
    minx, miny, maxx, maxy = provinces.total_bounds
    return (minx + maxx) / 2, (miny + maxy) / 2


def _map_scale(provinces: gpd.GeoDataFrame) -> float:
    minx, miny, maxx, maxy = provinces.total_bounds
    return max(maxx - minx, maxy - miny)


def _province_lookup(provinces: gpd.GeoDataFrame) -> tuple[pd.Series, pd.Series]:
    indexed = provinces.set_index("code")
    return indexed["anchor"], indexed["name"]


def _quadratic_bezier_points(
    start: Point,
    control: Point,
    end: Point,
    num_points: int,
) -> list[tuple[float, float]]:
    """Sample a quadratic Bezier curve."""

    ts = np.linspace(0.0, 1.0, num_points)
    points: list[tuple[float, float]] = []
    for t in ts:
        omt = 1.0 - t
        x = omt * omt * start.x + 2 * omt * t * control.x + t * t * end.x
        y = omt * omt * start.y + 2 * omt * t * control.y + t * t * end.y
        points.append((x, y))
    return points


def _curve_side(
    start: Point,
    end: Point,
    map_center: tuple[float, float],
) -> float:
    """Choose a deterministic side so curves bend away from the map center."""

    dx = end.x - start.x
    dy = end.y - start.y
    nx, ny = -dy, dx
    mx = (start.x + end.x) / 2
    my = (start.y + end.y) / 2
    cx, cy = map_center
    center_vec_x = mx - cx
    center_vec_y = my - cy
    dot = nx * center_vec_x + ny * center_vec_y
    return 1.0 if dot >= 0 else -1.0


def _control_point(
    start: Point,
    end: Point,
    map_center: tuple[float, float],
    curvature_scale: float,
    side: float | None = None,
) -> Point:
    """Construct one control point for a quadratic Bezier carrier."""

    dx = end.x - start.x
    dy = end.y - start.y
    length = float(np.hypot(dx, dy))
    if length == 0:
        return Point(start.x, start.y)

    nx, ny = -dy / length, dx / length
    mx = (start.x + end.x) / 2
    my = (start.y + end.y) / 2
    if side is None:
        side = _curve_side(start, end, map_center)
    bend = side * length * curvature_scale
    return Point(mx + bend * nx, my + bend * ny)


def _straight_line(start: Point, end: Point) -> LineString:
    """Create a straight carrier."""

    return LineString([(start.x, start.y), (end.x, end.y)])


def _curve_geometry(
    start: Point,
    end: Point,
    map_center: tuple[float, float],
    curvature_scale: float,
    samples_per_curve: int,
    side: float | None = None,
) -> tuple[LineString, Point]:
    """Create a curved carrier and return its control point."""

    control = _control_point(
        start,
        end,
        map_center,
        curvature_scale=curvature_scale,
        side=side,
    )
    curve = LineString(
        _quadratic_bezier_points(start, control, end, samples_per_curve)
    )
    return curve, control


def _finalize_carriers(carrier_df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized display scores shared by all strategies."""

    if carrier_df.empty:
        return carrier_df

    carrier_df = carrier_df.copy()
    width_scale = carrier_df["represented_flow"].max()
    carrier_df["width_score"] = (
        carrier_df["represented_flow"] / width_scale if width_scale else 0.0
    )
    carrier_df["arrow_score"] = carrier_df["imbalance_ratio"]
    carrier_df["alpha_score"] = 0.35 + 0.65 * carrier_df["width_score"]
    carrier_df["straight_distance"] = carrier_df.apply(
        lambda row: row.start.distance(row.end),
        axis=1,
    )
    carrier_df["geometry_length"] = carrier_df["geometry"].apply(lambda geom: geom.length)
    carrier_df["detour_ratio"] = np.where(
        carrier_df["straight_distance"] > 0,
        carrier_df["geometry_length"] / carrier_df["straight_distance"],
        1.0,
    )
    return carrier_df


def _pair_carrier_record(
    row: object,
    names: pd.Series,
    start: Point,
    end: Point,
    geometry: LineString,
    control: Point | None = None,
    waypoints: list[Point] | None = None,
) -> dict[str, object]:
    """Build the common carrier payload for pair-based strategies."""

    return {
        "a": row.a,
        "b": row.b,
        "source": row.dominant_origin,
        "target": row.dominant_destination,
        "source_name": names[row.dominant_origin],
        "target_name": names[row.dominant_destination],
        "flow_ab": row.flow_ab,
        "flow_ba": row.flow_ba,
        "represented_flow": row.represented_flow,
        "total_flow": row.total_flow,
        "net_flow": row.net_flow,
        "imbalance_ratio": row.imbalance_ratio,
        "start": start,
        "control": control,
        "end": end,
        "geometry": geometry,
        "waypoints": waypoints,
    }


def _ordered_pairs_for_routing(selected_pairs: pd.DataFrame) -> pd.DataFrame:
    """Route high-volume pairs first so later pairs can reuse their corridors."""

    return selected_pairs.sort_values(
        ["represented_flow", "net_flow"],
        ascending=[False, False],
    )


# ---------------------------------------------------------------------------
# Carrier construction
# ---------------------------------------------------------------------------

def build_directed_carriers(
    selected_flows: pd.DataFrame,
    provinces: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Build straight carriers for the raw directed baseline."""

    anchors, names = _province_lookup(provinces)

    carriers: list[dict[str, object]] = []
    for row in selected_flows.itertuples(index=False):
        start = anchors[row.origin]
        end = anchors[row.destination]
        carriers.append(
            {
                "a": row.origin,
                "b": row.destination,
                "source": row.origin,
                "target": row.destination,
                "source_name": names[row.origin],
                "target_name": names[row.destination],
                "flow_ab": row.flow,
                "flow_ba": 0.0,
                "represented_flow": row.represented_flow,
                "total_flow": row.flow,
                "net_flow": row.net_flow,
                "imbalance_ratio": row.imbalance_ratio,
                "start": start,
                "control": None,
                "end": end,
                "geometry": _straight_line(start, end),
                "waypoints": [start, end],
            }
        )

    return _finalize_carriers(pd.DataFrame(carriers))


def _tree_context(
    provinces: gpd.GeoDataFrame,
    node_buffer_ratio: float,
    hub_radius_ratio: float,
) -> tuple[pd.Series, pd.Series, tuple[float, float], float, float, float]:
    """Collect province lookups and map-scale parameters used by tree routing."""

    anchors, names = _province_lookup(provinces)
    scale = _map_scale(provinces)
    return (
        anchors,
        names,
        _map_center(provinces),
        scale,
        scale * node_buffer_ratio,
        scale * hub_radius_ratio,
    )


# ---------------------------------------------------------------------------
# Spiral-tree routing
# ---------------------------------------------------------------------------

def _source_angle(source: Point, point: Point) -> float:
    """Polar angle of one point around a source/root."""

    return float(np.arctan2(point.y - source.y, point.x - source.x))


def _ordered_targets_for_source(
    rows: list[object],
    source: Point,
    anchors: pd.Series,
) -> list[tuple[int, object, Point, float, float]]:
    """Order targets by radial order, cutting at the largest angular gap."""

    terminals = [
        (
            idx,
            row,
            anchors[row.dominant_destination],
            _source_angle(source, anchors[row.dominant_destination]),
            float(source.distance(anchors[row.dominant_destination])),
        )
        for idx, row in enumerate(rows)
    ]
    terminals.sort(key=lambda item: item[3])
    if len(terminals) <= 1:
        angle = terminals[0][3] if terminals else 0.0
        return [
            (idx, row, target, angle, radius)
            for idx, row, target, angle, radius in terminals
        ]

    cyclic_angles = [item[3] for item in terminals]
    gaps = []
    for i, angle in enumerate(cyclic_angles):
        nxt = cyclic_angles[(i + 1) % len(cyclic_angles)]
        gap = (nxt - angle) % (2.0 * np.pi)
        gaps.append(gap)
    cut = int(np.argmax(gaps))
    ordered = terminals[cut + 1 :] + terminals[: cut + 1]

    unwrapped: list[tuple[int, object, Point, float, float]] = []
    prev_angle: float | None = None
    for idx, row, target, angle, radius in ordered:
        current = angle
        if prev_angle is not None:
            while current <= prev_angle:
                current += 2.0 * np.pi
        unwrapped.append((idx, row, target, current, radius))
        prev_angle = current
    return unwrapped


def _leaf_node(
    node_id: int,
    leaf_id: int,
    point: Point,
    angle: float,
    radius: float,
    flow_sum: float,
) -> SpiralTreeNode:
    """Create a leaf node for one target terminal."""

    return SpiralTreeNode(
        node_id=node_id,
        point=point,
        radius=radius,
        angle=angle,
        angle_min=angle,
        angle_max=angle,
        flow_sum=flow_sum,
        leaf_ids=[leaf_id],
    )


def _merge_point_candidate(
    source: Point,
    left: SpiralTreeNode,
    right: SpiralTreeNode,
    shrink_factor: float,
    spiral_turns: float,
    angle_offset_factor: float = 0.0,
) -> tuple[Point, float, float]:
    """Create one candidate join point between two adjacent tree nodes."""

    span = right.angle_min - left.angle_max
    base_angle = (left.angle * left.flow_sum + right.angle * right.flow_sum) / (
        left.flow_sum + right.flow_sum
    )
    spiral_bias = 0.12 * max(0.0, spiral_turns - 1.0)
    angle = base_angle + (angle_offset_factor + spiral_bias) * span
    radius = max(0.25, shrink_factor * min(left.radius, right.radius) / max(0.8, spiral_turns))
    point = Point(
        source.x + radius * float(np.cos(angle)),
        source.y + radius * float(np.sin(angle)),
    )
    return point, angle, radius


def _source_group_intrusions(
    source: Point,
    candidate_point: Point,
    child_points: list[Point],
    endpoint_codes: set[str],
    anchors: pd.Series,
    radius: float,
) -> float:
    """Count local intrusions for the segments introduced by one merge."""

    local_segments = [
        LineString([(source.x, source.y), (candidate_point.x, candidate_point.y)]),
        *[
            LineString([(candidate_point.x, candidate_point.y), (child.x, child.y)])
            for child in child_points
        ],
    ]
    intrusions = 0.0
    for code, anchor in anchors.items():
        if code in endpoint_codes:
            continue
        buffered = anchor.buffer(radius)
        if any(segment.intersects(buffered) for segment in local_segments):
            intrusions += 1.0
    return intrusions


def _edge_key(p1: Point, p2: Point) -> tuple[tuple[float, float], tuple[float, float]]:
    """Canonical key for an edge between two points, for deduplication."""
    a = (round(p1.x, 6), round(p1.y, 6))
    b = (round(p2.x, 6), round(p2.y, 6))
    return (a, b) if a <= b else (b, a)


def _extract_tree_edges(
    source_paths: list[tuple[object, list[Point]]],
) -> list[LineString]:
    """Extract deduplicated edge segments from one source's tree paths."""
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    edges: list[LineString] = []
    for _, waypoints in source_paths:
        for start, end in zip(waypoints, waypoints[1:], strict=False):
            key = _edge_key(start, end)
            if key not in seen:
                seen.add(key)
                edges.append(LineString([(start.x, start.y), (end.x, end.y)]))
    return edges


def _obstacle_crossings(
    candidate_point: Point,
    child_points: list[Point],
    source: Point,
    obstacle_segments: list[LineString],
) -> int:
    """Count crossings between candidate merge segments and existing obstacles."""
    if not obstacle_segments:
        return 0

    local_segments = [
        LineString([(source.x, source.y), (candidate_point.x, candidate_point.y)]),
        *[
            LineString([(candidate_point.x, candidate_point.y), (child.x, child.y)])
            for child in child_points
        ],
    ]
    crossings = 0
    for seg in local_segments:
        for obs in obstacle_segments:
            if seg.crosses(obs):
                crossings += 1
    return crossings


def _best_merge_node(
    source: Point,
    left: SpiralTreeNode,
    right: SpiralTreeNode,
    *,
    next_node_id: int,
    spiral_turns: float,
    endpoint_codes: set[str],
    anchors: pd.Series,
    radius: float,
    enhanced: bool,
) -> SpiralTreeNode:
    """Choose a merge node for two adjacent subtrees."""

    if not enhanced:
        point, angle, merge_radius = _merge_point_candidate(
            source,
            left,
            right,
            shrink_factor=0.72,
            spiral_turns=spiral_turns,
        )
        return SpiralTreeNode(
            node_id=next_node_id,
            point=point,
            radius=merge_radius,
            angle=angle,
            angle_min=left.angle_min,
            angle_max=right.angle_max,
            flow_sum=left.flow_sum + right.flow_sum,
            leaf_ids=[*left.leaf_ids, *right.leaf_ids],
        )

    candidates: list[tuple[float, Point, float, float]] = []
    for shrink_factor in (0.58, 0.68, 0.78):
        for angle_offset_factor in (-0.20, 0.0, 0.20):
            point, angle, merge_radius = _merge_point_candidate(
                source, left, right,
                shrink_factor=shrink_factor,
                spiral_turns=spiral_turns,
                angle_offset_factor=angle_offset_factor,
            )
            local_length = (
                point.distance(left.point)
                + point.distance(right.point)
                + 0.4 * source.distance(point)
            )
            angle_balance = abs(angle - left.angle) + abs(right.angle - angle)
            intrusion_penalty = _source_group_intrusions(
                source,
                point,
                [left.point, right.point],
                endpoint_codes=endpoint_codes,
                anchors=anchors,
                radius=radius,
            )
            score = local_length + 0.35 * angle_balance + 1.50 * intrusion_penalty
            candidates.append((score, point, angle, merge_radius))

    _, point, angle, merge_radius = min(candidates, key=lambda item: item[0])
    return SpiralTreeNode(
        node_id=next_node_id,
        point=point,
        radius=merge_radius,
        angle=angle,
        angle_min=left.angle_min,
        angle_max=right.angle_max,
        flow_sum=left.flow_sum + right.flow_sum,
        leaf_ids=[*left.leaf_ids, *right.leaf_ids],
    )


def _source_tree_paths(
    rows: list[object],
    source_code: str,
    source: Point,
    anchors: pd.Series,
    anchors_codes: set[str],
    radius: float,
    spiral_turns: float,
    enhanced: bool,
    obstacle_segments: list[LineString] | None = None,
) -> list[tuple[object, list[Point]]]:
    """Build a source-rooted spiral tree and return one waypoint path per leaf."""

    ordered_targets = _ordered_targets_for_source(rows, source, anchors)
    if not ordered_targets:
        return []
    if len(ordered_targets) == 1:
        _, row, target, _, _ = ordered_targets[0]
        return [(row, [source, target])]

    nodes: dict[int, SpiralTreeNode] = {}
    leaf_node_ids: dict[int, int] = {}
    active: list[int] = []
    next_node_id = 0
    endpoint_codes = anchors_codes

    for leaf_id, row, target, angle, terminal_radius in ordered_targets:
        flow_sum = float(row.represented_flow)
        node = _leaf_node(
            node_id=next_node_id,
            leaf_id=leaf_id,
            point=target,
            angle=angle,
            radius=terminal_radius,
            flow_sum=flow_sum,
        )
        nodes[node.node_id] = node
        leaf_node_ids[leaf_id] = node.node_id
        active.append(node.node_id)
        next_node_id += 1

    while len(active) > 1:
        pair_candidates: list[tuple[float, int, SpiralTreeNode]] = []
        for idx in range(len(active) - 1):
            left = nodes[active[idx]]
            right = nodes[active[idx + 1]]
            merged = _best_merge_node(
                source,
                left,
                right,
                next_node_id=next_node_id,
                spiral_turns=spiral_turns,
                endpoint_codes=endpoint_codes,
                anchors=anchors,
                radius=radius,
                enhanced=enhanced,
            )
            # Check obstacle crossings for the chosen merge point
            merge_obs = _obstacle_crossings(
                merged.point,
                [left.point, right.point],
                source,
                obstacle_segments if obstacle_segments is not None else [],
            )
            # Primary sort: prefer crossing-free merges.
            # Secondary sort: outside-in by radius (original behavior).
            pair_candidates.append((merge_obs, -merged.radius, idx, merged))

        _, _, merge_idx, merged = min(pair_candidates, key=lambda item: (item[0], item[1]))
        left_id = active[merge_idx]
        right_id = active[merge_idx + 1]
        nodes[left_id].parent_id = merged.node_id
        nodes[right_id].parent_id = merged.node_id
        nodes[merged.node_id] = merged
        next_node_id += 1
        active = [*active[:merge_idx], merged.node_id, *active[merge_idx + 2 :]]

    paths: list[tuple[object, list[Point]]] = []
    root_id = active[0]
    root = nodes[root_id]
    for leaf_id, row, target, _, _ in ordered_targets:
        path_up = [target]
        current_id = leaf_node_ids[leaf_id]
        while nodes[current_id].parent_id is not None:
            current_id = nodes[current_id].parent_id
            path_up.append(nodes[current_id].point)
        if path_up[-1] != root.point:
            path_up.append(root.point)
        waypoints = [source, *reversed(path_up)]
        paths.append((row, waypoints))
    return paths

def _curve_through_waypoints(
    waypoints: list[Point],
    map_center: tuple[float, float],
    curvature_scale: float,
    samples_per_curve: int,
) -> tuple[LineString, Point | None]:
    """Create a smooth multi-leg carrier through one or more intermediate waypoints."""

    if len(waypoints) < 2:
        point = waypoints[0]
        return LineString([(point.x, point.y)]), None

    coords: list[tuple[float, float]] = []
    first_control: Point | None = None

    for start, end in zip(waypoints, waypoints[1:], strict=False):
        segment, control = _curve_geometry(
            start,
            end,
            map_center,
            curvature_scale=curvature_scale,
            samples_per_curve=samples_per_curve,
        )
        segment_coords = list(segment.coords)
        if coords:
            segment_coords = segment_coords[1:]
        coords.extend(segment_coords)
        if first_control is None and control is not None:
            first_control = control

    return LineString(coords), first_control


def _build_spiral_tree_carriers(
    selected_pairs: pd.DataFrame,
    provinces: gpd.GeoDataFrame,
    *,
    spiral_candidate_count: int,
    spiral_turns: float,
    samples_per_curve: int,
    node_buffer_ratio: float,
    length_penalty: float,
    hub_radius_ratio: float,
    merge_reward: float,
    corridor_reward: float,
    curvature_scale: float,
    attachment_samples: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Route salient pairs as source-rooted spiral trees, then overlay the trees."""

    anchors, names, map_center, _, radius, _ = _tree_context(
        provinces,
        node_buffer_ratio=node_buffer_ratio,
        hub_radius_ratio=hub_radius_ratio,
    )
    carriers: list[dict[str, object]] = []

    for source_code, group in _ordered_pairs_for_routing(selected_pairs).groupby(
        "dominant_origin",
        sort=False,
    ):
        source = anchors[source_code]
        rows = list(group.itertuples(index=False))
        endpoint_codes = {source_code, *(row.dominant_destination for row in rows)}
        source_paths = _source_tree_paths(
            rows,
            source_code=source_code,
            source=source,
            anchors=anchors,
            anchors_codes=endpoint_codes,
            radius=radius,
            spiral_turns=spiral_turns,
            enhanced=attachment_samples is not None,
        )

        for row, waypoints in source_paths:
            geometry, control = _curve_through_waypoints(
                waypoints,
                map_center,
                curvature_scale=curvature_scale,
                samples_per_curve=samples_per_curve,
            )
            carriers.append(
                _pair_carrier_record(
                    row,
                    names,
                    start=waypoints[0],
                    end=waypoints[-1],
                    geometry=geometry,
                    control=control,
                    waypoints=waypoints,
                )
            )

    return _finalize_carriers(pd.DataFrame(carriers))


def build_greedy_spiral_tree_carriers(
    selected_pairs: pd.DataFrame,
    provinces: gpd.GeoDataFrame,
    spiral_candidate_count: int = 6,
    spiral_turns: float = 1.10,
    samples_per_curve: int = 25,
    node_buffer_ratio: float = 0.018,
    length_penalty: float = 0.25,
    hub_radius_ratio: float = 0.10,
    merge_reward: float = 0.60,
    corridor_reward: float = 0.30,
) -> pd.DataFrame:
    """Build a simplified greedy spiral tree using shared global spiral hubs."""

    return _build_spiral_tree_carriers(
        selected_pairs,
        provinces,
        spiral_candidate_count=spiral_candidate_count,
        spiral_turns=spiral_turns,
        samples_per_curve=samples_per_curve,
        node_buffer_ratio=node_buffer_ratio,
        length_penalty=length_penalty,
        hub_radius_ratio=hub_radius_ratio,
        merge_reward=merge_reward,
        corridor_reward=corridor_reward,
        curvature_scale=0.12,
        attachment_samples=None,
    )


def build_enhanced_greedy_spiral_tree_carriers(
    selected_pairs: pd.DataFrame,
    provinces: gpd.GeoDataFrame,
    spiral_candidate_count: int = 6,
    spiral_turns: float = 1.10,
    samples_per_curve: int = 25,
    node_buffer_ratio: float = 0.018,
    length_penalty: float = 0.25,
    hub_radius_ratio: float = 0.10,
    merge_reward: float = 0.80,
    corridor_reward: float = 0.45,
    attachment_samples: Iterable[float] = (0.35, 0.50, 0.65),
) -> pd.DataFrame:
    """Allow later pairs to attach to previously accepted spiral-tree branches."""

    return _build_spiral_tree_carriers(
        selected_pairs,
        provinces,
        spiral_candidate_count=spiral_candidate_count,
        spiral_turns=spiral_turns,
        samples_per_curve=samples_per_curve,
        node_buffer_ratio=node_buffer_ratio,
        length_penalty=length_penalty,
        hub_radius_ratio=hub_radius_ratio,
        merge_reward=merge_reward,
        corridor_reward=corridor_reward,
        curvature_scale=0.10,
        attachment_samples=attachment_samples,
    )


# ---------------------------------------------------------------------------
# Obstacle-aware sequential spiral trees
# ---------------------------------------------------------------------------

def _order_source_groups(
    groups: dict[str, list],
    anchors: pd.Series,
    source_order: str,
) -> list[tuple[str, list]]:
    """Order source groups for sequential obstacle-aware tree construction."""

    items = list(groups.items())
    if source_order == "descending_outflow":
        items.sort(
            key=lambda kv: sum(float(r.represented_flow) for r in kv[1]),
            reverse=True,
        )
    elif source_order == "ascending_outflow":
        items.sort(
            key=lambda kv: sum(float(r.represented_flow) for r in kv[1]),
        )
    elif source_order == "west_to_east":
        items.sort(key=lambda kv: anchors[kv[0]].x)
    elif source_order == "north_to_south":
        items.sort(key=lambda kv: anchors[kv[0]].y, reverse=True)
    else:
        raise ValueError(
            "source_order must be one of 'descending_outflow', "
            "'ascending_outflow', 'west_to_east', or 'north_to_south'"
        )
    return items


def _build_obstacle_aware_spiral_tree_carriers(
    selected_pairs: pd.DataFrame,
    provinces: gpd.GeoDataFrame,
    *,
    spiral_turns: float,
    samples_per_curve: int,
    node_buffer_ratio: float,
    hub_radius_ratio: float,
    source_order: str,
) -> pd.DataFrame:
    """Route source trees sequentially, treating earlier trees as obstacles."""

    anchors, names, map_center, _, radius, _ = _tree_context(
        provinces,
        node_buffer_ratio=node_buffer_ratio,
        hub_radius_ratio=hub_radius_ratio,
    )

    # Group flows by source and order them
    source_groups: dict[str, list] = {}
    for row in _ordered_pairs_for_routing(selected_pairs).itertuples(index=False):
        source_groups.setdefault(row.dominant_origin, []).append(row)
    ordered_sources = _order_source_groups(source_groups, anchors, source_order)

    carriers: list[dict[str, object]] = []
    obstacle_segments: list[LineString] = []

    for source_code, rows in ordered_sources:
        source = anchors[source_code]
        endpoint_codes = {source_code, *(r.dominant_destination for r in rows)}

        source_paths = _source_tree_paths(
            rows,
            source_code=source_code,
            source=source,
            anchors=anchors,
            anchors_codes=endpoint_codes,
            radius=radius,
            spiral_turns=spiral_turns,
            enhanced=True,
            obstacle_segments=obstacle_segments,
        )

        # Build carrier records and accumulate curved geometries as obstacles
        for row, waypoints in source_paths:
            geometry, control = _curve_through_waypoints(
                waypoints,
                map_center,
                curvature_scale=0.10,
                samples_per_curve=samples_per_curve,
            )
            carriers.append(
                _pair_carrier_record(
                    row,
                    names,
                    start=waypoints[0],
                    end=waypoints[-1],
                    geometry=geometry,
                    control=control,
                    waypoints=waypoints,
                )
            )
            obstacle_segments.append(geometry)

    return _finalize_carriers(pd.DataFrame(carriers))


def build_obstacle_aware_spiral_tree_carriers(
    selected_pairs: pd.DataFrame,
    provinces: gpd.GeoDataFrame,
    spiral_turns: float = 1.10,
    samples_per_curve: int = 25,
    node_buffer_ratio: float = 0.018,
    hub_radius_ratio: float = 0.10,
    source_order: str = "descending_outflow",
) -> pd.DataFrame:
    """Build sequential obstacle-aware spiral trees."""

    return _build_obstacle_aware_spiral_tree_carriers(
        selected_pairs,
        provinces,
        spiral_turns=spiral_turns,
        samples_per_curve=samples_per_curve,
        node_buffer_ratio=node_buffer_ratio,
        hub_radius_ratio=hub_radius_ratio,
        source_order=source_order,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def count_crossings(carriers: pd.DataFrame) -> int:
    """Count pairwise curve crossings, excluding shared-endpoint touches."""

    geometries = list(carriers["geometry"])
    endpoints = list(zip(carriers["a"], carriers["b"], strict=False))
    crossings = 0

    for i, first in enumerate(geometries):
        first_nodes = set(endpoints[i])
        for j in range(i + 1, len(geometries)):
            second_nodes = set(endpoints[j])
            if first_nodes & second_nodes:
                continue
            if first.crosses(geometries[j]):
                crossings += 1

    return crossings


def count_node_intrusions(
    carriers: pd.DataFrame,
    provinces: gpd.GeoDataFrame,
    buffer_ratio: float = 0.018,
) -> int:
    """Count how often a carrier passes too close to an unrelated province anchor."""

    minx, miny, maxx, maxy = provinces.total_bounds
    scale = max(maxx - minx, maxy - miny)
    radius = scale * buffer_ratio

    anchors = provinces.set_index("code")["anchor"]
    intrusions = 0
    for row in carriers.itertuples(index=False):
        curve = row.geometry
        related = {row.a, row.b}
        for code, anchor in anchors.items():
            if code in related:
                continue
            if curve.intersects(anchor.buffer(radius)):
                intrusions += 1

    return intrusions


def coverage_ratio(
    selected: pd.DataFrame,
    total_represented_flow: float,
) -> float:
    """Share of represented flow retained by the selection."""

    total = float(total_represented_flow)
    if total == 0:
        return 0.0
    return float(selected["represented_flow"].sum()) / total


def _empty_metrics() -> dict[str, float]:
    """Return a zero-filled metric bundle for empty selections."""

    return {
        "pair_count": 0.0,
        "coverage": 0.0,
        "crossings": 0.0,
        "node_intrusions": 0.0,
        "mean_imbalance": 0.0,
        "mean_net_flow": 0.0,
        "total_tree_length": 0.0,
        "mean_detour_ratio": 0.0,
        "max_detour_ratio": 0.0,
    }


def evaluate_flow_map(
    total_represented_flow: float,
    selected: pd.DataFrame,
    carriers: pd.DataFrame,
    provinces: gpd.GeoDataFrame,
    buffer_ratio: float = 0.018,
) -> dict[str, float]:
    """Compute a small set of report-friendly quality measures."""

    if selected.empty or carriers.empty:
        return _empty_metrics()

    return {
        "pair_count": float(len(selected)),
        "coverage": coverage_ratio(selected, total_represented_flow),
        "crossings": float(count_crossings(carriers)),
        "node_intrusions": float(
            count_node_intrusions(carriers, provinces, buffer_ratio=buffer_ratio)
        ),
        "mean_imbalance": float(selected["imbalance_ratio"].mean()),
        "mean_net_flow": float(selected["net_flow"].mean()),
        "total_tree_length": float(carriers["geometry_length"].sum()),
        "mean_detour_ratio": float(carriers["detour_ratio"].mean()),
        "max_detour_ratio": float(carriers["detour_ratio"].max()),
    }


def _select_pairs_for_strategy(
    pair_summary: pd.DataFrame,
    config: FlowMapConfig,
) -> pd.DataFrame:
    """Apply the shared pair-selection step used by tree-based strategies."""

    return select_salient_pairs(
        pair_summary,
        top_k=config.top_k,
        min_total_flow=config.min_total_flow,
        selection_mode=config.selection_mode,
        hybrid_alpha=config.hybrid_alpha,
    )


# ---------------------------------------------------------------------------
# Strategy orchestration
# ---------------------------------------------------------------------------

def build_solution(
    provinces: gpd.GeoDataFrame,
    province_flows: pd.DataFrame,
    pair_summary: pd.DataFrame,
    config: FlowMapConfig = FlowMapConfig(),
) -> FlowMapResult:
    """Build one flow-map solution under the chosen strategy."""

    total_represented_flow = float(province_flows["flow"].sum())

    if config.strategy == "raw_directed":
        selected = select_salient_directed_flows(
            province_flows,
            top_k=config.top_k,
            min_flow=config.min_flow,
        )
        carriers = build_directed_carriers(selected, provinces)
    elif config.strategy == "greedy_spiral_tree":
        selected = _select_pairs_for_strategy(pair_summary, config)
        carriers = build_greedy_spiral_tree_carriers(
            selected,
            provinces,
            spiral_candidate_count=config.spiral_candidate_count,
            spiral_turns=config.spiral_turns,
            samples_per_curve=config.samples_per_curve,
            node_buffer_ratio=config.node_buffer_ratio,
            length_penalty=config.length_penalty,
            hub_radius_ratio=config.hub_radius_ratio,
            merge_reward=config.merge_reward,
            corridor_reward=config.corridor_reward,
        )
    elif config.strategy == "enhanced_greedy_spiral_tree":
        selected = _select_pairs_for_strategy(pair_summary, config)
        carriers = build_enhanced_greedy_spiral_tree_carriers(
            selected,
            provinces,
            spiral_candidate_count=config.spiral_candidate_count,
            spiral_turns=config.spiral_turns,
            samples_per_curve=config.samples_per_curve,
            node_buffer_ratio=config.node_buffer_ratio,
            length_penalty=config.length_penalty,
            hub_radius_ratio=config.hub_radius_ratio,
            merge_reward=config.merge_reward,
            corridor_reward=config.corridor_reward,
            attachment_samples=config.attachment_samples,
        )
    elif config.strategy == "obstacle_aware_spiral_tree":
        selected = _select_pairs_for_strategy(pair_summary, config)
        carriers = build_obstacle_aware_spiral_tree_carriers(
            selected,
            provinces,
            spiral_turns=config.spiral_turns,
            samples_per_curve=config.samples_per_curve,
            node_buffer_ratio=config.node_buffer_ratio,
            hub_radius_ratio=config.hub_radius_ratio,
            source_order=config.source_order,
        )
    else:
        raise ValueError(
            "strategy must be one of "
            "'raw_directed', 'greedy_spiral_tree', "
            "'enhanced_greedy_spiral_tree', or "
            "'obstacle_aware_spiral_tree'"
        )

    metrics = evaluate_flow_map(
        total_represented_flow,
        selected,
        carriers,
        provinces,
        buffer_ratio=config.node_buffer_ratio,
    )
    metrics["total_represented_flow"] = total_represented_flow
    return FlowMapResult(
        strategy=config.strategy,
        selected=selected,
        carriers=carriers,
        metrics=metrics,
    )


def compare_strategies(
    provinces: gpd.GeoDataFrame,
    province_flows: pd.DataFrame,
    pair_summary: pd.DataFrame,
    configs: Iterable[FlowMapConfig],
) -> pd.DataFrame:
    """Evaluate multiple strategies on the same dataset."""

    rows: list[dict[str, float | str]] = []
    for config in configs:
        result = build_solution(provinces, province_flows, pair_summary, config)
        rows.append(
            {
                "strategy": config.strategy,
                "top_k": config.top_k if config.top_k is not None else -1,
                "selection_mode": config.selection_mode,
                "spiral_turns": config.spiral_turns,
                "length_penalty": config.length_penalty,
                "source_order": config.source_order,
                **result.metrics,
            }
        )

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison

    comparison["clutter_score"] = (
        comparison["crossings"] + comparison["node_intrusions"]
    )
    comparison["coverage_minus_clutter"] = (
        comparison["coverage"] - 0.01 * comparison["clutter_score"]
    )
    return comparison.sort_values(
        ["coverage_minus_clutter", "coverage", "clutter_score"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_strategy_suite(
    provinces: gpd.GeoDataFrame,
    province_flows: pd.DataFrame,
    pair_summary: pd.DataFrame,
    configs: Iterable[FlowMapConfig],
) -> tuple[pd.DataFrame, dict[str, FlowMapResult]]:
    """Build both the comparison table and per-strategy results for a config set."""

    config_list = list(configs)
    comparison = compare_strategies(
        provinces,
        province_flows,
        pair_summary,
        config_list,
    )
    strategy_results = {
        config.strategy: build_solution(
            provinces,
            province_flows,
            pair_summary,
            config,
        )
        for config in config_list
    }
    return comparison, strategy_results


def default_strategy_configs(period: str = "2024JJ00") -> list[FlowMapConfig]:
    """Three default strategies for side-by-side comparison."""

    return [
        FlowMapConfig(
            strategy="raw_directed",
            period=period,
            top_k=40,
            min_flow=0.0,
        ),
        FlowMapConfig(
            strategy="greedy_spiral_tree",
            period=period,
            top_k=40,
            min_total_flow=0.0,
            selection_mode="total_flow",
            spiral_turns=1.05,
            spiral_candidate_count=6,
            hub_radius_ratio=0.10,
            merge_reward=0.60,
            corridor_reward=0.30,
        ),
        FlowMapConfig(
            strategy="enhanced_greedy_spiral_tree",
            period=period,
            top_k=40,
            min_total_flow=0.0,
            selection_mode="total_flow",
            length_penalty=0.25,
            spiral_turns=1.10,
            spiral_candidate_count=6,
            hub_radius_ratio=0.10,
            merge_reward=0.80,
            corridor_reward=0.45,
        ),
        FlowMapConfig(
            strategy="obstacle_aware_spiral_tree",
            period=period,
            top_k=40,
            min_total_flow=0.0,
            selection_mode="total_flow",
            spiral_turns=1.10,
            hub_radius_ratio=0.10,
            source_order="descending_outflow",
        ),
    ]


def enhanced_spiral_sweep_configs(
    period: str = "2024JJ00",
    top_ks: Iterable[int] = (10, 15, 20, 25, 30),
    spiral_turn_values: Iterable[float] = (0.90, 1.05, 1.20, 1.35),
) -> list[FlowMapConfig]:
    """Generate the enhanced spiral-tree configs used in the notebook sweep."""

    return [
        FlowMapConfig(
            strategy="enhanced_greedy_spiral_tree",
            period=period,
            top_k=top_k,
            min_total_flow=0.0,
            selection_mode="total_flow",
            spiral_turns=spiral_turn,
            spiral_candidate_count=6,
            hub_radius_ratio=0.10,
            merge_reward=0.80,
            corridor_reward=0.45,
        )
        for top_k in top_ks
        for spiral_turn in spiral_turn_values
    ]


def obstacle_aware_ordering_configs(
    period: str = "2024JJ00",
    source_orders: Iterable[str] = (
        "descending_outflow",
        "ascending_outflow",
        "west_to_east",
        "north_to_south",
    ),
) -> list[FlowMapConfig]:
    """Generate configs for the source-ordering experiment."""

    return [
        FlowMapConfig(
            strategy="obstacle_aware_spiral_tree",
            period=period,
            top_k=40,
            min_total_flow=0.0,
            selection_mode="total_flow",
            spiral_turns=1.10,
            hub_radius_ratio=0.10,
            source_order=order,
        )
        for order in source_orders
    ]
