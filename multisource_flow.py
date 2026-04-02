"""Core utilities for the 2IMA20 multi-source flow-map project.

The current codebase keeps four strategies:

- ``raw_directed``: thresholded direct OD baseline
- ``force_adjusted_curved_od``: curved OD with local control-point repulsion
- ``quality_aware_force_adjusted_od``: greedy quality-aware flow selection
- ``quality_aware_polyline_od``: guided polyline routing with diagonal escapes
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
    """Parameters for one strategy run."""

    strategy: str = "force_adjusted_curved_od"
    period: str = "2024JJ00"
    top_k: int | None = 40
    min_flow: float = 0.0
    greedy_candidate_limit: int | None = 60
    coverage_weight: float = 1.0
    crossing_weight: float = 0.008
    intrusion_weight: float = 0.004
    detour_weight: float = 0.20
    samples_per_curve: int = 25
    node_buffer_ratio: float = 0.018
    curvature_scale: float = 0.10
    force_iterations: int = 5
    selection_force_iterations: int = 0
    force_step_scale: float = 0.010
    force_proximity_scale: float = 0.028
    force_base_pull: float = 0.18
    polyline_lane_ratio: float = 0.010
    polyline_proximity_ratio: float = 0.020


@dataclass
class FlowMapResult:
    """Bundle the output of one strategy run."""

    strategy: str
    selected: pd.DataFrame
    carriers: pd.DataFrame
    metrics: dict[str, float]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def load_provinces(data_dir: str | Path = "data") -> gpd.GeoDataFrame:
    """Load Dutch provinces with a compact schema."""

    provinces = gpd.read_file(Path(data_dir) / "provinces.geojson")
    provinces = provinces.rename(columns={"statcode": "code", "statnaam": "name"})
    provinces = provinces[["code", "name", "geometry"]].copy()
    provinces["anchor"] = provinces.geometry.representative_point()
    return provinces


def load_observations(data_dir: str | Path = "data") -> pd.DataFrame:
    """Load the CBS observations table."""

    obs = pd.read_csv(Path(data_dir) / "Observations.csv", sep=";")
    obs["Value"] = pd.to_numeric(obs["Value"], errors="coerce")
    return obs


def build_province_flows(
    obs: pd.DataFrame,
    period: str = "2024JJ00",
) -> pd.DataFrame:
    """Return directed province-to-province flows for one period."""

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
            grouped[key] = {"a": a, "b": b, "flow_ab": 0.0, "flow_ba": 0.0}

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
    return summary.sort_values(
        ["total_flow", "net_flow", "a", "b"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def select_salient_directed_flows(
    province_flows: pd.DataFrame,
    top_k: int | None = None,
    min_flow: float = 0.0,
) -> pd.DataFrame:
    """Keep the most salient directed flows for direct-OD strategies."""

    selected = province_flows[province_flows["flow"] >= float(min_flow)].copy()
    selected = selected.sort_values(
        ["flow", "origin", "destination"],
        ascending=[False, True, True],
    )
    if top_k is not None:
        selected = selected.head(top_k).copy()

    selected["represented_flow"] = selected["flow"]
    selected["net_flow"] = selected["flow"]
    selected["imbalance_ratio"] = 1.0
    return selected.reset_index(drop=True)


def load_flow_dataset(
    data_dir: str | Path = "data",
    period: str = "2024JJ00",
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Load provinces and derive directed and unordered flow tables."""

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
    dot = nx * (mx - cx) + ny * (my - cy)
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
    bend_side = _curve_side(start, end, map_center) if side is None else side
    bend = bend_side * length * curvature_scale
    return Point(mx + bend * nx, my + bend * ny)


def _straight_line(start: Point, end: Point) -> LineString:
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
    curve = LineString(_quadratic_bezier_points(start, control, end, samples_per_curve))
    return curve, control


def _curve_geometry_from_control(
    start: Point,
    control: Point,
    end: Point,
    samples_per_curve: int,
) -> LineString:
    """Create a quadratic Bezier carrier from an explicit control point."""

    return LineString(_quadratic_bezier_points(start, control, end, samples_per_curve))


def _finalize_carriers(carrier_df: pd.DataFrame) -> pd.DataFrame:
    """Add display and geometry summary columns."""

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


# ---------------------------------------------------------------------------
# Carrier construction
# ---------------------------------------------------------------------------


def build_directed_carriers(
    selected_flows: pd.DataFrame,
    provinces: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Build straight carriers for the direct baseline."""

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


def build_force_adjusted_curved_carriers(
    selected_flows: pd.DataFrame,
    provinces: gpd.GeoDataFrame,
    curvature_scale: float = 0.10,
    samples_per_curve: int = 25,
    force_iterations: int = 5,
    force_step_scale: float = 0.010,
    force_proximity_scale: float = 0.028,
    force_base_pull: float = 0.18,
) -> pd.DataFrame:
    """Refine curved OD carriers by iteratively repelling crowded control points."""

    anchors, names = _province_lookup(provinces)
    map_center = _map_center(provinces)
    map_scale = _map_scale(provinces)

    records: list[dict[str, object]] = []
    base_controls: list[Point] = []
    for row in selected_flows.itertuples(index=False):
        start = anchors[row.origin]
        end = anchors[row.destination]
        base_side = _curve_side(start, end, map_center)
        direction_side = 1.0 if row.origin < row.destination else -1.0
        base_controls.append(
            _control_point(
                start,
                end,
                map_center,
                curvature_scale=curvature_scale,
                side=base_side * direction_side,
            )
        )
        records.append({"row": row, "start": start, "end": end})

    if not records:
        return _finalize_carriers(pd.DataFrame())

    base_coords = np.asarray([[p.x, p.y] for p in base_controls], dtype=float)
    control_coords = base_coords.copy()
    proximity_threshold = map_scale * force_proximity_scale
    base_step = map_scale * force_step_scale

    for _ in range(max(0, force_iterations)):
        geometries = [
            _curve_geometry_from_control(
                record["start"],
                Point(control[0], control[1]),
                record["end"],
                samples_per_curve,
            )
            for record, control in zip(records, control_coords, strict=False)
        ]

        deltas = np.zeros_like(control_coords)
        for i, geom_i in enumerate(geometries):
            nodes_i = {records[i]["row"].origin, records[i]["row"].destination}
            for j in range(i + 1, len(geometries)):
                nodes_j = {records[j]["row"].origin, records[j]["row"].destination}
                if nodes_i & nodes_j:
                    continue

                geom_j = geometries[j]
                crosses = geom_i.crosses(geom_j)
                distance = float(geom_i.distance(geom_j))
                if not crosses and distance >= proximity_threshold:
                    continue

                vec = control_coords[i] - control_coords[j]
                norm = float(np.hypot(vec[0], vec[1]))
                if norm < 1e-9:
                    vec = np.array([1.0, 0.0])
                    norm = 1.0

                strength = base_step * (1.6 if crosses else max(0.1, 1.0 - distance / proximity_threshold))
                direction = vec / norm
                deltas[i] += strength * direction
                deltas[j] -= strength * direction

        control_coords += deltas
        control_coords += force_base_pull * (base_coords - control_coords)

    carriers: list[dict[str, object]] = []
    for record, coords in zip(records, control_coords, strict=False):
        row = record["row"]
        start = record["start"]
        end = record["end"]
        control = Point(coords[0], coords[1])
        geometry = _curve_geometry_from_control(start, control, end, samples_per_curve)
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
                "control": control,
                "end": end,
                "geometry": geometry,
                "waypoints": [start, end],
            }
        )

    return _finalize_carriers(pd.DataFrame(carriers))


def _compress_polyline_points(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Drop duplicate and collinear interior points."""

    deduped: list[tuple[float, float]] = []
    for point in points:
        if not deduped or deduped[-1] != point:
            deduped.append(point)

    if len(deduped) <= 2:
        return deduped

    compressed = [deduped[0]]
    for idx, point in enumerate(deduped[1:-1], start=1):
        ax, ay = compressed[-1]
        bx, by = point
        cx, cy = deduped[idx + 1]
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(cross) > 1e-9:
            compressed.append(point)
    compressed.append(deduped[-1])
    return compressed


def _polyline_intrusions(
    geometry: LineString,
    provinces: gpd.GeoDataFrame,
    related_codes: set[str],
    radius: float,
) -> int:
    anchors = provinces.set_index("code")["anchor"]
    return sum(
        int(geometry.intersects(anchor.buffer(radius)))
        for code, anchor in anchors.items()
        if code not in related_codes
    )


def _polyline_route_cost(
    geometry: LineString,
    start_code: str,
    end_code: str,
    existing: list[dict[str, object]],
    provinces: gpd.GeoDataFrame,
    radius: float,
    proximity_threshold: float,
    straight_distance: float,
) -> tuple[float, float, float, float, float]:
    crossings = 0
    overlaps = 0
    near_hits = 0

    related = {start_code, end_code}
    for record in existing:
        if related & {record["a"], record["b"]}:
            continue

        other = record["geometry"]
        if geometry.crosses(other):
            crossings += 1
            continue

        if geometry.intersects(other):
            intersection = geometry.intersection(other)
            if getattr(intersection, "length", 0.0) > 0:
                overlaps += 1
            elif not intersection.is_empty:
                near_hits += 1
            continue

        if float(geometry.distance(other)) < proximity_threshold:
            near_hits += 1

    intrusions = _polyline_intrusions(geometry, provinces, related, radius)
    detour_penalty = 0.0 if straight_distance <= 0 else geometry.length / straight_distance - 1.0
    return (float(crossings), float(overlaps), float(near_hits), float(intrusions), float(detour_penalty))


def build_quality_aware_polyline_carriers(
    selected_flows: pd.DataFrame,
    provinces: gpd.GeoDataFrame,
    lane_ratio: float = 0.010,
    proximity_ratio: float = 0.020,
) -> pd.DataFrame:
    """Route each OD pair along the best polyline candidate with limited diagonal use."""

    anchors, names = _province_lookup(provinces)
    minx, miny, maxx, maxy = provinces.total_bounds
    width = maxx - minx
    height = maxy - miny
    map_scale = _map_scale(provinces)
    lane_offset = map_scale * lane_ratio
    intrusion_radius = map_scale * 0.018
    proximity_threshold = map_scale * proximity_ratio
    x_guides = [minx + ratio * width for ratio in (0.28, 0.40, 0.52, 0.64, 0.76)]
    y_guides = [miny + ratio * height for ratio in (0.22, 0.38, 0.54, 0.70)]

    rows = selected_flows.sort_values(
        ["represented_flow", "origin", "destination"],
        ascending=[False, True, True],
    )
    chosen: list[dict[str, object]] = []

    for row in rows.itertuples(index=False):
        start = anchors[row.origin]
        end = anchors[row.destination]
        sx, sy = start.x, start.y
        tx, ty = end.x, end.y
        lane_sign = 1.0 if row.origin < row.destination else -1.0

        candidate_paths: list[list[tuple[float, float]]] = [[(sx, sy), (tx, ty)]]

        for xg in x_guides:
            x_lane = xg + lane_sign * lane_offset
            candidate_paths.extend(
                [
                    [(sx, sy), (x_lane, sy), (x_lane, ty), (tx, ty)],
                    [(sx, sy), (x_lane, sy), (tx, ty)],
                    [(sx, sy), (x_lane, ty), (tx, ty)],
                ]
            )

        for yg in y_guides:
            y_lane = yg + lane_sign * lane_offset
            candidate_paths.extend(
                [
                    [(sx, sy), (sx, y_lane), (tx, y_lane), (tx, ty)],
                    [(sx, sy), (sx, y_lane), (tx, ty)],
                    [(sx, sy), (tx, y_lane), (tx, ty)],
                ]
            )

        best_points: list[tuple[float, float]] | None = None
        best_cost: tuple[float, float, float, float, float] | None = None
        best_geometry: LineString | None = None
        straight_distance = float(start.distance(end))

        for path in candidate_paths:
            points = _compress_polyline_points(path)
            if len(points) < 2:
                continue

            geometry = LineString(points)
            cost = _polyline_route_cost(
                geometry,
                row.origin,
                row.destination,
                chosen,
                provinces,
                intrusion_radius,
                proximity_threshold,
                straight_distance,
            )
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_points = points
                best_geometry = geometry

        assert best_points is not None and best_geometry is not None
        chosen.append(
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
                "geometry": best_geometry,
                "waypoints": [Point(x, y) for x, y in best_points],
            }
        )

    return _finalize_carriers(pd.DataFrame(chosen))


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
            if first_nodes & set(endpoints[j]):
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

    radius = _map_scale(provinces) * buffer_ratio
    anchors = provinces.set_index("code")["anchor"]
    intrusions = 0

    for row in carriers.itertuples(index=False):
        related = {row.a, row.b}
        for code, anchor in anchors.items():
            if code in related:
                continue
            if row.geometry.intersects(anchor.buffer(radius)):
                intrusions += 1

    return intrusions


def coverage_ratio(selected: pd.DataFrame, total_represented_flow: float) -> float:
    total = float(total_represented_flow)
    if total == 0:
        return 0.0
    return float(selected["represented_flow"].sum()) / total


def _empty_metrics() -> dict[str, float]:
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
    """Compute the report-facing quality measures."""

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


# ---------------------------------------------------------------------------
# Greedy quality-aware selection
# ---------------------------------------------------------------------------


def _build_direct_strategy_carriers(
    selected: pd.DataFrame,
    provinces: gpd.GeoDataFrame,
    config: FlowMapConfig,
) -> pd.DataFrame:
    if config.strategy == "raw_directed":
        return build_directed_carriers(selected, provinces)
    if config.strategy == "force_adjusted_curved_od":
        return build_force_adjusted_curved_carriers(
            selected,
            provinces,
            curvature_scale=config.curvature_scale,
            samples_per_curve=config.samples_per_curve,
            force_iterations=config.force_iterations,
            force_step_scale=config.force_step_scale,
            force_proximity_scale=config.force_proximity_scale,
            force_base_pull=config.force_base_pull,
        )
    if config.strategy == "quality_aware_polyline_od":
        return build_quality_aware_polyline_carriers(
            selected,
            provinces,
            lane_ratio=config.polyline_lane_ratio,
            proximity_ratio=config.polyline_proximity_ratio,
        )
    raise ValueError("unsupported direct strategy")


def _selection_proxy_config(config: FlowMapConfig) -> FlowMapConfig:
    """Use a cheaper routing proxy during greedy candidate evaluation."""

    return FlowMapConfig(
        **{
            **config.__dict__,
            "force_iterations": config.selection_force_iterations,
        }
    )


def quality_score(metrics: dict[str, float], config: FlowMapConfig) -> float:
    """Scalar objective used by the greedy selection algorithm."""

    detour_penalty = max(0.0, float(metrics["mean_detour_ratio"]) - 1.0)
    return (
        config.coverage_weight * float(metrics["coverage"])
        - config.crossing_weight * float(metrics["crossings"])
        - config.intrusion_weight * float(metrics["node_intrusions"])
        - config.detour_weight * detour_penalty
    )


def select_quality_aware_directed_flows(
    province_flows: pd.DataFrame,
    provinces: gpd.GeoDataFrame,
    config: FlowMapConfig,
    total_represented_flow: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Greedily add the candidate with the largest positive ΔQ until convergence."""

    candidates = select_salient_directed_flows(
        province_flows,
        top_k=config.greedy_candidate_limit,
        min_flow=config.min_flow,
    )
    if candidates.empty:
        empty = pd.DataFrame()
        return empty, empty, _empty_metrics()

    seed_k = min(config.top_k if config.top_k is not None else 40, len(candidates))
    selected = candidates.head(seed_k).copy().reset_index(drop=True)
    remaining = candidates.iloc[seed_k:].copy().reset_index(drop=True)

    proxy_config = _selection_proxy_config(config)
    carriers = _build_direct_strategy_carriers(selected, provinces, proxy_config)
    metrics = evaluate_flow_map(
        total_represented_flow,
        selected,
        carriers,
        provinces,
        buffer_ratio=config.node_buffer_ratio,
    )
    score = quality_score(metrics, config)

    while not remaining.empty:
        best_gain = 0.0
        best_idx = -1
        best_selected = selected
        best_carriers = carriers
        best_metrics = metrics

        for idx, candidate in remaining.iterrows():
            candidate_df = pd.DataFrame([candidate], columns=remaining.columns)
            trial_selected = pd.concat([selected, candidate_df], ignore_index=True)
            trial_carriers = _build_direct_strategy_carriers(trial_selected, provinces, proxy_config)
            trial_metrics = evaluate_flow_map(
                total_represented_flow,
                trial_selected,
                trial_carriers,
                provinces,
                buffer_ratio=config.node_buffer_ratio,
            )
            gain = quality_score(trial_metrics, config) - score
            if gain > best_gain:
                best_gain = gain
                best_idx = int(idx)
                best_selected = trial_selected
                best_carriers = trial_carriers
                best_metrics = trial_metrics

        if best_idx < 0:
            break

        selected = best_selected
        carriers = best_carriers
        metrics = best_metrics
        score = quality_score(metrics, config)
        remaining = remaining.drop(index=best_idx).reset_index(drop=True)

    final_carriers = _build_direct_strategy_carriers(selected, provinces, config)
    final_metrics = evaluate_flow_map(
        total_represented_flow,
        selected,
        final_carriers,
        provinces,
        buffer_ratio=config.node_buffer_ratio,
    )
    return selected, final_carriers, final_metrics


# ---------------------------------------------------------------------------
# Strategy orchestration
# ---------------------------------------------------------------------------


def build_solution(
    provinces: gpd.GeoDataFrame,
    province_flows: pd.DataFrame,
    pair_summary: pd.DataFrame,  # kept for call-site stability
    config: FlowMapConfig = FlowMapConfig(),
) -> FlowMapResult:
    """Build one flow-map solution under the chosen strategy."""

    del pair_summary
    total_represented_flow = float(province_flows["flow"].sum())

    if config.strategy == "quality_aware_force_adjusted_od":
        quality_config = FlowMapConfig(
            **{
                **config.__dict__,
                "strategy": "force_adjusted_curved_od",
            }
        )
        selected, carriers, metrics = select_quality_aware_directed_flows(
            province_flows,
            provinces,
            quality_config,
            total_represented_flow,
        )
    elif config.strategy in {
        "raw_directed",
        "force_adjusted_curved_od",
        "quality_aware_polyline_od",
    }:
        selected = select_salient_directed_flows(
            province_flows,
            top_k=config.top_k,
            min_flow=config.min_flow,
        )
        carriers = _build_direct_strategy_carriers(selected, provinces, config)
        metrics = evaluate_flow_map(
            total_represented_flow,
            selected,
            carriers,
            provinces,
            buffer_ratio=config.node_buffer_ratio,
        )
    else:
        raise ValueError(
            "strategy must be 'raw_directed', 'force_adjusted_curved_od', "
            "'quality_aware_force_adjusted_od', or 'quality_aware_polyline_od'"
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

    rows: list[dict[str, float | str | bool]] = []
    for config in configs:
        result = build_solution(provinces, province_flows, pair_summary, config)
        rows.append(
            {
                "strategy": config.strategy,
                "top_k": config.top_k if config.top_k is not None else -1,
                "greedy_candidate_limit": config.greedy_candidate_limit,
                "curvature_scale": config.curvature_scale,
                "force_iterations": config.force_iterations,
                "polyline_lane_ratio": config.polyline_lane_ratio,
                "quality_score": quality_score(result.metrics, config)
                if config.strategy == "quality_aware_force_adjusted_od"
                else np.nan,
                **result.metrics,
            }
        )

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison

    comparison["clutter_score"] = comparison["crossings"] + comparison["node_intrusions"]
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
    """Build both the comparison table and per-strategy results."""

    config_list = list(configs)
    comparison = compare_strategies(provinces, province_flows, pair_summary, config_list)
    results = {
        config.strategy: build_solution(
            provinces,
            province_flows,
            pair_summary,
            config,
        )
        for config in config_list
    }
    return comparison, results


def default_strategy_configs(period: str = "2024JJ00") -> list[FlowMapConfig]:
    """Default recommended comparison set."""

    return [
        FlowMapConfig(
            strategy="raw_directed",
            period=period,
            top_k=40,
            min_flow=0.0,
        ),
        FlowMapConfig(
            strategy="force_adjusted_curved_od",
            period=period,
            top_k=40,
            min_flow=0.0,
            curvature_scale=0.10,
            force_iterations=5,
        ),
        FlowMapConfig(
            strategy="quality_aware_force_adjusted_od",
            period=period,
            top_k=40,
            min_flow=0.0,
            greedy_candidate_limit=60,
            curvature_scale=0.10,
            force_iterations=5,
            coverage_weight=1.0,
            crossing_weight=0.008,
            intrusion_weight=0.004,
            detour_weight=0.20,
        ),
        FlowMapConfig(
            strategy="quality_aware_polyline_od",
            period=period,
            top_k=40,
            min_flow=0.0,
            polyline_lane_ratio=0.010,
            polyline_proximity_ratio=0.020,
        ),
    ]


def quality_aware_sweep_configs(
    period: str = "2024JJ00",
    top_k_values: Iterable[int] = (30, 40, 50),
    candidate_limits: Iterable[int] = (50, 60, 70),
) -> list[FlowMapConfig]:
    """Generate configs for the quality-aware greedy selection sweep."""

    return [
        FlowMapConfig(
            strategy="quality_aware_force_adjusted_od",
            period=period,
            top_k=top_k,
            min_flow=0.0,
            greedy_candidate_limit=limit,
            curvature_scale=0.10,
            force_iterations=5,
            coverage_weight=1.0,
            crossing_weight=0.008,
            intrusion_weight=0.004,
            detour_weight=0.20,
        )
        for top_k in top_k_values
        for limit in candidate_limits
        if limit >= top_k
    ]


def force_adjusted_sweep_configs(
    period: str = "2024JJ00",
    top_k: int = 40,
    iteration_values: Iterable[int] = (0, 5, 10, 15),
) -> list[FlowMapConfig]:
    """Generate configs for the force-iteration sweep."""

    return [
        FlowMapConfig(
            strategy="force_adjusted_curved_od",
            period=period,
            top_k=top_k,
            min_flow=0.0,
            curvature_scale=0.10,
            force_iterations=value,
        )
        for value in iteration_values
    ]


def polyline_sweep_configs(
    period: str = "2024JJ00",
    top_k: int = 40,
    lane_ratios: Iterable[float] = (0.0, 0.010, 0.020),
) -> list[FlowMapConfig]:
    """Generate configs for the polyline lane-separation sweep."""

    return [
        FlowMapConfig(
            strategy="quality_aware_polyline_od",
            period=period,
            top_k=top_k,
            min_flow=0.0,
            polyline_lane_ratio=value,
            polyline_proximity_ratio=0.020,
        )
        for value in lane_ratios
    ]
