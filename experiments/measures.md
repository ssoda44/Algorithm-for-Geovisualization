# Measures

## Independent Variables

- **strategy**: The routing and selection method used to construct the flow map.
  - `raw_directed`: Straight directed OD segments.
  - `force_adjusted_curved_od`: Curved OD segments with repulsive control-point updates.
  - `quality_aware_force_adjusted_od`: Greedy flow selection using a quality score, then force-adjusted curved routing.
  - `quality_aware_polyline_od`: Polyline routing with horizontal, vertical, and diagonal candidates plus directional lane offsets.
- **top_k**: Number of highest-flow directed pairs used as the initial displayed set.
- **force_iterations**: Number of repulsion iterations applied in the curved method.
- **greedy_candidate_limit**: Number of candidate directed flows considered by the quality-aware greedy selection.
- **polyline_lane_ratio**: Size of the lane offset used to separate opposite polyline directions.
- **geographic level**: Province level (12 regions) versus municipality level (342 regions).

## Dependent Variables

The current project evaluates each valid flow map using four primary quality measures.

## Coverage

Fraction of total directed flow value represented by the selected carriers:

\[
\mathrm{coverage}(S)=\frac{\sum_{e\in S} w_e}{\sum_{e\in E} w_e}
\]

Higher is better. This measure captures how much of the dataset remains visible after filtering or greedy selection.

## Crossing Count

Number of carrier pairs whose geometries intersect, excluding pairs that share an endpoint. Crossings are measured on the final rendered carrier geometries using Shapely's `crosses()`.

Lower is better. This is the main readability measure.

## Node Intrusions

Number of times a carrier passes too close to the anchor point of an unrelated region. This penalizes lines that visually interfere with node labels or region identity.

Lower is better.

## Mean Detour Ratio

Average ratio between route length and straight-line source-to-target distance:

\[
\mathrm{detour}(S)=\frac{1}{|S|}\sum_{e\in S}\frac{\mathrm{length}(e)}{\mathrm{dist}(e)}
\]

A value of `1.0` means the route is geometrically direct. Higher values indicate extra bending or routing overhead.

Lower is better.

## Secondary Reported Measures

- **total_tree_length**: Sum of all carrier lengths.
- **max_detour_ratio**: Worst-case detour ratio among the displayed carriers.
- **clutter_score**: `crossings + node_intrusions`.
- **coverage_minus_clutter**: Simple summary score `coverage - 0.01 * clutter_score`, used only as a compact comparison aid in the tables.

## Quality Objective For Algorithm 3

The greedy selection algorithm uses the scalar objective

\[
Q(S)=
\alpha \cdot \mathrm{coverage}(S)
-\beta \cdot \mathrm{crossings}(S)
-\gamma \cdot \mathrm{intrusions}(S)
-\delta \cdot (\mathrm{detour}(S)-1)
\]

with the current default weights

- \(\alpha = 1.0\)
- \(\beta = 0.008\)
- \(\gamma = 0.004\)
- \(\delta = 0.20\)

This objective is not intended as a universal truth; it is a heuristic way to balance information retention against visual clutter.
