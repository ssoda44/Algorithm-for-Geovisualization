# Experiment Design

## Goal

The goal of the project is not to find one universally best multi-source flow-map algorithm, but to study the trade-off between:

- preserving more flow information (`coverage`)
- keeping the map readable (`crossings`, `node_intrusions`)
- keeping routes geometrically efficient (`mean_detour_ratio`)

Our final setup therefore compares four heuristics that emphasize different parts of this trade-off.

## Algorithm Design

### Algorithm 1: Thresholded Direct OD Lines

This is the baseline. We sort directed OD pairs by flow magnitude and keep the top `k` pairs. Each selected pair is drawn as one straight directed segment from the source anchor to the target anchor.

- Strength: minimum geometric distortion
- Weakness: crossings increase quickly as density grows
- Type: simple deterministic baseline

### Algorithm 2: Force-Adjusted Curved OD

This method starts from the same top-`k` directed flows as Algorithm 1, but replaces each straight line by a quadratic curve. The control point first bends the curve away from the map center, then multiple repulsion iterations push nearby or crossing curves apart while a restoring term keeps the curve near its original shape.

- Strength: reduces clutter without changing the displayed flow set
- Weakness: adds detour
- Type: geometric routing heuristic

### Algorithm 3: Quality-Aware Force-Adjusted OD

This method extends Algorithm 2 by changing the selection stage. Instead of fixing the output to exactly the top `k` flows, it starts from a top-`k` seed and greedily adds extra flows from a candidate pool whenever they improve a scalar quality objective:

\[
Q(S)=
\alpha \cdot \mathrm{coverage}(S)
-\beta \cdot \mathrm{crossings}(S)
-\gamma \cdot \mathrm{intrusions}(S)
-\delta \cdot (\mathrm{detour}(S)-1)
\]

The final selected flows are then routed with the force-adjusted curved method.

- Strength: improves coverage in a controlled way
- Weakness: may reintroduce some clutter
- Type: greedy selection heuristic plus curved routing heuristic

### Algorithm 4: Quality-Aware Polyline OD

This method keeps the top-`k` flow set fixed, but changes the routing model. For each flow, it generates multiple candidate polylines using horizontal, vertical, and occasional diagonal segments. Opposite directions receive a small lane offset to preserve directional visibility. The flow is routed along the candidate with the lowest local cost, based on added crossings, overlap, near-conflicts, node intrusions, and detour.

\[
\mathrm{cost}(r)
=
w_1 \, \Delta \mathrm{crossings}
+
w_2 \, \Delta \mathrm{overlap}
+
w_3 \, \Delta \mathrm{intrusions}
+
w_4 \, \Delta \mathrm{detour}
\]

- Strength: strongly reduces crossings while preserving directional detail
- Weakness: detour is higher than the curved methods
- Type: greedy local routing heuristic

## Quality Measures

We evaluate all strategies using the following measures:

1. **Coverage**: fraction of total directed flow represented by the displayed carriers.
2. **Crossing count**: number of carrier pairs that intersect.
3. **Node intrusions**: number of times a carrier passes too close to an unrelated region anchor.
4. **Mean detour ratio**: average route length divided by straight-line distance.

These measures deliberately pull in different directions. A solution with low crossings may become too schematic; a solution with high coverage may become cluttered.

---

## Experiment 1: Main Strategy Comparison

**Question:** How do the four strategies compare under one shared default setting?

**Independent variable:** `strategy`

**Fixed parameters:** province-level data, `top_k = 40`, `force_iterations = 5`, `greedy_candidate_limit = 60`, `polyline_lane_ratio = 0.01`

**Output:** `exp1_strategy_comparison.csv`, `exp1_four_algorithms_comparison.png`

### Results

| Strategy | Coverage | Crossings | Intrusions | Mean Detour |
|----------|----------|-----------|------------|-------------|
| raw_directed | 0.743 | 28 | 7 | 1.000 |
| force_adjusted_curved_od | 0.743 | 19 | 5 | 1.032 |
| quality_aware_force_adjusted_od | 0.789 | 23 | 5 | 1.029 |
| quality_aware_polyline_od | 0.743 | 8 | 1 | 1.064 |

**Observations:**

- Algorithm 2 clearly improves the baseline at equal coverage.
- Algorithm 3 gives the highest coverage, but not the lowest crossings.
- Algorithm 4 gives the lowest clutter among the detailed-routing methods, but pays for it with extra detour.
- There is no single universal winner: Algorithm 3 is best when coverage matters most, while Algorithm 4 is best when minimizing clutter at fixed coverage matters most.

---

## Experiment 2: Effect of Flow Density

**Question:** How do the strategies behave as the map becomes denser?

**Independent variables:**

- `top_k` ∈ {10, 20, 30, 40, 50, 60}
- `strategy`

**Fixed parameters:** province-level data, `force_iterations = 5`, `greedy_candidate_limit = 60`, `polyline_lane_ratio = 0.01`

**Output:** `exp2_flow_density.csv`

### Key Results

At `top_k = 40`:

- `raw_directed`: `coverage = 0.743`, `crossings = 28`
- `force_adjusted_curved_od`: `coverage = 0.743`, `crossings = 19`
- `quality_aware_force_adjusted_od`: `coverage = 0.789`, `crossings = 23`
- `quality_aware_polyline_od`: `coverage = 0.743`, `crossings = 8`

At `top_k = 60`:

- `raw_directed`: `coverage = 0.857`, `crossings = 122`
- `force_adjusted_curved_od`: `coverage = 0.857`, `crossings = 102`
- `quality_aware_force_adjusted_od`: `coverage = 0.857`, `crossings = 102`
- `quality_aware_polyline_od`: `coverage = 0.857`, `crossings = 56`

**Observations:**

- All methods suffer as density increases, but Algorithm 4 scales better on crossings than the other detailed methods.
- Algorithm 3 is strongest at low and medium density because it quickly raises coverage by adding extra important flows.
- At very high density, the coverage advantage of Algorithm 3 disappears because the fixed top-`k` methods eventually catch up.
- Algorithm 4 remains the best low-crossing routing strategy across the denser settings.

---

## Experiment 3: Parameter Sensitivity

**Question:** Which parameter values work well as defaults for the curved, quality-aware, and polyline methods?

**Independent variables:**

- `force_iterations` ∈ {0, 5, 10, 15} for Algorithm 2
- `greedy_candidate_limit` ∈ {50, 60, 70} for Algorithm 3
- `polyline_lane_ratio` ∈ {0.00, 0.01, 0.02} for Algorithm 4

**Fixed parameters:** province-level data, `top_k = 40` unless the quality-aware seed size is explicitly varied inside Algorithm 3

**Output:** `exp3_parameter_sensitivity.csv`

### Key Results

Curved routing:

| Force Iterations | Crossings | Intrusions | Mean Detour |
|------------------|-----------|------------|-------------|
| 0 | 27 | 4 | 1.007 |
| 5 | 19 | 5 | 1.032 |
| 10 | 19 | 6 | 1.056 |
| 15 | 17 | 6 | 1.072 |

Quality-aware selection (`top_k = 40`):

| Candidate Limit | Pair Count | Coverage | Crossings | Intrusions |
|-----------------|------------|----------|-----------|------------|
| 50 | 44 | 0.768 | 20 | 5 |
| 60 | 48 | 0.789 | 23 | 5 |
| 70 | 50 | 0.797 | 23 | 6 |

Polyline routing:

| Lane Ratio | Crossings | Intrusions | Mean Detour |
|------------|-----------|------------|-------------|
| 0.00 | 5 | 3 | 1.133 |
| 0.01 | 8 | 1 | 1.064 |
| 0.02 | 12 | 3 | 1.091 |

**Observations:**

- `force_iterations = 5` gives most of the crossing reduction without excessive detour.
- `greedy_candidate_limit = 60` is a good default for Algorithm 3 because it substantially improves coverage while staying cleaner than the larger pool.
- `polyline_lane_ratio = 0.01` is the best balanced default for Algorithm 4: lower intrusion than the alternatives with moderate detour.

---

## Experiment 4: Geographic Granularity

**Question:** How do the strategies behave on a much denser municipality-level dataset?

**Independent variables:**

- geographic level ∈ {province, municipality}
- `strategy`

**Fixed parameters:** `top_k = 40`, `force_iterations = 5`, `greedy_candidate_limit = 60`, `polyline_lane_ratio = 0.01`

**Output:** `exp4_granularity.csv`

### Key Results

Province:

- `raw_directed`: `coverage = 0.743`, `crossings = 28`, `intrusions = 7`
- `force_adjusted_curved_od`: `coverage = 0.743`, `crossings = 19`, `intrusions = 5`
- `quality_aware_force_adjusted_od`: `coverage = 0.789`, `crossings = 23`, `intrusions = 5`
- `quality_aware_polyline_od`: `coverage = 0.743`, `crossings = 8`, `intrusions = 1`

Municipality:

- `raw_directed`: `coverage = 0.099`, `crossings = 0`, `intrusions = 118`
- `force_adjusted_curved_od`: `coverage = 0.099`, `crossings = 0`, `intrusions = 114`
- `quality_aware_force_adjusted_od`: `coverage = 0.107`, `crossings = 0`, `intrusions = 114`
- `quality_aware_polyline_od`: `coverage = 0.099`, `crossings = 0`, `intrusions = 94`

**Observations:**

- At municipality scale, the crossing metric becomes less informative because the selected flows are already sparse enough that none of the methods cross.
- The meaningful difference there is node intrusions, where Algorithm 4 performs best.
- Algorithm 3 still gives the best coverage at municipality level.

## Final Takeaway

The experiments show that the design problem is genuinely multi-objective:

- **Algorithm 2** is a clean improvement over the direct baseline when the selected flow set is fixed.
- **Algorithm 3** is the best method when the goal is to preserve more flow information while keeping clutter acceptable.
- **Algorithm 4** is the best method when the goal is to preserve detailed OD structure while minimizing crossings.

This is sufficient for the report requirement: we have a clear input model, explicit quality measures, multiple heuristics, and experiments that identify good default parameter values rather than claiming one globally optimal method.
