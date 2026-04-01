# Measures

## Independent Variables

- **source_order**: The sequence in which source trees are constructed in the obstacle-aware strategy. Determines which trees get obstacle-free routing and which must avoid earlier trees.
- **top_k**: The number of highest-flow region pairs selected for visualization. Controls how dense the flow map is.
- **geographic level**: The administrative granularity of the regions — provinces (12 regions) or municipalities (342 regions). Affects how much spatial freedom the algorithm has to place merge points.
- **spiral_turns**: The angle restriction parameter (α) controlling how tightly spiral merge candidates wrap around the source. Lower values produce tighter, more direct trees; higher values allow wider spirals with more merging freedom.

## Dependent Variables (Quality Measures)

Both measures are drawn from the flow tree quality criteria on lecture slide 54.

## Crossing Count

Number of pairs of flow lines whose geometries intersect. Carriers sharing an endpoint (same source or target region) are excluded — those touches are structural, not crossings. Measured using Shapely's `crosses()` on the rendered carrier geometries.

Lower is better. 0 is ideal.

## Total Tree Length

Sum of Euclidean lengths of all carrier geometries. Straight-line carriers (raw directed) give the minimum; tree strategies add length through merging detours. Reported alongside **mean detour ratio** (carrier length / straight-line distance, averaged over all carriers) as a normalized variant that is comparable across different top_k values and geographic levels.

Lower is better. A mean detour ratio of 1.0 means all paths are straight lines.
