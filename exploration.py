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
#     "scipy",
#     "networkx",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    from pathlib import Path

    DATA = Path("data")
    IMG = Path("__marimo__")
    IMG.mkdir(exist_ok=True)
    return DATA, IMG, gpd, mcolors, mo, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # Multi-Source Flow Map — Data Exploration
    """)
    return


@app.cell
def _(DATA, gpd, mo):
    provinces = gpd.read_file(DATA / "provinces.geojson")
    provinces = provinces.rename(columns={"statcode": "code", "statnaam": "name"})
    provinces = provinces[["code", "name", "geometry"]]
    mo.md(f"## Provinces\nLoaded **{len(provinces)}** provinces")
    return (provinces,)


@app.cell
def _(provinces):
    provinces
    return


@app.cell
def _(DATA, gpd, mo):
    municipalities = gpd.read_file(DATA / "municipalities.geojson")
    municipalities = municipalities.rename(columns={"statcode": "code", "statnaam": "name"})
    municipalities = municipalities[["code", "name", "geometry"]]
    mo.md(f"## Municipalities\nLoaded **{len(municipalities)}** municipalities")
    return (municipalities,)


@app.cell
def _(municipalities):
    municipalities.head(10)
    return


@app.cell
def _(DATA, mo, pd):
    obs = pd.read_csv(DATA / "Observations.csv", sep=";")
    mo.md(f"## Flow observations\nLoaded **{len(obs):,}** rows")
    return (obs,)


@app.cell
def _(obs):
    obs.head(10)
    return


@app.cell
def _(DATA, mo, pd):
    vestiging_codes = pd.read_csv(DATA / "RegioVanVestigingCodes.csv", sep=";")
    vertrek_codes = pd.read_csv(DATA / "RegioVanVertrekCodes.csv", sep=";")
    mo.md(
        f"### Region codes\n"
        f"- Vestiging (destination): **{len(vestiging_codes)}** codes\n"
        f"- Vertrek (origin): **{len(vertrek_codes)}** codes"
    )
    return (vestiging_codes,)


@app.cell
def _(mo, vestiging_codes):
    _groups = vestiging_codes["DimensionGroupId"].value_counts()
    mo.md(
        "### Region types in the data\n"
        + "\n".join(f"- **{k}**: {v} codes" for k, v in _groups.items())
    )
    return


@app.cell
def _(mo, obs, pd):
    province_flows = obs[
        (obs["RegioVanVestiging"].str.startswith("PV"))
        & (obs["RegioVanVertrek"].str.startswith("PV"))
        & (obs["Perioden"] == "2024JJ00")
    ].copy()

    province_flows = province_flows[
        province_flows["RegioVanVestiging"] != province_flows["RegioVanVertrek"]
    ]

    province_flows = province_flows[
        ["RegioVanVestiging", "RegioVanVertrek", "Value"]
    ].rename(
        columns={
            "RegioVanVestiging": "destination",
            "RegioVanVertrek": "origin",
            "Value": "flow",
        }
    )

    province_flows["flow"] = pd.to_numeric(province_flows["flow"], errors="coerce")

    mo.md(
        f"## Province-level flows (2024)\n"
        f"**{len(province_flows)}** origin-destination pairs "
        f"(excluding self-flows)\n\n"
        f"Total people moved between provinces: **{province_flows['flow'].sum():,.0f}**"
    )
    return (province_flows,)


@app.cell
def _(province_flows):
    province_flows.sort_values("flow", ascending=False).head(15)
    return


@app.cell
def _(mo, province_flows):
    od = province_flows.pivot(index="origin", columns="destination", values="flow").fillna(0)
    mo.md(
        f"## Origin-Destination matrix\n"
        f"{od.shape[0]} origins x {od.shape[1]} destinations\n\n"
        f"Min flow: **{od.values[od.values > 0].min():,.0f}** | "
        f"Max flow: **{od.values.max():,.0f}** | "
        f"Mean flow: **{od.values[od.values > 0].mean():,.0f}**"
    )
    return (od,)


@app.cell
def _(od):
    od
    return


@app.cell
def _(IMG, mcolors, od, plt):
    _fig, _ax = plt.subplots(figsize=(10, 8))
    _im = _ax.imshow(od.values, cmap="YlOrRd", norm=mcolors.LogNorm())
    _ax.set_xticks(range(len(od.columns)))
    _ax.set_xticklabels(od.columns, rotation=45, ha="right")
    _ax.set_yticks(range(len(od.index)))
    _ax.set_yticklabels(od.index)
    _ax.set_xlabel("Destination")
    _ax.set_ylabel("Origin")
    _ax.set_title("Inter-province migration (2024) — log scale")
    _fig.colorbar(_im, ax=_ax, label="Number of people")
    _fig.tight_layout()
    _fig.savefig(IMG / "01_heatmap.png", dpi=150, bbox_inches="tight")
    _fig
    return


@app.cell
def _(IMG, plt, provinces):
    _fig, _ax = plt.subplots(figsize=(10, 12))
    provinces.plot(ax=_ax, edgecolor="black", facecolor="lightblue", linewidth=0.8)
    for _idx, _row in provinces.iterrows():
        _c = _row.geometry.centroid
        _ax.annotate(
            _row["name"],
            xy=(_c.x, _c.y),
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
        )
    _ax.set_title("Dutch provinces")
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig.savefig(IMG / "02_provinces_map.png", dpi=150, bbox_inches="tight")
    _fig
    return


@app.cell
def _(IMG, plt, province_flows, provinces):
    _centroids = provinces.set_index("code").geometry.centroid

    _fig, _ax = plt.subplots(figsize=(10, 12))
    provinces.plot(ax=_ax, edgecolor="black", facecolor="#f0f0f0", linewidth=0.8)

    _max_flow = province_flows["flow"].max()
    for _, _row in province_flows.iterrows():
        _oc = _centroids.get(_row["origin"])
        _dc = _centroids.get(_row["destination"])
        if _oc is None or _dc is None:
            continue
        _w = 0.5 + 4.5 * (_row["flow"] / _max_flow)
        _a = 0.3 + 0.7 * (_row["flow"] / _max_flow)
        _ax.plot(
            [_oc.x, _dc.x],
            [_oc.y, _dc.y],
            color="steelblue",
            linewidth=_w,
            alpha=_a,
        )

    for _, _row in provinces.iterrows():
        _c = _row.geometry.centroid
        _ax.annotate(
            _row["name"],
            xy=(_c.x, _c.y),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        )

    _ax.set_title("All inter-province flows (2024) — line width ~ flow volume")
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig.savefig(IMG / "03_all_flows.png", dpi=150, bbox_inches="tight")
    _fig
    return


@app.cell
def _(IMG, mo, plt, province_flows, provinces):
    _centroids = provinces.set_index("code").geometry.centroid

    # Compute net flows
    _pairs = {}
    for _, _row in province_flows.iterrows():
        _key = tuple(sorted([_row["origin"], _row["destination"]]))
        if _key not in _pairs:
            _pairs[_key] = {"ab": 0, "ba": 0, "a": _key[0], "b": _key[1]}
        if _row["origin"] == _key[0]:
            _pairs[_key]["ab"] += _row["flow"]
        else:
            _pairs[_key]["ba"] += _row["flow"]

    net_flows = []
    for _key, _val in _pairs.items():
        _net = _val["ab"] - _val["ba"]
        if _net > 0:
            net_flows.append({"origin": _val["a"], "destination": _val["b"], "net": _net, "total": _val["ab"] + _val["ba"]})
        else:
            net_flows.append({"origin": _val["b"], "destination": _val["a"], "net": abs(_net), "total": _val["ab"] + _val["ba"]})

    _fig, _ax = plt.subplots(figsize=(10, 12))
    provinces.plot(ax=_ax, edgecolor="black", facecolor="#f0f0f0", linewidth=0.8)

    _max_net = max(f["net"] for f in net_flows)
    for _f in net_flows:
        _oc = _centroids.get(_f["origin"])
        _dc = _centroids.get(_f["destination"])
        if _oc is None or _dc is None:
            continue
        _w = 0.5 + 5 * (_f["net"] / _max_net)
        _a = 0.3 + 0.7 * (_f["net"] / _max_net)
        _ax.annotate(
            "",
            xy=(_dc.x, _dc.y),
            xytext=(_oc.x, _oc.y),
            arrowprops=dict(
                arrowstyle="-|>",
                color="steelblue",
                lw=_w,
                alpha=_a,
                mutation_scale=10 + 15 * (_f["net"] / _max_net),
            ),
        )

    for _, _row in provinces.iterrows():
        _c = _row.geometry.centroid
        _ax.annotate(
            _row["name"],
            xy=(_c.x, _c.y),
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

    _ax.set_title("Net inter-province migration (2024) — arrows show dominant direction")
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig.savefig(IMG / "04_net_flows.png", dpi=150, bbox_inches="tight")

    mo.md("## Net flow visualization\n66 pairs, arrows show dominant migration direction, width ~ net flow.")
    return


@app.cell
def _(DATA, IMG, municipalities, mo, obs, pd, plt):
    # Groningen municipality codes (province group GMPV20)
    _vest_codes = pd.read_csv(DATA / "RegioVanVestigingCodes.csv", sep=";")
    _gro_codes = set(
        _vest_codes[_vest_codes["DimensionGroupId"] == "GMPV20"]["Identifier"]
    )

    # Filter to municipalities that exist in the GeoJSON (current ones)
    gro_munis = municipalities[municipalities["code"].isin(_gro_codes)].copy()
    _gro_geo_codes = set(gro_munis["code"])

    # Get inter-municipality flows within Groningen, 2024
    gro_muni_flows = obs[
        (obs["Perioden"] == "2024JJ00")
        & obs["RegioVanVestiging"].isin(_gro_geo_codes)
        & obs["RegioVanVertrek"].isin(_gro_geo_codes)
        & (obs["RegioVanVestiging"] != obs["RegioVanVertrek"])
    ][["RegioVanVestiging", "RegioVanVertrek", "Value"]].rename(
        columns={
            "RegioVanVestiging": "destination",
            "RegioVanVertrek": "origin",
            "Value": "flow",
        }
    )
    gro_muni_flows["flow"] = pd.to_numeric(gro_muni_flows["flow"], errors="coerce")

    mo.md(
        f"## Groningen municipalities\n"
        f"**{len(gro_munis)}** municipalities, "
        f"**{len(gro_muni_flows)}** directed flow pairs"
    )
    return gro_muni_flows, gro_munis


@app.cell
def _(IMG, gro_muni_flows, gro_munis, plt):
    _centroids = gro_munis.set_index("code").geometry.centroid

    _fig, _ax = plt.subplots(figsize=(10, 10))
    gro_munis.plot(ax=_ax, edgecolor="black", facecolor="#f0f0f0", linewidth=0.8)

    _max_flow = gro_muni_flows["flow"].max()
    for _, _row in gro_muni_flows.iterrows():
        _oc = _centroids.get(_row["origin"])
        _dc = _centroids.get(_row["destination"])
        if _oc is None or _dc is None:
            continue
        _w = 0.5 + 4.5 * (_row["flow"] / _max_flow)
        _a = 0.3 + 0.7 * (_row["flow"] / _max_flow)
        _ax.plot(
            [_oc.x, _dc.x],
            [_oc.y, _dc.y],
            color="steelblue",
            linewidth=_w,
            alpha=_a,
        )

    for _, _row in gro_munis.iterrows():
        _c = _row.geometry.centroid
        _ax.annotate(
            _row["name"],
            xy=(_c.x, _c.y),
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9),
        )

    _ax.set_title("Groningen — inter-municipality flows (2024)")
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig.savefig(IMG / "05_groningen_munis_all.png", dpi=150, bbox_inches="tight")
    _fig
    return


@app.cell
def _(IMG, gro_muni_flows, gro_munis, mo, plt):
    _centroids = gro_munis.set_index("code").geometry.centroid

    # Compute net flows
    _pairs = {}
    for _, _row in gro_muni_flows.iterrows():
        _key = tuple(sorted([_row["origin"], _row["destination"]]))
        if _key not in _pairs:
            _pairs[_key] = {"ab": 0, "ba": 0, "a": _key[0], "b": _key[1]}
        if _row["origin"] == _key[0]:
            _pairs[_key]["ab"] += _row["flow"]
        else:
            _pairs[_key]["ba"] += _row["flow"]

    _net_flows = []
    for _key, _val in _pairs.items():
        _net = _val["ab"] - _val["ba"]
        if _net > 0:
            _net_flows.append({"origin": _val["a"], "destination": _val["b"], "net": _net, "total": _val["ab"] + _val["ba"]})
        else:
            _net_flows.append({"origin": _val["b"], "destination": _val["a"], "net": abs(_net), "total": _val["ab"] + _val["ba"]})

    _fig, _ax = plt.subplots(figsize=(10, 10))
    gro_munis.plot(ax=_ax, edgecolor="black", facecolor="#f0f0f0", linewidth=0.8)

    _max_net = max(_f["net"] for _f in _net_flows)
    for _f in _net_flows:
        _oc = _centroids.get(_f["origin"])
        _dc = _centroids.get(_f["destination"])
        if _oc is None or _dc is None:
            continue
        _w = 0.5 + 5 * (_f["net"] / _max_net)
        _a = 0.3 + 0.7 * (_f["net"] / _max_net)
        _ax.annotate(
            "",
            xy=(_dc.x, _dc.y),
            xytext=(_oc.x, _oc.y),
            arrowprops=dict(
                arrowstyle="-|>",
                color="steelblue",
                lw=_w,
                alpha=_a,
                mutation_scale=10 + 15 * (_f["net"] / _max_net),
            ),
        )

    for _, _row in gro_munis.iterrows():
        _c = _row.geometry.centroid
        _ax.annotate(
            _row["name"],
            xy=(_c.x, _c.y),
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9),
        )

    _ax.set_title("Groningen — net inter-municipality migration (2024)")
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig.savefig(IMG / "06_groningen_munis_net.png", dpi=150, bbox_inches="tight")

    mo.md("## Groningen net municipality flows\nArrows show dominant migration direction.")
    return


@app.cell
def _(mo, province_flows):
    # Find the source with the highest total outflow
    _outflows = province_flows.groupby("origin")["flow"].sum().sort_values(ascending=False)
    mo.md(
        "## Single-source flow tree\n"
        "### Total outflow by province\n"
        + "\n".join(f"- **{code}**: {val:,.0f}" for code, val in _outflows.items())
    )
    return


@app.cell
def _(np, province_flows, provinces):
    import networkx as nx
    from scipy.spatial import Delaunay

    # Pick source: highest total outflow
    source_code = province_flows.groupby("origin")["flow"].sum().idxmax()

    # Build centroid lookup
    centroids = {}
    for _, row in provinces.iterrows():
        c = row.geometry.centroid
        centroids[row["code"]] = np.array([c.x, c.y])

    # All province codes that appear in the flow data
    all_codes = sorted(centroids.keys())

    # Get flows from this source to all destinations
    source_flows = province_flows[province_flows["origin"] == source_code].set_index("destination")["flow"].to_dict()

    # Build complete graph on province centroids, weighted by Euclidean distance
    G = nx.Graph()
    for code in all_codes:
        G.add_node(code, pos=centroids[code])
    for i, c1 in enumerate(all_codes):
        for c2 in all_codes[i+1:]:
            dist = np.linalg.norm(centroids[c1] - centroids[c2])
            G.add_edge(c1, c2, weight=dist)

    # MST
    mst = nx.minimum_spanning_tree(G)

    # Root the MST at the source — get parent/children via BFS
    tree_edges = list(nx.bfs_edges(mst, source_code))
    children = {code: [] for code in all_codes}
    parent = {}
    for u, v in tree_edges:
        children[u].append(v)
        parent[v] = u

    # Compute subtree flow for each edge (sum of flows to all descendants)
    def subtree_flow(node):
        """Return total flow from source to this node and all its descendants."""
        total = source_flows.get(node, 0)
        for child in children[node]:
            total += subtree_flow(child)
        return total

    edge_flows = {}
    for u, v in tree_edges:
        edge_flows[(u, v)] = subtree_flow(v)

    return source_code, centroids, all_codes, source_flows, mst, tree_edges, children, edge_flows


@app.cell
def _(IMG, centroids, edge_flows, mst, np, plt, province_flows, provinces, source_code, source_flows, tree_edges):
    _names = provinces.set_index("code")["name"].to_dict()

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(20, 12))

    # --- Left: naive straight-line flows from source ---
    provinces.plot(ax=_ax1, edgecolor="black", facecolor="#f0f0f0", linewidth=0.8)

    _max_flow = max(source_flows.values()) if source_flows else 1
    _sc = centroids[source_code]
    for _dest, _fl in source_flows.items():
        _dc = centroids.get(_dest)
        if _dc is None:
            continue
        _w = 1 + 6 * (_fl / _max_flow)
        _a = 0.4 + 0.6 * (_fl / _max_flow)
        _ax1.annotate(
            "",
            xy=(_dc[0], _dc[1]),
            xytext=(_sc[0], _sc[1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color="steelblue",
                lw=_w,
                alpha=_a,
                mutation_scale=10 + 10 * (_fl / _max_flow),
            ),
        )

    for _code, _pos in centroids.items():
        _style = dict(boxstyle="round,pad=0.3", fc="gold", ec="black", alpha=0.9) if _code == source_code else dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9)
        _ax1.annotate(
            _names.get(_code, _code),
            xy=(_pos[0], _pos[1]),
            ha="center", va="center", fontsize=7, fontweight="bold",
            bbox=_style,
        )

    _ax1.set_title(f"Naive: straight-line flows from {_names.get(source_code, source_code)}", fontsize=11)
    _ax1.set_axis_off()

    # --- Right: MST-routed flow tree ---
    provinces.plot(ax=_ax2, edgecolor="black", facecolor="#f0f0f0", linewidth=0.8)

    _max_tree_flow = max(edge_flows.values()) if edge_flows else 1
    for (_u, _v), _fl in edge_flows.items():
        _uc, _vc = centroids[_u], centroids[_v]
        _w = 1 + 8 * (_fl / _max_tree_flow)
        _a = 0.5 + 0.5 * (_fl / _max_tree_flow)
        _ax2.plot(
            [_uc[0], _vc[0]],
            [_uc[1], _vc[1]],
            color="steelblue",
            linewidth=_w,
            alpha=_a,
            solid_capstyle="round",
        )

    # Draw small arrows at midpoints to show direction
    for (_u, _v), _fl in edge_flows.items():
        _uc, _vc = centroids[_u], centroids[_v]
        _mid = (_uc + _vc) / 2
        _dir = _vc - _uc
        _dir = _dir / (np.linalg.norm(_dir) + 1e-9) * 1000  # small arrow
        _ax2.annotate(
            "",
            xy=(_mid[0] + _dir[0], _mid[1] + _dir[1]),
            xytext=(_mid[0], _mid[1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color="crimson",
                lw=1.2,
                alpha=0.8,
                mutation_scale=10,
            ),
        )

    for _code, _pos in centroids.items():
        _style = dict(boxstyle="round,pad=0.3", fc="gold", ec="black", alpha=0.9) if _code == source_code else dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9)
        _ax2.annotate(
            _names.get(_code, _code),
            xy=(_pos[0], _pos[1]),
            ha="center", va="center", fontsize=7, fontweight="bold",
            bbox=_style,
        )

    _ax2.set_title(f"Flow tree (MST): flows from {_names.get(source_code, source_code)}", fontsize=11)
    _ax2.set_axis_off()

    _fig.suptitle(f"Single-source flow map — {_names.get(source_code, source_code)}", fontsize=14, fontweight="bold")
    _fig.tight_layout()
    _fig.savefig(IMG / "07_single_source_tree.png", dpi=150, bbox_inches="tight")
    _fig
    return


if __name__ == "__main__":
    app.run()
