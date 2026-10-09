"""
visualize_network.py

Static map visualizations of the three modal arc layers (sea, air, land)
plus a combined overlay. Pacific-centered projection, great-circle arcs,
nodes colored/sized by PPL tier.

Run: python visualize_network.py   (from aps_usarpac/)
Outputs: output/network_sea.png, network_air.png, network_land.png, network_combined.png
"""

import os
import sys
import pickle

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from network.network_builder import load_locations, load_sea_arcs, load_air_arcs, load_land_arcs

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CENTRAL_LON = 165.0

TIER_STYLE = {
    "PPL-1": {"color": "#d62728", "size": 90, "zorder": 5, "label": "PPL-1 Strategic Hub"},
    "PPL-2": {"color": "#ff7f0e", "size": 55, "zorder": 4, "label": "PPL-2 Operational Node"},
    "PPL-3": {"color": "#2ca02c", "size": 35, "zorder": 3, "label": "PPL-3 Contingency Site"},
    "None":  {"color": "#7f7f7f", "size": 18, "zorder": 2, "label": "Non-PPL Node"},
}

MODE_STYLE = {
    "sea":  {"color": "#1f77b4", "lw": 1.2, "alpha": 0.5, "ls": "-",  "label": "Sea (maritime)"},
    "air":  {"color": "#d62728", "lw": 0.6, "alpha": 0.25, "ls": "-", "label": "Air (airlift)"},
    "land": {"color": "#2ca02c", "lw": 2.0, "alpha": 0.7, "ls": "-",  "label": "Land (terrestrial)"},
}

SOLO_MODE_STYLE = {
    "sea":  {"color": "#1f77b4", "lw": 1.5, "alpha": 0.6},
    "air":  {"color": "#d62728", "lw": 0.8, "alpha": 0.35},
    "land": {"color": "#2ca02c", "lw": 2.5, "alpha": 0.8},
}


def _load_nodes():
    locations = load_locations()
    records = []
    for nid, data in locations.items():
        tier = str(data.get("tier", "None")).strip()
        if tier in ("nan", "NaN", ""):
            tier = "None"
        records.append({
            "id": nid,
            "name": data["name"],
            "lat": data["lat"],
            "lon": data["lon"],
            "tier": tier,
            "ppl_eligible": data.get("ppl_eligible", "No"),
        })
    return pd.DataFrame(records)


def _load_arcs():
    loaders = {"sea": load_sea_arcs, "air": load_air_arcs, "land": load_land_arcs}
    arcs = {}
    for mode, loader in loaders.items():
        rows = loader()
        arcs[mode] = [(int(r["from_node"]), int(r["to_node"])) for r in rows]
    return arcs


def _great_circle_points(lon1, lat1, lon2, lat2, n=50):
    """Interpolate n points along a great circle between two lon/lat coords."""
    lon1r, lat1r = np.radians(lon1), np.radians(lat1)
    lon2r, lat2r = np.radians(lon2), np.radians(lat2)

    d = np.arccos(
        np.clip(
            np.sin(lat1r) * np.sin(lat2r)
            + np.cos(lat1r) * np.cos(lat2r) * np.cos(lon2r - lon1r),
            -1.0, 1.0,
        )
    )
    if d < 1e-10:
        return np.array([lon1, lon2]), np.array([lat1, lat2])

    t = np.linspace(0, 1, n)
    A = np.sin((1 - t) * d) / np.sin(d)
    B = np.sin(t * d) / np.sin(d)

    x = A * np.cos(lat1r) * np.cos(lon1r) + B * np.cos(lat2r) * np.cos(lon2r)
    y = A * np.cos(lat1r) * np.sin(lon1r) + B * np.cos(lat2r) * np.sin(lon2r)
    z = A * np.sin(lat1r) + B * np.sin(lat2r)

    lats = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    lons = np.degrees(np.arctan2(y, x))
    return lons, lats


def _draw_arcs(ax, arcs, node_lookup, style, proj):
    for (i, j) in arcs:
        if i not in node_lookup or j not in node_lookup:
            continue
        n1, n2 = node_lookup[i], node_lookup[j]
        lons, lats = _great_circle_points(n1["lon"], n1["lat"], n2["lon"], n2["lat"])
        ax.plot(
            lons, lats,
            transform=ccrs.Geodetic(),
            color=style["color"],
            linewidth=style.get("lw", 1.0),
            alpha=style.get("alpha", 0.5),
            linestyle=style.get("ls", "-"),
            zorder=1,
        )


def _draw_nodes(ax, nodes_df):
    for tier in ["None", "PPL-3", "PPL-2", "PPL-1"]:
        sub = nodes_df[nodes_df["tier"] == tier]
        if sub.empty:
            continue
        s = TIER_STYLE[tier]
        ax.scatter(
            sub["lon"].values, sub["lat"].values,
            transform=ccrs.PlateCarree(),
            s=s["size"], c=s["color"], edgecolors="black", linewidths=0.3,
            zorder=s["zorder"], label=s["label"],
        )


def _add_tier_legend(ax):
    handles = []
    for tier in ["PPL-1", "PPL-2", "PPL-3", "None"]:
        s = TIER_STYLE[tier]
        handles.append(
            mlines.Line2D([], [], marker="o", color="w", markerfacecolor=s["color"],
                          markeredgecolor="black", markeredgewidth=0.3,
                          markersize=np.sqrt(s["size"]) * 0.8, label=s["label"],
                          linestyle="None")
        )
    ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.9)


def _compute_stats(mode, arcs, nodes_df):
    arc_list = arcs[mode]
    unique_arcs = set(arc_list)
    node_ids = set()
    for (i, j) in arc_list:
        node_ids.add(i)
        node_ids.add(j)

    all_node_ids = set(nodes_df["id"].values)
    dest_nodes = set(j for (_, j) in arc_list)
    isolated = all_node_ids - node_ids
    unreachable = all_node_ids - dest_nodes

    return {
        "mode": mode,
        "n_directed_arcs": len(arc_list),
        "n_unique_arcs": len(unique_arcs),
        "n_connected_nodes": len(node_ids),
        "n_total_nodes": len(all_node_ids),
        "n_isolated": len(isolated),
        "isolated_names": sorted(nodes_df[nodes_df["id"].isin(isolated)]["name"].values),
        "n_unreachable": len(unreachable),
        "unreachable_names": sorted(nodes_df[nodes_df["id"].isin(unreachable)]["name"].values),
    }


def make_figure(mode, arcs, nodes_df, node_lookup, title_suffix="", arc_style=None):
    proj = ccrs.PlateCarree(central_longitude=CENTRAL_LON)
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), subplot_kw={"projection": proj})

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="#999999")
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color="#cccccc")

    ax.set_extent([60, 280, -50, 60], crs=ccrs.PlateCarree())

    style = arc_style or SOLO_MODE_STYLE.get(mode, SOLO_MODE_STYLE["air"])
    _draw_arcs(ax, arcs[mode], node_lookup, style, proj)
    _draw_nodes(ax, nodes_df)
    _add_tier_legend(ax)

    stats = _compute_stats(mode, arcs, nodes_df)
    stat_text = (
        f"Directed arcs: {stats['n_directed_arcs']}  |  "
        f"Connected nodes: {stats['n_connected_nodes']}/{stats['n_total_nodes']}  |  "
        f"Isolated: {stats['n_isolated']}  |  "
        f"Unreachable (no inbound): {stats['n_unreachable']}"
    )
    ax.set_title(
        f"{mode.upper()} Modal Arc Layer{title_suffix}\n"
        f"{stat_text}",
        fontsize=11, fontweight="bold",
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    return fig, ax, stats


def make_combined(arcs, nodes_df, node_lookup):
    proj = ccrs.PlateCarree(central_longitude=CENTRAL_LON)
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), subplot_kw={"projection": proj})

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="#999999")
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color="#cccccc")
    ax.set_extent([60, 280, -50, 60], crs=ccrs.PlateCarree())

    for mode in ["air", "sea", "land"]:
        _draw_arcs(ax, arcs[mode], node_lookup, MODE_STYLE[mode], proj)

    _draw_nodes(ax, nodes_df)

    tier_handles = []
    for tier in ["PPL-1", "PPL-2", "PPL-3", "None"]:
        s = TIER_STYLE[tier]
        tier_handles.append(
            mlines.Line2D([], [], marker="o", color="w", markerfacecolor=s["color"],
                          markeredgecolor="black", markeredgewidth=0.3,
                          markersize=np.sqrt(s["size"]) * 0.8, label=s["label"],
                          linestyle="None")
        )
    mode_handles = []
    for mode in ["sea", "air", "land"]:
        ms = MODE_STYLE[mode]
        mode_handles.append(
            mlines.Line2D([], [], color=ms["color"], linewidth=ms["lw"] * 1.5,
                          alpha=min(1.0, ms["alpha"] * 2), label=ms["label"])
        )

    leg1 = ax.legend(handles=tier_handles, loc="lower left", fontsize=7, framealpha=0.9,
                     title="Node Tier", title_fontsize=7)
    ax.add_artist(leg1)
    ax.legend(handles=mode_handles, loc="lower right", fontsize=7, framealpha=0.9,
              title="Transport Mode", title_fontsize=7)

    counts = {m: len(arcs[m]) for m in ["sea", "air", "land"]}
    ax.set_title(
        f"Combined Modal Arc Network\n"
        f"Sea: {counts['sea']} arcs  |  Air: {counts['air']} arcs  |  Land: {counts['land']} arcs  |  "
        f"Total: {sum(counts.values())} directed arcs",
        fontsize=11, fontweight="bold",
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    return fig, ax


def main():
    print("Loading network data...")
    nodes_df = _load_nodes()
    arcs = _load_arcs()
    node_lookup = {row["id"]: row for _, row in nodes_df.iterrows()}

    print(f"Nodes: {len(nodes_df)}")
    for mode in ["sea", "air", "land"]:
        print(f"  {mode}: {len(arcs[mode])} directed arcs")

    all_stats = {}

    for mode in ["sea", "air", "land"]:
        print(f"\nGenerating {mode} map...")
        fig, ax, stats = make_figure(mode, arcs, nodes_df, node_lookup)
        all_stats[mode] = stats

        png_path = os.path.join(OUTPUT_DIR, f"network_{mode}.png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {png_path}")

        fig_path = os.path.join(OUTPUT_DIR, f"network_{mode}_fig.pkl")
        with open(fig_path, "wb") as f:
            pickle.dump(fig, f)
        print(f"  Saved: {fig_path}")

        plt.close(fig)

    print("\nGenerating combined map...")
    fig_c, ax_c = make_combined(arcs, nodes_df, node_lookup)
    png_path = os.path.join(OUTPUT_DIR, "network_combined.png")
    fig_c.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {png_path}")

    fig_path = os.path.join(OUTPUT_DIR, "network_combined_fig.pkl")
    with open(fig_path, "wb") as f:
        pickle.dump(fig_c, f)
    print(f"  Saved: {fig_path}")
    plt.close(fig_c)

    print("\n" + "=" * 70)
    print("NETWORK SUMMARY STATISTICS")
    print("=" * 70)
    for mode in ["sea", "air", "land"]:
        s = all_stats[mode]
        print(f"\n{mode.upper()}:")
        print(f"  Directed arcs: {s['n_directed_arcs']}")
        print(f"  Connected nodes: {s['n_connected_nodes']}/{s['n_total_nodes']}")
        print(f"  Isolated (no arc on this mode): {s['n_isolated']}")
        if s["isolated_names"]:
            for name in s["isolated_names"]:
                print(f"    - {name}")
        print(f"  Unreachable (no inbound arc): {s['n_unreachable']}")
        if s["unreachable_names"]:
            for name in s["unreachable_names"]:
                print(f"    - {name}")

    reachable_any = set()
    for mode in ["sea", "air", "land"]:
        for (_, j) in arcs[mode]:
            reachable_any.add(j)
    all_ids = set(nodes_df["id"].values)
    globally_unreachable = all_ids - reachable_any
    print(f"\nGlobally unreachable (no inbound arc on ANY mode): {len(globally_unreachable)}")
    for nid in sorted(globally_unreachable):
        name = nodes_df[nodes_df["id"] == nid]["name"].values[0]
        print(f"  - node {nid}: {name}")

    print(f"\nAll outputs in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
