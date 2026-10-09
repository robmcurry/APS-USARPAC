"""
build_modal_arcs.py

Generates the three modal arc layers (maritime, air, terrestrial) plus
intermodal transfer capacities from nodes.csv (S/A/L/R ratings, tier) and
config/model_parameters.yaml (modal_capacity block).

Step 2 of the network layer expansion. Writes:
  network/arcs_sea.csv
  network/arcs_air.csv
  network/arcs_land.csv
  network/transfer_capacities.csv
  output/step2_sea_arcs_summary.txt
  output/step2_air_arcs_summary.txt
  output/step2_land_arcs_summary.txt
  output/step2_transfer_summary.txt

Run as a script (python network/build_modal_arcs.py) to (re)generate all
six files. network_builder.py's load_*_arcs()/load_transfer_capacities()
read the CSVs produced here.
"""

import os

import pandas as pd
import yaml
from geopy.distance import geodesic

NETWORK_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(NETWORK_DIR)
NODES_CSV = os.path.join(NETWORK_DIR, "nodes.csv")
CONFIG_YAML = os.path.join(PROJECT_DIR, "config", "model_parameters.yaml")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

# manually defined contiguous terrestrial pairs (node_id_a, node_id_b, notes)
MANUAL_LAND_PAIRS = [
    (1, 5, "Tokyo/Yokosuka - Nagoya, Shinkansen + national highway"),
    (5, 2, "Nagoya - Osaka, Shinkansen + national highway"),
    (2, 4, "Osaka - Fukuoka/Sasebo, Shinkansen + national highway"),
    (1, 3, "Tokyo/Yokosuka - Sapporo, rail via Seikan tunnel, reduced capacity"),
    (6, 7, "Sydney - Melbourne, Hume Highway + freight rail"),
    (7, 10, "Melbourne - Adelaide, Western Highway + freight rail"),
    (8, 6, "Brisbane - Sydney, Pacific Highway + freight rail"),
    (10, 11, "Adelaide - Darwin, Stuart Highway, long route reduced capacity"),
    (43, 34, "Bangkok - Ho Chi Minh City, highway network"),
    (34, 37, "Ho Chi Minh City - Phnom Penh, highway"),
    (37, 36, "Phnom Penh - Vientiane, highway"),
    (36, 35, "Vientiane - Hanoi, highway"),
    (35, 39, "Hanoi - Da Nang, highway + rail"),
    (39, 34, "Da Nang - Ho Chi Minh City, highway + rail"),
    (43, 44, "Bangkok - Yangon, highway"),
]

# arcs where the land rating is capped regardless of the endpoints' actual
# L ratings (see TASK 5 special cases)
LAND_RATING_CAPS = {
    frozenset((1, 3)): 2,   # Seikan tunnel - rail only, no heavy convoy
    frozenset((10, 11)): 1,  # Stuart Highway - long, partially unsealed
}

TRANSFER_COST = {
    "sea_to_air": 3.0,
    "sea_to_land": 0.5,
    "air_to_land": 0.8,
    "land_to_sea": 0.7,
    "land_to_air": 1.2,
    "air_to_sea": 2.0,
}

TRANSFER_KAPPA = 500000  # person-days per rating unit (placeholder for SME calibration)


def load_nodes() -> pd.DataFrame:
    df = pd.read_csv(NODES_CSV)
    df["tier"] = df["tier"].astype(str).str.strip()
    df.loc[df["tier"].isin(["nan", "None", ""]), "tier"] = "None"
    return df


def load_modal_config() -> dict:
    with open(CONFIG_YAML) as f:
        cfg = yaml.safe_load(f)
    return cfg["modal_capacity"]


def _distance_km(node_a: dict, node_b: dict) -> float:
    return geodesic(
        (node_a["Latitude"], node_a["Longitude"]),
        (node_b["Latitude"], node_b["Longitude"]),
    ).km


def build_sea_arcs(nodes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    m = cfg["maritime"]
    rows = []
    node_list = nodes.to_dict("records")
    for ni in node_list:
        for nj in node_list:
            if ni["Node ID"] == nj["Node ID"]:
                continue
            if ni["S"] < m["min_rating"] or nj["S"] < m["min_rating"]:
                continue
            dist = _distance_km(ni, nj)
            if dist > m["max_distance_km"]:
                continue

            tier = ni["tier"]
            if tier in ("None", "PPL-3"):
                # origin has no maritime assets - arc retained for
                # transshipment topology but contributes zero capacity
                # (see TASK 8 check 3: U_sea must be 0 for PPL-3 origins)
                u_food = 0
                u_water = 0
                vessel_type = m["vessel_type"].get(tier, "none")
                n_assets = m["assets"].get(tier, 0)
            else:
                vessel_type = m["vessel_type"][tier]
                n_assets = m["assets"][tier]
                total_mt = n_assets * m["discharge_mt_per_window"][vessel_type]
                u_food = total_mt * m["pd_per_mt"]["food"]
                u_water = total_mt * m["pd_per_mt"]["water"]

            rows.append({
                "from_node": ni["Node ID"],
                "to_node": nj["Node ID"],
                "distance_km": round(dist, 1),
                "U_food": u_food,
                "U_water": u_water,
                "origin_tier": tier,
                "origin_sea_rating": ni["S"],
                "dest_sea_rating": nj["S"],
                "vessel_type": vessel_type,
                "n_assets": n_assets,
            })
    return pd.DataFrame(rows)


def build_air_arcs(nodes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    a = cfg["air"]
    rows = []
    node_list = nodes.to_dict("records")
    for ni in node_list:
        for nj in node_list:
            if ni["Node ID"] == nj["Node ID"]:
                continue
            if ni["A"] < a["min_rating"] or nj["A"] < a["min_rating"]:
                continue
            dist = _distance_km(ni, nj)
            if dist > a["max_distance_km"]:
                continue

            tier = ni["tier"]
            if tier == "None":
                fallback = min(ni["A"], nj["A"]) * 100000
                u_food = fallback
                u_water = fallback
                aircraft_type = "none"
                n_assets = 0
                n_sorties = 0
            else:
                aircraft_type = a["aircraft_type"][tier]
                n_assets = a["assets"][tier]
                n_sorties = a["sorties_per_window"][aircraft_type]
                total_mt = n_assets * a["payload_mt"][aircraft_type] * n_sorties
                u_food = total_mt * a["pd_per_mt"]["food"]
                u_water = total_mt * a["pd_per_mt"]["water"]

            rows.append({
                "from_node": ni["Node ID"],
                "to_node": nj["Node ID"],
                "distance_km": round(dist, 1),
                "U_food": u_food,
                "U_water": u_water,
                "origin_tier": tier,
                "origin_air_rating": ni["A"],
                "dest_air_rating": nj["A"],
                "aircraft_type": aircraft_type,
                "n_assets": n_assets,
                "n_sorties_per_window": n_sorties,
            })
    return pd.DataFrame(rows)


def build_land_arcs(nodes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    t = cfg["terrestrial"]
    nodes_by_id = nodes.set_index("Node ID").to_dict("index")
    rows = []
    for a, b, notes in MANUAL_LAND_PAIRS:
        na, nb = nodes_by_id[a], nodes_by_id[b]
        weak_L = min(na["L"], nb["L"])
        weak_R = min(na["R"], nb["R"])

        cap = LAND_RATING_CAPS.get(frozenset((a, b)))
        if cap is not None:
            weak_L = min(weak_L, cap)

        road = t["road_mt_per_day"].get(weak_L, 0)
        rail = t["rail_mt_per_day"].get(weak_R, 0)
        total_mt_per_day = road + rail
        total_mt = total_mt_per_day * t["window_days"]
        u_food = total_mt * t["pd_per_mt"]["food"]
        u_water = total_mt * t["pd_per_mt"]["water"]

        for (i, j) in [(a, b), (b, a)]:
            dist = _distance_km(nodes_by_id[i], nodes_by_id[j])
            rows.append({
                "from_node": i,
                "to_node": j,
                "distance_km": round(dist, 1),
                "U_food": u_food,
                "U_water": u_water,
                "weak_land_rating": weak_L,
                "weak_rail_rating": weak_R,
                "total_mt_per_day": total_mt_per_day,
                "notes": notes,
            })
    return pd.DataFrame(rows)


def get_land_active_nodes() -> set:
    nodes = set()
    for a, b, _ in MANUAL_LAND_PAIRS:
        nodes.add(a)
        nodes.add(b)
    return nodes


def build_transfer_capacities(nodes: pd.DataFrame) -> pd.DataFrame:
    land_nodes = get_land_active_nodes()
    rows = []
    for _, n in nodes.iterrows():
        has_sea = bool(n["S"] >= 2)
        has_air = bool(n["A"] >= 2)
        has_land = int(n["Node ID"]) in land_nodes

        row = {
            "node_id": int(n["Node ID"]),
            "node_name": n["Node Name"],
            "has_sea": has_sea,
            "has_air": has_air,
            "has_land": has_land,
        }

        mode_pairs = [
            ("sea_to_air", has_sea, has_air, n["S"], n["A"]),
            ("sea_to_land", has_sea, has_land, n["S"], n["L"]),
            ("air_to_land", has_air, has_land, n["A"], n["L"]),
            ("land_to_sea", has_land, has_sea, n["L"], n["S"]),
            ("land_to_air", has_land, has_air, n["L"], n["A"]),
            ("air_to_sea", has_air, has_sea, n["A"], n["S"]),
        ]
        for name, active1, active2, r1, r2 in mode_pairs:
            if active1 and active2:
                row[f"T_{name}"] = TRANSFER_KAPPA * min(r1, r2)
                row[f"cost_{name}"] = TRANSFER_COST[name]
            else:
                row[f"T_{name}"] = 0
                row[f"cost_{name}"] = None

        rows.append(row)
    return pd.DataFrame(rows)


def _write_sea_summary(df: pd.DataFrame, path: str) -> None:
    with open(path, "w") as f:
        f.write("STEP 2 - TASK 3: maritime arc set (arcs_sea.csv) summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total directed arcs: {len(df)}\n\n")
        f.write("Arcs by origin tier:\n")
        for tier, count in df["origin_tier"].value_counts().sort_index().items():
            f.write(f"  {tier}: {count}\n")
        f.write("\nDistance (km) distribution:\n")
        f.write(f"  min:    {df['distance_km'].min():.1f}\n")
        f.write(f"  median: {df['distance_km'].median():.1f}\n")
        f.write(f"  max:    {df['distance_km'].max():.1f}\n")


def _write_air_summary(df: pd.DataFrame, path: str) -> None:
    with open(path, "w") as f:
        f.write("STEP 2 - TASK 4: air arc set (arcs_air.csv) summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total directed arcs: {len(df)}\n\n")
        f.write("Arcs by origin tier:\n")
        for tier, count in df["origin_tier"].value_counts().sort_index().items():
            f.write(f"  {tier}: {count}\n")
        f.write("\nDistance (km) distribution:\n")
        f.write(f"  min:    {df['distance_km'].min():.1f}\n")
        f.write(f"  median: {df['distance_km'].median():.1f}\n")
        f.write(f"  max:    {df['distance_km'].max():.1f}\n")


def _write_land_summary(df: pd.DataFrame, path: str) -> None:
    with open(path, "w") as f:
        f.write("STEP 2 - TASK 5: terrestrial arc set (arcs_land.csv) summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total directed arcs: {len(df)} (expected: 30, from 15 manual pairs)\n\n")
        f.write("U_food range (person-days):\n")
        f.write(f"  min: {df['U_food'].min():.0f}\n")
        f.write(f"  max: {df['U_food'].max():.0f}\n\n")
        f.write("U_water range (person-days):\n")
        f.write(f"  min: {df['U_water'].min():.0f}\n")
        f.write(f"  max: {df['U_water'].max():.0f}\n")


def _write_transfer_summary(df: pd.DataFrame, path: str) -> None:
    n_modes = df[["has_sea", "has_air", "has_land"]].sum(axis=1)
    with open(path, "w") as f:
        f.write("STEP 2 - TASK 6: transfer capacities summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Nodes with all three modes (sea+air+land): {(n_modes == 3).sum()}\n")
        f.write(f"Nodes with exactly two modes:              {(n_modes == 2).sum()}\n")
        f.write(f"Nodes with exactly one mode:                {(n_modes == 1).sum()}\n")
        f.write(f"Nodes with zero modes active:               {(n_modes == 0).sum()}\n")
        f.write("\nNodes with all three modes:\n")
        for _, r in df[n_modes == 3].iterrows():
            f.write(f"  {r['node_id']:>3} {r['node_name']}\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nodes = load_nodes()
    cfg = load_modal_config()

    sea = build_sea_arcs(nodes, cfg)
    air = build_air_arcs(nodes, cfg)
    land = build_land_arcs(nodes, cfg)
    transfer = build_transfer_capacities(nodes)

    sea.to_csv(os.path.join(NETWORK_DIR, "arcs_sea.csv"), index=False)
    air.to_csv(os.path.join(NETWORK_DIR, "arcs_air.csv"), index=False)
    land.to_csv(os.path.join(NETWORK_DIR, "arcs_land.csv"), index=False)
    transfer.to_csv(os.path.join(NETWORK_DIR, "transfer_capacities.csv"), index=False)

    _write_sea_summary(sea, os.path.join(OUTPUT_DIR, "step2_sea_arcs_summary.txt"))
    _write_air_summary(air, os.path.join(OUTPUT_DIR, "step2_air_arcs_summary.txt"))
    _write_land_summary(land, os.path.join(OUTPUT_DIR, "step2_land_arcs_summary.txt"))
    _write_transfer_summary(transfer, os.path.join(OUTPUT_DIR, "step2_transfer_summary.txt"))

    print(f"sea arcs:      {len(sea)}")
    print(f"air arcs:      {len(air)}")
    print(f"land arcs:     {len(land)}")
    print(f"transfer rows: {len(transfer)}")


if __name__ == "__main__":
    main()
