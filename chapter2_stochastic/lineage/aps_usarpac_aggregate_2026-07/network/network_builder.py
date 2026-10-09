"""
network_builder.py

Network layer of the DST pipeline. Single source of truth for the node
table and the candidate graph that scenarios/ and model/ build on top of.

This module consolidates the locations-loading and graph-construction logic
shared by analysis/sensitivity_runner.py, plus a standalone ppl-eligibility
filter mirroring the inline one used by model/input_builder.py.
"""

import os
from typing import Dict, List

import networkx as nx
import pandas as pd


def load_locations(nodes_csv_path: str = None) -> Dict[int, Dict]:
    """
    loads the 50-node network from nodes.csv
    returns a dict keyed by node_id with all node attributes
    this is the single source of truth for network data in the model
    """
    # default path relative to this file
    if nodes_csv_path is None:
        nodes_csv_path = os.path.join(os.path.dirname(__file__), "nodes.csv")

    df = pd.read_csv(nodes_csv_path)

    # loads node id name lat lon population country country_iso region hub_type
    # ppl_eligible and tier for each node
    locations = {}
    for _, row in df.iterrows():
        node_id = int(row["Node ID"])
        locations[node_id] = {
            "name": row["Node Name"],
            "lat": float(row["Latitude"]),
            "lon": float(row["Longitude"]),
            "pop": float(row["Population"]),
            "country": row["Country"],
            "country_iso": str(row["Country ISO"]).strip(),
            "region": row["Region"],
            "hub_type": str(row["hub_type"]).strip().lower(),
            "ppl_eligible": str(row["ppl_eligible"]).strip(),
            "tier": str(row["tier"]).strip(),
            "S": int(row["S"]) if str(row.get("S", "")).strip() not in ("", "nan") else 0,
            "A": int(row["A"]) if str(row.get("A", "")).strip() not in ("", "nan") else 0,
            "L": int(row["L"]) if str(row.get("L", "")).strip() not in ("", "nan") else 0,
            "R": int(row["R"]) if str(row.get("R", "")).strip() not in ("", "nan") else 0,
        }

    # returns locations dict in the same format expected by all downstream functions
    return locations


def build_graph(locations: Dict[int, Dict]) -> nx.Graph:
    """
    builds a fully connected networkx graph over all 50 nodes
    every node pair gets an edge - arc pruning by distance happens later in input_builder
    returns the graph object passed to scenario_generator and input_builder
    """
    G = nx.Graph()
    for i in locations:
        G.add_node(i)

    for i in locations:
        for j in locations:
            if i != j:
                G.add_edge(i, j)

    return G


def get_ppl_nodes(locations: Dict[int, Dict]) -> List[int]:
    """
    returns list of node ids where ppl_eligible is Y or Y(C)
    these are the only nodes that can be selected as prepositioning sites
    """
    return [
        i for i in locations
        if locations[i].get("ppl_eligible", "No") in ("Y", "Y(C)")
    ]


def _load_arc_csv(filename: str) -> List[Dict]:
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{filename} not found at {path}. "
            "Run `python network/build_modal_arcs.py` to generate the modal "
            "arc CSVs from nodes.csv and config/model_parameters.yaml."
        )
    # keep_default_na=False: origin_tier holds the literal string "None" for
    # non-PPL nodes, which pandas would otherwise silently parse as NaN
    return pd.read_csv(path, keep_default_na=False, na_values=[""]).to_dict("records")


def load_sea_arcs() -> List[Dict]:
    """
    loads network/arcs_sea.csv
    returns a list of dicts, one per directed maritime arc
    """
    return _load_arc_csv("arcs_sea.csv")


def load_air_arcs() -> List[Dict]:
    """
    loads network/arcs_air.csv
    returns a list of dicts, one per directed air arc
    """
    return _load_arc_csv("arcs_air.csv")


def load_land_arcs() -> List[Dict]:
    """
    loads network/arcs_land.csv
    returns a list of dicts, one per directed terrestrial arc
    """
    return _load_arc_csv("arcs_land.csv")


def load_transfer_capacities() -> Dict[int, Dict]:
    """
    loads network/transfer_capacities.csv
    returns a dict keyed by node_id with intermodal transfer capacities/costs
    """
    path = os.path.join(os.path.dirname(__file__), "transfer_capacities.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"transfer_capacities.csv not found at {path}. "
            "Run `python network/build_modal_arcs.py` to generate it."
        )
    df = pd.read_csv(path)
    return {int(row["node_id"]): row.to_dict() for _, row in df.iterrows()}


def build_modal_graph(locations: Dict[int, Dict]) -> Dict:
    """
    builds the three modal arc layers (sea, air, land) plus intermodal
    transfer capacities from the CSVs produced by build_modal_arcs.py

    returns:
        {
          "sea":  list of (from_node, to_node, attrs_dict),
          "air":  list of (from_node, to_node, attrs_dict),
          "land": list of (from_node, to_node, attrs_dict),
          "transfer": {node_id: transfer_capacity_dict},
        }
    """

    def _to_edges(arcs: List[Dict]):
        edges = []
        for row in arcs:
            attrs = {k: v for k, v in row.items() if k not in ("from_node", "to_node")}
            edges.append((int(row["from_node"]), int(row["to_node"]), attrs))
        return edges

    return {
        "sea": _to_edges(load_sea_arcs()),
        "air": _to_edges(load_air_arcs()),
        "land": _to_edges(load_land_arcs()),
        "transfer": load_transfer_capacities(),
    }
