import pandas as pd
import networkx as nx
from geopy.distance import geodesic


def build_geospatial_network(csv_file):
    """
    Builds a NetworkX graph from a CSV of nodes with lat/lon and returns both:
    - G: the graph with weighted edges (km distance)
    - locations: a dict of node_id to {name, coords}
    """

    df = pd.read_csv(csv_file)
    locations = {}
    G = nx.Graph()

    # Step 1: Load and shift longitudes (for Pacific view)
    for _, row in df.iterrows():
        node_id = row["Node ID"]
        lat, lon = row["Latitude"], row["Longitude"]
        if lon < 0:
            lon += 360
        coords = (lat, lon)
        name = row["Node Name"]
        locations[node_id] = {"name": name, "coords": coords}
        G.add_node(node_id, name=name, coords=coords)

    # Step 2: Add weighted edges (distance in km)
    node_ids = list(locations.keys())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            n1, n2 = node_ids[i], node_ids[j]
            c1, c2 = locations[n1]["coords"], locations[n2]["coords"]
            dist = geodesic(c1, c2).kilometers
            G.add_edge(n1, n2, weight=dist)

    return G, locations