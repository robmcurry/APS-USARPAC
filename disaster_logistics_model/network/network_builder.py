import pandas as pd
import networkx as nx
from geopy.distance import geodesic
from math import radians, cos, sin, asin, sqrt

def haversine(coord1, coord2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface given as (lat, lon).
    Returns distance in kilometers.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers (mean radius)
    r = 6371
    return c * r

def build_geospatial_network(csv_file):
    df = pd.read_csv(csv_file)
    G = nx.Graph()
    locations = {}

    for _, row in df.iterrows():
        node_id = row['Node ID']
        G.add_node(node_id,
                   name=row['Node Name'],
                   coords=(row['Latitude'], row['Longitude']),
                   population=row['Population'])  # <-- NEW population attribute
        locations[node_id] = {
            'name': row['Node Name'],
            'coords': (row['Latitude'], row['Longitude']),
            'population': row['Population']  # <-- Also store it here
        }

    # Add edges based on distances
    nodes = list(locations.keys())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1, node2 = nodes[i], nodes[j]
            dist = haversine(locations[node1]['coords'], locations[node2]['coords'])
            G.add_edge(node1, node2, weight=dist)

    return G, locations