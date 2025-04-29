# flow_map.py

import folium
import pandas as pd
import networkx as nx
from geopy.distance import geodesic
import os

def adjusted_coords(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    # Adjust for International Date Line crossings
    if abs(lon1 - lon2) > 180:
        if lon1 > 0:
            lon1 -= 360
        else:
            lon2 -= 360
    return [(lat1, lon1), (lat2, lon2)]

def plot_network(G, locations, output_html_path="disaster_logistics_model/output/network_map.html"):
    avg_lat = sum(lat for lat, _ in [data["coords"] for data in locations.values()]) / len(locations)
    avg_lon = sum(lon for _, lon in [data["coords"] for data in locations.values()]) / len(locations)

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=3, tiles="cartodbpositron")

    # Plot nodes
    for node_id, data in locations.items():
        folium.CircleMarker(
            location=data['coords'],
            radius=5,
            color="red",
            fill=True,
            fill_opacity=0.7,
            popup=data['name']
        ).add_to(m)

    # Plot arcs (adjusting for long-distance wrap issues)
    for node1, node2, attributes in G.edges(data=True):
        coords = adjusted_coords(locations[node1]["coords"], locations[node2]["coords"])
        folium.PolyLine(coords, color="blue", weight=1, opacity=0.5).add_to(m)

    m.save(output_html_path)
    print(f"Network map saved as {output_html_path}")

if __name__ == "__main__":
    from disaster_logistics_model.network.network_builder import build_geospatial_network
    csv_path = "disaster_logistics_model/data/pacific_cities.csv"
    G, locations = build_geospatial_network(csv_path)
    plot_network(G, locations)