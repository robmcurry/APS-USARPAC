import pandas as pd
import networkx as nx
from math import radians, cos, sin, asin, sqrt

def haversine(coord1, coord2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface given as (lat, lon).
    Returns distance in kilometers.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def build_geospatial_network(csv_file):
    df = pd.read_csv(csv_file)
    G = nx.Graph()
    locations = {}

    for _, row in df.iterrows():
        node_id = row['Node ID']
        coords = (row['Latitude'], row['Longitude'])
        region = row.get("Region", 0)
        population = row["Population"]
        G.add_node(
            node_id,
            name=row['Node Name'],
            coords=coords,
            region=region,
            population=population,
            lat=coords[0],
            lon=coords[1]
        )
        locations[node_id] = {
            'name': row['Node Name'],
            'coords': coords,
            'region': region,
            'population': population
        }

    node_ids = list(locations.keys())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            n1, n2 = node_ids[i], node_ids[j]
            dist = haversine(locations[n1]['coords'], locations[n2]['coords'])
            G.add_edge(n1, n2, weight=dist)

    # Save city lookup as JSON
    import json
    import os
    city_lookup = {str(node_id): list(data["coords"]) for node_id, data in locations.items()}
    os.makedirs("disaster_logistics_model/data", exist_ok=True)
    with open("disaster_logistics_model/data/city_lookup.json", "w") as f:
        json.dump(city_lookup, f, indent=2)
    print("City lookup saved to disaster_logistics_model/data/city_lookup.json")

    return G, locations


# Visualization and saving function
import matplotlib.pyplot as plt
import os

# New imports for static and interactive map visualizations
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def draw_and_save_network(G, locations, output_path="disaster_logistics_model/output/maps/network_map.png"):
    pos = {node: (loc["coords"][1], loc["coords"][0]) for node, loc in locations.items()}  # lon, lat for plotting
    labels = {node: loc["name"] for node, loc in locations.items()}

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='blue')
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title("Pacific Cities Network")
    plt.axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Network visualization saved to {output_path}")


# New static map visualization using geopandas and contextily
def draw_static_map(G, locations, output_path="disaster_logistics_model/output/maps/static_map.png"):
    nodes = []
    for node_id, loc in locations.items():
        lat, lon = loc["coords"]
        nodes.append({
            "id": node_id,
            "name": loc["name"],
            "region": loc["region"],
            "geometry": Point(lon, lat)
        })

    gdf = gpd.GeoDataFrame(nodes, crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)

    ax = gdf.plot(figsize=(12, 8), column="region", cmap="Set1", legend=True, markersize=50)
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf["name"]):
        ax.text(x, y, label, fontsize=8)

    # Print bounds for debugging
    print("Map bounds:", gdf.total_bounds)
    # Ensure correct map extent
    ax.set_xlim(gdf.total_bounds[[0, 2]])
    ax.set_ylim(gdf.total_bounds[[1, 3]])

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_axis_off()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Static map saved to {output_path}")


# New interactive HTML map visualization using folium
def draw_interactive_map(locations, output_path="disaster_logistics_model/output/maps/network_map.html"):
    center_lat = sum(loc["coords"][0] for loc in locations.values()) / len(locations)
    center_lon = sum(loc["coords"][1] for loc in locations.values()) / len(locations)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=3, tiles="CartoDB positron")

    region_ids = sorted({loc["region"] for loc in locations.values()})
    color_map = cm.get_cmap("Set1", len(region_ids))
    region_colors = {rid: mcolors.to_hex(color_map(i)) for i, rid in enumerate(region_ids)}

    for node_id, loc in locations.items():
        lat, lon = loc["coords"]
        color = region_colors.get(loc["region"], "blue")
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=loc["name"]
        ).add_to(m)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"Interactive map saved to {output_path}")


# Example usage for visualization
if __name__ == "__main__":
    csv_file = "disaster_logistics_model/data/pacific_cities.csv"
    G, locations = build_geospatial_network(csv_file)
    draw_and_save_network(G, locations)
    draw_static_map(G, locations)
    draw_interactive_map(locations)