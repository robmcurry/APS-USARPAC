# disaster_logistics_model/visualization/flow_map.py

import folium
from folium.plugins import MarkerCluster
from math import radians, cos, sin, asin, sqrt

from disaster_logistics_model.config.loader import load_parameters


def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lon1, lon2 = lon1 - 360 if lon1 > 180 else lon1, lon2 - 360 if lon2 > 180 else lon2
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c  # Earth radius in km


def plot_full_scenario_map(scenario, aps_result, output_file):
    params = load_parameters()
    vis_params = params["visualization"]

    node_coords = {int(k): (v["lat"], v["lon"]) for k, v in scenario["locations"].items()}
    epicenter = int(scenario["epicenter"])
    severity = scenario["severity"]

    m = folium.Map(location=node_coords[epicenter], zoom_start=vis_params["map_zoom_start"],
                   tiles=vis_params["map_tiles"])

    marker_cluster = MarkerCluster().add_to(m)

    # Plot all nodes
    for node, (lat, lon) in node_coords.items():
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color='black',
            fill=True,
            fill_color='black',
            fill_opacity=0.7,
            popup=f"Node {node}"
        ).add_to(marker_cluster)

    # Highlight epicenter
    folium.CircleMarker(
        location=node_coords[epicenter],
        radius=8,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=1.0,
        popup="Epicenter"
    ).add_to(m)

    # Draw impact rings
    severity_km = severity * vis_params["impact_radius_multiplier"]
    folium.Circle(
        location=node_coords[epicenter],
        radius=severity_km * 1000,
        color=vis_params["impact_ring_color"],
        fill=False
    ).add_to(m)

    # Plot APS locations
    if "aps_locations" in aps_result:
        for aps_node in aps_result["aps_locations"]:
            if aps_node in node_coords:
                folium.Marker(
                    location=node_coords[aps_node],
                    icon=folium.Icon(color='blue', icon="star"),
                    popup=f"APS {aps_node}"
                ).add_to(m)

    # Draw flows
    if "flow_summary" in aps_result and aps_result["flow_summary"]:
        top_flows = sorted(aps_result["flow_summary"].items(), key=lambda x: -x[1])[:vis_params["top_k_flows"]]
        for (i, j), flow_value in top_flows:
            if i in node_coords and j in node_coords:
                folium.PolyLine(
                    locations=[node_coords[i], node_coords[j]],
                    color=vis_params["flow_color"],
                    weight=2,
                    opacity=0.7,
                    popup=f"Flow {i} → {j}: {flow_value:.1f}"
                ).add_to(m)

    m.save(output_file)