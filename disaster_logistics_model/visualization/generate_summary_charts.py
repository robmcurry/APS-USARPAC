import json
import folium
import argparse

def plot_tree_from_json(scenario_id, city_lookup, results_file, output_file):
    with open(results_file, 'r') as f:
        data = json.load(f)

    scenario_data = next((s for s in data if s["scenario_id"] == scenario_id), None)
    if scenario_data is None:
        print(f"Scenario {scenario_id} not found.")
        return

    tree_edges = scenario_data["tree_edges"]
    leaf_nodes = scenario_data.get("leaf_nodes", [])
    aps_assignments = scenario_data.get("aps_assignments", {})

    # Initialize map centered on average coordinates
    lats = [city_lookup.nodes[edge["from"]]["lat"] for edge in tree_edges] + \
           [city_lookup.nodes[edge["to"]]["lat"] for edge in tree_edges]
    lons = [city_lookup.nodes[edge["from"]]["lon"] for edge in tree_edges] + \
           [city_lookup.nodes[edge["to"]]["lon"] for edge in tree_edges]
    center = [sum(lats)/len(lats), sum(lons)/len(lons)]
    fmap = folium.Map(location=center, zoom_start=5)

    for edge in tree_edges:
        from_node = edge["from"]
        to_node = edge["to"]
        lat1, lon1 = city_lookup.nodes[from_node]["lat"], city_lookup.nodes[from_node]["lon"]
        lat2, lon2 = city_lookup.nodes[to_node]["lat"], city_lookup.nodes[to_node]["lon"]

        # Fix for International Date Line wrapping
        if abs(lon2 - lon1) > 180:
            if lon2 > lon1:
                lon2 -= 360
            else:
                lon2 += 360

        folium.PolyLine(locations=[(lat1, lon1), (lat2, lon2)],
                        color='blue', weight=2).add_to(fmap)

    for node in set(d["node"] for d in leaf_nodes):
        lat, lon = city_lookup.nodes[node]["lat"], city_lookup.nodes[node]["lon"]
        folium.CircleMarker(location=(lat, lon), radius=4,
                            color='orange', fill=True, fill_opacity=0.8,
                            popup=f"Leaf Node: {node}").add_to(fmap)

    for aps_entry in aps_assignments:
        if isinstance(aps_entry, dict) and "aps" in aps_entry:
            aps = aps_entry["aps"]
        elif isinstance(aps_entry, list) and len(aps_entry) > 0:
            aps = aps_entry[0]
        else:
            continue
        lat, lon = city_lookup.nodes[str(aps)]["lat"], city_lookup.nodes[str(aps)]["lon"]
        folium.Marker(location=(lat, lon),
                      popup=f"APS Node: {aps}",
                      icon=folium.Icon(color='blue')).add_to(fmap)

    fmap.save(output_file)
    print(f"Scenario {scenario_id} map saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="batch_detailed_results.json", help="Path to results JSON")
    parser.add_argument("--scenario", type=int, default=4, help="Scenario ID to visualize")
    parser.add_argument("--output", type=str, default="scenario_4_map.html", help="Output map file")

    args = parser.parse_args()

    from disaster_logistics_model.network.network_builder import build_geospatial_network
    city_lookup, _ = build_geospatial_network("disaster_logistics_model/data/pacific_cities.csv")

    plot_tree_from_json(args.scenario, city_lookup, args.results, args.output)