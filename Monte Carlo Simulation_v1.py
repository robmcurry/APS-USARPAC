"""
Monte Carlo Simulation of Pacific Disaster Logistics Network

This script generates a synthetic logistics network using real city locations,
simulates typhoon disruptions, and exports scenario data for use in optimization models.
"""

import pandas as pd
import networkx as nx
import random
import json
import csv
from geopy.distance import geodesic

# Step 1: Load City Data and Generate Base Network
def load_city_data(file_path):
    return pd.read_csv(file_path)

def generate_city_network(cities_df, commodities=['food', 'water']):
    G = nx.Graph()
    for _, row in cities_df.iterrows():
        G.add_node(
            row['Node ID'],
            name=row['Node Name'],
            country=row['Country'],
            latitude=row['Latitude'],
            longitude=row['Longitude'],
            supply={c: random.randint(100, 500) for c in commodities},
            demand={c: random.randint(50, 300) for c in commodities}
        )

    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                coord_u = (G.nodes[u]['latitude'], G.nodes[u]['longitude'])
                coord_v = (G.nodes[v]['latitude'], G.nodes[v]['longitude'])
                distance = geodesic(coord_u, coord_v).km
                capacity = {c: random.randint(50, 300) for c in commodities}
                G.add_edge(u, v, distance=distance, capacity=capacity)
    return G

# Step 2: Simulate Typhoon Impact
def typhoon_impact(G, severity):
    epicenter = random.choice(list(G.nodes()))
    radius = severity * 100
    affected_nodes = []

    for node in G.nodes():
        coord_node = (G.nodes[node]['latitude'], G.nodes[node]['longitude'])
        coord_epi = (G.nodes[epicenter]['latitude'], G.nodes[epicenter]['longitude'])
        distance = geodesic(coord_node, coord_epi).km

        if distance <= radius:
            impact_factor = max(0.1, 1 - distance / radius)
            for c in G.nodes[node]['supply']:
                G.nodes[node]['supply'][c] *= (1 - impact_factor)
                G.nodes[node]['demand'][c] *= (1 + impact_factor)
            affected_nodes.append(node)

            for neighbor in G.neighbors(node):
                if random.random() < impact_factor:
                    for c in G[node][neighbor]['capacity']:
                        G[node][neighbor]['capacity'][c] *= (1 - impact_factor)
    return G, affected_nodes, epicenter

# Step 3: Run Monte Carlo Simulations
def stringify_keys(d):
    return {f"{k[0]}_{k[1]}" if len(k) == 2 else f"{k[0]}_{k[1]}_{k[2]}": v for k, v in d.items()}

def run_simulation(G, commodities, num_scenarios=100):
    scenarios = []
    for s in range(num_scenarios):
        severity = random.randint(1, 5)
        sim_G, affected_nodes, epicenter = typhoon_impact(G.copy(), severity)

        demand, supply, capacity = {}, {}, {}
        for node in sim_G.nodes():
            for c in commodities:
                demand[(node, c)] = sim_G.nodes[node]['demand'][c]
                supply[(node, c)] = sim_G.nodes[node]['supply'][c]

        for u, v in sim_G.edges():
            for c in commodities:
                capacity[(u, v, c)] = sim_G[u][v]['capacity'][c]

        scenarios.append({
            'scenario_id': s,
            'epicenter': epicenter,
            'severity': severity,
            'affected_nodes': affected_nodes,
            'demand': demand,
            'supply': supply,
            'capacity': capacity
        })
    return scenarios

# Step 4: Export Simulation Results
def export_results(scenarios, json_file='simulation_scenarios.json', csv_file='simulation_scenarios_flat.csv'):
    with open(json_file, 'w') as f_json:
        exportable = [{
            'scenario_id': s['scenario_id'],
            'epicenter': s['epicenter'],
            'severity': s['severity'],
            'affected_nodes': s['affected_nodes'],
            'demand': stringify_keys(s['demand']),
            'supply': stringify_keys(s['supply']),
            'capacity': stringify_keys(s['capacity'])
        } for s in scenarios]
        json.dump(exportable, f_json, indent=2)

    with open(csv_file, 'w', newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=[
            'scenario_id', 'epicenter', 'severity', 'affected_nodes', 'node', 'commodity', 'demand', 'supply'
        ])
        writer.writeheader()
        for s in scenarios:
            for (i, c), d_val in s['demand'].items():
                s_val = s['supply'][(i, c)]
                writer.writerow({
                    'scenario_id': s['scenario_id'],
                    'epicenter': s['epicenter'],
                    'severity': s['severity'],
                    'affected_nodes': ';'.join(map(str, s['affected_nodes'])),
                    'node': i,
                    'commodity': c,
                    'demand': round(d_val, 2),
                    'supply': round(s_val, 2)
                })

# Main execution
if __name__ == "__main__":
    commodities = ['food', 'water']
    cities = load_city_data('pacific_cities.csv')
    G = generate_city_network(cities, commodities)
    scenarios = run_simulation(G, commodities, num_scenarios=100)
    export_results(scenarios)
    print(f"Simulation complete: {len(scenarios)} scenarios saved to JSON and CSV.")
