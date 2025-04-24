# test_simulator.py
from disaster_logistics_model.network.network_builder import build_geospatial_network
from disaster_logistics_model.monte_carlo.simulator import generate_scenarios

if __name__ == "__main__":
    csv_path = "disaster_logistics_model/data/pacific_cities.csv"

    # Build the geospatial network
    G, locations = build_geospatial_network(csv_path)
    print(f"{len(locations)} nodes loaded.")
    print("Sample node:", list(locations.items())[0])

    # Generate scenarios
    scenarios = generate_scenarios(G, locations)

    print(f"\n{len(scenarios)} scenarios generated.")
    for s in scenarios:
        print(f"\nScenario ID: {s['scenario_id']}")
        print(f"  Epicenter: {s['epicenter']}")
        print(f"  Severity: {s['severity']:.2f}")
        print(f"  Affected Nodes: {s['affected_nodes']}")
        print(f"  Sample Demand Entries: {list(s['demand'].items())[:5]}")