~~# Disaster Logistics Modeling Framework

This repository contains a modular and extensible framework for modeling prepositioning and delivery of humanitarian assistance commodities (e.g., food and water) in the aftermath of disasters. The system simulates disaster scenarios and solves deterministic optimization models using Gurobi. Future versions will include two-stage stochastic extensions.

## Project Structure

```
disaster_logistics_model/
├── config/
│   └── model_parameters.yaml         # Centralized configuration for parameters (dials/levers)
├── data/
│   └── pacific_cities.csv            # Location data for network nodes
├── monte_carlo/
│   └── simulator.py                  # Scenario generator based on disaster types and severity
├── network/
│   └── network_builder.py            # Constructs geospatial graph and arc distances
├── optimization/
│   ├── deterministic_model.py        # Deterministic MIP model with vehicle and inventory tracking
│   ├── stochastic_model.py           # Placeholder for future 2-stage stochastic model
│   └── run_batch_optimization.py     # Batch execution for all scenarios
├── visualization/
│   └── flow_map.py                   # (Optional) For generating maps and plots
├── main.py                           # Entry point if needed
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

## Quick Start

1. **Build the network**:
   ```python
   from disaster_logistics_model.network.network_builder import build_geospatial_network
   G, locations = build_geospatial_network("disaster_logistics_model/data/pacific_cities.csv")
   ```

2. **Generate scenarios**:
   ```python
   from disaster_logistics_model.monte_carlo.simulator import generate_scenarios
   scenarios = generate_scenarios(G, locations, num_scenarios=100)
   ```

3. **Run deterministic optimization**:
   ```python
   from disaster_logistics_model.optimization.deterministic_model import solve_deterministic_vrp_with_aps
   result = solve_deterministic_vrp_with_aps(scenarios[0])
   ```

4. **Batch process scenarios**:
   ```bash
   python -m disaster_logistics_model.optimization.run_batch_optimization
   ```

## Parameter Configuration

Edit `disaster_logistics_model/config/model_parameters.yaml` to tune model behavior:
- `commodities`: Commodity types (e.g., food, water)
- `arc_capacity`: Base and severity multiplier for arc capacities
- `default_disaster`: Severity and affected radius settings
- `seed`: Random seed for reproducibility

## Notes
- Built with extensibility in mind for disaster type, severity distributions, vehicle/commodity scaling.
- All modules are designed for reuse and clarity across network, simulation, optimization, and visualization stages.
- Model currently solves with Gurobi 11.0.3. Ensure `gurobipy` is installed and licensed.

## Coming Soon
- Two-stage stochastic version
- GUI interface for configuration
- Interactive dashboards and map overlays

---

