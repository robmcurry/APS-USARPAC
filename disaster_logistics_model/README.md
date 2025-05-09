# Disaster Logistics Modeling Framework

This repository contains a modular, extensible framework for modeling the prepositioning and delivery of humanitarian aid (e.g., food and water) in response to simulated disasters across the Pacific region. The system integrates scenario simulation and deterministic optimization, producing both analytical outputs and visualizations.

## Project Structure

```
disaster_logistics_model/
├── config/                     # Parameter loader and YAML config
├── data/                       # Location and network datasets
├── monte_carlo/                # Scenario generator
├── network/                    # Network graph builder
├── optimization/               # Gurobi models (deterministic)
├── visualization/              # Charts and map output
├── output/                     # Results and plots
control_tower.py                # Main entry point to run entire system
```

## Requirements

- Python 3.9+
- Gurobi + gurobipy
- matplotlib, pandas, folium, networkx, pyyaml

Install with:

```bash
pip install -r requirements.txt
```

## Quick Start

To run the full system:

```bash
python control_tower.py
```

This will:
- Build the network
- Generate simulated disaster scenarios
- Solve optimization models
- Save batch summary results
- Generate visualizations

## Key Outputs

- `output/batch_summary.csv`: Scenario-level results including demand, objective value, APS locations, and composite strategy scores
- `output/aps_frequency_chart.png`: Bar chart of how often each node was selected as an APS site
- `output/objective_values_chart.png`: Objective value per scenario
- `output/severity_vs_objective.png`: Scatter plot comparing disaster severity and cost
- `output/3d_severity_objective_connectivity.png`: 3D scatter of severity, objective, and network connectivity
- `output/interactive_3d_severity_objective_connectivity.html`: Interactive Plotly 3D chart of severity vs. objective vs. connectivity
- `output/aps_selection_map.html`: Map showing APS site frequency
- `output/epicenter_objective_map.html`: Map visualizing objective value based on epicenter location
- `output/composite_score_map.html`: Map of APS nodes color-coded by average composite strategy score

## Parameter Configuration

All simulation, model, and visualization settings are configured in:

```
disaster_logistics_model/config/model_parameters.yaml
```

Additional strategic scoring parameters:
- `config/political_scores.yaml`: Optional geopolitical scoring input for composite strategy analysis

## Roadmap

- Two-stage stochastic optimization extension
- Expanded disaster event library (earthquakes, tsunamis, conflict)
- Dynamic vehicle and routing constraints
- Web dashboard for scenario comparison and strategy exploration
