

# Stochastic Optimization for Humanitarian Prepositioning Under Network Capacity Uncertainty

**Author:** Clay D. Woody  
**Course:** INEG 5140V Final Project  
**Term:** Spring 2026  

---

## Overview

This project implements a two-stage stochastic optimization model for humanitarian prepositioning under transportation network capacity uncertainty.

The framework integrates Monte Carlo simulation, network construction, stochastic input generation, and mixed-integer optimization to evaluate how disruption to transportation infrastructure affects resource distribution. The model is specifically designed to isolate network capacity uncertainty while holding demand fixed across scenarios.

---

## Project Structure

```
/src
    stoch_loader.py
    stoch_simulator.py
    stochastic_input_builder.py
    stochastic_model_course.py

/experiments
    experiment_e1_baseline.py
    experiment_e2_low_fragility.py
    experiment_e3_high_fragility.py
    experiment_e4_low_risk.py
    experiment_e5_high_risk.py
    experiment_e6_low_capacity.py
    experiment_e7_high_capacity.py
    experiment_e8_tight_budget.py
    experiment_e9_loose_budget.py
    run_all_experiments.py

/data
    pacific_cities.csv

/config
    stoch_model_parameters.yaml

/results
    all_experiments_summary.csv
    all_experiments_plotting_summary.csv
    figures/

test_simulator.py
requirements.txt
README.md
```

---

## Configuration

Model parameters are defined in:

```
/config/stoch_model_parameters.yaml
```

Baseline configuration includes:

- Number of scenarios: 100  
- Random seed: 32  
- Commodities: food, water  
- Vessel speed: 1500 km/day  
- Maximum voyage time: 3 days  
- CVaR confidence level: β = 0.9  
- Prepositioning budget: 12  
- Safety stock fraction: 0.25  

---

## Requirements

Python 3.9+

Install dependencies:

```
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- networkx
- geopy
- pyyaml
- gurobipy

**Note:** A valid Gurobi license is required.

---

## How to Run

### Run baseline test

```
python test_simulator.py
```

### Run all experiments

```
python experiments/run_all_experiments.py
```

This will:
1. Generate disruption scenarios
2. Build stochastic optimization instances
3. Solve each experiment configuration
4. Save results to `/results`

---

## Model Description

### First Stage
- Select prepositioning locations (PPLs)
- Subject to:
  - Cardinality constraint
  - Budget constraint

### Second Stage
- Route resources under scenario-dependent disruption
- Respect arc capacity constraints
- Track:
  - Unmet demand
  - Transportation cost

---

## Uncertainty Modeling

Uncertainty is represented through Monte Carlo simulation:

- Random epicenter selection
- Severity sampled from [1, 5]
- Distance-based severity decay
- Scenario-dependent arc capacity degradation

Demand and inventory are deterministic and fixed across scenarios to isolate network effects.

---

## Objective Function

The model minimizes Conditional Value-at-Risk (CVaR) of total system loss:

- Unmet demand penalties  
- Transportation costs  

CVaR emphasizes worst-case scenarios and allows tuning via parameter β.

---

## Experiments

Nine experiments evaluate sensitivity to:

- Network fragility (γ)
- Risk aversion (β)
- Prepositioning capacity (P_max)
- Budget (B)

Results are stored in:

```
/results/all_experiments_summary.csv
```

---

## Notes

- Inventory is currently modeled as a function of population.
- Network is pruned based on a 3-day travel limit (4500 km).
- Scenario generation is synthetic.
- Model is designed to isolate network capacity uncertainty.

---

## Future Improvements

- Incorporate multi-modal transportation networks  
- Use historical disaster data for scenario generation  
- Improve inventory modeling using logistics capacity instead of population  

---

## Contact

Clay D. Woody