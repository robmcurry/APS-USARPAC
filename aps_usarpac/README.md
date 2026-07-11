# APS-USARPAC Stochastic Prepositioning Model

Two-stage stochastic CVaR optimization model for Army Prepositioned Stocks
site selection across the Indo-Pacific theater.

## Requirements

- Python 3.11+
- Gurobi 11+ with valid license (academic or commercial)
- Dependencies: `pip install -r requirements.txt`

## Quick Start

From the aps_usarpac/ directory:

```bash
# run full sensitivity analysis (100 scenarios, 5 gamma values)
python -m analysis.sensitivity_runner

# run saa convergence test
python -m analysis.convergence_test
```

## Directory Structure

```
aps_usarpac/
├── network/           network nodes, arc definitions, and network builder
├── scenarios/         scenario generator and sampling configuration
├── model/             stochastic MIP formulation and input builder
├── analysis/          sensitivity runner and convergence test
├── config/            parameter loader and legacy config shim
└── output/            results csv files written by analysis runs
```

## Key Parameters

Edit the config files to adjust model behavior:
- scenarios/scenario_config.yaml  -- scenario count, seed, epicenter weights, severity distribution
- model/model_config.yaml         -- cvar beta, site budget, inventory tiers, demand rates
- network/data/network_config.yaml -- arc distance limit, hub capacities

## Model Overview

The model selects up to P_max prepositioning locations from 22 PPL-eligible
nodes in a 50-node Indo-Pacific network. Demand at each node is fixed and
reflects expected disaster exposure weighted by population. Arc capacity
degrades with disaster severity through the gamma parameter. The CVaR
objective at beta=0.9 minimizes the expected loss in the worst 10% of
scenarios across the sampled disaster scenario set.
