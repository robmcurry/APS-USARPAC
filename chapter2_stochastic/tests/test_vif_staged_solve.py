import pytest

from model.model_distance_state import _distance_network
from scripts.vif_staged_solve import _subset_instance, solve_staged_vif


def _toy_instance():
    scenarios = [11, 22]
    nodes = [1, 2]
    demand = {
        (w, i, "food"): (25.0 if i == 2 else 0.0)
        for w in scenarios
        for i in nodes
    }
    return {
        "nodes": nodes,
        "ppl_nodes": [1],
        "commodities": ["food"],
        "scenarios": scenarios,
        "modes": ["air"],
        "modal_arcs": {"air": [(1, 2)]},
        "modal_residual": [{}, {}],
        "transfer_cap": {},
        "transfer_cost": {},
        "modal_arc_cost": {"air": {(1, 2): 0.1}},
        "modal_arc_distance": {"air": {(1, 2): 100.0}},
        "vehicle_types": {
            "T": {
                "mode": "air",
                "fleet_size": 1,
                "J_k": [1],
                "cap_tons": 100.0,
                "D_k": 1000.0,
                "pi_k": 0.0,
            }
        },
        "K_m": {"air": ["T"]},
        "probability": {11: 0.25, 22: 0.75},
        "demand": demand,
        "inventory_if_open": {(1, "food"): 100.0},
        "inventory_availability": {
            (w, 1, "food"): 1.0 for w in scenarios
        },
        "safety_stock_fraction": 0.0,
        "site_cost": {1: 1.0},
        "selection_budget": 1.0,
        "penalty": {(i, "food"): 500.0 for i in nodes},
        "P_max": 1,
        "beta": 0.6,
        "resource_weight": {"food": 1.0},
        "node_severity": {(w, i): 0.0 for w in scenarios for i in nodes},
        "disaster_type": {11: "none", 22: "none"},
        "degradation_matrix": {"air": {"none": 0.0}},
        "alpha": 1.0,
        "nominal_throughput": {"air": {(1, 2): 100.0}},
        "node_handling_capacity": {
            (1, "air"): 10.0,
            (2, "air"): 10.0,
        },
        "node_handling_bonus": {(1, "air"): 0.0},
    }


def test_subset_renormalizes_probabilities_and_filters_scenario_data():
    subset = _subset_instance(_toy_instance(), [22])
    assert subset["scenarios"] == [22]
    assert subset["probability"] == {22: 1.0}
    assert set(w for w, _i, _r in subset["demand"]) == {22}
    assert subset["disaster_type"] == {22: "none"}
    assert len(subset["modal_residual"]) == 1


def test_distance_network_is_conservative_and_excludes_over_budget_paths():
    vehicle_types = {
        "T": {
            "mode": "air",
            "J_k": [1],
            "D_k": 100.0,
            "pi_k": 0.0,
        }
    }
    arcs = {"air": [(1, 2), (2, 3), (3, 1)]}
    distances = {"air": {(1, 2): 41.0, (2, 3): 41.0, (3, 1): 1.0}}

    states, transitions, _outgoing, _incoming = _distance_network(
        vehicle_types, arcs, distances, distance_step_km=25.0
    )

    # Each 41 km leg rounds upward to two 25 km buckets.  The two-leg path
    # reaches the 100 km modeled budget; the return arc would exceed it.
    assert (1, 2, 0, 2) in transitions["T"]
    assert (2, 3, 2, 4) in transitions["T"]
    assert not any(i == 3 and j == 1 for i, j, _d, _next_d in transitions["T"])
    assert (3, 4) in states["T"]


def test_staged_vif_solves_toy_end_to_end():
    results, summary = solve_staged_vif(
        _toy_instance(),
        strategic_scenario_count=1,
        strategic_time=30,
        scenario_time=30,
        final_time=30,
        mip_gap=1e-6,
        site_neighborhood=0,
        base_neighborhood=0,
        verbose=False,
    )
    try:
        assert summary["final_solution_count"] > 0
        assert summary["vehicle_formulation"] == "distance_state"
        assert summary["final_mode"] == "fixed_route_resource_allocation_lp"
        assert summary["strategic_selected_sites"] == [1]
        assert summary["final_selected_sites"] == [1]
        assert len(summary["scenario_solves"]) == 2
        assert summary["combined_route_start_arcs"] == 2
        assert results["model"].NumIntVars == 0
        assert results["objective_value"] == pytest.approx(2.54, abs=1e-6)
    finally:
        results["model"].dispose()
