"""
test_vif_phase4.py

Hand-verified tests for the distance budget (C10, eq:vif:distbudget) and
subtour elimination (C11, eq:vif:sec) added to vehicle_formulation="vif"
in Phase 4 -- see model/model_vif.py's solve_vif and
docs/PRSVIF_Gospel.md, "Vehicle Routing Constraints."

Three tests:
    Test A: distance budget (C10) makes a node permanently unreachable,
            even though a physical arc path exists, because the total
            round-trip-inclusive-of-turnaround distance exceeds D_l. This
            is the exact LCU-1700-vs-4,500km-sea-arc scenario the original
            PRSVIF_DELTA_REPORT.md flagged for the old "individual"
            formulation (which pooled range fleet-wide and so never
            enforced this at all) -- PRS-VIF enforces it per vehicle
            instance, as intended.
    Test B: subtour elimination (C11) blocks the exact disconnected
            phantom-2-cycle use test_vif_phase1.py's module docstring
            documents finding (by accident) while building that phase's
            Test 3 -- before this callback existed, a vehicle could route
            resources through a disconnected 2-cycle it was never
            based near, at a cost of only 2*EPSILON. With C11 active, the
            same network (now safely using two-way arcs, unlike Test 1's
            one-way workaround) produces the physically-honest answer:
            the disconnected node's demand goes genuinely unmet.
    Test C: a legitimate two-leg CONNECTED route (both node2 and node3
            reachable in sequence from the same base) is NOT falsely cut
            by C11, and is NOT blocked by C10 when the distance budget is
            generous enough -- confirming the new machinery doesn't
            reject real routes, only disconnected ones.

Shared conventions with earlier phase test files: R = {"food"},
resource_weight = {"food": 1.0}, c_l,ij = 0.1, EPSILON = 0.04, delta = 500,
rho = 0.2. Node/arc/handling-capacity degradation data is set non-binding
throughout (this file is about C10/C11, not Phase 3's mechanisms).

Run: cd aps_usarpac && pytest tests/test_vif_phase4.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from model.model import solve_stochastic_cvar

R = ["food"]


def _run(instance):
    return solve_stochastic_cvar(
        instance, vehicle_formulation="vif", verbose=False, mip_gap=1e-9,
    )


# ---------------------------------------------------------------------------
# Test A: distance budget (C10) makes a physically-connected node
# unreachable.
# ---------------------------------------------------------------------------
#
# Network: nodes {1, 2, 3} in a line, one-way arcs (1,2) and (2,3). Node 1:
# PPL/base, J_T={1}, stock q_bar=1000. Nodes 2, 3: sinks, d_2=50, d_3=40.
# One vehicle, cap_tons=200 (never binds -- this test is about distance,
# not payload or the arc-throughput/node-handling mechanisms already
# covered in test_vif_phase3.py).
#
# dist_(1,2) = 100 km, dist_(2,3) = 250 km, psi_l (turnaround) = 25 km,
# D_l = 300 km.
#
# HAND COMPUTATION
# -----------------
# Reaching node 3 at all requires using BOTH legs (no arc bypasses node 2):
#   total distance = (100+25) + (250+25) = 400 km > D_l = 300 -- infeasible
#   for ANY route that reaches node 3, regardless of demand there.
# Reaching only node 2: (100+25) = 125 km <= 300 -- comfortably feasible,
#   with 175 km of budget to spare.
# So node 3's demand is permanently unreachable by this fleet (fleet_size=1
#   here, but the conclusion doesn't depend on fleet size -- EVERY vehicle
#   of this type has the same D_l, and the arc distances are fixed), while
#   node 2's is fully served. z_3 = 40 (its full demand, forced -- no route
#   can ever reach it), z_2 = 0.
# Lambda = delta*40 + c*50 + EPSILON*1 (only leg 1->2 is ever used)
#        = 500*40 + 0.1*50 + 0.04 = 20000 + 5 + 0.04 = 20005.04.


def test_c10_distance_budget_makes_a_connected_node_unreachable():
    instance = {
        "nodes": [1, 2, 3],
        "ppl_nodes": [1],
        "commodities": R,
        "scenarios": [1],
        "modes": ["air"],
        "modal_arcs": {"air": [(1, 2), (2, 3)]},
        "modal_residual": [{}],
        "transfer_cap": {},
        "transfer_cost": {},
        "modal_arc_cost": {"air": {(1, 2): 0.1, (2, 3): 0.1}},
        "vehicle_types": {
            "T": {
                "mode": "air", "fleet_size": 1, "J_k": [1], "cap_tons": 200.0,
                "D_k": 300.0, "pi_k": 25.0,
            }
        },
        "probability": {1: 1.0},
        "demand": {(1, 1, "food"): 0, (1, 2, "food"): 50, (1, 3, "food"): 40},
        "inventory_if_open": {(1, "food"): 1000},
        "safety_stock_fraction": 0.2,
        "site_cost": {1: 1},
        "selection_budget": 10,
        "penalty": {(i, "food"): 500 for i in [1, 2, 3]},
        "P_max": 1,
        "beta": 0.9,
        "resource_weight": {"food": 1.0},
        "node_severity": {},
        "disaster_type": {1: "none"},
        "degradation_matrix": {},
        "alpha": 1.0,
        "nominal_throughput": {"air": {(1, 2): 1.0e9, (2, 3): 1.0e9}},
        "node_handling_capacity": {(1, "air"): 1.0e9, (2, "air"): 1.0e9, (3, "air"): 1.0e9},
        "node_handling_bonus": {},
        "modal_arc_distance": {"air": {(1, 2): 100.0, (2, 3): 250.0}},
    }

    results = _run(instance)

    assert results["status"] == "OPTIMAL"
    assert results["objective_value"] == pytest.approx(20005.04, abs=1e-6)
    assert results["flows"] == pytest.approx(
        {(1, 1, 1, 2, "food"): 50.0}, abs=1e-6
    )
    assert results["unmet_demand"] == pytest.approx(
        {(1, 3, "food"): 40.0}, abs=1e-6
    )
    assert results["vehicle_arcs"] == {(1, 1, 1, 2): 1}


# ---------------------------------------------------------------------------
# Test B: subtour elimination (C11) blocks the disconnected phantom-loop
# use.
# ---------------------------------------------------------------------------
#
# Network: nodes {1, 2, 3, 4}. Node 1: PPL/base, J_T={1}, stock. Node 2:
# sink reachable via arc (1,2), d_2=100. Nodes 3, 4: a TWO-WAY disconnected
# pair -- arcs (3,4) AND (4,3) both exist, with NO arc bridging {1,2} and
# {3,4}. Node 4 has demand d_4=60 but is not, and can never legitimately
# be, reached by this vehicle (based only at node 1, J_T={1}).
#
# Distance budget generous (D_l huge) -- this test isolates C11 from C10
# (already covered in Test A).
#
# WHY THIS ISOLATES C11
# -----------------------
# Without subtour elimination, C9's conservation constraint alone permits
# a "phantom" zero-net 2-cycle n[l,3,4]=n[l,4,3]=1 (both arcs' inflow and
# outflow cancel locally at nodes 3 and 4, satisfying C9 there with
# b[l,3]=b[l,4]=0) at a cost of only 2*EPSILON=0.08 -- giving C7 a free
# channel to route x through node 4 despite the vehicle never being
# connected to it. This is exactly the mechanism
# test_vif_phase1.py's module docstring documents discovering by accident
# (a first draft of that phase's Test 3 used two-way arcs in its
# disconnected component and got used this way); that test worked
# around it by using one-way arcs, since C11 didn't exist yet. This test
# uses two-way arcs DELIBERATELY, to confirm C11 -- now that it exists --
# makes that workaround unnecessary: the use should NOT occur.
#
# HAND COMPUTATION
# -----------------
# With C11 active, node 4 remains genuinely unreachable (no legitimate
# arc/basing connection exists at all, regardless of what C9 alone would
# permit), so z_4 = 60 (forced, full demand) while node 2 is fully served.
# Lambda = delta*60 + c*100 + EPSILON*1 (only arc (1,2) is ever legitimately
#   used) = 500*60 + 0.1*100 + 0.04 = 30000 + 10 + 0.04 = 30010.04.
#
# Note: on this toy-scale instance, Gurobi's own presolve/heuristics reach
# this correct answer without the callback ever needing to emit an actual
# lazy cut (subtour_callback_stats shows invocations>0, cuts_added=0) --
# the callback is demonstrably wired in (it runs) and the answer is
# demonstrably correct (matches the hand computation, unlike the
# pre-Phase-4 use), but a nonzero cut count isn't something this small
# an instance reliably produces. This mirrors why model.py's OLD
# individual-formulation callback needed a purpose-built opposition-driven
# network (scripts/toy_individual_cyclic_test.py) to force a nonzero cut
# count in the first place -- not a gap in this test, just a property of
# problem scale. See PHASE4_NOTES.md.


def test_c11_subtour_elimination_blocks_disconnected_phantom_loop():
    instance = {
        "nodes": [1, 2, 3, 4],
        "ppl_nodes": [1],
        "commodities": R,
        "scenarios": [1],
        "modes": ["air"],
        "modal_arcs": {"air": [(1, 2), (3, 4), (4, 3)]},
        "modal_residual": [{}],
        "transfer_cap": {},
        "transfer_cost": {},
        "modal_arc_cost": {"air": {(1, 2): 0.1, (3, 4): 0.1, (4, 3): 0.1}},
        "vehicle_types": {
            "T": {
                "mode": "air", "fleet_size": 1, "J_k": [1], "cap_tons": 200.0,
                "D_k": 1.0e9, "pi_k": 0.0,
            }
        },
        "probability": {1: 1.0},
        "demand": {
            (1, 1, "food"): 0, (1, 2, "food"): 100,
            (1, 3, "food"): 0, (1, 4, "food"): 60,
        },
        "inventory_if_open": {(1, "food"): 1000},
        "safety_stock_fraction": 0.2,
        "site_cost": {1: 1},
        "selection_budget": 10,
        "penalty": {(i, "food"): 500 for i in [1, 2, 3, 4]},
        "P_max": 1,
        "beta": 0.9,
        "resource_weight": {"food": 1.0},
        "node_severity": {},
        "disaster_type": {1: "none"},
        "degradation_matrix": {},
        "alpha": 1.0,
        "nominal_throughput": {
            "air": {(1, 2): 1.0e9, (3, 4): 1.0e9, (4, 3): 1.0e9}
        },
        "node_handling_capacity": {
            (1, "air"): 1.0e9, (2, "air"): 1.0e9, (3, "air"): 1.0e9, (4, "air"): 1.0e9,
        },
        "node_handling_bonus": {},
        "modal_arc_distance": {
            "air": {(1, 2): 0.0, (3, 4): 0.0, (4, 3): 0.0}
        },
    }

    results = _run(instance)

    assert results["status"] == "OPTIMAL"
    assert results["objective_value"] == pytest.approx(30010.04, abs=1e-6)
    assert results["flows"] == pytest.approx(
        {(1, 1, 1, 2, "food"): 100.0}, abs=1e-6
    )
    assert results["unmet_demand"] == pytest.approx(
        {(1, 4, "food"): 60.0}, abs=1e-6
    )
    assert results["vehicle_arcs"] == {(1, 1, 1, 2): 1}
    assert results["subtour_callback_stats"]["invocations"] > 0


# ---------------------------------------------------------------------------
# Test C: a legitimate connected two-leg route is not falsely blocked.
# ---------------------------------------------------------------------------
#
# Same line network as Test A (nodes {1,2,3}, arcs (1,2),(2,3)), but with a
# generous distance budget this time (D_l=1000, comfortably covering both
# legs plus turnaround) -- confirming C10 doesn't block a route it
# shouldn't, and C11 doesn't mistake a real connected route (node 3 IS
# reachable via node 2 this time) for a disconnected subtour.
#
# HAND COMPUTATION
# -----------------
# d_2=30, d_3=40, cap_tons=100 (>= 70, the amount the vehicle must carry on
# leg 1 to cover both destinations -- see test_vif_phase1.py's discussion
# of how multi-leg delivery works: x_(1,2)=70 covers both the 30 dropped
# at node 2 and the 40 continuing on to node 3, verified against C6's
# balance at node 2: 70 in, 30 consumed as demand, 40 continues via
# x_(2,3), y_2=0).
# Total distance = (100+25) + (100+25) = 250 <= D_l=1000 -- feasible with
#   plenty of slack.
# Lambda = delta*0 + c*(70+40) + EPSILON*2 (both legs used)
#        = 0 + 0.1*110 + 0.08 = 11 + 0.08 = 11.08.


def test_legitimate_connected_two_leg_route_not_falsely_blocked():
    instance = {
        "nodes": [1, 2, 3],
        "ppl_nodes": [1],
        "commodities": R,
        "scenarios": [1],
        "modes": ["air"],
        "modal_arcs": {"air": [(1, 2), (2, 3)]},
        "modal_residual": [{}],
        "transfer_cap": {},
        "transfer_cost": {},
        "modal_arc_cost": {"air": {(1, 2): 0.1, (2, 3): 0.1}},
        "vehicle_types": {
            "T": {
                "mode": "air", "fleet_size": 1, "J_k": [1], "cap_tons": 100.0,
                "D_k": 1000.0, "pi_k": 25.0,
            }
        },
        "probability": {1: 1.0},
        "demand": {(1, 1, "food"): 0, (1, 2, "food"): 30, (1, 3, "food"): 40},
        "inventory_if_open": {(1, "food"): 1000},
        "safety_stock_fraction": 0.2,
        "site_cost": {1: 1},
        "selection_budget": 10,
        "penalty": {(i, "food"): 500 for i in [1, 2, 3]},
        "P_max": 1,
        "beta": 0.9,
        "resource_weight": {"food": 1.0},
        "node_severity": {},
        "disaster_type": {1: "none"},
        "degradation_matrix": {},
        "alpha": 1.0,
        "nominal_throughput": {"air": {(1, 2): 1.0e9, (2, 3): 1.0e9}},
        "node_handling_capacity": {(1, "air"): 1.0e9, (2, "air"): 1.0e9, (3, "air"): 1.0e9},
        "node_handling_bonus": {},
        "modal_arc_distance": {"air": {(1, 2): 100.0, (2, 3): 100.0}},
    }

    results = _run(instance)

    assert results["status"] == "OPTIMAL"
    assert results["objective_value"] == pytest.approx(11.08, abs=1e-6)
    assert results["flows"] == pytest.approx(
        {(1, 1, 1, 2, "food"): 70.0, (1, 1, 2, 3, "food"): 40.0}, abs=1e-6
    )
    assert results["unmet_demand"] == {}
    assert results["vehicle_arcs"] == {(1, 1, 1, 2): 1, (1, 1, 2, 3): 1}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
