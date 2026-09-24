"""
test_vif_phase1.py

Three hand-verified toy instances for vehicle_formulation="vif" (PRS-VIF
Phase 1 skeleton -- see model/model_vif.py's solve_vif and
docs/PRSVIF_Gospel.md, "Risk-Aware Vehicle-Indexed Model"). Each instance is
hand-constructed (no network_builder/scenario_generator/input_builder
pipeline -- that integration is Phase 5), with an optimum computed by hand
below so the assertions check actual correctness, not just "it solved."

Shared toy parameters across all three instances, chosen to keep arithmetic
simple:
    R = {"food"}, resource_weight = {"food": 1} (1 ton per unit -- lets
        cap_tons double as a plain unit cap).
    One vehicle type "T", mode "air", cap_tons = 150 (>= any single
        shipment below, so C7 never binds).
    delta_ir = 500 (unmet-demand penalty), c_l,ij = 0.1 (transport cost per
        unit per arc, via modal_arc_cost), EPSILON = 0.04 (hardcoded inside
        solve_vif, matching gospel Table 5) -- so moving 100 units
        of food down one arc with one vehicle costs 100*0.1 + 1*0.04 =
        10.04, versus a penalty of 500*100 = 50,000 for leaving it unmet.
        Transport is always the cheaper choice whenever a path exists.
    rho = 0.2 (safety-stock fraction) -> releasable_fraction = 0.8.
    Single scenario, w = 1, probability 1.0. a^w_i is hardcoded to 1 inside
        solve_vif this phase (no degradation model yet), so it does
        not appear as a modeled quantity here.

Run: cd aps_usarpac && pytest tests/test_vif_phase1.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from model.model import solve_stochastic_cvar

R = ["food"]
MODES = ["air"]


def _base_vehicle_type(J_k, fleet_size=2, cap_tons=150.0):
    return {
        "mode": "air",
        "fleet_size": fleet_size,
        "J_k": J_k,
        "cap_tons": cap_tons,
        # Phase 4 requires D_k/pi_k (distance budget / turnaround penalty
        # -- see solve_vif's docstring, "Schema note"). These tests
        # predate Phase 4 and were never about distance, so D_k is huge
        # (never binding) and pi_k is 0 (no turnaround penalty to worry
        # about either).
        "D_k": 1.0e9,
        "pi_k": 0.0,
    }


# Phase 3 requires several new instance keys (arc/node degradation, node
# handling capacity -- see model/model_vif.py's solve_vif docstring,
# "Schema note"). These tests predate Phase 3 and were never about
# degradation, so this helper supplies deliberately non-binding
# ("undamaged," effectively-unlimited-capacity) values for all of them,
# generated from the instance's own node/arc/scenario sets rather than
# hand-listed per test (hand-listing arcs here previously would have been
# an easy place for a silent typo to leave one arc's throughput at the
# unsafe 0.0 default -- see PHASE3_NOTES.md, "No safe default for
# nominal_throughput/node_handling_capacity"). Phase 4 added
# modal_arc_distance to this same non-binding bundle (0 km everywhere,
# paired with the D_k/pi_k above -- distance budget is never the thing
# under test in test_vif_phase1.py, see test_vif_phase4.py for that).
_BIG = 1.0e9


def _no_degradation_fields(nodes, modal_arcs, scenarios):
    return {
        "node_severity": {},                              # all default to 0.0 (undamaged)
        "disaster_type": {w: "none" for w in scenarios},   # degradation_matrix has no entry for "none" -> Gamma defaults to 0.0 regardless
        "degradation_matrix": {},
        "nominal_throughput": {
            mode: {arc: _BIG for arc in arcs} for mode, arcs in modal_arcs.items()
        },
        "node_handling_capacity": {
            (i, mode): _BIG for i in nodes for mode in modal_arcs
        },
        "node_handling_bonus": {},
        "modal_arc_distance": {
            mode: {arc: 0.0 for arc in arcs} for mode, arcs in modal_arcs.items()
        },
    }


def _run(instance, **kwargs):
    return solve_stochastic_cvar(
        instance, vehicle_formulation="vif", verbose=False, mip_gap=1e-6, **kwargs
    )


# ---------------------------------------------------------------------------
# Test 1: basic feasibility, hand-computed optimum
# ---------------------------------------------------------------------------
#
# Network: nodes {1, 2, 3}. N^P = {1, 3}. Arcs (air): (1,2),(2,1),(3,2),(2,3).
#   Node 1: PPL candidate, holds stock (q_bar = 1000 food).
#   Node 2: pure demand sink (not PPL-eligible), d_2 = 100 food.
#   Node 3: PPL candidate, but zero stock (q_bar = 0) -- a basing-only
#     candidate, distinct from node 1's "plausible PPL" role.
# Vehicle type T: fleet_size=2, J_T = {1, 3} (both candidates eligible).
# P_max = 1 (forces a single PPL choice -- see below for why this is what
#   makes the optimum unique rather than tied).
#
# HAND COMPUTATION
# -----------------
# Only node 1 has stock, so p_1=1 is required to deliver anything; with
# p_3=0 (P_max=1 already spent on node 1) any positive delivery beats the
# 500/unit penalty by four orders of magnitude, so p_1=1, p_3=0 is the
# unique optimum for C2/C3.
#
# C5 (b_l,j <= p_j) then forces b_{l,3} = 0 for BOTH vehicles (j=3 is
# ineligible since p_3=0), and C4 (exactly one base) forces b_{l,1} = 1 for
# both -- both vehicles end up based at node 1, uniquely (no tie: node 3 is
# the only alternative and it's blocked).
#
# Demand: deliver exactly 100 units 1->2 (cap_tons=150 >= 100, so one
# vehicle in one trip suffices; shipping more only adds transport cost with
# no offsetting benefit, so x=100 exactly, not more).
#   Vehicle v1 (l=1): n[1,1,2]=1 (departs home), nbar[1,2]=1 (route
#     terminates at 2, having delivered -- no return trip modeled this
#     phase), all other n/nbar for v1 = 0.
#     Conservation check at i=1: n[1,1,2] + nbar[1,1] - n[1,2,1]
#       = 1 + 0 - 0 = 1 = b[1,1]. OK.
#     Conservation check at i=2: (n[1,2,1]+n[1,2,3]) + nbar[1,2]
#       - (n[1,1,2]+n[1,3,2]) = 0 + 1 - 1 = 0 = b[1,2] (undefined -> 0,
#       since 2 not in J_T). OK.
#   Vehicle v2 (l=2): entirely unused. Conservation at i=1: 0 + nbar[2,1]
#     - 0 = b[2,1] = 1 -> nbar[2,1] = 1 (sits at its base, never departs).
#
# C6 at node 2 (r=food): inflow=100, release=0 (non-PPL), demand=100,
#   outflow=0 => z_2 + 100 = 100 + y_2 => z_2 = y_2. Only z costs anything
#   (500/unit) and y costs nothing, so minimizing forces z_2 = y_2 = 0.
# C6 at node 1 (r=food): inflow=0, release = 0.8*1000*1*p_1 = 800,
#   demand=0 (z_1 forced to 0 by its own UB = d_1 = 0), outflow=100
#   => 800 + 0 = 0 + y_1 + 100 => y_1 = 700 (forced, not chosen).
# C6 at node 3 (r=food): everything is 0 (no stock, no demand, no traffic)
#   => y_3 = 0.
#
# Objective (Lambda, single scenario): delta*z_2 (=0) + c*x (=0.1*100=10)
#   + EPSILON*n (=0.04*1, only n[1,1,2] is 1) = 10 + 0.04 = 10.04.


def test_c6_c7_c9_basic_feasibility_hand_computed_optimum():
    nodes = [1, 2, 3]
    modal_arcs = {"air": [(1, 2), (2, 1), (3, 2), (2, 3)]}
    instance = {
        "nodes": nodes,
        "ppl_nodes": [1, 3],
        "commodities": R,
        "scenarios": [1],
        "modes": MODES,
        "modal_arcs": modal_arcs,
        "modal_residual": [{}],
        "transfer_cap": {},
        "transfer_cost": {},
        "modal_arc_cost": {"air": {(1, 2): 0.1, (2, 1): 0.1, (3, 2): 0.1, (2, 3): 0.1}},
        "vehicle_types": {"T": _base_vehicle_type(J_k=[1, 3], fleet_size=2)},
        "probability": {1: 1.0},
        "demand": {
            (1, 1, "food"): 0, (1, 2, "food"): 100, (1, 3, "food"): 0,
        },
        "inventory_if_open": {(1, "food"): 1000, (3, "food"): 0},
        "safety_stock_fraction": 0.2,
        "site_cost": {1: 1, 3: 1},
        "selection_budget": 10,
        "penalty": {(i, "food"): 500 for i in [1, 2, 3]},
        "P_max": 1,
        "beta": 0.9,  # unused before Phase 2
        "resource_weight": {"food": 1.0},
        **_no_degradation_fields(nodes, modal_arcs, [1]),
    }

    results = _run(instance)

    assert results["status"] == "OPTIMAL"
    assert results["objective_value"] == pytest.approx(10.04, abs=1e-6)
    assert results["selected_sites"] == [1]
    assert results["basing"] == {(1, 1): 1, (2, 1): 1}
    assert results["vehicle_arcs"] == {(1, 1, 1, 2): 1}
    assert results["flows"] == pytest.approx({(1, 1, 1, 2, "food"): 100.0}, abs=1e-6)
    assert results["unmet_demand"] == {}
    assert results["retained"][(1, 1, "food")] == pytest.approx(700.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 2: isolates C5 (basing linked to activation) via the SELECTION
# BUDGET (C3) rather than the site count (C2), to confirm C5's forcing
# mechanism doesn't depend on which activation constraint is the binding
# one.
# ---------------------------------------------------------------------------
#
# Identical network/vehicle/demand to Test 1, except P_max=2 (not
# restrictive on its own) and selection_budget B=1 with f_1=1, f_3=5:
# activating node 3 alone already exceeds B (5 > 1), and activating both
# is even further out of reach, so C3 forces p_3=0 regardless of P_max.
# Everything downstream is identical to Test 1's hand computation: C5
# forces b_{l,3}=0 for both vehicles, C4 forces b_{l,1}=1 for both, and the
# delivery arithmetic (and objective) is unchanged at 10.04.
#
# This is the "C5 forces a vehicle's base to go unused" case: node 3 is a
# structurally valid basing candidate (J_T contains it) that becomes
# unusable purely because C5 links it to p_3, which C3 has forced to 0 --
# without C5, C4 alone would happily accept b_{l,3}=1 with no reference to
# p_3 at all.


def test_c5_basing_forced_unused_by_selection_budget():
    nodes = [1, 2, 3]
    modal_arcs = {"air": [(1, 2), (2, 1), (3, 2), (2, 3)]}
    instance = {
        "nodes": nodes,
        "ppl_nodes": [1, 3],
        "commodities": R,
        "scenarios": [1],
        "modes": MODES,
        "modal_arcs": modal_arcs,
        "modal_residual": [{}],
        "transfer_cap": {},
        "transfer_cost": {},
        "modal_arc_cost": {"air": {(1, 2): 0.1, (2, 1): 0.1, (3, 2): 0.1, (2, 3): 0.1}},
        "vehicle_types": {"T": _base_vehicle_type(J_k=[1, 3], fleet_size=2)},
        "probability": {1: 1.0},
        "demand": {
            (1, 1, "food"): 0, (1, 2, "food"): 100, (1, 3, "food"): 0,
        },
        "inventory_if_open": {(1, "food"): 1000, (3, "food"): 0},
        "safety_stock_fraction": 0.2,
        "site_cost": {1: 1, 3: 5},
        "selection_budget": 1,
        "penalty": {(i, "food"): 500 for i in [1, 2, 3]},
        "P_max": 2,
        "beta": 0.9,
        "resource_weight": {"food": 1.0},
        **_no_degradation_fields(nodes, modal_arcs, [1]),
    }

    results = _run(instance)

    assert results["status"] == "OPTIMAL"
    assert results["selected_sites"] == [1]
    # C5's effect: node 3 is J_T-eligible but never appears as a base,
    # solely because p_3 is forced to 0 by the (tight) selection budget.
    assert (1, 3) not in results["basing"]
    assert (2, 3) not in results["basing"]
    assert results["basing"] == {(1, 1): 1, (2, 1): 1}
    assert results["objective_value"] == pytest.approx(10.04, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 3: isolates C4 (exactly one base per vehicle) by giving a single
# vehicle two structurally DISCONNECTED demand pools it could reach from
# two different eligible bases.
# ---------------------------------------------------------------------------
#
# Network: nodes {1, 2, 3, 4}. N^P = {1, 3}. Arcs (air): (1,2) and (3,4)
# ONLY -- one-way, no return arcs, two disconnected components. (An earlier
# draft of this test included return arcs (2,1)/(4,3); see "PITFALL" below
# for why that breaks the test.)
#   Node 1: PPL, stock q_bar=1000, feeds sink node 2 (d_2=100).
#   Node 3: PPL, stock q_bar=1000, feeds sink node 4 (d_4=100).
# ONE vehicle only (fleet_size=1, l=1), J_T = {1, 3} (eligible at both).
# P_max=2, B=10 (both loose enough to afford activating both nodes, though
#   -- see the hand computation below -- only ONE of them ends up forced;
#   the other is a genuine tie the test does not assert on).
#
# WHY THIS ISOLATES C4
# ---------------------
# Nothing else in the constraint set links activity in the {1,2} component
# to activity in the {3,4} component for vehicle l=1 -- C9's conservation
# equation applies independently at each node, keyed off b[1,i]. If C4
# (sum_j b_{1,j} = 1) were absent, nothing would stop the solver setting
# BOTH b[1,1]=1 AND b[1,3]=1 simultaneously, which per C9 hands the SAME
# single vehicle index a free departure credit in EACH disconnected
# component -- i.e. one vehicle "serving" two disconnected trips at once,
# which is physically meaningless and exactly what C4 exists to prevent.
#
# PITFALL (found by running this test, not anticipated by hand): with a
# RETURN arc in the un-based component (e.g. (4,3) alongside (3,4)), C9
# alone -- with no subtour elimination (C11 is out of scope this phase) --
# permits a "phantom" zero-net 2-cycle n[1,3,4]=n[1,4,3]=1 even when
# b[1,3]=0: conservation nets to 0 at both nodes regardless, since the
# cycle's inflow and outflow cancel locally without ever routing through
# the vehicle's actual base. That phantom loop then gives C7 a costless
# channel to ship x through node 4 -- exactly the disconnected-subtour
# failure mode _build_subtour_callback's docstring documents for the old
# "individual" formulation, arising here for the same underlying reason
# (no subtour elimination), not a bug in C4 itself. Dropping the return
# arc closes this: a lone directed (3,4) with b[1,3]=0 forces
# n[1,3,4] + nbar[1,3] - 0 = 0 with both terms >= 0, so both are forced to
# 0 -- there is no zero-net way to use an arc with no matching return arc
# without a genuine base. This is why this test's network uses one-way
# arcs only, and it is documented in PHASE1_NOTES.md as a Phase-4
# (subtour elimination) dependency: C4 alone does not prevent a
# multi-vehicle-worth of phantom activity from a single based vehicle once
# an reachable cycle exists in the arc set.
#
# HAND COMPUTATION
# -----------------
# With C4 present, b[1,1] + b[1,3] = 1 exactly -- the vehicle can be based
# at 1 XOR 3, never both. Say b[1,1]=1, b[1,3]=0 (the symmetric case with
# b[1,3]=1 is identical with 2<->4 swapped for 1<->3):
#   Node 1 (home): n[1,1,2] + nbar[1,1] - 0 = b[1,1] = 1. Take n[1,1,2]=1,
#     nbar[1,1]=0 (departs).
#   Node 2 (no return arc exists): 0 + nbar[1,2] - n[1,1,2] = b[1,2] = 0
#     (2 not in J_T) => nbar[1,2] = 1 (terminates, having delivered).
#   Node 3 (b[1,3]=0, arc (3,4) is the only arc touching 3, no arc enters
#     3): n[1,3,4] + nbar[1,3] - 0 = 0, both terms >= 0 => both forced 0.
#   Node 4 (arc (3,4) is the only arc touching 4, no arc leaves 4):
#     0 + nbar[1,4] - n[1,3,4](=0) = b[1,4] = 0 (4 not in J_T) =>
#     nbar[1,4] = 0.
# So node 4 gets zero inflow, forcing z_4 = 100 (its full demand, penalty
# 500*100 = 50,000), while node 2 is fully served (cost 10.04 exactly as
# in Test 1). Node 1 must be activated (p_1=1) since C5 requires
# p[home] >= b[1,home] = 1 for the vehicle's actual base. Node 3's
# activation, by contrast, is a genuine tie: n[1,3,4] is forced to 0
# regardless of p_3 (shown above, driven by b[1,3]=0, not by p_3), and
# node 3 has zero demand of its own (d_3=0, so its own z is pinned to 0
# regardless), so p_3 has no effect on feasibility or the objective either
# way -- the test does not assert its value.
#
# So exactly one of the two sinks is served (cost 10.04) and the other is
# entirely unmet (cost 50,000). By the symmetry of the two components
# (identical stock, cost, demand), it does not matter which one Gurobi
# picks -- the objective is the same either way: 50,000 + 10.04 =
# 50,010.04. The test therefore asserts the tie-break-independent
# invariants: exactly one basing bit is set for the single vehicle, total
# unmet demand is exactly 100, and the objective is exactly 50,010.04 --
# rather than pinning which specific sink is served.


def test_c4_single_home_prevents_simultaneous_disconnected_service():
    nodes = [1, 2, 3, 4]
    modal_arcs = {"air": [(1, 2), (3, 4)]}
    instance = {
        "nodes": nodes,
        "ppl_nodes": [1, 3],
        "commodities": R,
        "scenarios": [1],
        "modes": MODES,
        "modal_arcs": modal_arcs,
        "modal_residual": [{}],
        "transfer_cap": {},
        "transfer_cost": {},
        "modal_arc_cost": {"air": {(1, 2): 0.1, (3, 4): 0.1}},
        "vehicle_types": {"T": _base_vehicle_type(J_k=[1, 3], fleet_size=1)},
        "probability": {1: 1.0},
        "demand": {
            (1, 1, "food"): 0, (1, 2, "food"): 100,
            (1, 3, "food"): 0, (1, 4, "food"): 100,
        },
        "inventory_if_open": {(1, "food"): 1000, (3, "food"): 1000},
        "safety_stock_fraction": 0.2,
        "site_cost": {1: 1, 3: 1},
        "selection_budget": 10,
        "penalty": {(i, "food"): 500 for i in [1, 2, 3, 4]},
        "P_max": 2,
        "beta": 0.9,
        "resource_weight": {"food": 1.0},
        **_no_degradation_fields(nodes, modal_arcs, [1]),
    }

    results = _run(instance)

    assert results["status"] == "OPTIMAL"

    # C4's invariant: the single vehicle (l=1) has exactly one basing bit
    # set, never both (1,1) and (1,3) simultaneously.
    assert set(results["basing"].keys()) <= {(1, 1), (1, 3)}
    assert len(results["basing"]) == 1

    # Whichever node is the vehicle's actual base must be activated (C5).
    # The OTHER node's activation is a genuine tie (see hand computation
    # above -- it affects nothing) and is deliberately not asserted on.
    home_node = 1 if (1, 1) in results["basing"] else 3
    assert home_node in results["selected_sites"]

    total_unmet = sum(results["unmet_demand"].values())
    assert total_unmet == pytest.approx(100.0, abs=1e-6)
    assert results["objective_value"] == pytest.approx(50010.04, abs=1e-6)

    # Whichever sink was served, delivery was for exactly 100 units on
    # exactly one vehicle-arc (no fractional/split delivery).
    assert sum(results["flows"].values()) == pytest.approx(100.0, abs=1e-6)
    assert len(results["vehicle_arcs"]) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
