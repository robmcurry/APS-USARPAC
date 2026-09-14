"""
test_vif_phase3.py

Hand-verified tests for the degradation and node-handling-capacity
machinery added to vehicle_formulation="vif" in Phase 3: real a^w_i
(eq:availability, replacing Phase 1/2's hardcoded 1), real
min{T^w_l,ij, cap_l} in C7 (eq:residual, replacing the collapsed-to-cap_l
placeholder), and the new C8 node handling capacity constraint
(eq:vif:transfercap, using Theta^w_i,m from eq:thetadegradation) -- see
model/model_vif.py's solve_vif and docs/PRSVIF_Gospel.md, "Disaster Impact
on Network Capacity."

Three tests, each isolating one of the three newly-active mechanisms, kept
on the smallest network that can exercise it (mostly reusing Phase 1's
2-node, single-arc, single-eligible-base pattern so C4/C5/C6/C7/C9's
baseline behavior -- already hand-verified in test_vif_phase1.py -- isn't
being re-derived here):

    Test 1: arc throughput degradation (eq:residual) actually binding C7,
            below the vehicle's own payload capacity.
    Test 2: node handling capacity Theta_i,m (eq:thetadegradation) binding
            C8, and its activation bonus DeltaTheta_i,m actually changing
            the optimal siting decision.
    Test 3: node availability a^w_i (eq:availability) blocking both a
            PPL's release (C6) and a based vehicle's departure credit (C9)
            in the one scenario where that node is severely hit, while
            leaving it fully available in a calmer scenario -- using the
            SAME first-stage basing decision in both.

Shared conventions with test_vif_phase1.py/test_vif_phase2.py: R = {"food"},
resource_weight = {"food": 1.0}, c_l,ij = 0.1, EPSILON = 0.04, delta = 500,
rho = 0.2 (releasable_fraction = 0.8).

Run: cd aps_usarpac && pytest tests/test_vif_phase3.py -v
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
# Test 1: arc throughput degradation (eq:residual) binds C7 below cap_l.
# ---------------------------------------------------------------------------
#
# Network: nodes {1, 2}, arc (1,2) only. Node 1: PPL, stock q_bar=1000
# (releasable = 800, non-binding). Node 2: pure sink, d_2 = 100.
# One vehicle, J_T={1} (forced base, no C4/C5 ambiguity), cap_tons=500
# (deliberately large -- payload is NOT what binds this test).
#
# Nominal arc throughput T_air,(1,2) = 100 (metric tons -- the
# INFRASTRUCTURE cap, distinct from the vehicle's own 500-ton payload).
# Disaster type "storm", Gamma_air,storm = 0.8, alpha = 1.0 -> gamma = 0.8.
# Node severities: sigma_1 = 0 (epicenter elsewhere), sigma_2 = 2.5 ->
# max(sigma_1, sigma_2) = 2.5.
#
# HAND COMPUTATION
# -----------------
# degradation_factor = max(0, 1 - 0.8 * 2.5/5) = max(0, 1 - 0.4) = 0.6
# T^w_(1,2) = 100 * 0.6 = 60 (metric tons) -- BELOW the vehicle's own
# 500-ton cap, so C7's min{T, cap_l} = 60 is what actually binds, not the
# vehicle's payload (already covered by test_vif_phase1.py).
# w_food = 1 ton/unit, so at most 60 units can move on this arc regardless
# of how many vehicles attempt it (only one exists here anyway).
#
# Deliver as much as physically possible (60 << 100, and transport is far
# cheaper than the 500/unit penalty, exactly as in test_vif_phase1.py):
# x = 60, z_2 = 100 - 60 = 40 (forced: C6 at node 2 gives
# 60 + z_2 = 100 + y_2, minimal z_2 = 40 with y_2 = 0).
# Lambda = delta*40 + c*60 + EPSILON*1 = 500*40 + 0.1*60 + 0.04
#        = 20000 + 6 + 0.04 = 20006.04.


def test_c7_arc_throughput_degradation_binds_below_payload():
    instance = {
        "nodes": [1, 2],
        "ppl_nodes": [1],
        "commodities": R,
        "scenarios": [1],
        "modes": ["air"],
        "modal_arcs": {"air": [(1, 2)]},
        "modal_residual": [{}],
        "transfer_cap": {},
        "transfer_cost": {},
        "modal_arc_cost": {"air": {(1, 2): 0.1}},
        "vehicle_types": {
            "T": {
                "mode": "air", "fleet_size": 1, "J_k": [1], "cap_tons": 500.0,
                "D_k": 1.0e9, "pi_k": 0.0,  # Phase 4: non-binding, not under test here
            }
        },
        "probability": {1: 1.0},
        "demand": {(1, 1, "food"): 0, (1, 2, "food"): 100},
        "inventory_if_open": {(1, "food"): 1000},
        "safety_stock_fraction": 0.2,
        "site_cost": {1: 1},
        "selection_budget": 10,
        "penalty": {(i, "food"): 500 for i in [1, 2]},
        "P_max": 1,
        "beta": 0.9,  # single scenario -> CVaR collapses to Lambda^1 regardless (Phase 2)
        "resource_weight": {"food": 1.0},
        "node_severity": {(1, 1): 0.0, (1, 2): 2.5},
        "disaster_type": {1: "storm"},
        "degradation_matrix": {"air": {"storm": 0.8}},
        "alpha": 1.0,
        "nominal_throughput": {"air": {(1, 2): 100.0}},
        "node_handling_capacity": {(1, "air"): 1.0e9, (2, "air"): 1.0e9},
        "node_handling_bonus": {},
        "modal_arc_distance": {"air": {(1, 2): 0.0}},  # Phase 4: paired with D_k above
    }

    results = _run(instance)

    assert results["status"] == "OPTIMAL"
    assert results["objective_value"] == pytest.approx(20006.04, abs=1e-6)
    assert results["flows"] == pytest.approx(
        {(1, 1, 1, 2, "food"): 60.0}, abs=1e-6
    )
    assert results["unmet_demand"] == pytest.approx(
        {(1, 2, "food"): 40.0}, abs=1e-6
    )


# ---------------------------------------------------------------------------
# Test 2: node handling capacity Theta_i,m (C8) binds, and its activation
# bonus DeltaTheta_i,m changes the optimal siting decision.
# ---------------------------------------------------------------------------
#
# Network: nodes {1, 2}, arc (1,2) only. Node 1: PPL/base, J_T={1}, stock
# q_bar=1000. Node 2: PPL-ELIGIBLE sink (N^P={1,2}) but zero stock -- its
# only role is to demonstrate the handling-capacity activation bonus.
# TWO vehicles (fleet_size=2), cap_tons=60 each (so one vehicle alone can
# cover at most 60 of the 100-unit demand). Arc throughput T_air,(1,2)=1000
# and severities all 0 (undamaged) -- deliberately non-binding, isolating
# C8 from C7's degradation (already covered in Test 1).
#
# Node 2's air handling capacity: Theta_2,air = 1 (baseline -- only ONE
# mode-air arrival allowed per horizon), DeltaTheta_2,air = 1 (activation
# bonus -- activating node 2 as a PPL doubles its capacity to 2 arrivals).
#
# HAND COMPUTATION
# -----------------
# If p_2 = 0: Theta^w_2,air = (1 + 1*0)*1 = 1 -- C8 permits only ONE of the
#   two vehicles to arrive; the other is blocked from using the arc at all
#   (n capped at 1 total arrival). Max deliverable = 60 (one vehicle's
#   cap), z_2 = 40. Lambda = 500*40 + 0.1*60 + 0.04*1 = 20006.04.
# If p_2 = 1: Theta^w_2,air = (1 + 1*1)*1 = 2 -- BOTH vehicles may arrive.
#   Max deliverable = 120 >= 100, so demand is fully served (some split of
#   100 across the two vehicles' <=60-unit legs, e.g. 60+40 -- the exact
#   split is not asserted, only that both vehicles fly). z_2 = 0.
#   Lambda = 500*0 + 0.1*100 + 0.04*2 = 10 + 0.08 = 10.08.
# Activating node 2 costs nothing in the objective (f_i only appears in
# C3's budget constraint, never in Lambda) and P_max=2/B=10 afford it, so
# p_2=1 strictly dominates (10.08 << 20006.04) -- this is not a tie, the
# solver MUST choose p_2=1. This is a direct instance of the gospel's own
# stated rationale for DeltaTheta: "Siting therefore buys throughput as
# well as inventory."


def test_c8_node_handling_capacity_activation_bonus_changes_siting():
    instance = {
        "nodes": [1, 2],
        "ppl_nodes": [1, 2],
        "commodities": R,
        "scenarios": [1],
        "modes": ["air"],
        "modal_arcs": {"air": [(1, 2)]},
        "modal_residual": [{}],
        "transfer_cap": {},
        "transfer_cost": {},
        "modal_arc_cost": {"air": {(1, 2): 0.1}},
        "vehicle_types": {
            "T": {
                "mode": "air", "fleet_size": 2, "J_k": [1], "cap_tons": 60.0,
                "D_k": 1.0e9, "pi_k": 0.0,  # Phase 4: non-binding, not under test here
            }
        },
        "probability": {1: 1.0},
        "demand": {(1, 1, "food"): 0, (1, 2, "food"): 100},
        "inventory_if_open": {(1, "food"): 1000, (2, "food"): 0},
        "safety_stock_fraction": 0.2,
        "site_cost": {1: 1, 2: 1},
        "selection_budget": 10,
        "penalty": {(i, "food"): 500 for i in [1, 2]},
        "P_max": 2,
        "beta": 0.9,
        "resource_weight": {"food": 1.0},
        "node_severity": {},
        "disaster_type": {1: "none"},
        "degradation_matrix": {},
        "alpha": 1.0,
        "nominal_throughput": {"air": {(1, 2): 1000.0}},
        "node_handling_capacity": {(1, "air"): 1.0e9, (2, "air"): 1.0},
        "node_handling_bonus": {(2, "air"): 1.0},
        "modal_arc_distance": {"air": {(1, 2): 0.0}},  # Phase 4: paired with D_k above
    }

    results = _run(instance)

    assert results["status"] == "OPTIMAL"
    assert results["objective_value"] == pytest.approx(10.08, abs=1e-6)
    assert sorted(results["selected_sites"]) == [1, 2]
    assert len(results["vehicle_arcs"]) == 2  # both vehicles fly
    assert sum(results["flows"].values()) == pytest.approx(100.0, abs=1e-6)
    assert results["unmet_demand"] == {}


# ---------------------------------------------------------------------------
# Test 3: node availability a^w_i (C6 + C9) -- the same first-stage base
# behaves differently across scenarios depending on realized severity.
# ---------------------------------------------------------------------------
#
# Network: nodes {1, 2}, arc (1,2) only. Node 1: PPL/base, J_T={1}, stock
# q_bar=1000. Node 2: sink, d_2^w=60 in BOTH scenarios. One vehicle,
# cap_tons=100 (>=60, non-binding). T and Theta both set large/undegraded
# (Gamma left empty -> gamma=0 everywhere), isolating 'a' from Tests 1-2's
# mechanisms entirely -- degradation_factor is always 1 regardless of
# severity here, so only eq:availability's hard 0/1 threshold is at work.
#
# Two equally-likely scenarios sharing the SAME first-stage basing (b is
# first-stage, chosen once): scenario 1 ("calm"), sigma_1=0 -> a^1_1=1;
# scenario 2 ("disaster at the base"), sigma_1=1.5 (>=1) -> a^2_1=0.
#
# HAND COMPUTATION
# -----------------
# C4/J_T={1} forces b[1,1]=1 exactly as in test_vif_phase1.py (the only
# eligible node) -- this decision is shared across both scenarios.
#
# Scenario 1 (a^1_1=1): C9 at node 1 gives departures + nbar - arrivals
#   = a^1_1 * b[1,1] = 1*1 = 1, exactly as in test_vif_phase1.py's Test 1
#   -- the vehicle can and does depart, delivering the full 60.
#   Lambda^1 = c*60 + EPSILON*1 = 0.1*60 + 0.04 = 6.04.
# Scenario 2 (a^2_1=0): C9 at node 1 gives departures + nbar - arrivals
#   = 0 * 1 = 0. With no incoming arc to node 1 (one-way arc only, per
#   test_vif_phase1.py's established practice against phantom loops),
#   arrivals=0, so departures + nbar = 0 forces BOTH to 0 -- the vehicle
#   is stranded, cannot depart AT ALL this scenario, regardless of b.
#   Simultaneously, C6's release term at node 1 is
#   releasable_fraction*q_bar*a^2_1*p_1 = 0.8*1000*0*1 = 0 -- even though
#   p_1=1 and stock exists, availability zeroes the release too.
#   With no inflow to node 2 and z_2's UB=d_2=60, z_2 is forced to exactly
#   60 (the C6 equation z_2 = 60 + y_2 with y_2>=0 forces z_2>=60, capped
#   at 60 by dom7). Lambda^2 = delta*60 = 500*60 = 30000.
#
# CVaR-mechanics note (this is what test_vif_phase2.py's Test A/C already
# established, applied here): at beta=0.9 with only 2 equally-likely
# scenarios, the tail (1-beta=0.1) is smaller than either scenario's own
# probability (0.5), so CVaR_0.9 collapses to pure max(Lambda^1, Lambda^2)
# REGARDLESS of Lambda^1's actual value (as long as it stays <=
# Lambda^2) -- which means at beta=0.9 the solver has ZERO incentive to
# actually optimize scenario 1's recourse well; any feasible Lambda^1 <=
# 30000 is equally optimal, including badly-routed ones. This isn't a
# hypothetical: it's what a first draft of this test (at beta=0.9) actually
# produced -- Gurobi reported MIPGap=0.0 (a certified true optimum) with
# scenario 1 delivering x=100 (over-supplying the 60-unit demand) and
# z_1=59.98 (a needless ~30,000 in avoidable penalty), because nothing in
# the beta=0.9 objective cared. This test therefore uses beta=0.0 (plain
# expectation) instead, which weights every scenario's recourse quality
# directly and eliminates that degeneracy -- see PHASE3_NOTES.md, "A test
# design mistake, not a code bug" for the full account.
#
# Objective = 0.5*Lambda^1 + 0.5*Lambda^2 = 0.5*(6.04 + 30000) = 15003.02.


def test_availability_blocks_release_and_departure_in_disaster_scenario():
    instance = {
        "nodes": [1, 2],
        "ppl_nodes": [1],
        "commodities": R,
        "scenarios": [1, 2],
        "modes": ["air"],
        "modal_arcs": {"air": [(1, 2)]},
        "modal_residual": [{}, {}],
        "transfer_cap": {},
        "transfer_cost": {},
        "modal_arc_cost": {"air": {(1, 2): 0.1}},
        "vehicle_types": {
            "T": {
                "mode": "air", "fleet_size": 1, "J_k": [1], "cap_tons": 100.0,
                "D_k": 1.0e9, "pi_k": 0.0,  # Phase 4: non-binding, not under test here
            }
        },
        "probability": {1: 0.5, 2: 0.5},
        "demand": {
            (1, 1, "food"): 0, (1, 2, "food"): 60,
            (2, 1, "food"): 0, (2, 2, "food"): 60,
        },
        "inventory_if_open": {(1, "food"): 1000},
        "safety_stock_fraction": 0.2,
        "site_cost": {1: 1},
        "selection_budget": 10,
        "penalty": {(i, "food"): 500 for i in [1, 2]},
        "P_max": 1,
        "beta": 0.0,  # NOT 0.9 -- see hand computation above ("CVaR-mechanics note")
        "resource_weight": {"food": 1.0},
        "node_severity": {(1, 1): 0.0, (1, 2): 0.0, (2, 1): 1.5, (2, 2): 0.0},
        "disaster_type": {1: "none", 2: "none"},
        "degradation_matrix": {},  # empty -> gamma=0 everywhere -> T/Theta undegraded; isolates 'a'
        "alpha": 1.0,
        "nominal_throughput": {"air": {(1, 2): 1000.0}},
        "node_handling_capacity": {(1, "air"): 1.0e9, (2, "air"): 1.0e9},
        "node_handling_bonus": {},
        "modal_arc_distance": {"air": {(1, 2): 0.0}},  # Phase 4: paired with D_k above
    }

    results = _run(instance)

    assert results["status"] == "OPTIMAL"
    assert results["node_availability"] == {
        (1, 1): 1.0, (1, 2): 1.0, (2, 1): 0.0, (2, 2): 1.0,
    }
    assert results["scenario_losses"] == pytest.approx(
        {1: 6.04, 2: 30000.0}, abs=1e-6
    )
    assert results["objective_value"] == pytest.approx(15003.02, abs=1e-6)
    # Scenario 1 delivers fully; scenario 2's vehicle cannot depart at all.
    assert results["flows"] == pytest.approx(
        {(1, 1, 1, 2, "food"): 60.0}, abs=1e-6
    )
    assert results["vehicle_arcs"] == {(1, 1, 1, 2): 1}
    assert results["unmet_demand"] == pytest.approx(
        {(2, 2, "food"): 60.0}, abs=1e-6
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
