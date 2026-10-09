"""
test_vif_phase2.py

Hand-verified tests for the CVaR risk wrapper added to vehicle_formulation
="vif" in Phase 2 (eta, xi, loss[w] as a variable, C1/cvar1
(xi^w >= Lambda^w - eta), and the objective eta + (1/(1-beta)) * sum_w
pi^w * xi^w -- see model/model_vif.py's solve_vif and
docs/PRSVIF_Gospel.md, "Risk-Aware Vehicle-Indexed Model", eq:vif:objective/
cvar1/cvar2). Phase 1's tests (test_vif_phase1.py) already hand-verify C4,
C5, C6, C7, C9 on single-scenario instances; this file's one instance is
deliberately as simple as possible on the C4/C5/C6/C7/C9 side (a single
vehicle with only one eligible base, so basing is forced and uninteresting)
so the arithmetic in each test is entirely about the CVaR mechanism itself.

ONE shared toy instance, four equally-likely scenarios (Omega = {1,2,3,4},
pi^w = 0.25 each), varying only in beta across the three tests below.
Network: nodes {1, 2}, arc (1,2) only. Node 1: PPL, J_T = {1} (the vehicle's
only eligible base, so C4 trivially forces b[1,1]=1 -- no basing ambiguity
to reason about here), stock q_bar = 10000 (never binding: releasable =
0.8*10000 = 8000 >= the largest single-scenario demand, 400). Node 2: pure
sink, demand d_2^w = 100, 200, 300, 400 for w = 1, 2, 3, 4 respectively.
cap_tons = 500 (>= 400, so one vehicle in one trip always suffices; C7
never binds). c_l,ij = 0.1, EPSILON = 0.04 (same values as Phase 1's
tests), delta = 500.

Because full delivery is always far cheaper than the 500/unit penalty
(exactly as established in test_vif_phase1.py), each scenario is fully
served: z^w = 0 always, and Lambda^w = 0.1 * d_2^w + 0.04 * 1 (one
vehicle-arc used). This gives four DISTINCT, easy-to-rank per-scenario
losses -- the "ground truth" every test below reasons about:

    Lambda^1 = 0.1*100 + 0.04 = 10.04
    Lambda^2 = 0.1*200 + 0.04 = 20.04
    Lambda^3 = 0.1*300 + 0.04 = 30.04
    Lambda^4 = 0.1*400 + 0.04 = 40.04

These four numbers were confirmed against the solver (not just asserted
from the formula) before being used as the basis for the three hand
computations below -- the first draft of this file's beta=0.6 case had an
arithmetic slip (computed 40.04 - 30.04 as 10.04 instead of 10.00) that a
solver cross-check caught; see the git history / PHASE2_NOTES.md for that
detail. Moral, restated for whoever reads this next: check hand arithmetic
against a numeric run before trusting it in a docstring.

Run: cd aps_usarpac && pytest tests/test_vif_phase2.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from model.model import solve_stochastic_cvar

R = ["food"]
LAMBDA = {1: 10.04, 2: 20.04, 3: 30.04, 4: 40.04}


def _instance(beta):
    return {
        "nodes": [1, 2],
        "ppl_nodes": [1],
        "commodities": R,
        "scenarios": [1, 2, 3, 4],
        "modes": ["air"],
        "modal_arcs": {"air": [(1, 2)]},
        "modal_residual": [{}, {}, {}, {}],
        "transfer_cap": {},
        "transfer_cost": {},
        "modal_arc_cost": {"air": {(1, 2): 0.1}},
        "vehicle_types": {
            "T": {
                "mode": "air", "fleet_size": 1, "J_k": [1], "cap_tons": 500.0,
                "D_k": 1.0e9, "pi_k": 0.0,  # Phase 4: non-binding, not under test here
            }
        },
        "probability": {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25},
        "demand": {
            (1, 1, "food"): 0, (1, 2, "food"): 100,
            (2, 1, "food"): 0, (2, 2, "food"): 200,
            (3, 1, "food"): 0, (3, 2, "food"): 300,
            (4, 1, "food"): 0, (4, 2, "food"): 400,
        },
        "inventory_if_open": {(1, "food"): 10000},
        "safety_stock_fraction": 0.2,
        "site_cost": {1: 1},
        "selection_budget": 10,
        "penalty": {(i, "food"): 500 for i in [1, 2]},
        "P_max": 1,
        "beta": beta,
        "resource_weight": {"food": 1.0},
        # Phase 3 degradation data -- deliberately non-binding ("undamaged")
        # values, since this file is entirely about the CVaR wrapper, not
        # degradation (see test_vif_phase1.py's _no_degradation_fields for
        # the same pattern, kept inline here rather than shared across test
        # files to keep each test file self-contained).
        "node_severity": {},
        "disaster_type": {w: "none" for w in [1, 2, 3, 4]},
        "degradation_matrix": {},
        "nominal_throughput": {"air": {(1, 2): 1.0e9}},
        "node_handling_capacity": {(1, "air"): 1.0e9, (2, "air"): 1.0e9},
        "node_handling_bonus": {},
        "modal_arc_distance": {"air": {(1, 2): 0.0}},  # Phase 4: paired with D_k above
    }


def _run(beta):
    return solve_stochastic_cvar(
        _instance(beta), vehicle_formulation="vif", verbose=False, mip_gap=1e-9,
    )


# ---------------------------------------------------------------------------
# Test A: beta = 0.6 -- genuine multi-scenario blending (the main case).
# ---------------------------------------------------------------------------
#
# HAND COMPUTATION
# -----------------
# The CVaR LP (for fixed Lambda^w) is: minimize f(eta) = eta +
# (1/(1-beta)) * sum_w pi^w * max(0, Lambda^w - eta), a convex, piecewise-
# linear function of eta with breakpoints at each Lambda^w. Its slope on
# the open interval between consecutive breakpoints is
# 1 - (1/(1-beta)) * P(Lambda > eta) (falls by 1/(1-beta) times the tail
# probability each time eta crosses one more Lambda^w from below). With
# beta=0.6, 1/(1-beta) = 2.5:
#   eta in (20.04, 30.04): P(Lambda > eta) = P({30.04, 40.04}) = 0.5
#     -> slope = 1 - 2.5*0.5 = -0.25 (still decreasing)
#   eta in (30.04, 40.04): P(Lambda > eta) = P({40.04}) = 0.25
#     -> slope = 1 - 2.5*0.25 = +0.375 (now increasing)
# The slope flips sign exactly at eta = 30.04 (= Lambda^3), so that is the
# UNIQUE minimizer (a genuine kink, not a flat tie -- unlike Test B below,
# beta=0.6 does not coincide with a cumulative-probability breakpoint of
# {0.25, 0.5, 0.75, 1.0}, so there is no flat optimal interval here).
#
# At eta = 30.04: xi^1 = xi^2 = xi^3 = max(0, Lambda^w - 30.04) = 0 (all
# <= 30.04); xi^4 = max(0, 40.04 - 30.04) = 10.00 exactly (NOT 10.04 --
# this is the arithmetic slip flagged in the module docstring).
# Objective = eta + 2.5 * sum_w pi^w * xi^w
#           = 30.04 + 2.5 * (0.25 * 10.00) = 30.04 + 6.25 = 36.29.


def test_cvar_beta_0p6_multi_scenario_blending():
    results = _run(0.6)

    assert results["status"] == "OPTIMAL"
    assert results["scenario_losses"] == pytest.approx(LAMBDA, abs=1e-6)
    assert results["eta"] == pytest.approx(30.04, abs=1e-6)
    assert results["xi"] == pytest.approx(
        {1: 0.0, 2: 0.0, 3: 0.0, 4: 10.00}, abs=1e-6
    )
    assert results["objective_value"] == pytest.approx(36.29, abs=1e-6)


# ---------------------------------------------------------------------------
# Test B: beta = 0.0 -- CVaR collapses to plain expectation.
# ---------------------------------------------------------------------------
#
# HAND COMPUTATION
# -----------------
# At beta=0, 1/(1-beta) = 1. For any eta <= min_w(Lambda^w) = 10.04, every
# scenario is in the "tail" (Lambda^w - eta > 0 for all w), so
# sum_w pi^w * max(0, Lambda^w - eta) = sum_w pi^w*(Lambda^w - eta)
#   = E[Lambda] - eta  (since sum_w pi^w = 1),
# making f(eta) = eta + (E[Lambda] - eta) = E[Lambda], a CONSTANT,
# independent of eta, on the entire interval eta <= 10.04. This is a
# genuine tie: any eta in that range (and the xi^w values that go with it)
# is optimal, achieving the same objective. This test therefore asserts
# ONLY the objective value -- not eta or xi, which are not unique here --
# unlike Test A, which has a genuine kink with no tie.
#
# E[Lambda] = 0.25*(10.04+20.04+30.04+40.04) = 0.25*100.16 = 25.04.


def test_cvar_beta_0_collapses_to_expectation():
    results = _run(0.0)

    assert results["status"] == "OPTIMAL"
    assert results["objective_value"] == pytest.approx(25.04, abs=1e-6)
    # eta/xi are not unique at beta=0 (see hand computation above) -- not
    # asserted. Sanity-check the tie's defining property instead: eta must
    # be <= the smallest scenario loss for the flat region's algebra to
    # apply, and xi^w must equal Lambda^w - eta exactly at whatever eta the
    # solver picked (i.e. every scenario is in the "tail" simultaneously).
    eta = results["eta"]
    assert eta <= LAMBDA[1] + 1e-6
    for w in [1, 2, 3, 4]:
        assert results["xi"][w] == pytest.approx(LAMBDA[w] - eta, abs=1e-6)


# ---------------------------------------------------------------------------
# Test C: beta = 0.99 -- CVaR collapses to the single worst scenario.
# ---------------------------------------------------------------------------
#
# HAND COMPUTATION
# -----------------
# At beta=0.99, 1/(1-beta) = 100. For eta in (30.04, 40.04), only scenario
# 4 is in the tail (prob 0.25): slope = 1 - 100*0.25 = -24 (steeply
# decreasing -- push eta up as far as possible). For eta > 40.04, no
# scenario is in the tail: slope = 1 - 0 = +1 (increasing -- don't go
# further). The unique minimizer is therefore eta = 40.04 (= Lambda^4,
# the worst scenario), where xi^w = max(0, Lambda^w - 40.04) = 0 for every
# w (nothing exceeds the maximum). Objective = 40.04 + 100*0 = 40.04
# exactly -- CVaR at high beta reduces to pure worst-case, matching the
# gospel's stated motivation for the risk measure (Section 1.2/2.2:
# protecting against tail outcomes).
#
# This is qualitatively the mirror image of Test B: beta=0 collapses to
# the (cheap, average-case) expectation; beta near 1 collapses to the
# (expensive, tail) worst case. Both are degenerate sanity checks on the
# same CVaR machinery Test A exercises in its "normal," blended form.


def test_cvar_beta_0p99_collapses_to_worst_case():
    results = _run(0.99)

    assert results["status"] == "OPTIMAL"
    assert results["eta"] == pytest.approx(40.04, abs=1e-6)
    assert results["xi"] == pytest.approx({1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}, abs=1e-6)
    assert results["objective_value"] == pytest.approx(40.04, abs=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
