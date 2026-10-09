# PRS-VIF Phase 3 Notes — Arc/node degradation + node handling capacity

Continuation of Phase 1/2 (`PHASE1_NOTES.md`, `PHASE2_NOTES.md`). Same
authoritative source: `aps_usarpac/docs/PRSVIF_Gospel.md`, "Disaster Impact
on Network Capacity" and "Risk-Aware Vehicle-Indexed Model." Scope was
exactly what Phase 1's TODOs flagged: arc/node degradation (`eq:residual`,
`eq:thetadegradation`, `eq:availability`) and node-handling capacity
(`eq:vif:transfercap`), taken directly from those TODOs, no new prompt.

## What was built

- `aps_usarpac/model/model_vif.py`:
  - New required `instance` keys, all read directly inside `solve_vif`
    (same pattern as Phase 1's `resource_weight`/`cap_tons`/`J_k` —
    nothing threaded through `model.py`'s call site, so **`model.py`
    needed zero changes this phase**, confirmed by `git diff --stat`
    showing the same +32/-1 as Phase 2): `node_severity`, `disaster_type`,
    `degradation_matrix`, `alpha` (optional, default 1.0),
    `nominal_throughput`, `node_handling_capacity`, `node_handling_bonus`.
  - `a[w,i]` (eq:availability) and `gamma[w,m]` (eq:gammascale)
    precomputed once up front — both are plain Python floats derived only
    from scenario data, no decision variables involved.
  - `_degradation_factor(gamma, severity_term)`: the `max(0, 1 -
    gamma*severity_term/5)` factor shared verbatim by `eq:residual` and
    `eq:thetadegradation` (same formula, different `severity_term` — arc
    uses `max(sigma_i, sigma_j)`, node uses `sigma_i` alone, exactly per
    gospel's "an arc's degradation is a function of its most affected
    endpoint... a node degrades by local severity alone").
  - C6 (`VifResourceBalance`) and C9 (`VifVehicleConservation`): the
    hardcoded `1` for `a^w_i` replaced with the real precomputed value.
  - C7 (`VifVehicleCapacity`): `cap_l` replaced with `min(T_residual, cap_l)`,
    `T_residual` precomputed once per `(w, mode, i, j)` (shared across
    every vehicle of that mode, per gospel's `T_l,ij := T_m,ij`).
  - **New** C8 (`VifNodeHandlingCapacity`, `eq:vif:transfercap`): total
    mode-`m` arrivals at node `i` capped by `Theta^w_i,m`. `L_m` (vehicles
    grouped by mode) added to support this.
  - `results["node_availability"]` added as a diagnostic (precomputed `a`
    values, not solution-dependent — always present even off-optimum).
- `aps_usarpac/tests/test_vif_phase3.py` (new file): three hand-verified
  tests, each isolating one mechanism — arc-throughput degradation binding
  below payload capacity, node-handling capacity's activation bonus
  actually changing the optimal siting decision, and node availability
  blocking both release and departure credit in a disaster scenario while
  the same first-stage basing decision stays fully usable in a calmer one.
- **Migrated** `test_vif_phase1.py`'s 3 instances and `test_vif_phase2.py`'s
  1 shared instance builder to supply the new required fields with
  deliberately non-binding ("undamaged," effectively-unlimited-capacity)
  values, so their original hand-verified numbers are preserved exactly —
  confirmed by re-running all 6 pre-existing tests, all still passing with
  identical asserted values.

## No McCormick auxiliary needed for Theta's activation bonus

`Theta^w_i,m = (Theta_i,m + DeltaTheta_i,m * p_i) * degradation_factor` —
the `DeltaTheta_i,m * p_i` term is a **constant coefficient times a
decision variable**, not a product of two decision variables. Gurobi
handles that natively as a linear term; no auxiliary variable is needed at
all. This is worth naming explicitly because the aggregate/individual
paths' *structurally similar-looking* turnaround-exemption term (their
`g_turn` McCormick linearization for `outbound_j * p_j`) genuinely does
need an auxiliary, because *that* term multiplies two decision-dependent
quantities together (an expression built from other decision variables,
times `p_j`). Seeing the shape `(baseline + bonus*p_i) * const` and
reaching for McCormick out of habit would have been over-engineering here.

## Recon-to-implementation link (Phase 1 → Phase 3)

Phase 1's node-handling-capacity recon (`PHASE1_NOTES.md`) found that
nothing in the pipeline implements `Theta_i,m`/`DeltaTheta_i,m` as the
gospel defines them, and that the closest existing data
(`modal_capacity.assets` in `config/model_parameters.yaml`) is consumed
entirely as an *arc*-capacity input by `network/build_modal_arcs.py`,
tier-indexed with no activation-conditional split. That recon holds up
unchanged after actually building C8: the new `instance["nominal_throughput"]`,
`instance["node_handling_capacity"]`, and `instance["node_handling_bonus"]`
keys this phase requires are **new schema, not a re-read of existing
config** — Phase 5 (real-network integration) will need to either
restructure `build_modal_arcs.py`'s consumption of `modal_capacity.assets`
into two separate outputs (arc throughput vs. node handling capacity) or
introduce fresh calibration data for `Theta`/`DeltaTheta` specifically.
Nothing about actually implementing C8 changed that assessment — it
confirms it.

## No safe default for `nominal_throughput`/`node_handling_capacity`

Unlike `node_severity`/`disaster_type`/`degradation_matrix`/`alpha`, which
have a genuinely safe "no data provided" default (severity 0, undamaged —
a real, meaningful no-op), `nominal_throughput` and `node_handling_capacity`
have no safe default: defaulting a missing arc/node entry to `0.0` would
silently make that arc/node completely unusable (C7/C8 would force `n=0`
there always), and defaulting to some large placeholder would risk
masking a real missing-data bug in Phase 5's eventual real-network
plumbing. Both are therefore **required** instance keys (validated with a
clear `ValueError` at the top of `solve_vif`, same pattern as Phase 1's
`J_k`/`cap_tons` check), and every test instance in this repo now supplies
them explicitly for every arc/node it uses — including the pre-existing
Phase 1/2 instances, migrated this phase (see "What was built" above).

## A test-design mistake, not a code bug — caught before it shipped

While verifying Test 3 (node availability) against the solver, my first
draft used `beta=0.9` (matching the other tests' habit) on a 2-scenario
instance. The solver returned a *certified* optimum (`MIPGap=0.0`) where
scenario 1 — the "calm," fully-deliverable scenario — delivered `x=100`
against a 60-unit demand and left `z≈59.98` unmet, an obviously terrible
routing choice, paying roughly 30,000 in avoidable penalty for no reason.

This looked like a bug in C6/C9's new availability logic. It wasn't. With
only 2 equally-likely scenarios (`pi=0.5` each) and `beta=0.9`, the CVaR
tail size `1-beta=0.1` is smaller than either scenario's own probability
(`0.5`), so `eta` always lands exactly at the worse scenario's loss and
`CVaR_0.9` collapses to `max(Lambda^1, Lambda^2)` **regardless of what
Lambda^1 actually is**, as long as it doesn't exceed `Lambda^2`. Scenario
1's recourse quality was genuinely, provably irrelevant to the objective
at that beta — the "terrible" solution and the "correct" one (full
delivery, `z=0`) are exactly tied, and Gurobi picked one arbitrarily. This
is the same mechanism Phase 2's `test_cvar_beta_0p99_collapses_to_worst_case`
already demonstrated deliberately; here it showed up *by accident* and
initially looked like a modeling defect.

Caught it the same way as Phase 2's arithmetic slip: by running the real
solve and comparing against hand expectations before trusting the test,
not by assuming a passing/plausible-looking number was correct. Fixed by
switching Test 3 to `beta=0.0` (plain expectation), which weights every
scenario's recourse directly and eliminates the degeneracy — confirmed the
solver then returns exactly the hand-predicted `Lambda^1=6.04`,
`Lambda^2=30000`. The lesson generalizes past this one test: **any
CVaR-wrapped test with few scenarios needs its beta chosen deliberately
against the actual scenario-probability structure, not copied from a
previous test's beta out of habit** — a beta that makes some scenario's
tail-probability smaller than that scenario's own weight will silently
stop caring about that scenario's recourse quality, and a "passing" test
built on top of that degeneracy would have verified nothing.

## Ambiguity in the gospel doc found this phase

None new. `eq:residual`, `eq:thetadegradation`, and `eq:availability` are
stated unambiguously, and Phase 1's already-flagged C6 ambiguity (whether
the release term is conditional on `i in N^P`) resolved identically here —
the `a^w_i` factor just multiplies into the same already-conditional term.

## Verification performed

- `python3 -m py_compile model/model.py model/model_vif.py` — passes.
- `pytest tests/test_vif_phase1.py tests/test_vif_phase2.py
  tests/test_vif_phase3.py -v` — 9/9 pass (6 pre-existing, unchanged
  after schema migration; 3 new this phase).
- `scripts/check_aggregate_regression.py` — **ALL PASS**, re-run after
  Phase 3's changes, every variable/constraint family count still matches
  the locked baseline exactly.
- `git diff --stat aps_usarpac/model/model.py` — still +32/-1, identical
  to Phase 2 — confirms this phase required zero changes to `model.py`.
- Did not run anything against the real 50-node network; did not modify
  `input_builder.py` or any `analysis/` script.
