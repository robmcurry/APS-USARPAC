# PRS-VIF Phase 4 Notes — Distance budget + subtour elimination

Continuation of Phase 1-3 (`PHASE1_NOTES.md`, `PHASE2_NOTES.md`,
`PHASE3_NOTES.md`). Same authoritative source:
`aps_usarpac/docs/PRSVIF_Gospel.md`, "Vehicle Routing Constraints." Scope
taken directly from Phase 1's original TODOs: C10 (`eq:vif:distbudget`)
and C11 (`eq:vif:sec`), the last two constraint families PRS-VIF's core
formulation needs (only symmetry-breaking, Phase 6, remains after this).

## What was built

- `aps_usarpac/model/model_vif.py`:
  - C10 (`VifDistanceBudget`): `sum_(i,j) (dist_l,ij + psi_l) * n[w,l,i,j]
    <= D_l`, per vehicle instance.
  - **Reused, not reinvented, three existing fields** for C10's
    parameters — `instance["modal_arc_distance"]` (the same key the
    aggregate/individual paths already populate from
    `build_modal_arcs()`), and `vehicle_types[k]["D_k"]`/`["pi_k"]` (the
    same fields `build_vehicle_params()` already computes, with formulas
    — `D_k=3.0*cruise_speed`, `pi_k=(turnaround_hours/24.0)*cruise_speed`
    — that already match `D_l=kappa*v_l` and `psi_l=(tau_l/24)*v_l`
    exactly, confirmed by direct comparison against
    `input_builder.py:526-527`). Unlike every Phase 1/3 schema addition,
    **C10 required zero new instance schema** — a genuine reuse win,
    not just a documented parallel.
  - C11 (`_build_vif_subtour_callback`, new function): a `MIPSOL` lazy
    callback separating the gospel's exact `eq:vif:sec` inequality family
    — `n[w,l,i,j] <= (external inflow to S) + (base credit in S)` — for
    each disconnected component found, rather than reusing
    `model.py`'s existing `_build_subtour_callback` (built for the old
    "individual" formulation's simpler `|S|-1` aggregate cut). See "Why
    the literal gospel SEC form" below for why the two aren't the same
    constraint even though they solve the same underlying problem.
  - `model.Params.LazyConstraints = 1` and `model.optimize(subtour_callback)`
    activated whenever `L` is non-empty (always true, since `vehicle_types`
    is required non-empty per Phase 1).
  - `results["subtour_callback_stats"]` added (`{"invocations",
    "cuts_added"}`), mirroring the old callback's reporting convention.
- `aps_usarpac/tests/test_vif_phase4.py` (new file): three hand-verified
  tests — distance budget making a physically-connected node permanently
  unreachable, subtour elimination blocking the exact phantom-2-cycle
  use Phase 1 found by accident, and a legitimate connected two-leg
  route confirmed NOT falsely blocked by either new mechanism.
- **Migrated** all 9 pre-existing tests (Phases 1-3) to supply `D_k`
  (huge), `pi_k` (0), and `modal_arc_distance` (0 km everywhere) — all
  non-binding, preserving every previously-asserted number exactly.

## `model.py` needed zero changes, again

Same as Phase 3: every new Phase 4 datum is read directly out of
`instance`/`vehicle_types` inside `solve_vif`, following the Phase 1
precedent, so `model.py`'s diff is still exactly +32/-1 — identical to
Phase 2 and Phase 3. Three phases in a row now where the "vif" branch's
growth has been entirely contained to `model_vif.py`, which is exactly
what the Phase-1-to-Phase-2 architectural split (moving `solve_vif` out of
`model.py` into its own module, done at your request) was for.

## Why the literal gospel SEC form, not the simpler `|S|-1` cut

`model.py`'s existing individual-formulation callback cuts a violated
component `S` with `sum_{a,b in S} n[...] <= |S|-1` — a well-known,
valid, and simpler subtour-elimination cut (a direct binary-count bound:
a genuine tour through `S` can use at most `|S|-1` of its internal arcs).
The gospel's `eq:vif:sec` is written differently: a **per-arc** inequality
bounding each individual `n_ij` by the external inflow to `S` plus the
base credit inside `S`, not an aggregate count bound. These are both
valid ways to eliminate the same disconnected-subtour solutions, but they
are not textually or structurally the same constraint family, and the
phase instructions (echoing the original Phase 1 prompt's standing
instruction) treat the gospel as authoritative over any prior
implementation that disagrees with it — including this repo's own
existing code. `_build_vif_subtour_callback` therefore separates the
gospel's literal form, adding one inequality per currently-selected arc
within a violated component (not eagerly enumerating the full family for
unused arcs in `S`, which is standard lazy-separation practice and still
yields a valid member of the gospel's family for each cut emitted).

## A finding, not a gap: toy-scale instances don't force a nonzero cut count

Both `test_vif_phase4.py::test_c11_...` and my manual verification runs
(including a version with node 4's demand pushed to 300, well past what a
"tempted" LP relaxation would need) produced `cuts_added=0` — the callback
runs (`invocations>0`, confirmed) and the final answer is correct (matches
the hand computation, unlike the pre-Phase-4 use), but Gurobi's own
presolve/heuristics reach the right answer on these tiny instances without
ever needing an explicit lazy cut. This isn't a defect in the test or the
callback — it mirrors why `model.py`'s existing individual-formulation
callback needed a *purpose-built opposition-driven network*
(`scripts/toy_individual_cyclic_test.py`) to force a nonzero cut count in
the first place; small toy instances generally don't. The callback's
correctness is verified by the answer (a disconnected node's demand stays
genuinely unmet, where the pre-Phase-4 formulation would have used
the phantom loop), not by cut count. Flagging this now so Phase 5 (real
50-node network) doesn't get read as "the callback stopped working" if the
cut count there is initially low too, and so nobody assumes a healthy cut
count is what "working" looks like at small scale.

## Ambiguity in the gospel doc found this phase

None. `eq:vif:distbudget` and `eq:vif:sec` are both stated unambiguously,
and — unlike Phase 3 — this phase didn't even need new parameter
definitions to be interpreted, since `dist_ijl`, `D_l`, and `psi_l` all
had exact, already-computed counterparts already sitting in the existing
schema.

## Verification performed

- `python3 -m py_compile model/model.py model/model_vif.py` — passes.
- `pytest tests/test_vif_phase1.py tests/test_vif_phase2.py
  tests/test_vif_phase3.py tests/test_vif_phase4.py -v` — 12/12 pass (9
  pre-existing, unchanged after the D_k/pi_k/modal_arc_distance
  migration; 3 new this phase).
- `scripts/check_aggregate_regression.py` — **ALL PASS**, re-run after
  Phase 4's changes, every variable/constraint family count still matches
  the locked baseline exactly.
- `git diff --stat aps_usarpac/model/model.py` — still +32/-1, identical
  to Phase 2 and Phase 3 — confirms this phase required zero changes to
  `model.py`.
- Did not run anything against the real 50-node network; did not modify
  `input_builder.py` or any `analysis/` script.

## What's left

Only Phase 6 ("Computational Considerations" — symmetry breaking,
`eq:vif:symbreak`, on EXPECTED/probability-weighted utilization across
scenarios) remains as an in-formulation TODO. Phase 5 (real-network
integration via `network_builder`/`scenario_generator`/`input_builder.py`)
is the other remaining item, and is the phase where every "reused
directly" schema field in this and prior phases' notes gets tested against
the real pipeline for the first time, and every "new schema, not yet
produced" field (Phase 1's `resource_weight`/`cap_tons`, Phase 3's
degradation data) needs actual backfill code written.
