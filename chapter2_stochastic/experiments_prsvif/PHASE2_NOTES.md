# PRS-VIF Phase 2 Notes — Multi-scenario CVaR wrapper

Continuation of Phase 1 (`PHASE1_NOTES.md`). Same authoritative source:
`aps_usarpac/docs/PRSVIF_Gospel.md`, section "Risk-Aware Vehicle-Indexed
Model." This phase's scope was exactly what Phase 1's own TODO flagged:
`eta`, `xi`, and the CVaR objective wrapper (`eq:vif:objective`, `cvar1`,
`cvar2`), replacing Phase 1's plain `sum_w Lambda^w` objective. No explicit
phase prompt was given this time (you said "let's go to phase 2"); scope
was taken directly from the TODO Phase 1 itself recorded, not re-derived.

## What was built

- `aps_usarpac/model/model_vif.py`:
  - `loss[w]` (code name for Lambda^w) is now a real Gurobi variable with
    a defining equality (`VifLossDefinition_w{w}`), not just an inline
    expression as in Phase 1 — needed because `cvar1` has to reference
    Lambda^w in a constraint, and the gospel's Variables section declares
    `Lambda^w >= 0` as a genuine decision variable, not merely a formula.
    Same pattern the aggregate/individual paths already use for their own
    `loss`.
  - `eta` (free, `lb=-GRB.INFINITY`, no scenario index — dom8) and `xi[w]`
    (`lb=0.0`, satisfying cvar2 as a variable bound rather than a named
    constraint row, matching the aggregate/individual paths' convention).
  - `VifCVaRExcess_w{w}`: `xi[w] >= loss[w] - eta` (cvar1).
  - Objective replaced: `eta + (1/(1-beta)) * sum_w prob[w] * xi[w]`
    (`eq:vif:objective`), in place of Phase 1's `sum_w Lambda^w`.
  - `solve_vif`'s signature gained two new required parameters, `beta` and
    `prob` — both already unpacked in `model.py` before the `"vif"`
    dispatch guard runs (`beta = instance["beta"]`,
    `prob = instance["probability"]`), so no new instance-schema field was
    needed (unlike Phase 1's `resource_weight`/`cap_tons`/`J_k`).
  - Results dict gained `eta`, `xi` (per-scenario), and `scenario_losses`
    (per-scenario Lambda^w), matching the aggregate/individual paths'
    naming for the same concepts.
- `aps_usarpac/model/model.py`: the `"vif"` dispatch call now passes
  `beta=beta, prob=prob`. One line added; nothing else touched (diff is
  +32/-1, up from Phase 1's +31/-1 by exactly that one line).
- `aps_usarpac/tests/test_vif_phase2.py` (new file, kept separate from
  `test_vif_phase1.py` — Phase 1's tests are specifically about C4/C5/C6/
  C7/C9 and shouldn't grow a second, unrelated concern): three hand-
  verified tests, all passing, sharing one 2-node/4-scenario instance and
  varying only `beta`:
  - `beta=0.6`: genuine multi-scenario blending, a real kink (not a tie).
  - `beta=0.0`: CVaR collapses to plain expectation (with a genuine `eta`
    tie — documented and not over-asserted).
  - `beta=0.99`: CVaR collapses to the single worst-case scenario.

## Rename: `solve_vif_phase1` → `solve_vif`

Also did a small pre-emptive cleanup before touching the logic: renamed
the function across `model_vif.py`, `model.py`, `test_vif_phase1.py`, and
this notes series. `solve_vif_phase1` was fine as a Phase-1-only name, but
it was about to keep absorbing Phase 2's logic (and will keep absorbing
Phases 3-6), so a name with "_phase1" baked in would become misleading
immediately. This is a pure rename — verified via `grep -rn
"solve_vif_phase1"` returning nothing anywhere in the repo afterward, and
`check_aggregate_regression.py` + both test files passing before touching
any Phase 2 logic, to confirm the rename itself introduced no behavior
change.

One thing the rename script got wrong on the first pass, worth flagging in
case a similar rename happens again: `test_vif_phase1.py`'s docstring had
pre-existing references to `model.py's _solve_vif_phase1` (written before
the Phase 1 module split, never updated at split time). The blind
`solve_vif_phase1` → `solve_vif` substitution left `_solve_vif` (still
wrong module, still has the stray leading underscore of the old private
name) in three places. Caught by grepping for the old name post-rename and
reading the diff, not by the tests (which don't check docstring prose) —
fixed by hand to `model_vif.py`'s `solve_vif`.

## Why single-scenario Phase 1 tests still pass unchanged

`test_vif_phase1.py`'s three tests all use `Omega = [1]` (single scenario)
and were never touched this phase, yet all still pass with their original
asserted `objective_value` (10.04 in two of them). This isn't a
coincidence to double-check nervously — it's a structural guarantee of the
CVaR construction: with exactly one scenario, `xi[1] >= Lambda^1 - eta`
and `xi[1] >= 0` combined with minimizing `eta + (1/(1-beta))*prob[1]*xi[1]`
(and `1/(1-beta) > 0` for any `beta < 1`) always drives the optimal
solution to `eta = Lambda^1`, `xi[1] = 0`, giving objective = Lambda^1
exactly, regardless of beta. Single-scenario CVaR always collapses to that
scenario's own loss. This is a nice free correctness check: if a Phase 1
test's objective had changed after adding the CVaR wrapper, that would
have meant the wrapper was wired wrong, not that the test needed updating.

## An arithmetic mistake, caught before it shipped

While hand-deriving the beta=0.6 test's expected values, I initially wrote
`xi^4 = Lambda^4 - eta = 40.04 - 30.04 = 10.04` — an actual subtraction
error (40.04 − 30.04 = 10.00, not 10.04; I'd carried the ".04" from both
operands into the result out of habit rather than actually subtracting the
decimals). This would have produced an objective of 36.315 instead of the
correct 36.29.

Caught it by running the real solve against my draft instance *before*
writing it into the test file (see the inline Python snippet used during
development), comparing against the hand-derived numbers, and finding the
mismatch immediately — then re-deriving by hand and confirming 36.29
matches the solver exactly, rather than adjusting the test to match
whatever the solver said (which would defeat the point of a hand-verified
test). The corrected arithmetic is what's in `test_vif_phase2.py` now, and
the module docstring there flags the mistake explicitly so a future reader
trusts the numbers but not blindly.

This is exactly the failure mode "hand-verified" testing is supposed to
catch, and it worked as intended — but it's worth naming directly: **hand
arithmetic on multi-term CVaR expressions is genuinely easy to get subtly
wrong (transposed digits, decimal-carrying errors), and every number in
these tests should be treated as "derived, then confirmed against a real
solve" rather than "derived, therefore correct."** I did that here; future
phases (3-6, all of which touch the objective or add new constraint
families) should keep doing it rather than trusting a docstring derivation
that "looks right."

## Ambiguity in the gospel doc found this phase

None beyond what Phase 1 already flagged. `eq:vif:objective`/`cvar1`/
`cvar2` are written unambiguously and matched the aggregate/individual
paths' existing CVaR implementation almost exactly (same Rockafellar–
Uryasev linearization, same variable roles) — this phase was mechanically
straightforward once Phase 1's `loss`/scenario-indexing groundwork was in
place, which is exactly what Phase 1's docstring said should happen
("multi-scenario-ready indexing by w is kept so Phase 2 only has to add
eta/xi and the CVaR objective, not re-index every variable" — held up in
practice; no variable needed re-indexing).

## Verification performed

- `python3 -m py_compile model/model.py model/model_vif.py` — passes.
- `pytest tests/test_vif_phase1.py tests/test_vif_phase2.py -v` — 6/6
  pass (3 from Phase 1, unchanged; 3 new this phase).
- `scripts/check_aggregate_regression.py` — **ALL PASS**, re-run after
  Phase 2's changes, every variable/constraint family count still matches
  the locked baseline exactly.
- `git diff --stat aps_usarpac/model/model.py` — +32/-1 (one new line,
  `beta=beta, prob=prob`, added to the existing "vif" dispatch call; no
  other lines touched beyond what Phase 1 already added).
- `grep -rn "solve_vif_phase1"` across the repo — no hits, confirming the
  rename was complete.
- Did not run anything against the real 50-node network; did not modify
  `input_builder.py` or any `analysis/` script.
