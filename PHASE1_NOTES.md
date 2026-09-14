# PRS-VIF Phase 1 Notes — Skeleton (joint siting + basing, deterministic)

Authoritative source used throughout: `aps_usarpac/docs/PRSVIF_Gospel.md`,
section "Risk-Aware Vehicle-Indexed Model" (lines 242-445), cross-checked
against the "Aggregate Type-Indexed Formulation" appendix for terminology
where the main section was silent. `docs/vehicle_indexed_formulation_section.tex`
was explicitly disregarded per instruction (stale, air-only scoping artifact).

## What was built

- `aps_usarpac/model/model_vif.py` (new file): `solve_vif(...)`,
  fully self-contained — no import back into `model.py` (would be
  circular, since `model.py` imports this module), so it carries its own
  tiny duplicate of `_status_to_string`. Everything else it needs is
  passed in explicitly by the caller.
- `aps_usarpac/model/model.py`: a `from model.model_vif import
  solve_vif` import, and a 2-line early-return guard inserted
  immediately after `p = model.addVars(...)` inside `solve_stochastic_cvar`,
  firing when `vehicle_formulation == "vif"` and delegating to
  `solve_vif`. Net diff to this file is now +31/-1 lines (was
  +365/-1 before the split below moved the bulk of it out) — confirmed by
  `git diff --stat`, and by re-running `scripts/check_aggregate_regression.py`
  after the split, which still passes with every variable/constraint
  family count matching the locked baseline exactly (see "Verification"
  below).

  This two-file split was a same-day follow-up, done at your request after
  you asked why the vif logic lived inside `model.py` at all. There's no
  runtime cost either way — Python doesn't care whether two functions live
  in one file or two — but `model.py` was already ~1750 lines before Phase
  1 and was going to keep absorbing every subsequent phase's additions
  right next to code that does nothing for vif, so splitting now (small
  diff) beats splitting after 5 more phases (large, riskier diff). See the
  "Deviation from the dispatch-point framing" section below, which still
  describes the *why-not-interleaved* reasoning accurately — only the
  *where the self-contained function lives* changed.
- `aps_usarpac/tests/test_vif_phase1.py`: three hand-verified toy tests, all
  passing (deterministic across repeated runs).
- `docs/PRSVIF_Gospel.md` copied into the repo as supplied (this deliverable
  reads it as authoritative; it was not modified).

In scope this phase, per the gospel doc's equation labels: sets N, N^P, K,
L, L_k, J_k; variables p, b, n, nbar, x, y, z; constraints C2
(`eq:vif:pmax`), C3 (`eq:vif:budget`), C4 (`eq:vif:baseassign`), C5
(`eq:vif:baselink`), C6 (`eq:vif:balance`, a^w_i fixed to 1), C7
(`eq:vif:vehcap`, min{T,cap_l} collapsed to cap_l), C9
(`eq:vif:conservation`, a^w_i fixed to 1); domain constraints dom1-dom7;
objective = sum_w Lambda^w (`eq:vif:loss`), no CVaR wrapper.

Explicitly not built (TODO markers with equation labels are in the code,
see `model_vif.solve_vif`'s docstring):
- Phase 2: eta, xi, CVaR objective (`eq:vif:objective`, `cvar1`, `cvar2`).
- Phase 3: arc/node degradation (`eq:residual`, `eq:thetadegradation`,
  `eq:availability`); node-handling capacity Theta_i,m
  (`eq:vif:transfercap`).
- Phase 4: distance budget (`eq:vif:distbudget`); subtour elimination
  (`eq:vif:sec`).
- Phase 6 ("Computational Considerations"): symmetry breaking
  (`eq:vif:symbreak`), on **expected** (probability-weighted, cross-scenario)
  utilization — not per-scenario, which is how the old "individual"
  formulation's `VehicleSymmetryBreak` does it. Flagging now so Phase 6
  doesn't copy that pattern by habit.

## Deviation from the "same three dispatch points" framing

The phase prompt asked for `vehicle_formulation="vif"` to be added at the
same three dispatch points as `"individual"` (variable creation, constraint
construction, solution extraction), implying it could be interleaved into
the existing code the way `"aggregate"`/`"individual"` share code today.

Reading the gospel doc surfaced a conflict with that framing: PRS-VIF's
variable and constraint set is not a variant of the aggregate/individual
math, it's structurally different. There is no `tau` (intermodal transfer)
in PRS-VIF at all — the gospel's Objective and Constraints sections never
mention it, and the "NOT IN PRS-VIF" framing from the delta-report phase
holds. There is no mode-indexed `x^w_m,ijr` — PRS-VIF's `x` is vehicle-indexed
(`x^w_l,ijr`) from the start, not something the aggregate/individual `x`
could be reinterpreted as. And there is no `release` decision variable —
PRS-VIF's C6 (`eq:vif:balance`) uses `(1-rho) q̄_ir a^w_i p_i` as a fixed
expression directly in the balance equation, not a bounded decision
variable the way `release[w,i,r]` works in the aggregate/individual paths.

Given that, threading `"vif"`'s logic into the same unconditional
variable-creation block (`model.py` lines ~383-478, which builds `x`, `tau`,
`z`, `release`, `n`, `eta`, `xi`, `loss` and sets the CVaR objective — none
of which PRS-VIF's Phase 1 skeleton wants) and the same downstream
extraction code would have meant either (a) building PRS-VIF-irrelevant
variables and constraints anyway and discarding them, or (b) surgically
guarding a dozen separate unconditional blocks scattered across ~700 lines
that `"aggregate"`/`"individual"` both currently depend on running
unconditionally — a much larger and riskier diff, for a formulation whose
math doesn't share those blocks' variables at all.

**Resolution:** `"vif"` branches away via an early `return` right after `p`
is created (the one variable genuinely shared across all three
formulations, per dom1/C2's variable definition being identical), into a
fully self-contained function that does its own variable creation →
constraint construction → objective → optimize → extraction, then returns
directly. This still touches variable creation, constraints, and
extraction — just consolidated into one function rather than interleaved
inline — and guarantees the `"aggregate"`/`"individual"` code paths are not
just behaviorally unchanged but literally untouched character-for-character
(verified: `git diff` shows the only line removed from the existing 1049
lines was one docstring sentence I extended, not any logic).

An unrecognized `vehicle_formulation` string still falls through to the
existing `raise ValueError(...)` at the "n" variable-creation dispatch
(`model.py`, now ~line 480) unchanged, since the new guard is an exact
`== "vif"` check with no catch-all — this was verified by inspection, not
just assumed.

I'm flagging this as a deviation from the letter of the prompt (not a
gospel-vs-prompt conflict in the sense the prompt anticipated, since the
prompt itself proposed the interleaving) because you should know the
architecture differs from what "same three dispatch points" literally
implies, in case Phase 2+ work assumed the interleaved shape.

## Design decision: J_k source (superseding the prompt's b_kj reinterpretation)

The prompt's instructed approach — reinterpret `vehicle_types[k]["b_kj"]` as
an eligibility map, `J_k = {j : b_kj[k][j] > 0}` — turned out to be
unnecessary and, on inspection, the wrong source: `build_vehicle_params()`
in `input_builder.py` (lines 493-501) already computes the **correct**
eligibility set — rating-threshold- and tier-filtered PPL nodes — and
stores it verbatim as `vehicle_types[k]["J_k"]`, *before* it computes
`b_kj` (lines 503-518) by splitting that same `J_k` via a floor-divide
tier-weighted allocation. `vehicle_types[k]["J_k"]` is currently unused by
both the aggregate and individual code paths in `model.py` (grep confirms
neither references it), but it exists, is correctly shaped, and is exactly
what the gospel's `J_k ⊆ N^P` set is defined to be.

Using `b_kj[j] > 0` instead, as the prompt proposed, would silently narrow
`J_k` to only the nodes that happened to receive a nonzero share under the
fixed tier-weighted split with floor-division and remainder-to-highest-tier
— which is a strictly smaller, allocation-dependent set, not an eligibility
set. On the real 50-node network this is very likely to be exactly the
degenerate case the prompt asked me to stop and report on: at low fleet
sizes relative to tier count, floor-division routinely zeroes out
lower-tier allocations entirely (see `build_vehicle_params`'s own
remainder-to-highest-tier logic, `input_builder.py:512-518`, which exists
specifically because floor-division leaves nodes at 0), which would make
`b_kj`-derived `J_k` a single node or a small strict subset of the true
eligibility set for several vehicle types — pointless as a decision
variable's domain.

**Decision:** `model_vif.solve_vif` reads `vehicle_types[k]["J_k"]` directly
and does not touch `b_kj` or `input_builder.py` at all, per the prompt's own
preference ("prefer not touching it"). This is reported per the prompt's
explicit instruction to stop and report a degenerate reinterpretation
rather than silently working around it — here "working around it" meant
using a field that was already sitting in the instance dict, unused, doing
exactly what was needed.

## New instance-schema fields this phase requires (Phase 5 will need to backfill)

`model_vif.solve_vif` reads two fields the current `input_builder.py`
pipeline does not produce. Neither was invented for convenience — they're
direct reads of gospel parameters (`w_r`, `cap_k`) that the aggregate/
individual paths never needed because their C7 analog (`VehicleCapFlow`)
uses the already-person-day-converted `vehicle_types[k]["capacity"][r]`
directly, with no combined-weight cap (that's the "known simplification"
flagged in the earlier delta-report). PRS-VIF's actual C7 needs the raw
metric-ton capacity:

- `instance["resource_weight"][r]` — w_r, metric tons per unit of resource
  r. Not present in `input_builder.py` output today.
- `vehicle_types[k]["cap_tons"]` — cap_k, payload capacity in metric tons.
  The raw number exists in `config/model_parameters.yaml` per vehicle type
  as `payload_kg` (e.g. C-17: 77500), but `build_vehicle_params()` never
  reads that key — it only reads `vconfig["capacity"]` (the pre-converted
  person-day figures). Phase 5 will need to add `cap_tons = payload_kg /
  1000.0` (or read directly from `docs/PRSVIF_Gospel.md`'s Table 4 values,
  which match) to `build_vehicle_params()`'s output dict, and add a
  top-level `resource_weight` key to the instance built by
  `build_stochastic_instance()`, sourced from the gospel's `w_food =
  5.4e-4`, `w_water = 1.5e-2` (MT per person-day, Table 5).

Both toy instances in `tests/test_vif_phase1.py` supply these directly by
hand (not through `input_builder.py`), so this phase's tests do not exercise
or depend on that backfill — it's a Phase 5 (real-network integration) item,
flagged now so it isn't rediscovered from scratch.

## Toy tests — hand calculations

Full arithmetic lives as comments directly above each test function in
`tests/test_vif_phase1.py` (kept there rather than duplicated here, so
there's exactly one place to check the numbers against the code they
justify). Summary of what each test checks and its result:

**Test 1** — `test_c6_c7_c9_basic_feasibility_hand_computed_optimum`. 3
nodes, 1 resource, 1 vehicle type with 2 vehicles, P_max=1 forcing a unique
PPL choice. Hand-computed optimum: `p=[1]`, both vehicles based at node 1,
one delivers 100 units on arc (1,2), objective = 10.04 exactly
(0.1×100 transport + 0.04 tie-break, zero unmet demand, 700 units retained
at node 1 from its 800-unit release budget). **PASSED**, exact match
including the retained-inventory value.

**Test 2** — `test_c5_basing_forced_unused_by_selection_budget`. Same
network, but node 3's activation is blocked by the *selection budget*
(f_3=5 > B=1) rather than P_max, to confirm C5's forcing mechanism doesn't
depend on which activation constraint (C2 or C3) is the binding one.
Identical delivery arithmetic and objective (10.04) as Test 1; the point of
the test is that neither vehicle's basing dict ever contains `(l, 3)`.
**PASSED**.

**Test 3** — `test_c4_single_home_prevents_simultaneous_disconnected_service`.
4 nodes, 2 disconnected one-way arc pairs, 1 vehicle, 2 separately-PPL'd
demand pools. This is the test that actually caught a real gap (see next
section) before landing on a correct, passing form. Final hand-computed
result: exactly one basing bit set for the single vehicle (C4's invariant),
exactly one of the two 100-unit demand pools served (objective = 50,010.04:
50,000 unmet-demand penalty + 10.04 delivery cost), tie-break-independent
on which pool is served. **PASSED**.

## A finding from building the tests, not just running them

My first draft of Test 3 used two-way arcs in both components — (1,2)/(2,1)
and (3,4)/(4,3) — mirroring Test 1's arc set out of habit. It failed with
zero unmet demand instead of the hand-predicted 100: Gurobi found a way to
serve *both* disconnected demand pools with the single vehicle.

The reason is not a bug in C4. With a return arc present in the
*un-based* component, C9's conservation constraint — which nets flow
*locally at each node*, with no requirement that a vehicle's active arcs
form a connected path back to wherever it's actually based — permits a
"phantom" zero-net 2-cycle (`n[l,3,4] = n[l,4,3] = 1`) even though
`b[l,3] = 0`: the cycle's inflow and outflow cancel at both nodes regardless
of whether the vehicle has any real presence there. That phantom loop then
gives C7 a free channel to route `x` through the disconnected component at
a cost of only 2×epsilon, instead of eating the 500/unit unmet-demand
penalty.

This is precisely the disconnected-subtour failure mode the existing
`_build_subtour_callback` docstring in `model.py` documents for the old
"individual" formulation (citing `toy_individual_cyclic_test.py`) — same
underlying cause (no subtour elimination), same symptom (a vehicle getting
free, physically-impossible arc activity via a closed loop disconnected
from its actual base), now confirmed to reproduce in PRS-VIF's C9 for
exactly the same structural reason. It is fully expected given C11
(`eq:vif:sec`) is explicitly out of scope this phase, and does not indicate
a defect in C4, C9, or C7 individually — each is doing exactly what its
equation specifies. But it is worth flagging plainly: **the Phase 1
skeleton, as scoped, does not by itself prevent a single vehicle from
generating unlimited disconnected phantom-loop activity anywhere a
closed 2-cycle (or longer) exists in the arc set with zero net flow at
every node on it.** Phase 4 (subtour elimination) is not just a
distance/routing refinement — it's load-bearing for basic correctness the
moment the arc topology contains any cycle disconnected from truth, which
the real 50-node air/sea/land network certainly does (all three arc layers
contain many cycles). Recommend Phase 4 not be deferred past any phase that
starts reporting real per-vehicle itineraries as meaningful output, even
if CVaR/degradation land first.

I fixed this in the tests by using one-way arcs only, which makes the
phantom loop provably infeasible (shown in the test file's comments) rather
than by adding any subtour logic to `model_vif.solve_vif` — that stays out of
scope per the phase instructions.

## Node-handling-capacity (Theta_i,m) recon — read-only, nothing built

Searched `aps_usarpac/network/nodes.csv`, `aps_usarpac/network/transfer_capacities.csv`,
`aps_usarpac/config/model_parameters.yaml`, and `aps_usarpac/model/input_builder.py`
(plus `network/build_modal_arcs.py`, which is where the config values
actually get consumed) for anything resembling Theta_i,m (vehicle arrivals
per horizon a node can process) or Delta-Theta_i,m (the activation-conditional
capacity bonus).

**Nothing implements Theta_i,m or Delta-Theta_i,m as the gospel defines
them** — a node-level, mode-specific, PPL-activation-conditional vehicle-
arrival cap, decoupled from arc flow. Confirmed absent from all four files
by direct inspection, not just grep.

What *does* exist and is adjacent enough to be worth flagging for Phase 3:

- `config/model_parameters.yaml`'s `modal_capacity` block (lines 127-201)
  has a tier-indexed `assets` count per mode (e.g. maritime PPL-1: 2
  T-AKR-equivalent vessels, air PPL-1: 4 C-17-equivalents) combined with a
  per-asset throughput rate (`discharge_mt_per_window`, `payload_mt` ×
  `sorties_per_window`, `road_mt_per_day`/`rail_mt_per_day`). This is
  structurally close to "how much can this node's infrastructure handle,"
  but it is consumed by `network/build_modal_arcs.py` (lines 107-181,
  confirmed by reading the code, not inferring from the config comments)
  entirely as an input to computing **arc**-level nominal capacity
  (`U_food`/`U_water` on each directed arc row), keyed off the arc's
  **origin** node's tier only. It is baked into arc capacity once, at
  instance-build time, with no dependency on the eventual `p_i` decision —
  there is no equivalent of the gospel's `Delta-Theta_i,m` activation bonus
  anywhere in this pipeline; tier alone (a static node attribute, not
  `p_i`) determines the number.
  - This means Phase 3 cannot just "read Theta from here" — it would need
    to either (a) decouple a genuine node-arrival cap from what's currently
    an arc-capacity input (splitting one config table into two different
    consumption paths), or (b) introduce Theta_i,m as new data entirely,
    and either way will need to add the `p_i`-conditional Delta-Theta split
    from scratch, since nothing like it exists today.
- `network/nodes.csv` has per-node capability ratings `S`/`A`/`L` (sea/air/
  land, 1-3) and `tier` (PPL-1/2/3) — these gate arc *existence* and vehicle
  *basing eligibility* (`J_k`, per `build_vehicle_params`), not handling
  throughput. No arrivals-per-horizon field exists on this file at all.
- `network/transfer_capacities.csv` has per-node, per-mode-pair intermodal
  **transfer** capacities (`T_sea_to_air`, etc.) and costs — this is the
  data behind the aggregate/individual formulations' `tau`/`TransferCapacity`
  constraint, conceptually adjacent to Theta but answers a different
  question (how much can move *between* two modes at a node) than Theta_i,m
  (how many mode-m vehicle *arrivals* the node can process). Since PRS-VIF
  has no `tau` at all (see "Deviation" section above), this file's data is
  not a source for Theta_i,m either, though it's the closest existing
  precedent for a node-level capacity number that isn't just arc capacity.

**Bottom line for Phase 3:** Theta_i,m and Delta-Theta_i,m will need to be
invented essentially from scratch — no field, config table, or code path in
the current pipeline computes or stores them as the gospel defines them.
The `modal_capacity.assets` tier table is the best available raw material
(it's already a tier-indexed "how many assets can operate here" number) but
reusing it will require restructuring `build_modal_arcs.py`'s consumption
of it, not just reading an existing output.

## Ambiguities/underspecification found in the gospel doc while implementing

1. **C6's release term at non-PPL nodes.** `eq:vif:balance` is written as
   one equation "for all i in N," including the term
   `(1-rho) q̄_ir a^w_i p_i`, but `p_i` is only defined for `i ∈ N^P`
   (`eq:vif:dom1`) and the appendix's analogous constraint (`eq:balppl`/
   `eq:balnon`) is explicitly split into PPL/non-PPL cases with the release
   term appearing only in the PPL case. I resolved this the way the
   appendix resolves it: the release term is included only when `i ∈ N^P`,
   treated as 0 otherwise. This seems like the obviously intended reading
   (there's no other sensible way to evaluate `p_i` for a non-PPL node),
   but the main-text equation as literally written doesn't state the
   restriction, so I'm flagging it rather than silently assuming.
2. **Upper bound on `y^w_ir`.** `eq:vif:dom6` states only `y ≥ 0`, with no
   upper bound, and `y` doesn't appear in the objective (`eq:vif:loss`) at
   all. That combination means the model is, by construction, indifferent
   to arbitrarily large "retained" values as long as they're consistent
   with C6 — which they always trivially are, since `y` is whatever value
   makes the equality balance once everything else is fixed. This isn't a
   bug (the domain constraint is what it is), but it does mean `y` in this
   formulation is closer to a slack/accounting variable than a modeled
   quantity with real-world meaning (e.g. it can't be read as "shelf space
   used," since nothing bounds it by actual storage capacity `q̄_ir`). Not
   fixing this — it's exactly what dom6 specifies — but noting it since a
   reader might expect `y ≤ q̄_ir` or similar and it isn't there.
3. **`nbar`'s role is implicit rather than stated as a constraint.** The
   gospel's prose ("ensure each vehicle... terminates exactly where its
   route ends, possibly at the base itself") describes `nbar` as marking
   route termination, but nothing in C9 or the domain constraints actually
   *requires* `nbar[l,i] ∈ {0,1}` or that at most one node per vehicle per
   scenario have `nbar > 0` — the gospel's own closing remark after
   `eq:vif:dom8` notes integrality is *implied* by C9 given `n`, `b`
   integral, which I verified holds in the toy tests (all resulting `nbar`
   values were exactly 0 or 1), but "at most one termination node" is
   something I could not find stated or proven anywhere in the provided
   text — it's presumably a byproduct of C9 plus the eventual subtour
   constraint (C11, out of scope this phase), not something Phase 1's
   in-scope constraints establish on their own. Flagging in case Phase 4
   needs to prove or explicitly add it.

## Verification performed

- `python3 -m py_compile model/model.py model/model_vif.py` — passes.
- `pytest tests/test_vif_phase1.py -v` — 3/3 pass against the split module,
  repeated 3 times, consistent results (no flakiness observed).
- `scripts/check_aggregate_regression.py` — **ALL PASS**, re-run after the
  module split, every variable and constraint family count still matches
  the locked baseline (`output/baseline_aggregate_toy.json`) exactly,
  confirming `vehicle_formulation="aggregate"` is unaffected.
- `git diff --stat aps_usarpac/model/model.py` — +31/-1 (an import line,
  the early-return guard, and one docstring extension; no other lines
  touched). `model_vif.py` is a new, untracked file.
- Did not run anything against the real 50-node network, and did not modify
  `input_builder.py`, `analysis/problem_size_certificate.py`, or any other
  `analysis/` script, per the phase instructions.
