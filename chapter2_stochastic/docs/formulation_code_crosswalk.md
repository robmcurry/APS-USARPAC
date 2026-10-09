# Formulation ↔ Code Crosswalk

**Source documents:** `docs/main.md` (extracted via `docs/extract_formulation.py` →
`docs/formulation_only.md`, Methods section, §3.1–§3.13) vs. `model/model.py`
(`solve_stochastic_cvar`, both `vehicle_formulation` branches), cross-referenced against
`model/input_builder.py` (`build_vehicle_params`) for parameter derivations.

**Method:** every named equation in the extracted Methods section was matched against the
constraint/variable that implements it in `model.py`, by exact line number. This is not a
re-derivation from memory — each match below was re-read from the current file.

---

## 1. Bottom line

The **aggregate formulation** (§3.1–§3.11, Eq. `\eqref{eq:objective}` through
`\eqref{eq:integrality}`) matches the code closely and accurately, including two places
where the paper explicitly documents a simplification that the code also makes (per-commodity
independent vehicle capacity; the non-binding inventory ceiling `\bar{Q}_r`). No material
formulation-vs-code discrepancy was found in this part of the document.

The **individual-vehicle-indexing section** (§3.12, "Individual Vehicle Indexing for Air
Assets (Experimental)") is the part that needs real revision. It describes an **earlier,
superseded version** of the code:

1. It presents the static "single departure per instance" constraint as *the* mechanism
   preventing a vehicle from being double-booked across two origin nodes. In the current
   code, that static constraint is **off by default** (`_debug_skip_departure_single_node:
   bool = True`) and has been replaced by a Gurobi lazy-constraint callback. The paper
   contains no mention of the callback, the phantom-cycle/subtour failure mode it exists to
   fix, or the symmetry-breaking constraints added alongside it.
2. The equation the paper writes for that static constraint (`\eqref{eq:singledeparture}`)
   does not actually implement what its own prose claims, and does not match what the code
   implements either. Detailed below (§3).
3. The paper's closing "Computational Status" paragraph is an intentional placeholder
   ("Results ... are reported in a subsequent section once available") — this is exactly the
   material now sitting in `docs/individual_vehicle_lazy_subtour_report.md` and
   `output/intractability_evidence_report.pdf` (Runs 1–8), which has not yet been folded back
   into `main.md`.

---

## 2. Section-by-section crosswalk — aggregate formulation (§3.1–§3.11)

| Paper | Code | Match |
|---|---|---|
| Objective, Eq. `\eqref{eq:objective}` — CVaR | `model.py:474-478` | Exact |
| First-stage budgets, Eq. `\eqref{eq:pmax}`, `\eqref{eq:budget}` | `model.py:483-487` (`SiteBudget`, `SelectionBudget`) | Exact |
| Inventory ceiling, Eq. `\eqref{eq:invceiling}` | — | **Correctly documented as not implemented** in the paper itself (§3.8, "not implemented in the current model"). Verified: no `\bar{Q}_r` constraint exists in `model.py`. Consistent. |
| Inventory release, Eq. `\eqref{eq:release}`, `\eqref{eq:norelease}` | `model.py:490-506` | Exact |
| Aggregate flow balance, Eq. `\eqref{eq:balppl}`, `\eqref{eq:balnon}` | `model.py:513-537` | Exact |
| Per-mode outbound feasibility, Eq. `\eqref{eq:permode}` | `model.py:545-567` | Exact |
| Transfer backing, Eq. `\eqref{eq:transferbacking}` | `model.py:587-604` | Exact, including the paper's stated rationale (self-sustaining cross-mode cycle) matching the code comment (`model.py:579-586`) word for word in substance |
| Capacity constraints, Eq. `\eqref{eq:arccap}`, `\eqref{eq:transfercap}` | `model.py:572-577`, `609-615` | Exact |
| Vehicle conservation, Eq. `\eqref{eq:vehcons}` | `model.py:620-640` | Exact, including the removal of the old fleet-cardinality constraint — the paper's justification (§3.8) matches the code comment (`model.py:642-649`) point for point |
| Vehicle-capacity flow, Eq. `\eqref{eq:vehcap}` | `model.py:651-664` | Exact. Paper explicitly flags the per-resource independence simplification (§3.3.3, "modestly overstates throughput on mixed sorties") — this is documented, not a gap |
| Fleet-wide distance budget + McCormick, Eq. `\eqref{eq:distbudget}` | `model.py:666-733` | Exact. Paper describes the McCormick linearization narratively without spelling out the three inequalities; code has all three (`TurnExemptUB1/UB2/LB`). Stylistic, not a discrepancy |
| Nonnegativity/integrality, Eq. `\eqref{eq:nonneg1}`–`\eqref{eq:integrality}` | variable declarations, `model.py:385-471` | Exact |

**Minor, non-substantive note:** the paper's loss function (Eq. `\eqref{eq:loss}`) writes a
single vehicle-movement tiebreaker term `$\varepsilon = 0.04$`. The code implements this as
two named constants, `EPSILON_DEPLOY = 0.01` and `EPSILON_EMPTY = 0.03` (`model.py:1082-1083`),
summed (`model.py:1101`). The numeric effect is identical (0.04) and the two-part code
comment explains a distinct rationale for each half (tie-break vs. suppressing empty
round-trips) — worth folding into the paper's discussion of Eq. `\eqref{eq:loss}` for
completeness, but not a numerical discrepancy.

---

## 3. Section-by-section crosswalk — individual vehicle indexing (§3.12)

| Paper | Code | Match |
|---|---|---|
| Per-instance binary, Eq. `\eqref{eq:individualvehiclevar}` | `model.py:447-455` (`n_ind`) | Exact |
| Aggregation link, Eq. `\eqref{eq:vehicleagglink}` | `model.py:771-780` (`VehicleAggregation`) | Exact |
| Per-vehicle conservation, Eq. `\eqref{eq:indvehcons}` | `model.py:792-810` (`VehicleConservationIndiv`) | Exact, including the `p_j`-gated home term |
| Per-vehicle distance budget, Eq. `\eqref{eq:indvehdistbudget}` | `model.py:874-965` (`DistanceBudgetIndiv`) | Structurally exact (`D_k` not `F_k D_k`). **But** the paper does not describe the linearization mechanics at the individual level at all — see finding below |
| Single departure per instance, Eq. `\eqref{eq:singledeparture}` | `model.py:1010-1047` (`DepartureSingleNode`/`DepartureNodeLink`) | **Does not match — see below** |

### Finding A: the paper's `\eqref{eq:singledeparture}` doesn't do what its own prose says, and doesn't match the code

The paper's prose (§3.12.3): *"a single vehicle instance cannot be asserted as departing more
than one node's outbound arc set within a scenario."*

The paper's equation:
```
sum_{j:(i,j) in A_air} n^omega_{k,l,air,ij}  <=  1     for all i in N, k, l, omega
```

Read literally, this is a **separate constraint for every node `i`**, each one bounding *that
node's own* outbound arc count for instance `ℓ` to at most 1. It says nothing about two
*different* nodes both being active at once — a vehicle could satisfy this constraint at
`i = Tokyo` (1 outbound arc) **and** at `i = Manila` (1 outbound arc) **simultaneously**, since
each is its own independent inequality. That is exactly the double-booking failure mode the
prose claims to rule out.

The code implements something structurally different and correct for the stated goal: a
linking binary `dep_node[w,k,l,j]` per node (`model.py:1021-1029`), with
`n_ind[...,j,j_dst] <= dep_node[w,k,l,j]` (so any positive outbound flow from `j` forces
`dep_node[j]=1`), and then a **single constraint summed across all nodes**:
```python
gp.quicksum(dep_node[w, k, l, j] for j in N if modal_outgoing["air"][j]) <= 1
```
(`model.py:1042-1047`). This is the constraint that actually enforces "at most one departure
node, period" — it requires the auxiliary binary precisely because a plain sum of the
continuous/count expression across nodes isn't linear in the way the paper's per-`i` equation
implies. The paper's equation is missing this linking step; as written, it is a weaker (and
largely redundant — see below) constraint.

**Recommendation:** replace Eq. `\eqref{eq:singledeparture}` with the linked formulation:
```
n^omega_{k,l,air,ij} <= y^omega_{k,l,j}         for all (i,j) in A_air, k, l, omega
sum_{j in N} y^omega_{k,l,j} <= 1                for all k, l, omega
```
with `y^omega_{k,l,j} \in \{0,1\}` a new auxiliary ("has instance ℓ departed node j at all").

### Finding B: this whole mechanism is superseded in the current code, and the paper doesn't say so

`_debug_skip_departure_single_node` defaults to `True` (`model.py:203`) — meaning the static
constraint discussed above is **not built at all** under normal operation. In its place, a
Gurobi lazy-constraint callback (`_build_subtour_callback`, `model.py:38-192`) is the actual,
default subtour-prevention mechanism: it inspects every integer-feasible candidate, finds any
set of an instance's selected arcs that is disconnected from its home node (a "phantom
cycle" — a vehicle appearing to fly a loop without ever leaving its base), and adds a
targeted cut ruling out exactly that disconnected pattern.

This is not a cosmetic implementation detail — it exists because the static constraint
family is provably **both insufficient and overly restrictive**:

- **Insufficient:** on a toy network with an actual cycle in the air-arc graph, running with
  *no* subtour mechanism at all produced 3 vehicle instances the solver "flew" on fully
  disconnected loops for zero cost (`toy_individual_cyclic_test.py`) — proving conservation
  alone (Eq. `\eqref{eq:indvehcons}` / `\eqref{eq:vehicleagglink}`) does not prevent phantom
  cycles, contrary to what the paper's current text implies by presenting the static
  constraint as sufficient.
- **Overly restrictive:** the static constraint (correctly linked, per Finding A) blocks any
  legitimate multi-leg route (e.g., home → A → B), because it caps a vehicle to outbound arcs
  from only one node total, when a real multi-leg trip needs to depart from more than one
  node in sequence.

None of this — the phantom-cycle failure mode, the lazy callback, the symmetry-breaking
constraints (`VehicleSymmetryBreak`, `model.py:967-1008`, also with no paper analog) — appears
anywhere in §3.12. This is the largest gap between code and paper in the document.

**Recommendation:** §3.12.3 needs to be rewritten to (a) drop or correct Eq.
`\eqref{eq:singledeparture}` per Finding A, (b) describe the phantom-cycle failure mode and
why static per-instance constraints cannot both permit multi-leg routing and prevent
disconnected cycles, and (c) introduce the lazy-constraint (DFJ-style subtour elimination)
mechanism as the actual current approach, with a citation to the standard vehicle-routing
literature on subtour elimination.

### Finding C: the individual-formulation McCormick linearization isn't described, and has an undocumented restriction

Eq. `\eqref{eq:indvehdistbudget}` references `h^omega_{k,ℓ,j}` (turnaround-exempt outbound
flow) without describing how the exemption is linearized at the individual level, unlike the
aggregate case where §3.8 explicitly names the McCormick auxiliary `g^omega_{k,j}`. The code's
individual-level auxiliary `g_ind[w,k,l]` (`model.py:914-947`) mirrors the aggregate's
three-inequality McCormick envelope, but with the constant upper bound replaced by **1**
(one vehicle) instead of `F_k`:

```python
g_ind[w,k,l] <= p[home_j]                        # UB1
g_ind[w,k,l] <= outbound_home                     # UB2
g_ind[w,k,l] >= outbound_home - (1 - p[home_j])   # LB
```

Because the bound is exactly 1, when the home base is selected these three inequalities
together **force `outbound_home <= 1`** — i.e., a vehicle instance may depart its home node on
at most one arc per scenario. This is stricter than what conservation (Eq.
`\eqref{eq:indvehcons}`) alone requires, which would permit a legitimate
return-refuel-relaunch (arrive home, then depart again). This was already identified
independently in the accuracy audit (`output/model_py_accuracy_audit.pdf`, Finding F-1) as a
genuine over-restriction: it never produces an incorrect number, but it can silently exclude
a valid multi-sortie-from-home route, and it is not mentioned anywhere in §3.12.

**Recommendation:** either (a) add a sentence to §3.12 documenting single-sortie-per-scenario
as an accepted modeling restriction of the individual formulation (simplest, if the
restriction is judged acceptable), or (b) rework the exemption so it doesn't cap
`outbound_home` at 1, and update the paper accordingly once fixed.

### Finding D: §3.12's "Computational Status" is a placeholder — the actual results now exist

The paper currently reads: *"This formulation is experimental and under active evaluation for
computational tractability at theater scale ... Results, including a determination of
computational feasibility at theater scale, are reported in a subsequent section once
available."*

That determination has now been made, across 8 runs on the real 50-node network at the real
fleet size (F_k=12, 16), documented in full in `docs/individual_vehicle_lazy_subtour_report.md`
and `output/intractability_evidence_report.pdf`. Summary of what belongs in this section:

- At real fleet scale, the compact individual-vehicle MIP does not find a single feasible
  solution within 45 minutes (Run 1) or even within 3 hours (Run 8), at the real production
  β=0.90.
- An isolating experiment (Run 2, fleet size reduced to 1 per type) confirms the formulation
  and lazy-callback logic are correct and performant — 306,544 nodes explored, 4 solutions
  found, 12% gap, all within 45 minutes — so the wall is specifically the combinatorial scale
  of 28 interchangeable named vehicles, not a defect in the modeling approach.
- Symmetry-breaking constraints and solver-focus tuning were each tried and neither closed the
  gap (Runs 3–4); extending the time budget alone did not either, at either the diagnostic
  β=0.5 or the real β=0.90 (Runs 5, 8).

**Recommendation:** replace the "Computational Status" placeholder with a summary of these
findings, and add a forward-looking paragraph naming column generation / Dantzig-Wolfe
decomposition as the proposed next step, per the reasoning already captured in
`individual_vehicle_lazy_subtour_report.md` §7.

---

## 4. Items that exist in the code with no formulation counterpart (and don't need one)

These are solver/engineering infrastructure, correctly out of scope for a formulation
section — listed here only so nothing is mistaken for an omission:

- All `_debug_*` parameters (`model.py:203-207`) and their diagnostic hooks (periodic
  node-count logging, incumbent-flag marker files)
- The graceful `SIGTERM` → `model.terminate()` handler (`model.py:1130-1148`)
- `_assign_individual_homes` (`model.py:1389-1414`) — a deterministic bookkeeping helper
  implementing the basing-to-instance assignment already described in prose (§3.3.1,
  §3.12); the paper's `\mathbb{1}[\ell \text{ based at } j]` notation is the formulation-level
  statement of exactly what this function computes
- `Method=2` (barrier root LP) solver-parameter choice and its measured performance
  justification (`model.py:364-371`) — a solve-time tuning decision, not a modeling choice;
  the paper's Computational Implementation section (§3.11) already discusses solve times at
  the right level of abstraction without needing this specific parameter name

---

## 5. Suggested edit list for `main.md`, in priority order

1. **Rewrite §3.12.3** ("New Constraint: Single Departure Per Instance") to describe the
   phantom-cycle problem and the lazy-callback mechanism instead of (or in addition to) the
   static constraint, and fix Eq. `\eqref{eq:singledeparture}` per Finding A.
2. **Replace the Computational Status placeholder** in §3.12 with the Run 1–8 findings
   (Finding D above).
3. **Add a documentation sentence** for the individual-formulation turnaround exemption's
   single-sortie-per-scenario restriction (Finding C), or fix the McCormick bound and update
   accordingly.
4. Optional/minor: note the two-part epsilon (deploy vs. empty-trip) in the Eq.
   `\eqref{eq:loss}` discussion for completeness.

---

## 6. On which model to use for this kind of work

You asked whether Fable 5 or Opus would do this better than the current default (Sonnet 5).
For this specific kind of task — precise line-by-line reconciliation between a formal
mathematical write-up and an implementation, where the value is in catching a subtle
mismatch like Finding A (an equation that looks plausible at a glance but doesn't actually
enforce what its own prose claims) — **Opus 4.8 is the one I'd actually recommend switching
to.** It's Anthropic's most capable model, and this task rewards exactly the kind of patient,
high-effort cross-checking (re-deriving what an equation *actually* constrains, not just
what it's labeled as, then verifying against real code) where the higher-effort model
tends to catch more. I don't have a specific basis to recommend Fable 5 over Opus for this
particular kind of formal-verification task — I'd default to Opus for future crosswalk/audit
passes like this one, and keep Sonnet for faster iterative work (running scripts, drafting
code, quick edits).
