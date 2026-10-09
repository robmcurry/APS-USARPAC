# PRS-VIF vs. `vehicle_formulation="individual"` — Delta Report

Read-only verification. All line numbers refer to `aps_usarpac/model/model.py` unless
otherwise noted. Repo root: `/Users/claywoody/PycharmProjects/APS-USARPAC`.

## A. Constraint-by-constraint delta

| # | PRS-VIF | Status | Code form / file:line |
|---|---|---|---|
| C1 | CVaR: ξʷ ≥ Λʷ − η, ξ ≥ 0 | **Implemented as written** | `model.addConstr(xi[w] >= loss[w] - eta, ...)`, `model.py:1115` ("CVaRExcess"); `xi = model.addVars(Omega, lb=0.0, ...)`, `model.py:470`. |
| C2 | Site count: Σ pᵢ ≤ P_max | **Implemented as written** | `model.addConstr(gp.quicksum(p[i] for i in PPL) <= P_max, name="SiteBudget")`, `model.py:483`. |
| C3 | Activation budget: Σ fᵢpᵢ ≤ B | **Implemented as written** | `model.addConstr(gp.quicksum(site_cost[i]*p[i] for i in PPL) <= selection_budget, name="SelectionBudget")`, `model.py:484-487`. `f_i`→`site_cost[i]`, `B`→`selection_budget`. |
| C4 | One base each: Σⱼ b_l,j = 1 | **Not implemented as a constraint** | `b` is not a Gurobi variable (see B2). "One base each" is enforced as a Python-side data invariant in `_assign_individual_homes`, `model.py:1389-1414`: it raises `ValueError` unless `sum(b_kj.values()) == fleet_size`, then deterministically assigns each `l` exactly one `home_j`. No model constraint exists; the invariant is enforced before the model is even built. |
| C5 | Basing linked: b_l,j ≤ p_j | **Not implemented as such** | No such linking constraint exists because `b` isn't a variable. The closest analog is `home_term = p[j] if (j == home_j and j in PPL_set) else 0` inside `VehicleConservationIndiv`, `model.py:806`, which gates a vehicle's *departure credit* on `p[home_j]`, but does not constrain `b` itself — the vehicle is still "based" there (by data construction) regardless of whether the optimizer ever selects that node as a PPL. |
| C6 | Resource balance over all i∈N, with retained-resource term y | **Implemented differently** | `FlowBalance_PPL`/`FlowBalance_nonPPL`, `model.py:513-537`. Unconditional (outside the formulation branch, applies to both paths identically). Differences from PRS-VIF: (1) flow is indexed by mode `x[w,m,i,j,r]`, never by vehicle `l` — there is no `x^w_l,ijr`; (2) `release[w,i,r]` is a full decision variable bounded by `InventoryReleaseBound` (`model.py:489-500`), not the fixed parameter expression `(1-ρ)q̄ᵢᵣaʷᵢpᵢ`; (3) there is **no `y` (retained-resource) variable anywhere** — confirmed by grep, no analog exists in either formulation. |
| C7 | Vehicle capacity, per vehicle: Σᵣ wᵣ x^w_l,ijr ≤ min{Tʷ_l,ij, cap_l}·n^w_l,ij | **Implemented differently** | `VehicleCapFlow`, byte-identical in both branches: aggregate `model.py:651-664`, individual `model.py:812-827`. Actual form: `x[w,m,i,j,r] <= gp.quicksum(vehicle_types[k]["capacity"][r] * n[w,k,m,i,j] for k in K_m.get(m, []))`, one constraint **per resource `r` independently** (code's own comment at `model.py:652-653` calls this "a known simplification" — a vehicle could be credited a full food load and a full water load simultaneously). Uses the pinned aggregate `n[w,k,"air",i,j]` (via VehicleAggregation, see below), never `n_ind` directly, even for air. There is no `min{T^w_l,ij, cap_l}` term — arc throughput degradation is handled entirely separately via `ArcCapacity`/`modal_residual`, uncoupled from vehicle count. Not per-vehicle in any sense. |
| C8 | Node handling: Σ_{l∈L_m} Σ n^w_l,ji ≤ Θʷ_i,m | **Not implemented at all** | No constraint family bounds vehicle count/arrivals at a node by any handling capacity, in either formulation. Confirmed by (a) grep for `Theta`/`handling`/`airfield`/`port_cap` across `model.py`/`input_builder.py` — no hits; (b) `problem_size_certificate.py`'s exhaustive `CONSTRAINT_FAMILIES` list (`analysis/problem_size_certificate.py:29-46`) has no such family, and the certificate's family-sum reconciles exactly against `model.NumConstrs` (813202 = 813202), i.e. there is nothing unclassified/missing from that enumeration. |
| C9 | Vehicle conservation (equality, with nbar and aʷᵢ): Σout n + n̄ʷ_l,i − Σin n = aʷᵢ·b_l,i | **Implemented differently** | `VehicleConservationIndiv`, `model.py:792-810`: `arrivals_l + home_term >= departures_l` — an **inequality**, no `nbar` variable exists anywhere in the codebase (grep confirms), and no `aʷᵢ` severity-deactivation term — `home_term` is unconditionally `p[j]` whenever `j == home_j`, never gated by scenario severity. (A per-resource severity factor *does* exist elsewhere — `inventory_availability`/`a[w,i,r]` built at `input_builder.py:303-331` and consumed in `InventoryReleaseBound`, `model.py:494-495` — but it is never applied to vehicle basing/conservation.) |
| C10 | Distance budget, per vehicle: Σ n^w_l,ij(dist+ψ_l) ≤ D_l | **Implemented as written, air only** | `DistanceBudgetIndiv`, `model.py:952-965`: `dist_term_l + pi_k*(total_outbound_l - exempted_l) <= D_k`, one constraint per `(w, k, l)` for `k` in `air_indiv_types`. See **B.1** below — this is a real per-vehicle budget, but only for C-17/C-130J. |
| C11 | Subtour elimination (exponential, lazily separated) | **Implemented as written** | `_build_subtour_callback`, `model.py:38-192`. A `MIPSOL` lazy-constraint callback: for each `(w,k,l)`, finds nodes unreachable from `home` via selected `n_ind` arcs, groups them into connected components `S`, and adds `sum_{a,b in S} n_ind[w,k,l,"air",a,b] <= |S|-1` (`model.py:161-166`). Activated via `model.Params.LazyConstraints = 1` (`model.py:1059`) and `model.optimize(subtour_callback)` (`model.py:1146`) whenever `vehicle_formulation=="individual"` and `n_ind` is non-empty (`model.py:1058`). This is a faithful match to "exponential, lazily separated." (Distinct from `DepartureSingleNode`/`DepartureNodeLink`, `model.py:1010-1047` — see F.) |
| C12 | Symmetry breaking on expected utilization within type | **Implemented differently (weaker)** | `VehicleSymmetryBreak`, `model.py:967-1008`: within each `(k, home_j)` group, orders instances by `total_outbound[w,k,l] <= total_outbound[w,k,l+1]` **per scenario w**, not by "expected" (probability-weighted, cross-scenario) utilization. Ordering key is a scalar count of outbound air-arcs in that one scenario, not a full lexicographic measure — the code's own comment (`model.py:982-984`) calls this weaker than full lexicographic ordering. |
| C13 | Domains (incl. nbar continuous) | **Partial** | `p` binary (`model.py:381`) ✓. `n_ind` binary (`model.py:455`) ✓, matches `n^w_l,ij`. `b` has no domain because it isn't a variable. `nbar` has no domain because it doesn't exist. `x^w_l,ijr` in the paper is per-vehicle; code's `x` (`model.py:385`) has no `l` index at all. `y^w_ir` has no domain because it doesn't exist. `z` (`model.py:392-397`) is created with `lb=0.0` only — **no explicit upper bound `z ≤ dʷ_ir`** is set on the variable or via any constraint found; z's ceiling is only an indirect consequence of `FlowBalance` + `ModalOutboundFeasibility` algebra, which I could not verify actually forces `z ≤ d` in all cases (flagged, not confirmed either way — see F). |

## B. The four questions

### 1. Distance budget — per-vehicle or fleet-wide?

**Both, depending on mode**, under `vehicle_formulation="individual"`:

- **Air (C-17, C-130J):** per-vehicle, `DistanceBudgetIndiv`, `model.py:962-965`:
  ```python
  model.addConstr(
      dist_term_l + pi_k * (total_outbound_l - exempted_l) <= D_k,
      name=f"DistanceBudgetIndiv_w{w}_k{k}_l{l}",
  )
  ```
  `D_k` here is the per-vehicle-instance budget (identical value for every vehicle of type `k`, since `D_k = 3 * cruise_speed_km_day` is type-constant — `input_builder.py:526`). This constraint is added once per `(w, k, l)`, so it *is* `D_l` in the paper's notation.

- **Sea (LCU-1700) and land (M1083):** still the **fleet-wide aggregate form**, even when `vehicle_formulation="individual"`. `model.py:874-902` (inside the `elif vehicle_formulation == "individual":` branch) is the exact same code as the aggregate branch (`model.py:707-733`), just skipping `k_name in air_indiv_types`:
  ```python
  for w in Omega:
      for k_name, vtype in vehicle_types.items():
          if k_name in air_indiv_types:
              continue
          ...
          model.addConstr(
              dist_term + pi_k * (total_outbound - exempted) <= F_k * D_k,
              name=f"DistanceBudget_w{w}_k{k_name}",
          )
  ```
  Because `air_indiv_types = [k for k in K_m.get("air", []) if k in ("C-17", "C-130J")]` (`model.py:741`), sea and land are **never** individually indexed under any value of `vehicle_formulation` — there is no `vehicle_formulation` value that gives LCU-1700 a per-vehicle budget.

**Conclusion:** your concern is confirmed for LCU-1700. Under `vehicle_formulation="individual"`, the 8 LCU-1700s still pool to `F_k * D_k = 8 * 1224 km = 9792 km` in one shared constraint (`model.py:730-732` / `899-901`), and the model can and will route flow-equivalents across a 4,500 km sea arc using that pooled budget, exactly as it would under `"aggregate"`. Vehicle indexing buys nothing for range enforcement on any mode except air.

### 2. Basing

`b_l,j` is **not a Gurobi decision variable** anywhere in the codebase (confirmed by grep — no `addVar`/`addVars` call produces a `b`-named variable). Basing is read as fixed input data from `vehicle_types[k]["b_kj"]`, computed once in `input_builder.py:build_vehicle_params` (`input_builder.py:456-535`) via a deterministic tier-weighted (3:2:1) allocation over `J_k` (rating- and tier-eligible nodes), independent of the optimizer's eventual choice of `p`.

Because `b` is a parameter, **C4 and C5 are not present as model constraints** (see A, rows C4/C5). What exists instead:
- C4's "exactly one home" property is enforced by `_assign_individual_homes` raising `ValueError` if `sum(b_kj.values()) != fleet_size` (`model.py:1401-1407`), and by construction each `l ∈ {1..fleet_size}` gets exactly one `home_j` (`model.py:1408-1413`).
- C5's linkage is approximated only through `home_term = p[j] if j==home_j else 0` in `VehicleConservationIndiv` (`model.py:806`) — this zeroes a vehicle's departure credit if its home node isn't selected as a PPL, but does not touch `b` (which doesn't exist) and has no effect on other constraints that might reference basing.

### 3. Transfer variables

Yes — `tau` is created **unconditionally**, outside the `vehicle_formulation` branch entirely: `model.py:389`, `tau = model.addVars(transfer_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="tau")`. So are its constraints: `TransferBacking` (`model.py:587-604`) and `TransferCapacity` (`model.py:609-615`), plus `ModalOutboundFeasibility` (`model.py:545-567`) which makes `tau` meaningful, and the transfer cost term in the objective (`transfer_cost_expr`, `model.py:1095-1098`).

None of this is gated on `vehicle_formulation`. So under `vehicle_formulation="individual"`, `tau` is built, constrained, and optimized exactly as under `"aggregate"` — it is **populated** with genuine solution values (extracted at `model.py:1189, 1203-1206, 1262-1264`), not empty and not copied from anything. This is also the PRS-VIF-vs-code gap flagged in "NOT IN PRS-VIF" in your ground truth: the code's `Λʷ` (`LossDefinition`, `model.py:1085-1111`) includes `transfer_cost_expr`, a term the PRS-VIF objective does not have at all, regardless of which `vehicle_formulation` is active.

### 4. Mode coverage

From `model.py:741-742` and `config/model_parameters.yaml:362-409`:

```python
air_indiv_types = [k for k in K_m.get("air", []) if k in ("C-17", "C-130J")]
non_indiv_air_types = [k for k in K_m.get("air", []) if k not in air_indiv_types]
```

- **Individual treatment:** `C-17` (air, fleet_size 12), `C-130J` (air, fleet_size 16) — both configured `mode: air`, both are the entirety of `K_m["air"]`, so `non_indiv_air_types` is empty in the current config.
- **Aggregate fallback (always, regardless of `vehicle_formulation`):** `LCU-1700` (sea, fleet_size 8), `M1083` (land, fleet_size 60).

## C. Mixed-formulation mechanics

- **One distance-budget family or two?** Two, coexisting under `vehicle_formulation="individual"`: `DistanceBudget_w{w}_k{k_name}` (fleet-wide, sea/land) and `DistanceBudgetIndiv_w{w}_k{k}_l{l}` (per-vehicle, air). See A/C10, B.1.
- **C7 (VehicleCapFlow):** No — it uses `n[w,k,m,i,j]` for **every** mode, including air, never `n_ind` directly (`model.py:812-827`, byte-identical to the aggregate branch). Air's `n[w,k,"air",i,j]` values are numerically pinned to `sum_l n_ind[w,k,l,"air",i,j]` by `VehicleAggregation` (`model.py:771-780`), so `VehicleCapFlow` operates on the *aggregate* variable for air too, just one that happens to be constrained equal to a sum of binaries.
- **C8 (node handling):** N/A — there is no C8-equivalent constraint in the model at all (see A row C8), so the question of whether it sums both variable families doesn't arise; no node's vehicle traffic (airfield or port) is capacity-limited by anything but arc-level `ArcCapacity`/`modal_residual` acting on commodity flow `x`, not vehicle counts.
- **C6 (resource balance):** No mixing question applies — `FlowBalance_PPL`/`FlowBalance_nonPPL` (`model.py:513-537`) sum only over `x[w,m,i,j,r]`, which is unconditional and identical in both formulations; it never references `n`, `n_ind`, or vehicles at all. Resource flow is not attributed to vehicles (aggregate or individual) anywhere in the model.

## D. Aggregate vs. individual diff

The individual branch's comment at `model.py:734-739` ("fully replaces, rather than supplements... duplicated here unchanged") was checked line-by-line against the aggregate branch (`model.py:618-733`):

- **(16) VehicleConservation**, sea/land/non-individual-air: aggregate `model.py:622-640` vs. individual `model.py:746-765` — **identical body**, the only difference is the loop header restricting `k_list` to `non_indiv_air_types` when `m == "air"` (`model.py:748`). Not drifted.
- **(18) VehicleCapFlow**: aggregate `model.py:654-664` vs. individual `model.py:817-827` — **byte-identical**, not even scoped by type (both iterate `for k in K_m.get(m, [])` unrestricted). Not drifted.
- **(19) DistanceBudget** (fleet-wide, sea/land): aggregate `model.py:670-733` vs. individual `model.py:832-902` — identical logic, with `if k_name in air_indiv_types: continue` guards added in three places (`g_keys`, TurnExempt loop, DistanceBudget loop) to skip air-individual types. Not drifted.

**Verdict: the claim holds.** The duplication is real but has not drifted from the aggregate branch — it is a faithful copy scoped to a type subset, not an independently-evolved variant.

What is genuinely new in the individual branch (no aggregate analog):
- `VehicleAggregation` (`model.py:771-780`) — pins `n` to `Σ n_ind`.
- `VehicleConservationIndiv` (`model.py:792-810`) — per-vehicle-instance conservation.
- `g_ind` / `DistanceBudgetIndiv` (`model.py:914-965`) — per-vehicle distance budget + turnaround McCormick linearization.
- `VehicleSymmetryBreak` (`model.py:967-1008`).
- `dep_node` / `DepartureSingleNode` / `DepartureNodeLink` (`model.py:1010-1047`, off by default).
- The subtour-elimination lazy callback (`model.py:38-192`, `1058-1071`).

So: roughly two-thirds of the vehicle-heterogeneity constraint code (16/18/19 for sea+land) is shared logic re-run over a smaller type set; the rest is new machinery that exists only to make the air-individual binaries behave like the aggregate integers did (VehicleAggregation), plus the per-vehicle constraints PRS-VIF actually wants (conservation, distance, symmetry-break, subtour).

## E. Model size

No live certificate exists for the individual path — `problem_size_certificate.py:71` calls
```python
build = solve_stochastic_cvar(instance, build_only=True, verbose=False)
```
with no `vehicle_formulation` kwarg, so it defaults to `"aggregate"` (`model.py:202`) and only ever measures that path. I did not modify or re-invoke it (out of scope / read-only), and did not run the model myself to produce individual-path numbers — so the individual-path figures below are **derived symbolically from the code**, not measured, and are flagged as such.

**Aggregate path — measured** (`aps_usarpac/output/problem_size_certificate.md`, generated 2026-07-10, N=100 scenarios, 50-node network, seed=32):

| Family | Count | Formula |
|---|---|---|
| p | 22 | \|N^P\| |
| x | 353,600 | \|Ω\| · Σ_m \|A_m\| · \|R\| |
| z | 10,000 | \|Ω\| · \|N\| · \|R\| |
| release (y) | 10,000 | \|Ω\| · \|N\| · \|R\| |
| tau | 21,200 | \|Ω\| · Σ_i\|T_i\| · \|R\| |
| n | 327,000 | \|Ω\| · Σ_m(\|K_m\|·\|A_m\|) |
| eta | 1 | 1 |
| xi | 100 | \|Ω\| |
| loss | 100 | \|Ω\| |
| g_turn | 3,400 | \|Ω\| · Σ_k \|based-PPL-nodes(k)\| |
| **NumVars** | **725,423** | — |
| **NumConstrs** | **813,202** | — |

**Individual path — symbolic, not measured.** Relative to the aggregate model built with the same instance, replace/add, for `k ∈ air_indiv_types = {C-17, C-130J}` with `L_k = fleet_size(k)`:

- Removes: aggregate `VehicleConservation` for air (was `|Ω|·|K_air|·|N|`, drops to 0 since `non_indiv_air_types` is empty), and `TurnExemptUB1/UB2/LB` + `DistanceBudget` for air types (those `g`/`base_nodes_by_k` entries are skipped).
- Adds:
  - `n_ind` vars: `|Ω| · Σ_{k∈air_indiv} L_k · |A_air|`
  - `VehicleAggregation` constrs: `|Ω| · Σ_{k∈air_indiv} |A_air|`
  - `VehicleConservationIndiv` constrs: `|Ω| · Σ_{k∈air_indiv} L_k · |N|`
  - `g_ind` vars / `TurnExemptIndiv*` constrs (×3): `|Ω| · Σ_{k∈air_indiv} |{l : home_l ∈ N^P}|` each
  - `DistanceBudgetIndiv` constrs: `|Ω| · Σ_{k∈air_indiv} L_k`
  - `VehicleSymmetryBreak` constrs: `≤ |Ω| · Σ_{k∈air_indiv}(L_k − 1)` (fewer if a home group has size 1)
  - `dep_node` vars / `DepartureSingleNode`/`DepartureNodeLink` constrs: **0 by default** (`_debug_skip_departure_single_node=True` default, `model.py:203`); if disabled, `|Ω| · Σ_{k∈air_indiv} L_k · |N|` (with outgoing arcs) for `dep_node`, similar order for the two constraint families.

With `L_{C-17}=12`, `L_{C-130J}=16` and the measured 50-node / |A_air|=1502 / |Ω|=100 instance, `n_ind` alone would be on the order of `100 · 28 · 1502 ≈ 4.2M` binaries — over an order of magnitude larger than the entire aggregate model's 725,423 variables. This is a rough order-of-magnitude estimate from the formula above, not a measured count.

## F. Gaps and surprises

- **C8 (node handling capacity) is entirely unimplemented**, in both formulations — not a debug-gated omission, just absent from the model (A, row C8).
- **C9's `nbar` and `aʷᵢ` are absent**, so the individual formulation's vehicle conservation is a plain inequality with no severity-based deactivation of home-based departure credit (A, row C9).
- **C7 is never actually per-vehicle**, even under `"individual"` — resource flow `x` has no vehicle index in either formulation, and the "known simplification" comment at `model.py:652-653` (independent per-resource capacity, no combined-weight cap) applies identically to both.
- **`_debug_*` kwargs** (`model.py:203-207`, documented `233-272`), all scoped to `vehicle_formulation="individual"` only:
  - `_debug_skip_departure_single_node` (default `True`): skips the static `DepartureSingleNode`/`DepartureNodeLink` constraints; subtour elimination relies solely on the lazy callback by default. Passing `False` adds the static constraints back for A/B comparison — the two mechanisms are independent (docstring, `model.py:246-247`) and either alone is claimed sufficient.
  - `_debug_skip_symmetry_break` (default `False`): if `True`, skips `VehicleSymmetryBreak` entirely — exists to reproduce a pre-symmetry-break baseline.
  - `_debug_mip_focus` (default `1`): sets `Params.MIPFocus`; scoped only to the individual branch (`model.py:1066`) since, per the comment, the aggregate branch's regression baseline must not be perturbed.
  - `_debug_node_log_interval_sec` / `_debug_incumbent_flag_path` (default `None`): diagnostic-only instrumentation on the subtour callback (periodic node-count/bound logging; first-incumbent marker file). No effect on the optimization itself when `None`.
  - None of these affect `vehicle_formulation="aggregate"` runs at all.
- **The `model_parameters.yaml:352` `vehicle_formulation: "aggregate"` key is dead** for any caller in `analysis/`. Grep across `.py`/`.yaml` shows: `analysis/sensitivity_runner.py`, `analysis/convergence_stage1.py`, `analysis/convergence_stage2.py`, and `analysis/problem_size_certificate.py` all call `solve_stochastic_cvar(...)` **without** a `vehicle_formulation` argument and without reading `params["vehicle_formulation"]` — so they always run `"aggregate"` regardless of the YAML value. The only code that reads the config key at all is `toy_vehicle_test.py:417` (a standalone toy script, not in `analysis/`). **The YAML comment claiming "individual" is "not yet implemented — raises NotImplementedError if selected" (`config/model_parameters.yaml:349-351`) is confirmed stale**: the actual code has no `NotImplementedError` anywhere; unknown `vehicle_formulation` values raise `ValueError` (`model.py:460`, `1049`), and `"individual"` is fully implemented and exercised by multiple scripts under `aps_usarpac/scripts/` (`toy_individual_cyclic_test.py`, `toy_individual_fk2_departure_test.py`, `toy_individual_smoke_test.py`, `lazy_subtour_run6_3hr.py`, `lazy_subtour_run6_repeat.py`, `lazy_subtour_staged_run.py`).
- **`z`'s upper bound (`z ≤ dʷ_ir`)** is not set as an explicit variable bound or constraint — flagged in A/C13 as unconfirmed rather than asserted broken; I could not determine from the constraint set alone whether it is always implied.
