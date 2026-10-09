# 03 — Data contract (DRAFT, schema 0.1)

## 1. The rule

The UI never reads solver-native files. The solver writes a Gurobi `.sol` (about 211,000 lines for one run, mostly internal variables) and a `summary_*.json` whose keys are solver internals (`final_bound`, `distance_states_by_type`, `callback_stats`). Neither is stable or readable by a UI developer.

Instead, a small exporter turns each run into a **run bundle** with stable names and plain units. The contract is between the exporter and the UI. The solver can change underneath it.

## 2. Bundle layout

```
bundle/
  index.json                 list of runs: id, date, method, beta, status, headline numbers
  network.json               static: nodes, arcs, vehicle types   (changes rarely)
  runs/<run_id>/
    run.json                 provenance, status, warnings, objective, sites, basing
    scenarios.json           per-scenario: id, disaster, epicenter, severity, loss
    fleet.json               per vehicle type: available, used, share delivered
    scenario_detail/<id>.json  per-scenario deliveries and unmet demand (lazy-loaded)
```

Why split: a map view needs `network.json` and `run.json` only (small). Per-scenario flows are large and are fetched only when a planner opens a scenario.

## 3. Conventions

- IDs are integers (node ids match `network/nodes.csv`). Vehicle types are strings: `C-17`, `C-130J`, `LCU-1700`, `T-AKR`, `T-AKE`, `EPF`, `M1083`, `PLS`.
- Distances in km, coordinates in decimal degrees, longitudes as in `nodes.csv`.
- The objective is **unitless** (model cost units). Never show a currency symbol.
- `null` means "unknown or not recorded". An absent key means "this producer does not export it". The UI must treat both without crashing and show "not recorded".
- Every file carries `schema_version`. Breaking changes bump the major number.

## 4. What exists today, and what does not

This is the part to read before scoping work. Evidence is from `output/vif_staged_proxy/summary_20260929_101829.json`, `output/vif_staged/summary_20260929_101949.json`, and `network/*.csv`.

| UI need (story) | Field | Today |
|---|---|---|
| Sites on map (A1) | `selected_sites` | **Exists** as node ids in `final_selected_sites`. Names and coordinates come from `network/nodes.csv` (50 nodes). |
| Site detail: type, capacity (A1) | node attributes | **Exists** in `nodes.csv` (`hub_type`, `ppl_eligible`, `tier`). Handling capacity (Θ) is in `config/model_parameters.yaml`, not per run. |
| Vehicles based at each site (A2) | `basing` | **Exists** in `strategic_proxy.basing` / `strategic_basing` as `[type, node, count]`. |
| Objective, gap, status (E1) | `objective`, `gap`, `status` | **Exists.** |
| Per-scenario loss (B2, D) | `scenario_losses` | **Exists** in proxy output (`final_scenario_losses`, 20 scenarios). The detailed solve records one objective per solved scenario only. |
| Risk setting, seed, scenario counts (B1, E1) | `run.beta`, `run.seed`, ... | **Exists only from commit 3b0863d onward** (`summary["run"]`). Every earlier output has no record of β or seed; the sample payload shows these as `null`. |
| Solver warnings, e.g. under-resolved tail (E2) | `run.warnings` | **Missing.** The warning is printed to the terminal and not saved. Small change: add `summary["warnings"]`. |
| Vehicle utilization (C1, C2) | `fleet.json` | **Missing.** Not in any summary. Must be computed from the `.sol` movement variables (`ds_n`, 172,470 entries in the example) plus fleet sizes from config. |
| Delivered vs unmet demand per node per scenario (D1, D2) | `scenario_detail` | **Missing.** The `.sol` has `ds_z` (bounded above by demand, so it is demand served) and `ds_y`; I have not confirmed what `ds_y` means. The exporter author must confirm in `model/model_distance_state.py` before building this. |
| Scenario description: disaster type, epicenter, severity (D2) | `scenarios.json` | **Missing from outputs.** Scenarios are regenerated from the seed by `scenarios/scenario_generator.py`. The exporter must persist them, and old runs without a recorded seed cannot be reconstructed. |
| Plain-language risk label (B1) | UI mapping | UI-side. We supply a table from β to wording. |
| Notes on a view (F1) | UI-owned storage | Not part of the solver contract. |

Consequence for the roadmap: the map and recommendation views (Epic A, risk setting B1) can be built now against existing outputs. Fleet and bad-day views (C, D) depend on new exporter work.

## 5. Schemas (v0.1)

### run.json

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | `"0.1-draft"` |
| `run.id` | string | stable, from the output file timestamp |
| `run.method` | string | e.g. `all_scenario_strategic_proxy_then_detailed_routing` |
| `run.scenario_count` | int | scenarios evaluated |
| `run.strategic_scenarios` | int or null | scenarios in the strategic decision |
| `run.beta` | number or null | risk level; see label table below |
| `run.seed` | int or null | |
| `run.status` | string | `OPTIMAL`, ... |
| `run.gap` | number | relative |
| `run.elapsed_seconds` | number | |
| `run.warnings` | string[] | plain text, shown with the run |
| `objective.value` | number | unitless |
| `selected_sites[]` | object | `id, name, country, region, lat, lon, population, hub_type, site_eligible, tier` |
| `basing[]` | object | `vehicle_type, node_id, node_name, count` |

### fleet.json (proposed)

`[{ vehicle_type, available, used, share_of_cargo_delivered, binding }]` where `binding` is true when `used == available`.

### scenario_detail/<id>.json (proposed)

`{ scenario_id, deliveries: [{from, to, vehicle_type, commodity, quantity}], unmet: [{node_id, commodity, quantity}] }`. Commodities in the model are food and water.

### network.json (proposed)

`nodes[]` as above; `arcs[]` with `from, to, mode (air|sea|land), distance_km`; `vehicle_types[]` with `name, mode, fleet_size`. Source files: `network/nodes.csv`, `arcs_air.csv` (1,502), `arcs_sea.csv` (236), `arcs_land.csv` (30), and `config/model_parameters.yaml`.

### Risk label table (UI-owned)

| β | Label (draft) |
|---|---|
| 0.5 | Moderately cautious: protects against the worst half of days |
| 0.8–0.9 | Cautious: protects against the worst 1 in 5 to 1 in 10 days |
| 0.95 | Very cautious: protects against the worst 1 in 20 days |

Show the label only if the run's tail was resolved (`warnings` empty). Otherwise show the number and the warning.

## 6. Sample

`samples/run_sample.json` is a real payload assembled by hand from the 29 Sep proxy run: 3 selected sites, 4 basing records, 20 scenario losses. It is a mock for UI development, not output from an exporter. Fields in section 4 marked Missing are absent from it on purpose.

## 7. Work items for the exporter (outside the UI team)

1. Persist `warnings` into the summary (about a 5-line change in `vif_staged_solve.py`).
2. Persist scenario definitions with each run.
3. Compute `fleet.json` and `scenario_detail` from solver variables; confirm `ds_y` and `ds_z` meaning first.
4. Write `index.json` and `network.json`, and validate every file against JSON Schema files (to be written once these fields are agreed).
