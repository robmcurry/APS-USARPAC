# Chapter 2 — stochastic prepositioning

Type-indexed, distance-state CVaR model and the staged matheuristic (Stage 1A/1B, Stage 2 per-scenario routing, Stage 3 CVaR allocation).

## Layout

| Path | Contents |
|---|---|
| `model/` | `model.py` (dispatch), `model_distance_state.py` (the Chapter 2 formulation), `model_vif.py` (PRS-VIF individual-vehicle formulation), `input_builder.py` |
| `config/` | `model_parameters.yaml`: fleet, risk (`beta`), capacities, ranges |
| `network/` | `nodes.csv` (50 nodes), `arcs_air/sea/land.csv`, builders |
| `scenarios/` | scenario generator and validator |
| `scripts/` | `vif_staged_solve.py` (staged solve, main entry), `vif_staged_proxy_solve.py` (Stage 1B), `vif_tail_scenario_selection.py`, regression and diagnostic scripts |
| `analysis/` | convergence, sensitivity, and problem-size analyses |
| `tests/` | pytest suite |
| `docs/` | method write-ups, `MODEL_CHANGELOG.md`, `DECISIONS.md` |
| `experiments_prsvif/` | PRS-VIF phase notes and reports. Kept until the final stochastic formulation is chosen |
| `lineage/` | frozen July 2026 snapshot (`aps_usarpac`) that this code grew from |
| `output/` | run outputs (not tracked) |

## Running (from this folder)

```bash
pip install -r requirements.txt
python scripts/vif_staged_solve.py --scenarios 150 --strategic-scenarios 50 --seed 42 --beta 0.9
pytest
```

Note `--strategic-scenarios` must be large enough to resolve the CVaR tail at your `beta`; the script warns when it is not. See `docs/DECISIONS.md`.

## Known state (2026-10-09)
- `pytest`: 3 pass, 12 fail. The 12 failures are the PRS-VIF phase 1-4 tests, whose hand-computed optima no longer match. They failed before the restructure too and need triage.
- `scripts/check_aggregate_regression.py` compares against a locked July baseline (`output/baseline_aggregate_toy.json`, not tracked; copy it from the `individual-vehicle-indexing` workspace). It currently reports a difference caused by configuration drift: `mip_gap` is 0.1 in `config/model_parameters.yaml` and was 0.01 in the baseline. Solved to gap 0, the model reproduces the July objective (see `lineage/README.md`).
