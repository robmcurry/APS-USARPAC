# experiments_prsvif/ — PRS-VIF research trail

Notes and reports from the individual-vehicle-indexed (PRS-VIF) formulation. They stay under Chapter 2 until the final stochastic formulation is chosen. The solver code remains in `../model/model_vif.py`, because `model.py` dispatches to it.

Contents: `PHASE1_NOTES.md` to `PHASE5_NOTES.md`, `PRSVIF_DELTA_REPORT.md`, `PRSVIF_MEMORY_WALL.html/.pdf`.

## Layout and paths

- `scripts/`: PRS-VIF diagnostics and runs (`toy_individual_*`, `lazy_subtour_*`, `vif_phase5_*`, `vehcap_diagnostic`)
- `tests/`: the phase 1-4 hand-computed-optimum tests (currently 12 failing; see `../README.md`)

Run everything from `chapter2_stochastic/`, for example `python experiments_prsvif/scripts/toy_individual_smoke_test.py` or `pytest experiments_prsvif/tests`. Outputs go to `chapter2_stochastic/output/`.

The `PHASE*_NOTES.md` files and reports are historical records and keep the paths as they were written. Translate them like this: `aps_usarpac/` is `chapter2_stochastic/`, `scripts/{toy_individual,lazy_subtour,vif_phase5,vehcap}*` is `experiments_prsvif/scripts/`, and `tests/test_vif_phase[1-4]*` is `experiments_prsvif/tests/`.
