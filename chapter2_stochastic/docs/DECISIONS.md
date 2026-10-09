# Decision log

Newest first. Each entry: decision, evidence, what it constrains. Dates are session dates.

## 2026-10-06 — Stage 1A "top-50 by probability" is really scenarios 0–49
Scenario probabilities are uniform (`scenario_generator.py`: `1.0 / num_scenarios`), so `_selected_scenarios` sorts by `(-probability, id)` and degenerates to ids 0–49. In the measured instance it captured 5 of the 15 worst scenarios, a weak fit for CVaR.
Constrains: Stage 1A results at β=0.9 reflect an arbitrary subset, not a tail-aware one.

## 2026-10-06 — CVaR tail resolution is (1−β) × strategic scenarios
β=0.9 needs at least 21 strategic scenarios, and β=0.95 needs at least 40, for a tail of 2 or more. The script default of 3 strategic scenarios gives a 0.3-scenario tail. Renormalizing a worst-k subset silently raises effective β (worst-50-of-150 ≈ CVaR 0.967).
Implemented: `--beta` flag, tail-resolution warning, run provenance block in `vif_staged_solve.py` (uncommitted as of this entry).
Constrains: any risk-dial (β) sweep must be run at honest tail resolution.

## 2026-10-06 — Advisor's tail-aware selector is not a routine screen as shipped
`vif_tail_selection` is not wired into anything. Its lexicographic second pass assumes an exact optimum from pass 1, which Gurobi's crossover provides. The crossover run did not finish in 20 minutes (1.5M+ simplex iterations). Barrier-only (`--barrier-only`, 1e-4 band) takes about 640 s for pass 1 but is not yet verified end to end.
Open: ask the advisor whether leaving it unwired was deliberate.

## 2026-10-01 — Fleet is 8 types; range, duty-factor, and Θ conventions documented
See `MODEL_CHANGELOG.md` for the equation-level impact. Θ bonus set at 10–25% (land highest), not 75%, after a plausibility argument. Four open audit items remain at the bottom of the changelog.

## 2026-09-30 — Stage 1A runs on a 50-scenario subset of 150, but never silently
The user's instruction to run 150 scenarios was substituted with 50 without saying so; this was reversed. Rule: when asked for N scenarios, run N, and label any subset explicitly.
