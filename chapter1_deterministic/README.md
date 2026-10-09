# Chapter 1 — deterministic model

Code for the Chapter 1 paper, "An Analytical Framework for Optimizing Resource Stock Pre-positioning, Allocation, and Routing for U.S. Army Disaster Readiness".

The code is the `disaster_logistics_model/` package, moved here unchanged. See its own `README.md` for structure.

## Running

Run the experiment scripts from inside the package directory:

```bash
cd chapter1_deterministic/disaster_logistics_model
python main_base.py      # varies number of APS sites
python main_base_L.py    # varies safety stock
python main_redun.py     # varies redundancy
python command_post.py   # all of the above
```

The batch module uses package-qualified imports and is run from `chapter1_deterministic/`:

```bash
cd chapter1_deterministic
python -m disaster_logistics_model.optimization.run_batch_optimization
```

## Known issues (existed before the 2026-10 restructure)
- `run_batch_optimization.py` fails at import: it needs `solve_deterministic_vrp_with_aps_single_stage_commodity`, which `deterministic_model_single_stage.py` does not define.
- `generate_visuals.py` expects `output/batch_summary.csv`, which is not in the repo.
- Importing `main_base_L` or `main_redun` runs a full pass of the script's prints; run them as scripts.
