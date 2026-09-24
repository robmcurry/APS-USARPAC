"""
vif_phase5_structural_validation.py

Phase 5, step 3: build the real 50-node PRS-VIF instance and confirm it
assembles -- variable/constraint counts, then LP-relaxation feasibility.
Explicitly NOT a MIP solve (per phase instructions).

num_scenarios is a CLI arg (default 3) rather than hardcoded to the
gospel's full |Omega|=100. C7 (VifVehicleCapacity) and C9
(VifVehicleConservation) are now built via bulk model.addConstrs()
(converted after |Omega|=100 was confirmed to exhaust memory under the
prior per-row addConstr() loop -- see PHASE5_NOTES.md, "addConstrs
conversion"), but their constraint COUNT is unchanged and still scales
as |Omega| * |L| * |A_mode|: at |Omega|=100 this is ~4.57M C7
constraints alone (28 air + 8 sea + 60 land vehicles, ~45,744 (l,i,j)
triples per scenario). Run this script with a small num_scenarios first
to get real per-scenario build-time and
memory-cost data on this machine before deciding whether a full
|Omega|=100 structural build is worth the wall-clock/memory cost --
see PHASE5_NOTES.md for the timing this produced and the resulting
recommendation.

Run: python scripts/vif_phase5_structural_validation.py [num_scenarios]
     (from aps_usarpac/)
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_parameters
from model.input_builder import build_stochastic_instance
from model.model import solve_stochastic_cvar
from network.network_builder import build_graph, load_locations
from scenarios.scenario_generator import generate_scenarios


def main():
    num_scenarios = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    params = load_parameters()
    locations = load_locations()
    G = build_graph(locations)

    t0 = time.time()
    scenarios = generate_scenarios(G, locations, num_scenarios=num_scenarios, seed=32)
    t_scenarios = time.time() - t0

    t0 = time.time()
    instance = build_stochastic_instance(locations, scenarios, params, alpha=1.0)
    t_instance = time.time() - t0

    print(f"num_scenarios={num_scenarios}")
    print(f"scenario generation: {t_scenarios:.2f}s")
    print(f"instance assembly:   {t_instance:.2f}s")
    print(f"|N|={len(instance['nodes'])}  |N^P|={len(instance['ppl_nodes'])}  "
          f"|R|={len(instance['commodities'])}  |Omega|={len(instance['scenarios'])}")
    print(f"|A_sea|={len(instance['modal_arcs']['sea'])}  "
          f"|A_air|={len(instance['modal_arcs']['air'])}  "
          f"|A_land|={len(instance['modal_arcs']['land'])}")
    vt = instance["vehicle_types"]
    print(f"|K|={len(vt)}  fleet sizes={ {k: v['fleet_size'] for k, v in vt.items()} }  "
          f"|L|={sum(v['fleet_size'] for v in vt.values())}")

    t0 = time.time()
    build_result = solve_stochastic_cvar(
        instance, vehicle_formulation="vif", build_only=True, verbose=False,
    )
    t_build = time.time() - t0
    model = build_result["model"]

    print(f"\nGurobi model build (build_only=True): {t_build:.2f}s")
    print(f"num_vars    = {model.NumVars}")
    print(f"num_bin_vars= {model.NumBinVars}")
    print(f"num_constrs = {model.NumConstrs}")

    var_families = build_result["variables"]
    print("\nVariable family sizes:")
    for name, v in var_families.items():
        try:
            n = len(v)
        except TypeError:
            n = 1
        print(f"  {name:6s}: {n}")

    # split on whichever of "_" or "[" comes first: the still-per-row-named
    # families (e.g. "VifBaseAssign_k...") use "_", while C7/C9 now use
    # bulk addConstrs()'s auto-generated "Name[idx1,idx2,...]" bracket
    # style -- both need to collapse to their family name, not one row
    # per unique index tuple.
    constr_family_counts = {}
    for c in model.getConstrs():
        prefix = re.split(r"[_\[]", c.ConstrName, maxsplit=1)[0]
        constr_family_counts[prefix] = constr_family_counts.get(prefix, 0) + 1
    print("\nConstraint family sizes (by name prefix before first '_' or '['):")
    for name, n in sorted(constr_family_counts.items()):
        print(f"  {name:26s}: {n}")

    # --- LP relaxation feasibility (NOT a MIP solve) ---
    t0 = time.time()
    relaxed = model.relax()
    relaxed.Params.OutputFlag = 0
    relaxed.optimize()
    t_relax = time.time() - t0
    status_map = {2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED"}
    print(f"\nLP relaxation solve: {t_relax:.2f}s  status={status_map.get(relaxed.Status, relaxed.Status)}"
          + (f"  obj={relaxed.ObjVal:.4f}" if relaxed.Status == 2 else ""))


if __name__ == "__main__":
    main()
