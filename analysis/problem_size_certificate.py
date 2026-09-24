"""
problem_size_certificate.py

Standalone, read-only diagnostic: builds the exact Gurobi model a standard
solve would build (same locations/scenarios/params, default alpha=1.0), but
never calls model.optimize() (via solve_stochastic_cvar(..., build_only=True)).
Prints set sizes, per-family variable counts, and per-family constraint
counts as actually realized in the model object -- not estimated from the
formulation, counted directly off model.getVars()/model.getConstrs().

Run as: python -m analysis.problem_size_certificate from aps_usarpac/ root.
Writes output/problem_size_certificate.md.
"""
import os
from collections import Counter

from config.loader import load_parameters
from model.input_builder import build_stochastic_instance
from model.model import solve_stochastic_cvar
from network.network_builder import build_graph, load_locations
from scenarios.scenario_generator import generate_scenarios

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

# Constraint family name prefixes exactly as used in model/model.py's
# model.addConstr(..., name=f"<FAMILY>_...") calls. Checked longest/most
# specific first is unnecessary here since every family name below is a
# distinct full token followed by "_" (or, for the two budget constraints,
# an exact match with no suffix at all).
CONSTRAINT_FAMILIES = [
    "SiteBudget",
    "SelectionBudget",
    "InventoryReleaseBound",
    "NonPPLReleaseZero",
    "FlowBalance_PPL",
    "FlowBalance_nonPPL",
    "ModalOutboundFeasibility",
    "ArcCapacity",
    "TransferBacking",
    "TransferCapacity",
    "VehicleConservation",
    "VehicleCapFlow",
    "TurnExemptUB1",
    "TurnExemptUB2",
    "TurnExemptLB",
    "DistanceBudget",
    "LossDefinition",
    "CVaRExcess",
]


def classify_constraint(name: str) -> str:
    for fam in CONSTRAINT_FAMILIES:
        if name == fam or name.startswith(fam + "_"):
            return fam
    return "UNCLASSIFIED"


def main():
    params = load_parameters()
    locations = load_locations()
    G = build_graph(locations)
    scenarios = generate_scenarios(
        G, locations,
        save_path=os.path.join(OUTPUT_DIR, "problem_size_certificate_scenarios.csv"),
    )
    instance = build_stochastic_instance(
        locations=locations, scenarios=scenarios, params=params,
    )

    build = solve_stochastic_cvar(instance, build_only=True, verbose=False)
    model = build["model"]
    v = build["variables"]

    # --- Set sizes ---
    N = instance["nodes"]
    NP = instance["ppl_nodes"]
    Omega = instance["scenarios"]
    modal_arcs = instance["modal_arcs"]
    K_m = instance["K_m"]
    transfer_cap = instance["transfer_cap"]
    has_vehicles = bool(instance.get("vehicle_types"))
    sum_T_i = sum(len(pairs) for pairs in transfer_cap.values())

    # --- Variable family counts (ground truth: len() of each tupledict as
    # actually built; eta is a single scalar Var, not a tupledict) ---
    var_counts = {
        "p": len(v["p"]),
        "x": len(v["x"]),
        "z": len(v["z"]),
        "y (release)": len(v["release"]),
        "tau": len(v["tau"]),
        "n": len(v["n"]),
        "eta": 1,
        "xi": len(v["xi"]),
        "loss (code-only, no dissertation symbol)": len(v["loss"]),
        "g_turn (code-only McCormick auxiliary, eq. 19 linearization)": len(v["g"]),
    }
    var_total_check = sum(var_counts.values())

    # --- Constraint family counts: bucket every constraint actually in the
    # model by its name prefix, cross-checked against model.NumConstrs ---
    model.update()
    constr_counts = Counter()
    for c in model.getConstrs():
        constr_counts[classify_constraint(c.ConstrName)] += 1
    constr_total_check = sum(constr_counts.values())

    # --- y (release) domain check: does a release[w,i,r] var exist for
    # i outside N^P, or only for i in N^P? ---
    release_keys = list(v["release"].keys())
    release_nodes = {i for (_w, i, _r) in release_keys}
    y_over_all_N = set(N).issubset(release_nodes) and release_nodes == set(N)
    y_over_NP_only = release_nodes == set(NP)

    lines = []
    lines.append("# Problem Size Certificate")
    lines.append("")
    lines.append("**Generated:** 2026-07-10")
    lines.append(
        "**Method:** model built via `solve_stochastic_cvar(instance, build_only=True)` "
        "-- every variable and constraint exists in the live Gurobi model object; "
        "`model.optimize()` is never called. Counts below are read directly from "
        "`model.getVars()`/`model.getConstrs()` (via `len()` of each variable "
        "tupledict and a name-prefix bucketing of every constraint), not estimated "
        "from the formulation."
    )
    lines.append(
        "**Instance:** default `build_stochastic_instance` call (alpha=1.0), "
        f"seed={params.get('seed')}, N={params.get('num_scenarios')} -- the same "
        "instance a standard solve builds."
    )
    lines.append("")

    lines.append("## Set sizes")
    lines.append("")
    lines.append("| Set | Size |")
    lines.append("|---|---|")
    lines.append(f"| \\|N\\| (all nodes) | {len(N)} |")
    lines.append(f"| \\|N^P\\| (PPL candidates) | {len(NP)} |")
    lines.append(f"| \\|A_sea\\| | {len(modal_arcs['sea'])} |")
    lines.append(f"| \\|A_air\\| | {len(modal_arcs['air'])} |")
    lines.append(f"| \\|A_land\\| | {len(modal_arcs['land'])} |")
    lines.append(f"| \\|Omega\\| (scenarios) | {len(Omega)} |")
    lines.append(f"| \\|K_sea\\| | {len(K_m.get('sea', []))} |")
    lines.append(f"| \\|K_air\\| | {len(K_m.get('air', []))} |")
    lines.append(f"| \\|K_land\\| | {len(K_m.get('land', []))} |")
    lines.append(f"| has_vehicles | {has_vehicles} |")
    lines.append(f"| sum_i \\|T_i\\| (transfer-eligible mode-pairs, summed over nodes) | {sum_T_i} |")
    lines.append("")

    lines.append("## Variable family counts (ground truth: `len()` of each built tupledict)")
    lines.append("")
    lines.append("The 8 families explicitly requested are p, x, z, y, tau, n, eta, xi. Two more")
    lines.append("families exist in the model with no dissertation symbol (`loss` is a code-only")
    lines.append("convenience variable; `g_turn` is the McCormick linearization auxiliary for the")
    lines.append("p_j-coupled turnaround exemption in constraint 19) -- both included below so the")
    lines.append("total reconciles exactly against `model.NumVars`.")
    lines.append("")
    lines.append("| Family (code name / dissertation symbol) | Count |")
    lines.append("|---|---|")
    lines.append(f"| p (site selection) | {var_counts['p']} |")
    lines.append(f"| x (modal flow) | {var_counts['x']} |")
    lines.append(f"| z (unmet demand) | {var_counts['z']} |")
    lines.append(f"| release / **y** (inventory release) | {var_counts['y (release)']} |")
    lines.append(f"| tau (intermodal transfer) | {var_counts['tau']} |")
    lines.append(f"| n (vehicle count) | {var_counts['n']} |")
    lines.append(f"| eta (CVaR threshold, scalar) | {var_counts['eta']} |")
    lines.append(f"| xi (CVaR excess) | {var_counts['xi']} |")
    lines.append(f"| loss (code-only, no dissertation symbol) | {var_counts['loss (code-only, no dissertation symbol)']} |")
    lines.append(f"| g_turn (code-only McCormick auxiliary, eq. 19 linearization) | {var_counts['g_turn (code-only McCormick auxiliary, eq. 19 linearization)']} |")
    lines.append(f"| **Sum of families above** | **{var_total_check}** |")
    lines.append(f"| `model.NumVars` (model ground truth) | **{model.NumVars}** |")
    lines.append(
        f"| Match | {'YES' if var_total_check == model.NumVars else 'MISMATCH -- unaccounted variables exist, see below'} |"
    )
    lines.append("")

    lines.append("## Constraint family counts (bucketed by name prefix, every constraint in the model)")
    lines.append("")
    lines.append("| Family | Count |")
    lines.append("|---|---|")
    for fam in CONSTRAINT_FAMILIES:
        lines.append(f"| {fam} | {constr_counts.get(fam, 0)} |")
    if constr_counts.get("UNCLASSIFIED", 0):
        lines.append(f"| **UNCLASSIFIED (name didn't match any known family)** | {constr_counts['UNCLASSIFIED']} |")
    lines.append(f"| **Sum of families above** | **{constr_total_check}** |")
    lines.append(f"| `model.NumConstrs` (model ground truth) | **{model.NumConstrs}** |")
    lines.append(
        f"| Match | {'YES' if constr_total_check == model.NumConstrs else 'MISMATCH -- see note below'} |"
    )
    lines.append("")

    lines.append("## y (release) domain")
    lines.append("")
    lines.append(
        f"release/y variables exist for **{len(release_nodes)}** distinct nodes "
        f"out of \\|N\\|={len(N)} total, \\|N^P\\|={len(NP)}."
    )
    if y_over_all_N:
        lines.append(
            "**y is defined (as a decision variable) over all of N, not only N^P** "
            "-- matching the dissertation exactly (`docs/formulation_only.md` "
            "eq. 9/10: the release bound applies for i in N^P, and a separate "
            "constraint sets y=0 for i in N\\N^P; both are constraints ON a "
            "variable that exists over all of N, not a restriction on the "
            "variable's index set itself). In code: `model.py` builds "
            "`release = model.addVars(((w, i, r) for w in Omega for i in N for r "
            "in R), ...)` and then adds `NonPPLReleaseZero` constraints "
            "(`model.py:226-231`) forcing release[w,i,r]==0 for i not in N^P."
        )
    elif y_over_NP_only:
        lines.append("**y is defined only over N^P** (variable index set restricted at creation).")
    else:
        lines.append(
            f"**Unexpected domain** -- release exists for a node set that is "
            f"neither all of N nor exactly N^P. release_nodes size={len(release_nodes)}."
        )
    lines.append("")

    lines.append("## sum_i |T_i|")
    lines.append("")
    lines.append(
        f"T_i = the set of transfer-eligible (m1, m2) mode-pairs at node i "
        f"(`instance[\"transfer_cap\"][i].keys()`, only pairs with capacity > 0 "
        f"per `build_transfer_capacity`, `model/input_builder.py:146-182`). "
        f"**sum_i |T_i| = {sum_T_i}** across the "
        f"{len(transfer_cap)} nodes with at least one active transfer pair. "
        f"This is exactly `len(instance[\"transfer_cap\"])`-node-summed pair "
        f"count, and (sum_i |T_i|) x |Omega| x |R| = "
        f"{sum_T_i} x {len(Omega)} x {len(instance['commodities'])} = "
        f"{sum_T_i * len(Omega) * len(instance['commodities'])} should equal the "
        f"tau variable count above."
    )
    lines.append("")

    model.dispose()

    report = "\n".join(lines)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "problem_size_certificate.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
