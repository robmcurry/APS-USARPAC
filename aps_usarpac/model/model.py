"""
model.py

Third and final stage of the pipeline (stochastic instance -> gurobi model
-> results dict). Builds and solves the extensive-form two-stage stochastic
prepositioning model with a CVaR objective.

Model structure:
    First stage (here-and-now decision, made before the disaster realizes):
        p[i]         = 1 if node i is selected as a prepositioning location (ppl)

    Second stage (recourse decisions, made after scenario w is observed):
        x[w,i,j,r]   = flow of commodity r on arc (i,j) in scenario w
        z[w,j,r]     = unmet demand of commodity r at node j in scenario w
        release[w,i,r] = inventory released from site i in scenario w

    Risk variables:
        eta          = VaR-like threshold for CVaR
        xi[w]        = amount by which scenario w's loss exceeds eta

CVaR setup:
    objective = eta + (1/(1-beta)) * sum_w prob[w] * xi[w]
    this targets the expected loss within the worst (1-beta) fraction of
    scenarios rather than the average loss across all scenarios, so site
    selection is steered toward protecting against severe/tail disasters
    rather than just performing well on average.

Unit convention:
    1 unit of any commodity = 1 person-day of support
"""

from typing import Dict, Any, Tuple, List

import gurobipy as gp
from gurobipy import GRB


def _build_subtour_callback(
    n_ind: Dict,
    home_by_k: Dict[str, Dict[int, int]],
    air_indiv_types: List[str],
    Omega: List[int],
    air_arcs: List[Tuple[int, int]],
    stats: Dict[str, int],
):
    """
    Build a Gurobi lazy-constraint callback enforcing subtour elimination for
    the "individual" vehicle_formulation's per-vehicle-instance binaries
    (n_ind), replacing the static DepartureSingleNode/DepartureNodeLink
    constraints (5).

    On every integer-feasible incumbent (MIPSOL), for each (scenario w,
    air-individual type k, vehicle instance l): computes which nodes are
    reachable from l's home node by following its selected (value > 0.5)
    n_ind arcs, then groups any leftover selected arcs (touching nodes never
    reached from home) into weakly-connected components. Each such component
    S is a genuine disconnected subtour -- a self-sustaining loop the MIP got
    "for free" without consuming a home-departure credit (see
    toy_individual_cyclic_test.py, which found and confirmed this failure
    mode with constraint 5 removed and no callback in place). For each S,
    adds the direct subtour-elimination cut:
        sum_{a in S, b in S, (a,b) in air_arcs} n_ind[w,k,l,"air",a,b] <= |S| - 1
    which forbids exactly that disconnected arc set (and any other full
    closure within S) without touching legitimate home-reachable arcs.

    stats: mutable dict, updated in place with "invocations" (MIPSOL calls
    seen) and "cuts_added" (total lazy constraints emitted) so the caller can
    report them after model.optimize() returns.
    """
    air_arc_set = set(air_arcs)

    # Precompute once: (w, k, l) -> [(i, j), ...] over all air arcs. n_ind is
    # fully dense over (w, k in air_indiv_types, l, "air", i, j) for (i, j)
    # in air_arcs, so this just re-groups the existing dict's keys.
    group_arcs: Dict[Tuple[int, str, int], List[Tuple[int, int]]] = {}
    for (w, k, l, _m, i, j) in n_ind:
        group_arcs.setdefault((w, k, l), []).append((i, j))

    def callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return
        stats["invocations"] += 1

        all_keys = list(n_ind.keys())
        all_vals = model.cbGetSolution([n_ind[key] for key in all_keys])
        val_map = dict(zip(all_keys, all_vals))

        for (w, k, l), arcs in group_arcs.items():
            selected = [
                (i, j) for (i, j) in arcs
                if val_map[(w, k, l, "air", i, j)] > 0.5
            ]
            if not selected:
                continue

            home = home_by_k[k][l]
            reached = {home}
            changed = True
            while changed:
                changed = False
                for (i, j) in selected:
                    if i in reached and j not in reached:
                        reached.add(j)
                        changed = True

            touched = {node for arc in selected for node in arc}
            phantom_nodes = touched - reached
            if not phantom_nodes:
                continue

            remaining = set(phantom_nodes)
            while remaining:
                seed = next(iter(remaining))
                component = {seed}
                stack = [seed]
                while stack:
                    node = stack.pop()
                    for (i, j) in selected:
                        if i == node and j in remaining and j not in component:
                            component.add(j)
                            stack.append(j)
                        elif j == node and i in remaining and i not in component:
                            component.add(i)
                            stack.append(i)
                remaining -= component
                if len(component) < 2:
                    continue
                lhs = gp.quicksum(
                    n_ind[w, k, l, "air", a, b]
                    for a in component for b in component
                    if (a, b) in air_arc_set
                )
                model.cbLazy(lhs <= len(component) - 1)
                stats["cuts_added"] += 1

    return callback


def solve_stochastic_cvar(
    instance: Dict[str, Any],
    time_limit: float = None,
    mip_gap: float = None,
    verbose: bool = True,
    detailed_extraction: bool = False,
    build_only: bool = False,
    vehicle_formulation: str = "aggregate",
    _debug_skip_departure_single_node: bool = True,
    _debug_skip_symmetry_break: bool = False,
    _debug_mip_focus: int = 1,
) -> Dict[str, Any]:
    """
    Solve the extensive-form CVaR stochastic prepositioning model.

    Args:
        instance: stochastic instance dictionary produced by model/input_builder.py
        time_limit: optional Gurobi time limit in seconds
        mip_gap: optional relative MIP gap target
        verbose: if True, allow standard solver output
        detailed_extraction: if True, also populate the diagnostic result keys
            scenario_transport_cost, scenario_unmet_penalty, and
            node_flow_summary. Off by default: these passes re-walk every flow
            variable and no sweep consumer reads them (R8 audit finding).
        build_only: if True, return immediately after all variables and
            constraints are added (model.update() called, but NOT
            model.optimize()). Returns {"model": model, "variables": {...}}
            only -- no solution fields. For problem-size inspection
            (analysis/problem_size_certificate.py) without paying for a solve.
        vehicle_formulation: "aggregate" (default) uses the existing
            n^omega_{k,m,ij} integer-vehicle-count formulation, unchanged,
            for every mode. "individual" uses per-vehicle-instance binaries
            n^omega_{k,l,m,ij} for air-mode k in {C-17, C-130J} only; sea,
            land, and any other air type still use the aggregate n
            unconditionally. Callers that don't pass this at all get
            "aggregate", i.e. current behavior.
        _debug_skip_departure_single_node: When True (default) and
            vehicle_formulation="individual", the static DepartureSingleNode/
            DepartureNodeLink constraints (5) are NOT built. Subtour
            elimination is instead enforced by a Gurobi lazy-constraint
            callback (see _build_subtour_callback below), which detects
            disconnected (non-home-reachable) arc components on each
            integer-feasible incumbent and cuts them directly -- validated on
            the toy cyclic network (toy_individual_cyclic_test.py's added
            2<->3 non-home cycle): with no mitigation the optimizer selects 3
            phantom/disconnected instances; with this callback active, an
            independent post-solve reachability check confirms 0 phantom
            instances remain (1 lazy cut sufficed, over 3 MIPSOL callback
            invocations). Pass False to fall back to the static constraint
            (DIAGNOSTIC/comparison use only; does not disable the callback --
            the two mechanisms are independent and either alone suffices).
            Does not touch VehicleConservationIndiv (3), which is the sole
            carrier of the p_j/home-node linkage.
        _debug_skip_symmetry_break: DIAGNOSTIC ONLY. When True and
            vehicle_formulation="individual", skips the VehicleSymmetryBreak
            constraints (scalar ordering of interchangeable same-home-node
            vehicle instances). Default False (constraints active). Exists
            to reproduce the pre-symmetry-break baseline for isolated A/B
            comparison against later runs that added this constraint family.
        _debug_mip_focus: DIAGNOSTIC ONLY. Value passed to Gurobi's
            Params.MIPFocus when vehicle_formulation="individual". Default 1
            (prioritize finding feasible incumbents). Pass 0 to restore
            Gurobi's own default (balanced) focus, to reproduce the
            baseline behavior from before MIPFocus tuning was introduced.

    Returns:
        results dictionary with selected sites, objective value, and solution details
    """

    # --- Unpack sets ---
    N: List[int] = instance["nodes"]
    PPL: List[int] = instance["ppl_nodes"]
    R: List[str] = instance["commodities"]
    Omega: List[int] = instance["scenarios"]
    modes: List[str] = instance.get("modes", ["sea", "air", "land"])

    # --- Unpack modal network data ---
    modal_arcs = instance["modal_arcs"]           # {mode: [(i,j),...], "all": [...]}
    modal_residual = instance["modal_residual"]   # list[scenario_id] -> {mode: {(i,j): {r: cap}}}
    transfer_cap = instance["transfer_cap"]       # {node_id: {(m1,m2): capacity}}
    transfer_cost_mult = instance["transfer_cost"]  # {(m1,m2): multiplier}
    modal_arc_cost = instance["modal_arc_cost"]   # {mode: {(i,j): distance_km/1000}}
    modal_arc_distance = instance.get("modal_arc_distance", {})  # {mode: {(i,j): km}}

    # --- Vehicle heterogeneity (constraints 16-19) ---
    vehicle_types = instance.get("vehicle_types", {})
    K_m = instance.get("K_m", {m: [] for m in modes})
    has_vehicles = bool(vehicle_types)

    # --- Unpack parameters (unchanged from pre-Step-3) ---
    prob = instance["probability"]
    demand = instance["demand"]
    inventory_if_open = instance["inventory_if_open"]
    inventory_availability = instance.get("inventory_availability", None)
    safety_stock_fraction = float(instance.get("safety_stock_fraction", 0.0))
    releasable_fraction = 1.0 - safety_stock_fraction
    site_cost = instance["site_cost"]
    selection_budget = instance["selection_budget"]
    penalty = instance["penalty"]
    P_max = instance["P_max"]
    beta = instance["beta"]

    PPL_set = set(PPL)
    non_ppl_nodes = [i for i in N if i not in PPL_set]

    # --- Precompute per-mode adjacency maps ---
    # modal_incoming[m][j] = list of (i,j) arcs on mode m arriving at j
    # modal_outgoing[m][i] = list of (i,j) arcs on mode m leaving i
    modal_incoming: Dict[str, Dict[int, List]] = {m: {j: [] for j in N} for m in modes}
    modal_outgoing: Dict[str, Dict[int, List]] = {m: {i: [] for i in N} for m in modes}
    for m in modes:
        for i, j in modal_arcs[m]:
            modal_outgoing[m][i].append((i, j))
            modal_incoming[m][j].append((i, j))

    # precompute which (m2) modes each node can transfer OUT to per source mode m1,
    # and which (m1) modes each node can transfer IN from per destination mode m2
    transfer_out_modes: Dict[int, Dict[str, List[str]]] = {}
    transfer_in_modes: Dict[int, Dict[str, List[str]]] = {}
    for node_id, pairs in transfer_cap.items():
        transfer_out_modes[node_id] = {m: [] for m in modes}
        transfer_in_modes[node_id] = {m: [] for m in modes}
        for (m1, m2) in pairs:
            transfer_out_modes[node_id][m1].append(m2)
            transfer_in_modes[node_id][m2].append(m1)

    # --- Build flow and transfer key lists ---
    # explicit list comprehension keeps variable creation fast and unambiguous
    flow_keys = [
        (w, m, i, j, r)
        for w in Omega
        for m in modes
        for (i, j) in modal_arcs[m]
        for r in R
    ]
    transfer_keys = [
        (w, i, m1, m2, r)
        for w in Omega
        for i in N
        if i in transfer_cap
        for (m1, m2) in transfer_cap[i]
        for r in R
    ]

    # per-scenario index for efficient loss definition construction
    flow_keys_by_w: Dict[int, List] = {w: [] for w in Omega}
    for (w, m, i, j, r) in flow_keys:
        flow_keys_by_w[w].append((m, i, j, r))
    transfer_keys_by_w: Dict[int, List] = {w: [] for w in Omega}
    for (w, i, m1, m2, r) in transfer_keys:
        transfer_keys_by_w[w].append((i, m1, m2, r))

    # --- Build model ---
    model = gp.Model("stochastic_prepositioning_cvar")
    model.Params.OutputFlag = 1 if verbose else 0
    # M2 audit fix: barrier-only root LP instead of the concurrent default.
    # The concurrent root runs primal simplex + dual simplex + barrier
    # simultaneously (3x root-LP working memory, plus reported 45-60s of
    # "concurrent spin time" per solve). Measured on the full alpha=1.0,
    # N=100, 1%-gap instance (2026-07-09): Method=2 solved in 628s at 4.0 GB
    # peak RSS vs 1139s at 5.2 GB for the concurrent default — same sites,
    # equivalent gap. Solver-parameter change only; formulation unaffected.
    model.Params.Method = 2

    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)

    # --- Decision variables ---

    # p[i] - first-stage binary, unchanged from pre-Step-3
    p = model.addVars(PPL, vtype=GRB.BINARY, name="p")

    # x[w,m,i,j,r] - second-stage continuous flow of commodity r on mode-m
    # arc (i,j) in scenario w; indexed only over existing mode-arc combos
    x = model.addVars(flow_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

    # tau[w,i,m1,m2,r] - intermodal transfer of commodity r at node i from
    # mode m1 to mode m2 in scenario w; indexed only over active transfer pairs
    tau = model.addVars(transfer_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="tau")

    # z[w,j,r] - unmet demand; unchanged from pre-Step-3 (node-level, not per-mode)
    z = model.addVars(
        ((w, j, r) for w in Omega for j in N for r in R),
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="z",
    )

    # release[w,i,r] - inventory released from site i; unchanged from pre-Step-3
    release = model.addVars(
        ((w, i, r) for w in Omega for i in N for r in R),
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="release",
    )

    # n_ind/dep_node/g_ind: individual-formulation-only variable families
    # (air-mode only). Defaulted here, before either branch runs, so they're
    # always defined even under vehicle_formulation="aggregate" or
    # has_vehicles=False -- the "individual" branches below overwrite them.
    n_ind: Dict = {}
    dep_node: Dict = {}
    g_ind: Dict = {}

    # n[w,k,m,i,j] - integer vehicle count: number of type-k vehicles of mode m
    # traversing arc (i,j) in scenario w; only defined for k in K_m[m]
    if vehicle_formulation == "aggregate":
        if has_vehicles:
            n_keys = [
                (w, k, m, i, j)
                for w in Omega
                for m in modes
                for k in K_m.get(m, [])
                for (i, j) in modal_arcs[m]
            ]
            n = model.addVars(n_keys, lb=0, vtype=GRB.INTEGER, name="n")
        else:
            n = {}
    elif vehicle_formulation == "individual":
        # Individual-vehicle indexing is scoped to air mode only (C-17,
        # C-130J) -- sea and land keep the same aggregate n unconditionally.
        # n itself is built exactly as in the aggregate branch (same keys,
        # all modes) so constraint (18) below can reference it unmodified;
        # for air-individual types its value is pinned by the new
        # VehicleAggregation equality constraint further down, not
        # optimized directly.
        if has_vehicles:
            n_keys = [
                (w, k, m, i, j)
                for w in Omega
                for m in modes
                for k in K_m.get(m, [])
                for (i, j) in modal_arcs[m]
            ]
            n = model.addVars(n_keys, lb=0, vtype=GRB.INTEGER, name="n")

            air_indiv_types = [k for k in K_m.get("air", []) if k in ("C-17", "C-130J")]
            n_ind_keys = [
                (w, k, l, "air", i, j)
                for k in air_indiv_types
                for w in Omega
                for l in range(1, vehicle_types[k]["fleet_size"] + 1)
                for (i, j) in modal_arcs["air"]
            ]
            n_ind = model.addVars(n_ind_keys, vtype=GRB.BINARY, name="n_ind") if n_ind_keys else {}
        else:
            n = {}
            n_ind = {}
    else:
        raise ValueError(f"unknown vehicle_formulation: {vehicle_formulation!r}")

    # g[w,k,j] (McCormick auxiliary for the p_j-coupled turnaround exemption in
    # constraint 19) is created further below, only inside the has_vehicles
    # block, and only if g_keys is non-empty. Default here so it's always
    # defined by the time build_only reads it back.
    g: Dict = {}

    # eta, xi[w], loss[w] - CVaR variables; unchanged from pre-Step-3
    eta = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="eta")
    xi = model.addVars(Omega, lb=0.0, vtype=GRB.CONTINUOUS, name="xi")
    loss = model.addVars(Omega, lb=0.0, vtype=GRB.CONTINUOUS, name="loss")

    # --- Objective: CVaR (unchanged) ---
    cvar_multiplier = 1.0 / (1.0 - beta)
    model.setObjective(
        eta + cvar_multiplier * gp.quicksum(prob[w] * xi[w] for w in Omega),
        GRB.MINIMIZE,
    )

    # --- Constraints ---

    # SiteBudget / SelectionBudget - first-stage constraints, UNCHANGED
    model.addConstr(gp.quicksum(p[i] for i in PPL) <= P_max, name="SiteBudget")
    model.addConstr(
        gp.quicksum(site_cost[i] * p[i] for i in PPL) <= selection_budget,
        name="SelectionBudget",
    )

    # InventoryReleaseBound / NonPPLReleaseZero - constraint (8), UNCHANGED
    for w in Omega:
        for i in PPL:
            for r in R:
                availability_factor = 1.0
                if inventory_availability is not None:
                    availability_factor = inventory_availability[w, i, r]
                model.addConstr(
                    release[w, i, r]
                    <= inventory_if_open[i, r] * releasable_fraction * availability_factor * p[i],
                    name=f"InventoryReleaseBound_w{w}_i{i}_r{r}",
                )
        for i in non_ppl_nodes:
            for r in R:
                model.addConstr(
                    release[w, i, r] == 0,
                    name=f"NonPPLReleaseZero_w{w}_i{i}_r{r}",
                )

    # --- Flow balance (replaces NodeBalance constraint (9)) ---
    # Aggregate modal balance: inflow across ALL modes + release (PPL only) + z
    # == demand + outflow across ALL modes (one constraint per (w,i,r)).
    # Transfer tau cancels in the aggregate sum (internal node-level conversion)
    # but is made meaningful by the ModalOutboundFeasibility constraints below.
    for w in Omega:
        for i in N:
            for r in R:
                inflow_all = gp.quicksum(
                    x[w, m, j, i, r]
                    for m in modes
                    for (j, _) in modal_incoming[m][i]
                )
                outflow_all = gp.quicksum(
                    x[w, m, i, j, r]
                    for m in modes
                    for (_, j) in modal_outgoing[m][i]
                )
                if i in PPL_set:
                    model.addConstr(
                        inflow_all + release[w, i, r] + z[w, i, r]
                        == demand[w, i, r] + outflow_all,
                        name=f"FlowBalance_PPL_w{w}_i{i}_r{r}",
                    )
                else:
                    model.addConstr(
                        inflow_all + z[w, i, r]
                        == demand[w, i, r] + outflow_all,
                        name=f"FlowBalance_nonPPL_w{w}_i{i}_r{r}",
                    )

    # --- Per-mode outbound feasibility with intermodal transfer ---
    # Replaces the single-mode OutboundFeasibility (constraint (10)).
    # Outbound flow on mode m (plus tau leaving mode m) cannot exceed inbound
    # flow on mode m (plus tau arriving at mode m) plus release at PPL nodes.
    # This makes tau[w,i,m1,m2,r] meaningful: a node can forward via mode m2
    # only by receiving on mode m2 OR by transferring from another mode m1.
    for w in Omega:
        for m in modes:
            for i in N:
                for r in R:
                    outflow_m = gp.quicksum(
                        x[w, m, i, j, r] for (_, j) in modal_outgoing[m][i]
                    )
                    inflow_m = gp.quicksum(
                        x[w, m, j, i, r] for (j, _) in modal_incoming[m][i]
                    )
                    tau_out_m = gp.quicksum(
                        tau[w, i, m, m2, r]
                        for m2 in transfer_out_modes.get(i, {}).get(m, [])
                    )
                    tau_in_m = gp.quicksum(
                        tau[w, i, m1, m, r]
                        for m1 in transfer_in_modes.get(i, {}).get(m, [])
                    )
                    release_term = release[w, i, r] if i in PPL_set else 0
                    model.addConstr(
                        outflow_m + tau_out_m <= inflow_m + tau_in_m + release_term,
                        name=f"ModalOutboundFeasibility_w{w}_m{m}_i{i}_r{r}",
                    )

    # --- Arc capacity per mode and commodity (replaces ArcCapacity constraint (11)) ---
    # Each modal arc (i,j) on mode m is bounded by its per-commodity residual
    # capacity, which degrades with scenario severity via the degradation_matrix.
    for w, m, i, j, r in flow_keys:
        cap = modal_residual[w].get(m, {}).get((i, j), {}).get(r, 0.0)
        model.addConstr(
            x[w, m, i, j, r] <= cap,
            name=f"ArcCapacity_w{w}_m{m}_i{i}_j{j}_r{r}",
        )

    # --- Transfer backing: tau_out backed by arc inflow only, not by tau_in ---
    # Constraint (13) permits tau_in_m to back tau_out_m, which creates a
    # self-referential cycle: sea->air->land->sea can spin at max capacity
    # even at nodes with zero arc inflow, because each tau_in backs the next
    # tau_out in a closed loop. This constraint breaks that cycle by requiring
    # tau_out from mode m to be backed by *actual* arc inflow on m (plus any
    # inventory release at PPL nodes). The existing (13) still governs outbound
    # arc flow, which legitimately uses tau_in_m to route received transfers.
    for w in Omega:
        for m in modes:
            for i in N:
                if i not in transfer_cap:
                    continue
                out_modes_m = transfer_out_modes.get(i, {}).get(m, [])
                if not out_modes_m:
                    continue
                for r in R:
                    tau_out_m = gp.quicksum(tau[w, i, m, m2, r] for m2 in out_modes_m)
                    inflow_m = gp.quicksum(
                        x[w, m, j, i, r] for (j, _) in modal_incoming[m][i]
                    )
                    release_term = release[w, i, r] if i in PPL_set else 0
                    model.addConstr(
                        tau_out_m <= inflow_m + release_term,
                        name=f"TransferBacking_w{w}_m{m}_i{i}_r{r}",
                    )

    # --- Transfer capacity (Task 8, new constraint) ---
    # Total commodity transferred from mode m1 to mode m2 at node i across
    # both commodities cannot exceed the node's intermodal handling capacity.
    for i, pairs in transfer_cap.items():
        for (m1, m2), cap in pairs.items():
            for w in Omega:
                model.addConstr(
                    gp.quicksum(tau[w, i, m1, m2, r] for r in R) <= cap,
                    name=f"TransferCapacity_w{w}_i{i}_m{m1}_{m2}",
                )

    # --- Vehicle-heterogeneity constraints (16-19) ---
    if vehicle_formulation == "aggregate":
        if has_vehicles:
            # (16) Vehicle Conservation with p_j coupling:
            # vehicles departing node j <= vehicles arriving at j + b_{k,j}*p_j
            for w in Omega:
                for m in modes:
                    for k in K_m.get(m, []):
                        b_kj = vehicle_types[k]["b_kj"]
                        for j in N:
                            arrivals = gp.quicksum(
                                n[w, k, m, i_src, j]
                                for (i_src, _) in modal_incoming[m][j]
                            )
                            departures = gp.quicksum(
                                n[w, k, m, j, j_dst]
                                for (_, j_dst) in modal_outgoing[m][j]
                            )
                            b_val = b_kj.get(j, 0)
                            basing_term = b_val * p[j] if (b_val > 0 and j in PPL_set) else 0
                            model.addConstr(
                                arrivals + basing_term >= departures,
                                name=f"VehicleConservation_w{w}_k{k}_j{j}",
                            )

            # (17) Fleet Size — REMOVED. The original constraint capped total
            # vehicle-arc-traversals at F_k, which prevented multi-leg routing:
            # a vehicle flying A->B->C consumed 2 of the F_k budget, limiting
            # 12 C-17s to 12 total legs rather than 12 vehicles each flying
            # multiple legs. Vehicle conservation (16) already prevents using
            # more vehicles than are based+arrived, and the distance budget (19)
            # already caps total fleet-wide travel. Together they bound vehicle
            # usage without artificially restricting multi-leg operations.

            # (18) Vehicle-Capacity-Constrained Flow, per resource (option A):
            # independent per-resource capacity — a vehicle could be credited with
            # a full food load and full water load simultaneously (known simplification)
            for w in Omega:
                for m in modes:
                    for (i, j) in modal_arcs[m]:
                        for r in R:
                            model.addConstr(
                                x[w, m, i, j, r] <= gp.quicksum(
                                    vehicle_types[k]["capacity"][r] * n[w, k, m, i, j]
                                    for k in K_m.get(m, [])
                                ),
                                name=f"VehicleCapFlow_w{w}_m{m}_i{i}_j{j}_r{r}",
                            )

            # (19) Fleet-Wide Distance Budget with p_j-coupled turnaround:
            # Turnaround charged at ALL nodes; exemption at base nodes only when
            # the site is selected (b_{k,j}*p_j > 0). The product outbound*p_j
            # is linearized via auxiliary variable g (McCormick, exact for binary p).
            base_nodes_by_k: Dict[str, List[int]] = {}
            for k_name, vtype in vehicle_types.items():
                b_kj = vtype["b_kj"]
                base_nodes_by_k[k_name] = [
                    j for j in N if b_kj.get(j, 0) > 0 and j in PPL_set
                ]

            g_keys = [
                (w, k_name, j)
                for k_name in vehicle_types
                for w in Omega
                for j in base_nodes_by_k[k_name]
            ]
            g = model.addVars(g_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="g_turn") if g_keys else {}

            for k_name, vtype in vehicle_types.items():
                m_veh = vtype["mode"]
                F_k = vtype["fleet_size"]
                for w in Omega:
                    for j in base_nodes_by_k[k_name]:
                        outbound_j = gp.quicksum(
                            n[w, k_name, m_veh, j, j_dst]
                            for (_, j_dst) in modal_outgoing[m_veh][j]
                        )
                        model.addConstr(
                            g[w, k_name, j] <= F_k * p[j],
                            name=f"TurnExemptUB1_w{w}_k{k_name}_j{j}",
                        )
                        model.addConstr(
                            g[w, k_name, j] <= outbound_j,
                            name=f"TurnExemptUB2_w{w}_k{k_name}_j{j}",
                        )
                        model.addConstr(
                            g[w, k_name, j] >= outbound_j - F_k * (1 - p[j]),
                            name=f"TurnExemptLB_w{w}_k{k_name}_j{j}",
                        )

            for w in Omega:
                for k_name, vtype in vehicle_types.items():
                    m_veh = vtype["mode"]
                    F_k = vtype["fleet_size"]
                    D_k = vtype["D_k"]
                    pi_k = vtype["pi_k"]

                    dist_term = gp.quicksum(
                        modal_arc_distance.get(m_veh, {}).get((i, j), 0.0)
                        * n[w, k_name, m_veh, i, j]
                        for (i, j) in modal_arcs[m_veh]
                    )

                    total_outbound = gp.quicksum(
                        n[w, k_name, m_veh, j, j_dst]
                        for j in N
                        for (_, j_dst) in modal_outgoing[m_veh][j]
                    )

                    exempted = gp.quicksum(
                        g[w, k_name, j] for j in base_nodes_by_k[k_name]
                    ) if base_nodes_by_k[k_name] else 0

                    model.addConstr(
                        dist_term + pi_k * (total_outbound - exempted) <= F_k * D_k,
                        name=f"DistanceBudget_w{w}_k{k_name}",
                    )
    elif vehicle_formulation == "individual":
        # Individual-vehicle indexing (air mode only: C-17, C-130J). Sea and
        # land -- and any other air-mode type not in this pair -- keep the
        # exact aggregate (16)/(18)/(19) logic from the branch above,
        # duplicated here unchanged (not shared) because this branch fully
        # replaces, rather than supplements, the has_vehicles constraint set.
        if has_vehicles:
            air_indiv_types = [k for k in K_m.get("air", []) if k in ("C-17", "C-130J")]
            non_indiv_air_types = [k for k in K_m.get("air", []) if k not in air_indiv_types]

            # --- (16) Vehicle Conservation -- sea/land/non-individual-air,
            # UNCHANGED aggregate logic, restricted to those modes/types ---
            for w in Omega:
                for m in modes:
                    k_list = non_indiv_air_types if m == "air" else K_m.get(m, [])
                    for k in k_list:
                        b_kj = vehicle_types[k]["b_kj"]
                        for j in N:
                            arrivals = gp.quicksum(
                                n[w, k, m, i_src, j]
                                for (i_src, _) in modal_incoming[m][j]
                            )
                            departures = gp.quicksum(
                                n[w, k, m, j, j_dst]
                                for (_, j_dst) in modal_outgoing[m][j]
                            )
                            b_val = b_kj.get(j, 0)
                            basing_term = b_val * p[j] if (b_val > 0 and j in PPL_set) else 0
                            model.addConstr(
                                arrivals + basing_term >= departures,
                                name=f"VehicleConservation_w{w}_k{k}_j{j}",
                            )

            # --- VehicleAggregation (new, no aggregate analog): pins the
            # aggregate n^omega_{k,air,ij} for each individually-indexed air
            # type to the sum of its per-vehicle-instance binaries, so (18)
            # below needs no modification. ---
            for w in Omega:
                for k in air_indiv_types:
                    F_k = vehicle_types[k]["fleet_size"]
                    for (i, j) in modal_arcs["air"]:
                        model.addConstr(
                            n[w, k, "air", i, j] == gp.quicksum(
                                n_ind[w, k, l, "air", i, j] for l in range(1, F_k + 1)
                            ),
                            name=f"VehicleAggregation_w{w}_k{k}_i{i}_j{j}",
                        )

            # --- (16-individual) Vehicle Conservation per vehicle instance l:
            # l may only depart node j if it is based there (home_j == j,
            # gated by p[home_j]) or arrived there via n_ind on a prior leg
            # in the same scenario. Scenario-local: l has no identity across
            # scenarios w, so this is per-w like the aggregate version. ---
            home_by_k: Dict[str, Dict[int, int]] = {
                k: _assign_individual_homes(vehicle_types[k]["b_kj"], vehicle_types[k]["fleet_size"])
                for k in air_indiv_types
            }

            for w in Omega:
                for k in air_indiv_types:
                    homes = home_by_k[k]
                    for l in range(1, vehicle_types[k]["fleet_size"] + 1):
                        home_j = homes[l]
                        for j in N:
                            arrivals_l = gp.quicksum(
                                n_ind[w, k, l, "air", i_src, j]
                                for (i_src, _) in modal_incoming["air"][j]
                            )
                            departures_l = gp.quicksum(
                                n_ind[w, k, l, "air", j, j_dst]
                                for (_, j_dst) in modal_outgoing["air"][j]
                            )
                            home_term = p[j] if (j == home_j and j in PPL_set) else 0
                            model.addConstr(
                                arrivals_l + home_term >= departures_l,
                                name=f"VehicleConservationIndiv_w{w}_k{k}_l{l}_j{j}",
                            )

            # --- (18) Vehicle-Capacity-Constrained Flow -- UNCHANGED, byte-
            # identical to the aggregate branch's code. n[w,k,m,i,j] is valid
            # for every k here regardless of formulation: sea/land/non-
            # individual-air are optimized directly, air-individual types are
            # pinned by VehicleAggregation above. No modification required. ---
            for w in Omega:
                for m in modes:
                    for (i, j) in modal_arcs[m]:
                        for r in R:
                            model.addConstr(
                                x[w, m, i, j, r] <= gp.quicksum(
                                    vehicle_types[k]["capacity"][r] * n[w, k, m, i, j]
                                    for k in K_m.get(m, [])
                                ),
                                name=f"VehicleCapFlow_w{w}_m{m}_i{i}_j{j}_r{r}",
                            )

            # --- (19) Fleet-Wide Distance Budget -- sea/land/non-individual-
            # air, UNCHANGED aggregate logic (F_k*D_k, McCormick g), just
            # skipping air_indiv_types (handled by the per-l version below). ---
            base_nodes_by_k: Dict[str, List[int]] = {}
            for k_name, vtype in vehicle_types.items():
                if k_name in air_indiv_types:
                    continue
                b_kj = vtype["b_kj"]
                base_nodes_by_k[k_name] = [
                    j for j in N if b_kj.get(j, 0) > 0 and j in PPL_set
                ]

            g_keys = [
                (w, k_name, j)
                for k_name in vehicle_types
                if k_name not in air_indiv_types
                for w in Omega
                for j in base_nodes_by_k[k_name]
            ]
            g = model.addVars(g_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="g_turn") if g_keys else {}

            for k_name, vtype in vehicle_types.items():
                if k_name in air_indiv_types:
                    continue
                m_veh = vtype["mode"]
                F_k = vtype["fleet_size"]
                for w in Omega:
                    for j in base_nodes_by_k[k_name]:
                        outbound_j = gp.quicksum(
                            n[w, k_name, m_veh, j, j_dst]
                            for (_, j_dst) in modal_outgoing[m_veh][j]
                        )
                        model.addConstr(
                            g[w, k_name, j] <= F_k * p[j],
                            name=f"TurnExemptUB1_w{w}_k{k_name}_j{j}",
                        )
                        model.addConstr(
                            g[w, k_name, j] <= outbound_j,
                            name=f"TurnExemptUB2_w{w}_k{k_name}_j{j}",
                        )
                        model.addConstr(
                            g[w, k_name, j] >= outbound_j - F_k * (1 - p[j]),
                            name=f"TurnExemptLB_w{w}_k{k_name}_j{j}",
                        )

            for w in Omega:
                for k_name, vtype in vehicle_types.items():
                    if k_name in air_indiv_types:
                        continue
                    m_veh = vtype["mode"]
                    F_k = vtype["fleet_size"]
                    D_k = vtype["D_k"]
                    pi_k = vtype["pi_k"]

                    dist_term = gp.quicksum(
                        modal_arc_distance.get(m_veh, {}).get((i, j), 0.0)
                        * n[w, k_name, m_veh, i, j]
                        for (i, j) in modal_arcs[m_veh]
                    )

                    total_outbound = gp.quicksum(
                        n[w, k_name, m_veh, j, j_dst]
                        for j in N
                        for (_, j_dst) in modal_outgoing[m_veh][j]
                    )

                    exempted = gp.quicksum(
                        g[w, k_name, j] for j in base_nodes_by_k[k_name]
                    ) if base_nodes_by_k[k_name] else 0

                    model.addConstr(
                        dist_term + pi_k * (total_outbound - exempted) <= F_k * D_k,
                        name=f"DistanceBudget_w{w}_k{k_name}",
                    )

            # --- (19-individual) Fleet-Wide Distance Budget per vehicle
            # instance l: own D_k budget (not F_k*D_k, since F_k=1 for a
            # single instance). Turnaround exemption re-derived at the
            # individual level: g_ind McCormick-linearizes outbound_home_j *
            # p[home_j], with the constant upper bound replaced by 1 (a
            # single vehicle instance) instead of F_k. Since home_j is a
            # FIXED, precomputed node per l (not a decision variable), only
            # one candidate base node exists per l -- the McCormick trick
            # transfers directly (same UB1/UB2/LB structure, F_k -> 1,
            # base_nodes_by_k -> {home_j}); no new derivation was needed. ---
            g_ind_keys = [
                (w, k, l)
                for k in air_indiv_types
                for w in Omega
                for l in range(1, vehicle_types[k]["fleet_size"] + 1)
                if home_by_k[k][l] in PPL_set
            ]
            g_ind = model.addVars(g_ind_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="g_turn_indiv") if g_ind_keys else {}

            for k in air_indiv_types:
                vtype = vehicle_types[k]
                D_k = vtype["D_k"]
                pi_k = vtype["pi_k"]
                homes = home_by_k[k]
                for w in Omega:
                    for l in range(1, vtype["fleet_size"] + 1):
                        home_j = homes[l]
                        if home_j in PPL_set:
                            outbound_home = gp.quicksum(
                                n_ind[w, k, l, "air", home_j, j_dst]
                                for (_, j_dst) in modal_outgoing["air"][home_j]
                            )
                            model.addConstr(
                                g_ind[w, k, l] <= p[home_j],
                                name=f"TurnExemptIndivUB1_w{w}_k{k}_l{l}",
                            )
                            model.addConstr(
                                g_ind[w, k, l] <= outbound_home,
                                name=f"TurnExemptIndivUB2_w{w}_k{k}_l{l}",
                            )
                            model.addConstr(
                                g_ind[w, k, l] >= outbound_home - (1 - p[home_j]),
                                name=f"TurnExemptIndivLB_w{w}_k{k}_l{l}",
                            )
                            exempted_l = g_ind[w, k, l]
                        else:
                            exempted_l = 0

                        dist_term_l = gp.quicksum(
                            modal_arc_distance.get("air", {}).get((i, j), 0.0)
                            * n_ind[w, k, l, "air", i, j]
                            for (i, j) in modal_arcs["air"]
                        )
                        total_outbound_l = gp.quicksum(
                            n_ind[w, k, l, "air", j, j_dst]
                            for j in N
                            for (_, j_dst) in modal_outgoing["air"][j]
                        )
                        model.addConstr(
                            dist_term_l + pi_k * (total_outbound_l - exempted_l) <= D_k,
                            name=f"DistanceBudgetIndiv_w{w}_k{k}_l{l}",
                        )

            # --- VehicleSymmetryBreak (new, no aggregate analog): breaks
            # permutation symmetry among individually-indexed vehicle
            # instances that share both type k and home node -- they are
            # fully interchangeable (same D_k, pi_k, capacity, home node),
            # differing only by the arbitrary l label _assign_individual_homes
            # gave them. Scoped STRICTLY within (k, home_j) groups: instances
            # at different home nodes are already structurally distinguishable
            # and must never be constrained against each other, or the cut
            # could exclude every optimum. Orders instances by a cheap scalar
            # summary (total outbound air-arc count in scenario w) rather than
            # a full lexicographic ordering on the arc-incidence vector:
            #   total_outbound[w,k,l] <= total_outbound[w,k,l+1]
            # for consecutive l within a home group. Always valid -- for any
            # feasible/optimal solution, relabeling within an interchangeable
            # group to satisfy non-decreasing order changes nothing else, so
            # at least one optimum survives the cut. Weaker than full
            # lexicographic ordering (ties leave residual symmetry) but far
            # cheaper: one constraint per adjacent pair per scenario.
            for k in (air_indiv_types if not _debug_skip_symmetry_break else []):
                homes = home_by_k[k]
                group_by_home: Dict[int, List[int]] = {}
                for l, home_j in homes.items():
                    group_by_home.setdefault(home_j, []).append(l)
                for home_j, l_list in group_by_home.items():
                    if len(l_list) < 2:
                        continue
                    l_list = sorted(l_list)
                    for w in Omega:
                        outbound_by_l = {
                            l: gp.quicksum(
                                n_ind[w, k, l, "air", j, j_dst]
                                for j in N
                                for (_, j_dst) in modal_outgoing["air"][j]
                            )
                            for l in l_list
                        }
                        for idx in range(len(l_list) - 1):
                            l_a, l_b = l_list[idx], l_list[idx + 1]
                            model.addConstr(
                                outbound_by_l[l_a] <= outbound_by_l[l_b],
                                name=f"VehicleSymmetryBreak_w{w}_k{k}_home{home_j}_l{l_a}_l{l_b}",
                            )

            # --- DepartureSingleNode (new, no aggregate analog): each
            # vehicle instance l may have positive outbound arcs from at most
            # one node per scenario, preventing the MIP from asserting the
            # same physical vehicle simultaneously departing two different
            # origin nodes. dep_node[w,k,l,j] indicates l has >=1 outbound
            # arc from j; capped to at most one j per (w,k,l). Carries no
            # p_j/home-node information -- that linkage lives entirely in
            # VehicleConservationIndiv above, via home_term. Skippable via
            # _debug_skip_departure_single_node for isolating its effect
            # (e.g. testing for double-booking with it removed). ---
            if not _debug_skip_departure_single_node:
                dep_node_keys = [
                    (w, k, l, j)
                    for k in air_indiv_types
                    for w in Omega
                    for l in range(1, vehicle_types[k]["fleet_size"] + 1)
                    for j in N
                    if modal_outgoing["air"][j]
                ]
                dep_node = model.addVars(dep_node_keys, vtype=GRB.BINARY, name="dep_node") if dep_node_keys else {}

                for k in air_indiv_types:
                    for w in Omega:
                        for l in range(1, vehicle_types[k]["fleet_size"] + 1):
                            for j in N:
                                if not modal_outgoing["air"][j]:
                                    continue
                                for (_, j_dst) in modal_outgoing["air"][j]:
                                    model.addConstr(
                                        n_ind[w, k, l, "air", j, j_dst] <= dep_node[w, k, l, j],
                                        name=f"DepartureNodeLink_w{w}_k{k}_l{l}_j{j}_jd{j_dst}",
                                    )
                            model.addConstr(
                                gp.quicksum(
                                    dep_node[w, k, l, j] for j in N if modal_outgoing["air"][j]
                                ) <= 1,
                                name=f"DepartureSingleNode_w{w}_k{k}_l{l}",
                            )
    else:
        raise ValueError(f"unknown vehicle_formulation: {vehicle_formulation!r}")

    # --- Lazy subtour-elimination callback (individual formulation only) ---
    # n_ind is only non-empty when vehicle_formulation="individual" AND
    # has_vehicles AND air_indiv_types is non-empty -- which also implies
    # home_by_k/air_indiv_types were populated by the branch above, so it's
    # safe to reference them here.
    subtour_stats = {"invocations": 0, "cuts_added": 0}
    subtour_callback = None
    if vehicle_formulation == "individual" and n_ind:
        model.Params.LazyConstraints = 1
        # MIPFocus=1: prioritize finding feasible incumbents over proving
        # optimality. Root-scale diagnostics (F_k=12/16 on the real 50-node
        # network) never leave node 0 in 45 minutes under the default focus
        # (SolCount=0 throughout) -- scoped to the individual formulation
        # only, since the aggregate branch already converges well under the
        # default and this must not perturb its locked regression baseline.
        model.Params.MIPFocus = _debug_mip_focus
        subtour_callback = _build_subtour_callback(
            n_ind, home_by_k, air_indiv_types, Omega, modal_arcs["air"], subtour_stats,
        )

    # --- Scenario loss definition (updated for modal arc cost + transfer cost) ---
    # Two-tier vehicle epsilon (both are flat per-vehicle-arc costs):
    #   EPSILON_DEPLOY (0.01): breaks exact ties so the solver prefers fewer
    #     vehicle movements when delivery outcomes are equal.
    #   EPSILON_EMPTY  (0.03): additional penalty making any vehicle-arc
    #     strictly more expensive than not moving. At optimality (0% gap) this
    #     eliminates empty round trips; within a 1% gap the display layer
    #     suppresses them from itinerary output. Combined epsilon = 0.04 per
    #     vehicle-arc — negligible relative to delta=500 per unit of unmet demand.
    EPSILON_DEPLOY = 0.01
    EPSILON_EMPTY  = 0.03

    for w in Omega:
        unmet_penalty_expr = gp.quicksum(
            penalty[j, r] * z[w, j, r] for j in N for r in R
        )
        # arc transport cost: c[m,i,j] = distance_km * cost_per_unit_km
        arc_transport_expr = gp.quicksum(
            modal_arc_cost[m].get((i, j), 0.0) * x[w, m, i, j, r]
            for (m, i, j, r) in flow_keys_by_w[w]
        )
        # intermodal transfer cost: multiplier * tau
        transfer_cost_expr = gp.quicksum(
            transfer_cost_mult.get((m1, m2), 0.0) * tau[w, i, m1, m2, r]
            for (i, m1, m2, r) in transfer_keys_by_w[w]
        )
        # vehicle deployment tiebreaker (both epsilons combined into one term)
        vehicle_deploy_expr = gp.quicksum(
            (EPSILON_DEPLOY + EPSILON_EMPTY) * n[w, k, m, i, j]
            for m in modes
            for k in K_m.get(m, [])
            for (i, j) in modal_arcs[m]
        ) if has_vehicles else 0

        model.addConstr(
            loss[w] == unmet_penalty_expr + arc_transport_expr
            + transfer_cost_expr + vehicle_deploy_expr,
            name=f"LossDefinition_w{w}",
        )

    # CVaRExcess - unchanged from pre-Step-3
    for w in Omega:
        model.addConstr(xi[w] >= loss[w] - eta, name=f"CVaRExcess_w{w}")

    if build_only:
        model.update()
        return {
            "model": model,
            "variables": {
                "p": p, "x": x, "tau": tau, "z": z,
                "release": release, "eta": eta, "xi": xi, "loss": loss,
                "n": n, "g": g,
                "n_ind": n_ind, "dep_node": dep_node, "g_ind": g_ind,
            },
        }

    # --- Optimize ---
    if subtour_callback is not None:
        model.optimize(subtour_callback)
    else:
        model.optimize()

    # --- Prepare results ---
    results: Dict[str, Any] = {
        "status_code": model.Status,
        "status": _status_to_string(model.Status),
        "objective_value": None,
        "eta": None,
        "selected_sites": [],
        "scenario_losses": {},
        "scenario_transport_cost": {},
        "scenario_unmet_penalty": {},
        "unmet_demand": {},
        "flows": {},               # keyed by (w, m, i, j, r)
        "tau": {},                 # keyed by (w, i, m1, m2, r)
        "release": {},
        "node_flow_summary": [],
        "flow_by_mode": {"sea": 0.0, "air": 0.0, "land": 0.0},  # total flow per mode across all scenarios
        "total_transfer_vol": 0.0,   # total intermodal transfer volume across all scenarios
        "safety_stock_fraction": safety_stock_fraction,
        "model": model,
        "vehicle_flows": {},       # keyed by (w, k, m, i, j) -> integer count
        "vehicle_flows_individual": {},  # keyed by (w, k, l, m, i, j) -> 0/1 (individual formulation, air only)
        "subtour_callback_stats": subtour_stats,  # {"invocations": int, "cuts_added": int}
        "variables": {
            "p": p, "x": x, "tau": tau, "z": z,
            "release": release, "eta": eta, "xi": xi, "loss": loss,
            "n": n, "n_ind": n_ind,
        },
    }

    if model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} and model.SolCount > 0:
        results["objective_value"] = model.ObjVal
        results["eta"] = eta.X
        results["selected_sites"] = [i for i in PPL if p[i].X > 0.5]

        # Batched value extraction (R8 audit finding): one getAttr call per
        # variable family instead of ~1M individual .X attribute queries.
        x_vals = model.getAttr("X", x)
        tau_vals = model.getAttr("X", tau)
        z_vals = model.getAttr("X", z)
        release_vals = model.getAttr("X", release)
        loss_vals = model.getAttr("X", loss)

        for w in Omega:
            results["scenario_losses"][w] = loss_vals[w]

        if detailed_extraction:
            for w in Omega:
                arc_transport_val = sum(
                    modal_arc_cost[m].get((i, j), 0.0) * x_vals[w, m, i, j, r]
                    for (m, i, j, r) in flow_keys_by_w[w]
                )
                transfer_cost_val = sum(
                    transfer_cost_mult.get((m1, m2), 0.0) * tau_vals[w, i, m1, m2, r]
                    for (i, m1, m2, r) in transfer_keys_by_w[w]
                )
                unmet_penalty_val = sum(
                    penalty[j, r] * z_vals[w, j, r] for j in N for r in R
                )
                results["scenario_transport_cost"][w] = arc_transport_val + transfer_cost_val
                results["scenario_unmet_penalty"][w] = unmet_penalty_val

        for (w, i, r), val in release_vals.items():
            if val > 1e-6:
                results["release"][(w, i, r)] = val

        # Node flow summary - aggregate inflow/outflow across all modes
        # (diagnostic only; gated because no sweep consumer reads it)
        if detailed_extraction:
            for w in Omega:
                for i in N:
                    for r in R:
                        inflow_val = sum(
                            x_vals[w, m, j, i, r]
                            for m in modes
                            for (j, _) in modal_incoming[m][i]
                        )
                        outflow_val = sum(
                            x_vals[w, m, i, j, r]
                            for m in modes
                            for (_, j) in modal_outgoing[m][i]
                        )
                        release_val = release_vals[w, i, r]
                        demand_val = demand[w, i, r]
                        unmet_val = z_vals[w, i, r]
                        retained_val = inflow_val + release_val - outflow_val
                        satisfied_val = demand_val - unmet_val

                        if (inflow_val > 1e-6 or outflow_val > 1e-6
                                or release_val > 1e-6 or unmet_val > 1e-6):
                            results["node_flow_summary"].append({
                                "scenario": w,
                                "node": i,
                                "commodity": r,
                                "inflow": inflow_val,
                                "release": release_val,
                                "outflow": outflow_val,
                                "demand": demand_val,
                                "unmet": unmet_val,
                                "retained": retained_val,
                                "satisfied": satisfied_val,
                            })

        for (w, j, r), val in z_vals.items():
            if val > 1e-6:
                results["unmet_demand"][(w, j, r)] = val

        for key, val in x_vals.items():
            if val > 1e-6:
                results["flows"][key] = val

        for key, val in tau_vals.items():
            if val > 1e-6:
                results["tau"][key] = val

        # Vehicle flow extraction
        if has_vehicles:
            n_vals = model.getAttr("X", n)
            for key, val in n_vals.items():
                if val > 0.5:
                    results["vehicle_flows"][key] = int(round(val))

        if n_ind:
            n_ind_vals = model.getAttr("X", n_ind)
            for key, val in n_ind_vals.items():
                if val > 0.5:
                    results["vehicle_flows_individual"][key] = int(round(val))

        # Aggregate flow totals by mode and total transfer volume
        flow_by_mode: Dict[str, float] = {"sea": 0.0, "air": 0.0, "land": 0.0}
        for (w, m, i, j, r), val in results["flows"].items():
            if m in flow_by_mode:
                flow_by_mode[m] += val
        results["flow_by_mode"] = flow_by_mode
        results["total_transfer_vol"] = sum(results["tau"].values())

    return results



def print_solution_summary(results: Dict[str, Any], max_flows: int = 20) -> None:
    """
    Print a compact summary of the stochastic model solution.

    results: results dict returned by solve_stochastic_cvar
    max_flows: max number of rows to print in each per-item listing

    returns nothing - prints directly to stdout
    """
    print("\n=== STOCHASTIC MODEL SUMMARY ===")
    print(f"Status: {results['status']}")

    if results["objective_value"] is None:
        print("No feasible solution available.")
        return

    print(f"Objective value: {results['objective_value']:.4f}")
    print(f"Eta: {results['eta']:.4f}")
    print(f"Selected sites: {results['selected_sites']}")

    print("\nScenario losses:")
    for w, val in sorted(results["scenario_losses"].items()):
        print(f"  Scenario {w}: {val:.4f}")

    if results.get("scenario_transport_cost"):
        print("\nScenario transport costs:")
        for w, val in sorted(results["scenario_transport_cost"].items()):
            print(f"  Scenario {w}: {val:.4f}")

    if results.get("scenario_unmet_penalty"):
        print("\nScenario unmet-demand penalties:")
        for w, val in sorted(results["scenario_unmet_penalty"].items()):
            print(f"  Scenario {w}: {val:.4f}")

    # --- Aggregate release vs demand ---
    total_release = sum(results.get("release", {}).values())
    total_unmet = sum(results.get("unmet_demand", {}).values())
    total_demand_implied = total_release + total_unmet

    print("\nAggregate supply-demand summary:")
    print(f"  Total released from PPLs: {total_release:.4f}")
    print(f"  Total unmet demand: {total_unmet:.4f}")
    print(f"  Total demand (implied): {total_demand_implied:.4f}")

    # --- Release from non-PPL nodes (should be zero) ---
    selected = set(results.get("selected_sites", []))
    non_ppl_release = sum(
        val for (w, i, r), val in results.get("release", {}).items() if i not in selected
    )

    print("\nNon-PPL Flow Check:")
    print(f"  Total release from non-PPL nodes: {non_ppl_release:.4f}")

    print("\nNode flow balance samples:")
    for row in results.get("node_flow_summary", [])[:max_flows]:
        print(
            "  "
            f"w={row['scenario']}, node={row['node']}, {row['commodity']}: "
            f"in={row['inflow']:.4f}, rel={row['release']:.4f}, out={row['outflow']:.4f}, "
            f"dem={row['demand']:.4f}, unmet={row['unmet']:.4f}, "
            f"retained={row['retained']:.4f}, satisfied={row['satisfied']:.4f}"
        )

    if results["unmet_demand"]:
        print("\nPositive unmet demand entries:")
        for key, val in list(results["unmet_demand"].items())[:max_flows]:
            print(f"  {key}: {val:.4f}")
    else:
        print("\nNo unmet demand in the reported solution.")

    if results.get("flows"):
        print("\nPositive flows (first few):")
        for (w, m, i, j, r), val in list(results["flows"].items())[:max_flows]:
            print(f"  w={w}, mode={m}, {i}->{j}, {r}: {val:.4f}")



def _status_to_string(status_code: int) -> str:
    """
    Convert Gurobi status code to readable text.

    status_code: model.Status integer from gurobipy

    returns a human-readable status string (e.g. "OPTIMAL"), or
    "STATUS_<code>" if the code isn't one of the common ones listed
    """
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return status_map.get(status_code, f"STATUS_{status_code}")


def _assign_individual_homes(b_kj: Dict[int, int], fleet_size: int) -> Dict[int, int]:
    """
    Deterministically assign each individual vehicle instance l in
    {1,...,fleet_size} a fixed home/basing node, by splitting the aggregate
    per-node basing counts b_kj (the same counts the aggregate formulation
    uses) into per-instance slots. Nodes are visited in sorted order so the
    assignment is reproducible run to run.

    Raises ValueError if sum(b_kj.values()) != fleet_size -- individual
    vehicle indexing requires every instance to have exactly one home node
    (a vehicle with no home could never legally depart under constraint 16).
    """
    total = sum(b_kj.values())
    if total != fleet_size:
        raise ValueError(
            f"basing allocation {b_kj} sums to {total}, expected fleet_size="
            f"{fleet_size} -- individual vehicle indexing requires every "
            "instance to have a home node"
        )
    homes: Dict[int, int] = {}
    l = 1
    for node in sorted(b_kj.keys()):
        for _ in range(b_kj[node]):
            homes[l] = node
            l += 1
    return homes
