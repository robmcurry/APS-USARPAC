"""
model_vif.py

PRS-VIF (docs/PRSVIF_Gospel.md, "Risk-Aware Vehicle-Indexed Model") --
individually indexed, jointly sited-and-based vehicle formulation. Kept in
its own module, separate from model.py's aggregate/individual formulations,
because PRS-VIF's variable/constraint set has no analog for those paths'
mode-indexed x/tau or their "release" decision variable (see
solve_vif's docstring, and PHASE1_NOTES.md's "Deviation from the
dispatch-point framing" for why this isn't interleaved into model.py the
way "individual" shares code with "aggregate").

Called from model.py's solve_stochastic_cvar() via an early return when
vehicle_formulation == "vif", before any aggregate/individual variable is
built. Deliberately has no import back into model.py (that would be
circular, since model.py imports this module) -- everything this function
needs is passed in explicitly by the caller, and the one small piece of
shared logic (Gurobi status-code -> string) is duplicated locally rather
than imported.
"""

from typing import Any, Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB
#from typing import Any, Dict, List, Tuple
from collections import defaultdict, deque
import time
import traceback

def _status_to_string(status_code: int) -> str:
    """
    Convert Gurobi status code to readable text. Duplicated from
    model.py's helper. Need to update model.py if anything changes here
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


def _degradation_factor(gamma: float, severity_term: float) -> float:
    """
    Shared max{0, 1 - gamma * severity_term / 5} factor used by both
    eq:residual (severity_term = max(sigma_i, sigma_j)) and
    eq:thetadegradation (severity_term = sigma_i).
    """
    return max(0.0, 1.0 - gamma * severity_term / 5.0)


#def _build_vif_subtour_callback(
#    n: Dict,
#    b: Dict,
#    a: Dict[Tuple[int, int], float],
#    L: List[int],
#    J_l: Dict[int, set],
#    Omega: List[int],
#    modal_arcs: Dict[str, List[Tuple[int, int]]],
#    mode_of: Dict[int, str],
#    stats: Dict[str, int],
#):
#    """
#    Build a Gurobi lazy-constraint callback enforcing C11 (eq:vif:sec,
#    subtour elimination) for PRS-VIF's n[w,l,i,j]/b[l,i] variables.
#    Analogous in spirit to model.py's _build_subtour_callback for the old
#    "individual" formulation's n_ind, but separates the GOSPEL's exact
#    per-arc inequality family (n_ij <= external inflow to S + base credit
#    in S) rather than that callback's simpler aggregate |S|-1 cut -- see
#    PHASE4_NOTES.md, "Why the literal gospel SEC form, not the simpler
#    |S|-1 cut" for the reasoning.
#
#    On every integer-feasible incumbent (MIPSOL), for each (scenario w,
#    vehicle l): computes which nodes are reachable from l's currently
#    active, available base(s) this incumbent (nodes i' in J_l[l] with
#    a[w,i']=1 and b[l,i']=1) by following l's selected (value > 0.5) n
#    arcs, then groups any leftover selected arcs (touching nodes never
#    reached from an active base) into weakly-connected components -- each
#    is a disconnected subtour: exactly the failure mode documented in
#    test_vif_phase1.py's module docstring (a single vehicle useing a
#    disconnected 2-cycle with zero net flow at every node on it, found
#    while building that phase's Test 3, before this callback existed --
#    that test used one-way arcs specifically to sidestep the issue since
#    this callback didn't exist yet; with it active, two-way arcs in a
#    disconnected component are safe, as Phase 4's own subtour test
#    confirms).
#
#    For each violated component S, adds -- for every currently-selected
#    arc (i,j) with i,j in S -- the exact gospel inequality:
#        n[w,l,i,j] <= sum_{(i',j') in A_l: i' not in S, j' in S} n[w,l,i',j']
#                     + sum_{i' in S, i' in J_l[l]} a[w,i'] * b[l,i']
#    (the b-sum is restricted to i' in J_l[l] since b[l,i'] only exists as
#    a variable for l's actual basing-eligible nodes; it's 0 if S contains
#    none of them). Adding cuts only for the arcs actually selected in the
#    current violation, rather than eagerly enumerating the full family for
#    S, is standard lazy-separation practice -- each cut added is still a
#    valid member of the gospel's eq:vif:sec family for that specific
#    (S, l, w, arc) instance.
#
#    stats: mutable dict updated in place with "invocations" (MIPSOL calls
#    seen) and "cuts_added" (total lazy constraints emitted), mirroring
#    model.py's callback for a consistent post-solve report.
#    """
#
#    def callback(model, where):
#        if where != GRB.Callback.MIPSOL:
#            return
#        stats["invocations"] += 1
#
#        for w in Omega:
#            for l in L:
#                m = mode_of[l]
#                arcs = modal_arcs[m]
#                arc_vals = model.cbGetSolution([n[w, l, i, j] for (i, j) in arcs])
#                val_map = dict(zip(arcs, arc_vals))
#                selected = [(i, j) for (i, j) in arcs if val_map[(i, j)] > 0.5]
#                if not selected:
#                    continue
#
#                base_nodes = list(J_l[l])
#                base_vals = (
#                    model.cbGetSolution([b[l, i] for i in base_nodes])
#                    if base_nodes else []
#                )
#                active_bases = {
#                    i for i, val in zip(base_nodes, base_vals)
#                    if val > 0.5 and a[w, i] > 0.5
#                }
#
#                reached = set(active_bases)
#                changed = True
#                while changed:
#                    changed = False
#                    for (i, j) in selected:
#                        if i in reached and j not in reached:
#                            reached.add(j)
#                            changed = True
#
#                touched = {node for arc in selected for node in arc}
#                phantom_nodes = touched - reached
#                if not phantom_nodes:
#                    continue
#
#                remaining = set(phantom_nodes)
#                while remaining:
#                    seed = next(iter(remaining))
#                    component = {seed}
#                    stack = [seed]
#                    while stack:
#                        node = stack.pop()
#                        for (i, j) in selected:
#                            if i == node and j in remaining and j not in component:
#                                component.add(j)
#                                stack.append(j)
#                            elif j == node and i in remaining and i not in component:
#                                component.add(i)
#                                stack.append(i)
#                    remaining -= component
#                    if len(component) < 2:
#                        continue
#
#                    external_inflow = gp.quicksum(
#                        n[w, l, ip, jp]
#                        for (ip, jp) in arcs
#                        if ip not in component and jp in component
#                    )
#                    base_credit = gp.quicksum(
#                        a[w, i2] * b[l, i2]
#                        for i2 in component if i2 in J_l[l]
#                    )
#                    rhs = external_inflow + base_credit
#
#                    for (i, j) in selected:
#                        if i in component and j in component:
#                            model.cbLazy(n[w, l, i, j] <= rhs)
#                            stats["cuts_added"] += 1
#
#    return callback

def _build_vif_subtour_callback(
    n: Dict,
    b: Dict,
    a: Dict[Tuple[int, int], float],
    L: List[int],
    J_l: Dict[int, set],
    Omega: List[int],
    modal_arcs: Dict[str, List[Tuple[int, int]]],
    mode_of: Dict[int, str],
    stats: Dict[str, Any],
):
    """
    Separate disconnected vehicle components at integer MIP solutions.

    Performance features:
      * Retrieves all n values in one callback call.
      * Retrieves all b values in one callback call.
      * Examines only scenario/vehicle pairs with selected arcs.
      * Uses adjacency lists for graph searches.
      * Adds one aggregate lazy constraint per disconnected component.
      * Limits the number of cuts generated during one callback invocation.
    """

    # Construct these lists once rather than during every callback.
    n_keys = list(n.keys())
    n_vars = [n[key] for key in n_keys]

    b_keys = list(b.keys())
    b_vars = [b[key] for key in b_keys]

    # Precompute incoming arcs for efficient boundary-cut construction.
    incoming_by_mode = {}

    for mode, arcs in modal_arcs.items():
        incoming = defaultdict(list)

        for i, j in arcs:
            incoming[j].append((i, j))

        incoming_by_mode[mode] = dict(incoming)

    # Avoid adding an excessive number of rows from one candidate.
    max_cuts_per_callback = 20

    def callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return

        start = time.perf_counter()
        cuts_this_call = 0
        stats["invocations"] += 1

        try:
            # One Gurobi query for all vehicle-arc variables.
            n_values = model.cbGetSolution(n_vars)

            selected_by_pair = defaultdict(list)

            for (w, l, i, j), value in zip(n_keys, n_values):
                if value > 0.5:
                    selected_by_pair[w, l].append((i, j))

            if not selected_by_pair:
                return

            # b is not scenario-indexed, so retrieve it only once.
            b_values = model.cbGetSolution(b_vars)

            selected_bases_by_vehicle = defaultdict(set)

            for (l, node), value in zip(b_keys, b_values):
                if value > 0.5:
                    selected_bases_by_vehicle[l].add(node)

            # Only inspect scenario/vehicle pairs that use at least one arc.
            for (w, l), selected in selected_by_pair.items():
                mode = mode_of[l]

                active_bases = {
                    node
                    for node in selected_bases_by_vehicle.get(l, ())
                    if a[w, node] > 0.5
                }

                directed_out = defaultdict(list)
                weak_neighbors = defaultdict(set)
                touched = set()

                for i, j in selected:
                    directed_out[i].append(j)
                    weak_neighbors[i].add(j)
                    weak_neighbors[j].add(i)
                    touched.add(i)
                    touched.add(j)

                # Find nodes reachable from an available assigned base.
                reached = set(active_bases)
                queue = deque(active_bases)

                while queue:
                    node = queue.popleft()

                    for next_node in directed_out.get(node, ()):
                        if next_node not in reached:
                            reached.add(next_node)
                            queue.append(next_node)

                phantom_nodes = touched - reached

                if not phantom_nodes:
                    continue

                # Split unreachable nodes into weakly connected components.
                remaining = set(phantom_nodes)

                while remaining:
                    seed = next(iter(remaining))
                    component = set()
                    queue = deque([seed])
                    remaining.remove(seed)

                    while queue:
                        node = queue.popleft()
                        component.add(node)

                        for neighbor in weak_neighbors.get(node, ()):
                            if neighbor in remaining:
                                remaining.remove(neighbor)
                                queue.append(neighbor)

                    internal_selected = [
                        (i, j)
                        for i, j in selected
                        if i in component and j in component
                    ]

                    if not internal_selected:
                        continue

                    # All model arcs entering this component.
                    boundary_arcs = []

                    for destination in component:
                        for origin, destination2 in incoming_by_mode[mode].get(
                            destination, ()
                        ):
                            if origin not in component:
                                boundary_arcs.append((origin, destination2))

                    external_inflow = gp.quicksum(
                        n[w, l, i, j]
                        for i, j in boundary_arcs
                    )

                    base_credit = gp.quicksum(
                        a[w, node] * b[l, node]
                        for node in component
                        if node in J_l[l]
                    )

                    rhs = external_inflow + base_credit

                    # This is the sum of the original per-arc inequalities
                    # for the internal arcs selected by this candidate.
                    #
                    # It cuts off the current component using one row instead
                    # of one nearly identical row per selected internal arc.
                    model.cbLazy(
                        gp.quicksum(
                            n[w, l, i, j]
                            for i, j in internal_selected
                        )
                        <= len(internal_selected) * rhs
                    )

                    cuts_this_call += 1
                    stats["cuts_added"] += 1

                    if cuts_this_call >= max_cuts_per_callback:
                        return

        except Exception:
            stats["errors"] += 1
            stats["last_error"] = traceback.format_exc()
            print(stats["last_error"])

            # Callback exceptions are otherwise ignored by Python/Gurobi.
            model.terminate()

        finally:
            stats["callback_seconds"] += time.perf_counter() - start

            if cuts_this_call > 0:
                stats["candidates_rejected"] += 1
                
            if stats["invocations"] % 10 == 0:
                runtime = model.cbGet(GRB.Callback.RUNTIME)

                print(
                    "[subtour callback]"
                    f" runtime={runtime:.1f}s,"
                    f" invocations={stats['invocations']},"
                    f" cuts={stats['cuts_added']},"
                    f" rejected={stats['candidates_rejected']},"
                    f" callback_seconds="
                    f"{stats['callback_seconds']:.2f}"
                )

    return callback


def solve_vif(
    model: "gp.Model",
    instance: Dict[str, Any],
    p: Dict,
    N: List[int],
    PPL: List[int],
    PPL_set: set,
    R: List[str],
    Omega: List[int],
    modes: List[str],
    modal_arcs: Dict[str, List[Tuple[int, int]]],
    modal_incoming: Dict[str, Dict[int, List]],
    modal_outgoing: Dict[str, Dict[int, List]],
    modal_arc_cost: Dict[str, Dict[Tuple[int, int], float]],
    vehicle_types: Dict[str, Dict],
    demand: Dict,
    penalty: Dict,
    inventory_if_open: Dict,
    releasable_fraction: float,
    site_cost: Dict,
    selection_budget: float,
    P_max: int,
    beta: float,
    prob: Dict,
    build_only: bool,
    verbose: bool,
    solve_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    PRS-VIF, through Phase 4: joint siting + basing (Phase 1), the
    multi-scenario CVaR risk wrapper (Phase 2), arc/node degradation +
    node-handling capacity (Phase 3), and the vehicle distance budget +
    subtour elimination (Phase 4). See docs/PRSVIF_Gospel.md, section
    "Risk-Aware Vehicle-Indexed Model" -- the sole authoritative source for
    this branch; equation labels below (eq:vif:*) refer to that document.

    Fully self-contained: PRS-VIF's variable/constraint set has no analog
    for the aggregate/individual formulations' mode-indexed x, tau, or the
    "release" decision variable (PRS-VIF has no intermodal transfer
    variables at all, and its release term is a fixed expression on p_i,
    not a decision variable -- see PHASE1_NOTES.md, "Deviation from the
    dispatch-point framing"). This function is therefore called via an
    early return from solve_stochastic_cvar() BEFORE any of that shared
    aggregate/individual code executes, rather than interleaved into it.

    IN SCOPE through Phase 4 (of 6):
        Sets N, N^P, K, L, L_k, J_k; variables p, b, n, nbar, x, y, z, loss
        (code name for Lambda^w), eta, xi; constraints C2 (site count), C3
        (activation budget), C4 (eq:vif:baseassign), C5 (eq:vif:baselink),
        C6 (eq:vif:balance, with the real a^w_i from eq:availability), C7
        (eq:vif:vehcap, with the real min{T^w_l,ij, cap_l} from
        eq:residual), C8 (eq:vif:transfercap -- node handling capacity
        Theta^w_i,m from eq:thetadegradation), C9 (eq:vif:conservation,
        with the real a^w_i), C10 (eq:vif:distbudget, new this phase --
        per-vehicle distance budget D_l inclusive of turnaround psi_l),
        C11 (eq:vif:sec, new this phase -- subtour elimination, lazily
        separated via _build_vif_subtour_callback); domain constraints
        dom1-dom7; the loss-definition equality (eq:vif:loss), C1/cvar1
        (eq:vif:cvar1), and dom-for-xi (xi >= 0, via the variable's own lb,
        matching cvar2). Objective: minimize
        eta + (1/(1-beta)) * sum_w pi^w * xi^w (eq:vif:objective).

    OUT OF SCOPE through Phase 4 (left as TODOs citing gospel equation
    labels):
        TODO(Phase 6, "Computational Considerations"): symmetry breaking
            (eq:vif:symbreak), on EXPECTED (probability-weighted) utilization
            across scenarios, not per-scenario as the old individual
            formulation's VehicleSymmetryBreak does.

    Schema note: this phase expects several instance keys not produced by
    the current input_builder.py pipeline (that pipeline is untouched --
    see PHASE1_NOTES.md, PHASE3_NOTES.md, and PHASE4_NOTES.md):
        instance["resource_weight"][r]      -- w_r, metric tons per unit of
                                                resource r.
        vehicle_types[k]["cap_tons"]         -- cap_k, payload capacity in
                                                metric tons (distinct from
                                                vehicle_types[k]["capacity"],
                                                which is the aggregate/
                                                individual paths' already-
                                                per-resource-converted
                                                person-day figure and is not
                                                used here).
        vehicle_types[k]["J_k"]              -- basing-eligible node list.
                                                Already computed and stored
                                                by build_vehicle_params()
                                                (input_builder.py:493-501)
                                                but unused by the aggregate/
                                                individual paths; reused
                                                as-is here. See
                                                PHASE1_NOTES.md, "Design
                                                decision: J_k source" for why
                                                this is used instead of the
                                                b_kj-eligibility
                                                reinterpretation the phase
                                                prompt proposed.
        instance["node_severity"][(w,i)]     -- sigma^w_i, node severity
                                                (eq:decay's output),
                                                default 0.0 (undamaged) if
                                                a key is missing.
        instance["disaster_type"][w]         -- nu(w), the disaster type
                                                realized in scenario w.
        instance["degradation_matrix"]
            [mode][disaster_type]            -- Gamma_m,nu, baseline
                                                degradation sensitivity
                                                (same structure/semantics
                                                as the aggregate/individual
                                                pipeline's identically-named
                                                config key -- see
                                                PHASE3_NOTES.md for why this
                                                is threaded through as a
                                                fresh instance key rather
                                                than reused from that
                                                pipeline).
        instance["alpha"]                    -- global degradation scale,
                                                default 1.0 if absent
                                                (matches the aggregate/
                                                individual pipeline's
                                                default).
        instance["nominal_throughput"]
            [mode][(i,j)]                    -- T_m,ij, nominal arc
                                                throughput in metric tons
                                                (NOT the same quantity as
                                                the aggregate/individual
                                                paths' modal_residual --
                                                see PHASE3_NOTES.md).
        instance["node_handling_capacity"]
            [(i,mode)]                       -- Theta_i,m, baseline node
                                                handling capacity (vehicle
                                                arrivals per horizon).
                                                Missing entries default to
                                                0.0.
        instance["node_handling_bonus"]
            [(i,mode)]                       -- DeltaTheta_i,m, activation
                                                bonus. Only applied for
                                                i in N^P (forced to 0
                                                otherwise, per gospel's
                                                explicit convention
                                                DeltaTheta_i,m := 0 for
                                                i not in N^P); missing
                                                entries default to 0.0.
        instance["modal_arc_distance"]
            [mode][(i,j)]                    -- dist_l,ij, km (== dist_ij;
                                                gospel notes this doesn't
                                                actually depend on which
                                                vehicle traverses a given
                                                arc). REUSED from the
                                                aggregate/individual paths'
                                                identically-named,
                                                identically-shaped instance
                                                key (already populated by
                                                build_modal_arcs() in the
                                                real pipeline) -- not a new
                                                schema field, unlike
                                                everything else on this
                                                list.
        vehicle_types[k]["D_k"], ["pi_k"]    -- D_l, psi_l. REUSED from the
                                                aggregate/individual paths'
                                                identically-computed fields
                                                (build_vehicle_params():
                                                D_k=3.0*cruise_speed,
                                                pi_k=(turnaround_hours/24.0)
                                                *cruise_speed -- exactly
                                                D_l=kappa*v_l and
                                                psi_l=(tau_l/24)*v_l with
                                                kappa=3). See
                                                PHASE4_NOTES.md.
    """
    if not vehicle_types:
        raise ValueError(
            "vehicle_formulation='vif' requires a non-empty vehicle_types "
            "dict -- PRS-VIF has no vehicle-free mode (b, n, nbar, x are "
            "all defined over L, which is empty without vehicle_types)."
        )
    for k, vtype in vehicle_types.items():
        if not all(key in vtype for key in ("J_k", "cap_tons", "D_k", "pi_k")):
            raise ValueError(
                f"vehicle_formulation='vif': vehicle_types[{k!r}] is missing "
                f"one or more of 'J_k', 'cap_tons', 'D_k', 'pi_k' -- all are "
                f"required by the vif branch. 'D_k'/'pi_k' are already "
                f"computed by build_vehicle_params() with formulas matching "
                f"D_l/psi_l exactly (see solve_vif's docstring, 'Schema "
                f"note'), so only 'J_k' and 'cap_tons' are genuinely new "
                f"relative to the aggregate/individual schema."
            )
    if "resource_weight" not in instance:
        raise ValueError(
            "vehicle_formulation='vif' requires instance['resource_weight'] "
            "(w_r, metric tons per unit of resource r) -- see "
            "solve_vif's docstring, 'Schema note'."
        )
    resource_weight = instance["resource_weight"]

    for key in (
        "node_severity", "disaster_type", "degradation_matrix",
        "nominal_throughput", "node_handling_capacity", "node_handling_bonus",
    ):
        if key not in instance:
            raise ValueError(
                f"vehicle_formulation='vif' requires instance[{key!r}] "
                f"(Phase 3 degradation data) -- see solve_vif's docstring, "
                f"'Schema note'."
            )
    node_severity = instance["node_severity"]            # {(w,i): sigma^w_i}
    disaster_type = instance["disaster_type"]             # {w: nu(w)}
    degradation_matrix = instance["degradation_matrix"]    # {mode: {disaster_type: Gamma_m,nu}}
    alpha = float(instance.get("alpha", 1.0))              # global degradation scale (default 1.0, matching the aggregate/individual paths' pipeline default)
    nominal_throughput = instance["nominal_throughput"]    # {mode: {(i,j): T_m,ij}}
    node_handling_capacity = instance["node_handling_capacity"]  # {(i,m): Theta_i,m}
    node_handling_bonus = instance["node_handling_bonus"]  # {(i,m): DeltaTheta_i,m}

    if "modal_arc_distance" not in instance:
        raise ValueError(
            "vehicle_formulation='vif' requires instance['modal_arc_distance'] "
            "(dist_l,ij, km) -- the same key the aggregate/individual paths "
            "already populate from build_modal_arcs(); see solve_vif's "
            "docstring, 'Schema note'."
        )
    modal_arc_distance = instance["modal_arc_distance"]  # {mode: {(i,j): km}}

    # --- Derived per-scenario quantities (eq:availability, eq:gammascale) ---
    # a[w,i]: node availability (eq:availability). Missing node_severity
    # entries default to 0 (undamaged), matching the aggregate/individual
    # pipeline's convention for the same quantity (input_builder.py's
    # build_modal_residual_capacity: "Missing node_severity entries
    # default to 0").
    a: Dict[Tuple[int, int], float] = {
        (w, i): (0.0 if node_severity.get((w, i), 0.0) >= 1.0 else 1.0)
        for w in Omega for i in N
    }

    # gamma[w,m] = Gamma_{m,nu(w)} * alpha (eq:gammascale), one value per
    # (scenario, mode) -- shared by every vehicle of that mode (gospel:
    # "written gamma_l for l in L_m").
    gamma: Dict[Tuple[int, str], float] = {
        (w, m): degradation_matrix.get(m, {}).get(disaster_type[w], 0.0) * alpha
        for w in Omega for m in modes
    }

    # --- Sets: K, L, L_k, and per-vehicle mode/eligibility lookups ---
    K = sorted(vehicle_types.keys())
    L: List[int] = []
    L_k: Dict[str, List[int]] = {}
    type_of: Dict[int, str] = {}
    mode_of: Dict[int, str] = {}
    _next_l = 1
    for k in K:
        fleet_size = int(vehicle_types[k]["fleet_size"])
        L_k[k] = []
        for _ in range(fleet_size):
            l = _next_l
            _next_l += 1
            L.append(l)
            L_k[k].append(l)
            type_of[l] = k
            mode_of[l] = vehicle_types[k]["mode"]

    # L_m: vehicles operating on mode m (needed by C8's per-mode arrival
    # sum -- A_l := A_m for l in L_m makes every vehicle of a given mode
    # share the same incoming-arc structure at any node).
    L_m: Dict[str, List[int]] = {m: [] for m in modes}
    for l in L:
        L_m[mode_of[l]].append(l)

    # --- Variables ---
    # p[i] is created by the caller (identical to the aggregate/individual
    # branches' p -- dom1 matches exactly, no reason to duplicate it).

    # b[l,j] in {0,1}, l in L, j in J_k(l). dom2.
    b_keys = [(l, j) for l in L for j in vehicle_types[type_of[l]]["J_k"]]
    b = model.addVars(b_keys, vtype=GRB.BINARY, name="vif_b") if b_keys else {}

    # n[w,l,i,j] in {0,1}, l in L, (i,j) in A_l := A_{mode(l)}. dom3.
    n_keys = [
        (w, l, i, j)
        for w in Omega for l in L
        for (i, j) in modal_arcs[mode_of[l]]
    ]
    n = model.addVars( n_keys, lb=0,ub=1,vtype=GRB.BINARY,name="vif_n",) if n_keys else {}

    # nbar[w,l,i] >= 0, continuous (dom4 -- gospel eq:vif:dom4 declares this
    # continuous, not binary; integrality follows from C9 given n, b
    # integral -- see gospel's remark after eq:vif:dom8).
    nbar_keys = [(w, l, i) for w in Omega for l in L for i in N]
    nbar = model.addVars(nbar_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="vif_nbar")

    # x[w,l,i,j,r] >= 0, l in L, (i,j) in A_l, r in R. dom5.
    x_keys = [
        (w, l, i, j, r)
        for w in Omega for l in L
        for (i, j) in modal_arcs[mode_of[l]]
        for r in R
    ]
    x = model.addVars(x_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="vif_x") if x_keys else {}

    # y[w,i,r] >= 0 (retained resource -- NOT the aggregate/individual
    # paths' "release"; see docstring). dom6.
    node_r_keys = [(w, i, r) for w in Omega for i in N for r in R]
    y = model.addVars(node_r_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="vif_y")

    # z[w,i,r], 0 <= z <= d^w_ir (dom7 -- box bound, not a separate
    # constraint row).
    z = model.addVars(node_r_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="vif_z")
    for (w, i, r) in node_r_keys:
        z[w, i, r].UB = demand[w, i, r]

    # --- Constraints ---

    # C2 (eq:vif:pmax), C3 (eq:vif:budget). Byte-identical in spirit to the
    # aggregate/individual paths' SiteBudget/SelectionBudget, duplicated
    # here (2 lines) rather than shared, since this function is reached
    # only via an early return that occurs before those lines run.
    model.addConstr(gp.quicksum(p[i] for i in PPL) <= P_max, name="Vif_SiteBudget")
    model.addConstr(
        gp.quicksum(site_cost[i] * p[i] for i in PPL) <= selection_budget,
        name="Vif_SelectionBudget",
    )

    # C4 (eq:vif:baseassign): every vehicle assigned exactly one base.
    for k in K:
        for l in L_k[k]:
            model.addConstr(
                gp.quicksum(b[l, j] for j in vehicle_types[k]["J_k"]) == 1,
                name=f"VifBaseAssign_k{k}_l{l}",
            )

    # C5 (eq:vif:baselink): basing permitted only at an activated node.
    for k in K:
        for l in L_k[k]:
            for j in vehicle_types[k]["J_k"]:
                model.addConstr(
                    b[l, j] <= p[j],
                    name=f"VifBaseLink_k{k}_l{l}_j{j}",
                )

    # C6 (eq:vif:balance), a^w_i now the real eq:availability value (was
    # hardcoded to 1 in Phase 1/2).
    for w in Omega:
        for i in N:
            for r in R:
                inflow = gp.quicksum(
                    x[w, l, jj, i, r]
                    for l in L for (jj, _jj2) in modal_incoming[mode_of[l]][i]
                )
                outflow = gp.quicksum(
                    x[w, l, i, jj, r]
                    for l in L for (_ii2, jj) in modal_outgoing[mode_of[l]][i]
                )
                release_term = (
                    releasable_fraction * inventory_if_open[i, r] * a[w, i] * p[i]
                    if i in PPL_set else 0
                )
                model.addConstr(
                    inflow + release_term + z[w, i, r]
                    == demand[w, i, r] + y[w, i, r] + outflow,
                    name=f"VifResourceBalance_w{w}_i{i}_r{r}",
                )

    # C7 (eq:vif:vehcap), now min{T^w_l,ij, cap_l} (eq:residual) -- was
    # collapsed to cap_l alone in Phase 1/2. T^w_l,ij depends only on
    # scenario data (no decision variables), so it's a plain precomputed
    # float, computed once per (w, mode, i, j) rather than per vehicle
    # (every vehicle of a mode shares the same residual throughput on a
    # given arc -- gospel: "T_l,ij := T_m,ij for l in L_m").
    T_residual: Dict[Tuple[int, str, int, int], float] = {}
    for w in Omega:
        for m in modes:
            T_nominal_m = nominal_throughput.get(m, {})
            for (i, j) in modal_arcs.get(m, []):
                sigma_i = node_severity.get((w, i), 0.0)
                sigma_j = node_severity.get((w, j), 0.0)
                T_residual[w, m, i, j] = T_nominal_m.get((i, j), 0.0) * _degradation_factor(
                    gamma[w, m], max(sigma_i, sigma_j)
                )

    # talk: switched this from one addConstr call per row to one bulk
    # addConstrs call. checked its the same math by rebuilding at omega=3
    # and getting identical variable/constraint counts and lp objective to
    # the decimal. made zero measurable speed difference at any size tested
    # PHASE5 addConstrs conversion: identical per-(w,l,i,j) inequality as
    # before, built in one bulk model.addConstrs() call instead of one
    # model.addConstr() Python call per row. See PHASE5_NOTES.md,
    # "addConstrs conversion" for why this is a construction-mechanics
    # change only (same LHS/RHS expression per row, same coefficients,
    # same feasible region) -- NOT a reformulation of eq:vif:vehcap.
    # cap_l/m are still looked up once per vehicle l via
    # vehicle_lookup[l] rather than recomputed per row.
    vehicle_lookup = {l: (vehicle_types[type_of[l]]["cap_tons"], mode_of[l]) for l in L}
    
        # Explicit per-resource flow bounds implied by vehicle capacity.
    #
    # The aggregate C7 capacity constraint remains necessary because it
    # couples the different resource flows on the same vehicle arc.
    for r in R:
        if resource_weight[r] <= 0:
            raise ValueError(
                f"resource_weight[{r!r}] must be positive; "
                f"received {resource_weight[r]!r}"
            )

    zero_capacity_x = 0

    for w in Omega:
        for l in L:
            vehicle_capacity, mode = vehicle_lookup[l]

            for i, j in modal_arcs[mode]:
                usable_capacity = min(
                    T_residual[w, mode, i, j],
                    vehicle_capacity,
                )

                for r in R:
                    x[w, l, i, j, r].UB = (
                        usable_capacity / resource_weight[r]
                    )

                    if usable_capacity <= 0:
                        zero_capacity_x += 1

    if verbose:
        print(
            f"Explicitly fixed {zero_capacity_x} "
            f"zero-capacity resource-flow variables."
        )
    
    model.addConstrs(
        (
            gp.quicksum(resource_weight[r] * x[w, l, i, j, r] for r in R)
            <= min(T_residual[w, vehicle_lookup[l][1], i, j], vehicle_lookup[l][0])
            * n[w, l, i, j]
            for w in Omega
            for l in L
            for (i, j) in modal_arcs[vehicle_lookup[l][1]]
        ),
        name="VifVehicleCapacity",
    )

    # C8 (eq:vif:transfercap, new this phase): total mode-m vehicle
    # arrivals at i cannot exceed the residual node handling capacity
    # Theta^w_i,m (eq:thetadegradation). Theta_i,m + DeltaTheta_i,m*p_i is
    # LINEAR in the decision variable p_i (a constant times p_i, not a
    # product of two decision variables), so no McCormick auxiliary is
    # needed here -- unlike the aggregate/individual paths' turnaround-
    # exemption linearization (their g_turn variable), which multiplies
    # two decision-dependent quantities together.
    for w in Omega:
        for m in modes:
            for i in N:
                baseline = node_handling_capacity.get((i, m), 0.0)
                if i in PPL_set:
                    bonus_term = node_handling_bonus.get((i, m), 0.0) * p[i]
                else:
                    bonus_term = 0  # DeltaTheta_i,m := 0 for i not in N^P (gospel)
                theta_w_i_m = _degradation_factor(gamma[w, m], node_severity.get((w, i), 0.0)) * (
                    baseline + bonus_term
                )
                arrivals = gp.quicksum(
                    n[w, l, jj, i]
                    for l in L_m[m] for (jj, _ii) in modal_incoming[m][i]
                )
                model.addConstr(
                    arrivals <= theta_w_i_m,
                    name=f"VifNodeHandlingCapacity_w{w}_m{m}_i{i}",
                )

    # C9 (eq:vif:conservation), a^w_i now the real eq:availability value
    # (was hardcoded to 1 in Phase 1/2) -- a vehicle based at a node whose
    # severity has reached the availability cutoff gets zero departure
    # credit there this scenario, regardless of its (first-stage, fixed)
    # basing decision b.
    # PHASE5 addConstrs conversion: identical per-(w,l,i) equality as
    # before (same out_i/in_i/rhs terms), built via one bulk
    # model.addConstrs() call. J_l_sets is precomputed once per vehicle
    # (96 entries) instead of being rebuilt on every (w,l) pass -- a
    # redundant-recomputation removal, not a semantic change (the set
    # membership test it feeds is identical either way). See
    # PHASE5_NOTES.md, "addConstrs conversion."
    J_l_sets = {l: set(vehicle_types[type_of[l]]["J_k"]) for l in L}
    model.addConstrs(
        (
            gp.quicksum(n[w, l, i, jj] for (_ii, jj) in modal_outgoing[mode_of[l]][i])
            + nbar[w, l, i]
            - gp.quicksum(n[w, l, jj, i] for (jj, _ii) in modal_incoming[mode_of[l]][i])
            == (a[w, i] * b[l, i] if i in J_l_sets[l] else 0)
            for w in Omega
            for l in L
            for i in N
        ),
        name="VifVehicleConservation",
    )

    # C10 (eq:vif:distbudget): total distance traveled (inclusive of
    # turnaround) over the horizon, per vehicle instance. dist_l,ij is
    # read from instance["modal_arc_distance"] -- the SAME key the
    # aggregate/individual paths already populate from the real pipeline
    # (input_builder.py's build_modal_arcs()), reused as-is rather than
    # inventing a new one, since it's genuinely the same physical
    # quantity (gospel: "dist_ijl, l in L: Length of arc (i,j) for
    # vehicle l" -- distance doesn't actually depend on which vehicle
    # traverses a given arc, only on the arc itself). D_l and psi_l are
    # likewise read from vehicle_types[k]["D_k"]/["pi_k"], already
    # computed by build_vehicle_params() with formulas that already match
    # D_l=kappa*v_l and psi_l=(tau_l/24)*v_l exactly (see
    # PHASE4_NOTES.md).
    for w in Omega:
        for l in L:
            m = mode_of[l]
            D_l = vehicle_types[type_of[l]]["D_k"]
            psi_l = vehicle_types[type_of[l]]["pi_k"]
            dist_expr = gp.quicksum(
                (modal_arc_distance.get(m, {}).get((i, j), 0.0) + psi_l) * n[w, l, i, j]
                for (i, j) in modal_arcs[m]
            )
            model.addConstr(dist_expr <= D_l, name=f"VifDistanceBudget_w{w}_l{l}")


    # Symmetry breaking for identical vehicles within each vehicle type.
    #
    # Any solution can be relabeled so that vehicles of the same type are
    # ordered by expected arc utilization. This removes equivalent solutions
    # created by permuting otherwise identical vehicle indices.
#    for k in K:
#        vehicles = L_k[k]
#
#        for position in range(len(vehicles) - 1):
#            l_current = vehicles[position]
#            l_next = vehicles[position + 1]
#            mode = mode_of[l_current]
#
#            expected_utilization_current = gp.quicksum(
#                prob[w] * n[w, l_current, i, j]
#                for w in Omega
#                for i, j in modal_arcs[mode]
#            )
#
#            expected_utilization_next = gp.quicksum(
#                prob[w] * n[w, l_next, i, j]
#                for w in Omega
#                for i, j in modal_arcs[mode]
#            )
#
#            model.addConstr(
#                expected_utilization_current
#                >= expected_utilization_next,
#                name=(
#                    f"VifVehicleSymmetry_"
#                    f"k{k}_l{l_current}_l{l_next}"
#                ),
#            )

    # C11 (eq:vif:sec, subtour elimination): exponential in |N|, not
    # enumerated -- separated lazily via callback (_build_vif_subtour_callback,
    # analogous to model.py's _build_subtour_callback for the old
    # "individual" formulation's n_ind, rebuilt here against n[w,l,i,j]/
    # b[l,i] since PRS-VIF's variable set differs). See its docstring for
    # the detection/cut-generation algorithm.
#    subtour_stats = {"invocations": 0, "cuts_added": 0}

    subtour_stats = {
        "invocations": 0,
        "cuts_added": 0,
        "candidates_rejected": 0,
        "callback_seconds": 0.0,
        "errors": 0,
        "last_error": None,
    }
    if L:
        J_l = {l: set(vehicle_types[type_of[l]]["J_k"]) for l in L}
        model.Params.LazyConstraints = 1
        subtour_callback = _build_vif_subtour_callback(
            n, b, a, L, J_l, Omega, modal_arcs, mode_of, subtour_stats,
        )
    else:
        subtour_callback = None

    # --- Scenario loss Lambda^w (eq:vif:loss) and CVaR wrapper
    # (eq:vif:objective, cvar1, cvar2) ---
    # EPSILON: gospel Table 5 (Model Parameter Values), vehicle-movement
    # tie-breaker = 0.04. Hardcoded to match the gospel-stated value, same
    # pattern the aggregate/individual paths use for their epsilon terms
    # (not read from instance -- TODO(later phase): source from config if
    # this needs to become a swept parameter).
    EPSILON = 40

    # loss[w] is the code name for Lambda^w. The gospel's Variables section
    # declares Lambda^w >= 0 as a genuine decision variable (not just an
    # expression), pinned to the eq:vif:loss formula -- Phase 1 computed
    # the formula directly as an expression since nothing else referenced
    # it; cvar1 now needs Lambda^w as a reusable quantity, so it becomes a
    # variable with a defining equality, matching the aggregate/individual
    # paths' identical "loss" pattern (see PHASE2_NOTES.md).
    loss = model.addVars(Omega, lb=0.0, vtype=GRB.CONTINUOUS, name="vif_loss")
    for w in Omega:
        unmet_term = gp.quicksum(penalty[i, r] * z[w, i, r] for i in N for r in R)
        transport_term = gp.quicksum(
            modal_arc_cost.get(mode_of[l], {}).get((i, j), 0.0) * x[w, l, i, j, r]
            for l in L for (i, j) in modal_arcs[mode_of[l]] for r in R
        )
        movement_term = EPSILON * gp.quicksum(
            n[w, l, i, j] for l in L for (i, j) in modal_arcs[mode_of[l]]
        )
        model.addConstr(
            loss[w] == unmet_term + transport_term + movement_term,
            name=f"VifLossDefinition_w{w}",
        )

    # eta (dom8: eta in R, no scenario index -- chosen once, shared across
    # all scenarios, same as the aggregate/individual paths' eta).
    eta = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="vif_eta")

    # xi^w >= 0 (cvar2 -- enforced as the variable's own lower bound, not a
    # separate named constraint row; same convention the aggregate/
    # individual paths use for their xi).
    xi = model.addVars(Omega, lb=0.0, vtype=GRB.CONTINUOUS, name="vif_xi")

    # C1/cvar1 (eq:vif:cvar1): xi^w >= Lambda^w - eta.
    for w in Omega:
        model.addConstr(xi[w] >= loss[w] - eta, name=f"VifCVaRExcess_w{w}")

    # Objective (eq:vif:objective): min eta + (1/(1-beta)) * sum_w pi^w * xi^w.
    model.setObjective(
        eta + (1.0 / (1.0 - beta)) * gp.quicksum(prob[w] * xi[w] for w in Omega),
        GRB.MINIMIZE,
    )

    variables = {
        "p": p, "b": b, "n": n, "nbar": nbar, "x": x, "y": y, "z": z,
        "loss": loss, "eta": eta, "xi": xi,
    }

    solve_config = solve_config or {}

    # Optional guidance for the staged VIF workflow. All of this is applied
    # after the complete model has been built but before optimization, so it
    # cannot accidentally alter the mathematical formulation used by normal
    # calls that omit solve_config.
    fixed_defaults = solve_config.get("fix_defaults", {})
    fixed_values = solve_config.get("fix", {})
    start_values = solve_config.get("start", {})
    start_defaults = solve_config.get("start_defaults", {})

    def fix_variable(var, value):
        value = float(value)
        # Set UB first so a binary previously fixed to zero can safely be
        # overridden to one by an explicit sparse value below.
        var.UB = value
        var.LB = value

    for family_name, default_value in fixed_defaults.items():
        family = variables.get(family_name)
        if family is None:
            raise ValueError(
                f"unknown VIF variable family in fix_defaults: {family_name!r}"
            )
        for var in family.values():
            fix_variable(var, default_value)

    for family_name, values in fixed_values.items():
        family = variables.get(family_name)
        if family is None:
            raise ValueError(f"unknown VIF variable family in fix: {family_name!r}")
        for key, value in values.items():
            if key not in family:
                raise KeyError(f"unknown {family_name} key in fix: {key!r}")
            fix_variable(family[key], value)

    # When p, b, and n are all fixed, converting them to continuous
    # variables makes the final resource-allocation stage a genuine LP.
    # The caller must fully fix every variable in each listed family.
    for family_name in solve_config.get("relax_fixed_families", []):
        family = variables.get(family_name)
        if family is None:
            raise ValueError(
                f"unknown VIF variable family in relax_fixed_families: "
                f"{family_name!r}"
            )
        for var in family.values():
            var.VType = GRB.CONTINUOUS

    for family_name, default_value in start_defaults.items():
        family = variables.get(family_name)
        if family is None:
            raise ValueError(
                f"unknown VIF variable family in start_defaults: {family_name!r}"
            )
        for var in family.values():
            var.Start = float(default_value)

    for family_name, values in start_values.items():
        family = variables.get(family_name)
        if family is None:
            raise ValueError(f"unknown VIF variable family in start: {family_name!r}")
        for key, value in values.items():
            if key in family:
                family[key].Start = float(value)

    strategic_p = solve_config.get("strategic_p")
    p_radius = solve_config.get("p_neighborhood")
    if strategic_p is not None and p_radius is not None:
        strategic_p = {i: int(value > 0.5) for i, value in strategic_p.items()}
        model.addConstr(
            gp.quicksum(
                (1 - p[i]) if strategic_p.get(i, 0) else p[i]
                for i in PPL
            ) <= int(p_radius),
            name="VifStagedSiteNeighborhood",
        )

    strategic_b = solve_config.get("strategic_b")
    b_radius = solve_config.get("b_neighborhood")
    if strategic_b is not None and b_radius is not None:
        chosen_base = {
            l: j for (l, j), value in strategic_b.items() if value > 0.5
        }
        missing = [l for l in L if l not in chosen_base]
        if missing:
            raise ValueError(
                "strategic_b must select one base for every vehicle; "
                f"missing vehicles: {missing}"
            )
        # Count reassigned vehicles, rather than binary Hamming distance.
        # A single reassignment changes two b entries but consumes one unit
        # of this neighborhood.
        model.addConstr(
            gp.quicksum(1 - b[l, chosen_base[l]] for l in L)
            <= int(b_radius),
            name="VifStagedBaseNeighborhood",
        )

    for parameter_name, value in solve_config.get("params", {}).items():
        model.setParam(parameter_name, value)

    if build_only:
        model.update()
        return {"model": model, "variables": variables}

    if subtour_callback is not None:
        model.optimize(subtour_callback)
    else:
        model.optimize()

    if verbose and subtour_callback is not None:
        print(
            "Subtour callback:"
            f" invocations={subtour_stats['invocations']},"
            f" cuts={subtour_stats['cuts_added']},"
            f" rejected_candidates={subtour_stats['candidates_rejected']},"
            f" callback_seconds={subtour_stats['callback_seconds']:.2f},"
            f" errors={subtour_stats['errors']}"
        )

    results: Dict[str, Any] = {
        "status_code": model.Status,
        "status": _status_to_string(model.Status),
        "objective_value": None,
        "eta": None,
        "scenario_losses": {},  # keyed w -> Lambda^w value
        "xi": {},              # keyed w -> CVaR excess value
        # node_availability: keyed (w, i) -> a^w_i (eq:availability), a
        # precomputed diagnostic (not solution-dependent), always present
        # regardless of solve status.
        "node_availability": dict(a),
        "subtour_callback_stats": subtour_stats,  # {"invocations": int, "cuts_added": int}
        "selected_sites": [],
        "basing": {},        # keyed (l, j) -> 1 for l based at j
        "vehicle_arcs": {},  # keyed (w, l, i, j) -> 1 for arcs traversed
        "vehicle_terminations": {},  # keyed (w, l, i) -> nbar value
        "flows": {},         # keyed (w, l, i, j, r)
        "retained": {},      # keyed (w, i, r) -> y value
        "unmet_demand": {},  # keyed (w, i, r) -> z value
        "model": model,
        "variables": variables,
    }

    if model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} and model.SolCount > 0:
        results["objective_value"] = model.ObjVal
        results["eta"] = eta.X
        for w, var in loss.items():
            results["scenario_losses"][w] = var.X
        for w, var in xi.items():
            results["xi"][w] = var.X
        results["selected_sites"] = [i for i in PPL if p[i].X > 0.5]

        for (l, j), var in b.items():
            if var.X > 0.5:
                results["basing"][(l, j)] = 1

        for key, var in n.items():
            if var.X > 0.5:
                results["vehicle_arcs"][key] = 1

        for key, var in nbar.items():
            if var.X > 1e-6:
                results["vehicle_terminations"][key] = var.X

        for key, var in x.items():
            if var.X > 1e-6:
                results["flows"][key] = var.X

        for key, var in y.items():
            if var.X > 1e-6:
                results["retained"][key] = var.X

        for key, var in z.items():
            if var.X > 1e-6:
                results["unmet_demand"][key] = var.X

    return results
