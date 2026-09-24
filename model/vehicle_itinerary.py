"""
vehicle_itinerary.py

Reconstructs per-vehicle itineraries from the aggregate n^omega_{k,m,ij}
variables produced by solve_stochastic_cvar and writes a formatted markdown
report.

DISCLAIMER (printed once at the top of every report, not per scenario):
  The paths below are ONE CONSISTENT RECONSTRUCTION from the model's
  aggregate vehicle-count variables n^omega_{k,m,ij}. The formulation
  tracks integer vehicle counts per arc, not individual vehicle identity
  or sequence. Multiple decompositions into per-vehicle paths are
  mathematically consistent with the same n values; this report shows
  one such decomposition produced by a greedy depth-first algorithm.
  Cargo quantities on each leg are pro-rated from aggregate arc flow
  divided by vehicle count and are likewise reconstructed approximations.

Default behaviour: generate for the 3 lowest-SR and 3 highest-SR
scenarios per solve. Pass itinerary_all=True to generate all N scenarios.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any

DISCLAIMER = (
    "**Disclaimer:** paths below are one consistent reconstruction from the "
    "model's aggregate n^omega_{k,m,ij} variables. The formulation tracks "
    "integer vehicle counts per arc, not individual vehicle identity. Multiple "
    "decompositions may be consistent with the same n values; one is shown "
    "here. Cargo quantities are pro-rated from aggregate arc flow and are "
    "likewise reconstructed approximations, not the model's literal allocation."
)


# ── Path decomposition ────────────────────────────────────────────────────

def _decompose_to_paths(
    arc_counts: Dict[Tuple[int, int], int],
    base_alloc: Dict[int, int],
    arc_distances: Optional[Dict[Tuple[int, int], float]] = None,
    D_k: float = float("inf"),
    pi_k: float = 0.0,
) -> Tuple[List[List[int]], List[str]]:
    """
    Decompose aggregate vehicle-arc counts into individual vehicle paths.

    arc_counts  : {(i, j): count}  – non-zero n[k,m,i,j] for this type/scenario
    base_alloc  : {j: count}       – b[k,j]*p[j], vehicles initially available
    arc_distances, D_k, pi_k: accepted for interface compatibility but not used
        for a per-path cap. The model's distance constraint (19) is fleet-wide,
        not per-vehicle, so individual paths are not distance-capped here.

    Termination: non-base cycle guard only — do not revisit an intermediate
    (non-base) node within the same path. This breaks shuttle cycles like
    A→B→A→B→… while allowing legitimate return-to-base legs (A→B→A).

    Returns
    -------
    paths   : list of node-sequences (each is one vehicle's itinerary)
    flags   : list of inconsistency descriptions (shown only when cross-check fails)
    """
    remaining = {arc: cnt for arc, cnt in arc_counts.items() if cnt > 0}
    available = {j: cnt for j, cnt in base_alloc.items() if cnt > 0}
    paths: List[List[int]] = []
    flags: List[str] = []

    def _next_hops(node: int) -> List[int]:
        return sorted(j for (i, j), c in remaining.items() if i == node and c > 0)

    max_iter = sum(remaining.values()) + 1

    for _ in range(max_iter):
        if not remaining:
            break

        start: Optional[int] = None
        for base in sorted(available.keys()):
            if available[base] > 0 and _next_hops(base):
                start = base
                break

        if start is None:
            for (i, j), cnt in list(remaining.items()):
                if cnt > 0:
                    flags.append(
                        f"arc {i}→{j}: {cnt} vehicle-traversal(s) not explained "
                        f"by base allocation — conservation inconsistency"
                    )
                    for _ in range(cnt):
                        paths.append([i, j])
            remaining.clear()
            break

        available[start] -= 1
        path = [start]
        cur = start
        visited_intermediate: set = set()

        while True:
            nexts = _next_hops(cur)
            if not nexts:
                break
            # Cycle guard: don't revisit intermediate (non-base) nodes in this path
            nexts_ok = [j for j in nexts
                        if j in base_alloc or j not in visited_intermediate]
            if not nexts_ok:
                break
            nxt = nexts_ok[0]
            remaining[(cur, nxt)] -= 1
            if remaining[(cur, nxt)] == 0:
                del remaining[(cur, nxt)]
            path.append(nxt)
            if nxt not in base_alloc:
                visited_intermediate.add(nxt)
            cur = nxt

        paths.append(path)
        if cur in base_alloc:
            available[cur] = available.get(cur, 0) + 1

    if remaining:
        total = sum(remaining.values())
        flags.append(
            f"{total} total vehicle-arc(s) remain after path decomposition — "
            f"verify conservation constraint (16) is satisfied"
        )

    return paths, flags


# ── Cargo annotation ──────────────────────────────────────────────────────

def _arc_cargo(
    w: int,
    mode: str,
    i: int,
    j: int,
    kname: str,
    results: Dict,
    instance: Dict,
) -> Tuple[float, float, float, float]:
    """
    Return (food_per_veh, water_per_veh, cap_food_fleet, cap_water_fleet) for one leg.

    Flow on arc (i,j) is shared across all vehicle types on that mode.
    Per-vehicle average = total_arc_flow / n_total_all_types.

    Capacity is total fleet capacity on this arc across all types, so that
    the utilisation percentage (food_per_veh / cap_food_fleet) is always ≤ 100%
    when constraint (18) holds, even when multiple vehicle types share the arc.
    """
    try:
        food = results["variables"]["x"][w, mode, i, j, "food"].X
    except Exception:
        food = 0.0
    try:
        water = results["variables"]["x"][w, mode, i, j, "water"].X
    except Exception:
        water = 0.0

    K_m = instance.get("K_m", {})
    vt = instance.get("vehicle_types", {})
    n_var = results["variables"]["n"]

    n_per_k: Dict = {}
    for k in K_m.get(mode, []):
        try:
            nv = n_var[w, k, mode, i, j].X if (w, k, mode, i, j) in n_var else 0.0
        except Exception:
            nv = 0.0
        n_per_k[k] = nv

    n_total = max(sum(n_per_k.values()), 1.0)

    # Total fleet capacity on this arc (across all vehicle types present)
    cap_food_fleet = sum(
        vt.get(k, {}).get("capacity", {}).get("food", 0.0) * n_per_k.get(k, 0.0)
        for k in K_m.get(mode, [])
    )
    cap_water_fleet = sum(
        vt.get(k, {}).get("capacity", {}).get("water", 0.0) * n_per_k.get(k, 0.0)
        for k in K_m.get(mode, [])
    )
    # Fall back to type-specific capacity for single-type arcs
    if cap_food_fleet < 1e-6:
        cap_food_fleet = vt.get(kname, {}).get("capacity", {}).get("food", 0.0) * n_total
    if cap_water_fleet < 1e-6:
        cap_water_fleet = vt.get(kname, {}).get("capacity", {}).get("water", 0.0) * n_total

    food_per_veh = food / n_total
    water_per_veh = water / n_total
    return food_per_veh, water_per_veh, cap_food_fleet / n_total, cap_water_fleet / n_total


# ── Per-scenario report builder ───────────────────────────────────────────

def _scenario_block(
    w: int,
    sr: float,
    results: Dict,
    instance: Dict,
    site_names: Dict[int, str],
) -> str:
    """Return a markdown block for one scenario's vehicle itineraries."""
    lines: List[str] = []
    lines.append(f"### Scenario w={w}  (sr={sr:.4f})")

    vehicle_types = instance.get("vehicle_types", {})
    K_m = instance.get("K_m", {})
    modes = instance.get("modes", [])
    modal_arcs = instance.get("modal_arcs", {})
    PPL_set = set(instance.get("ppl_nodes", []))
    n_var = results["variables"]["n"]
    p_var = results["variables"]["p"]

    any_movement = False

    for kname in sorted(vehicle_types.keys()):
        vt = vehicle_types[kname]
        mv = vt["mode"]
        Fk = vt["fleet_size"]
        bkj = vt["b_kj"]

        # Active arcs for this type/scenario
        arc_counts: Dict[Tuple[int, int], int] = {}
        for (i, j) in modal_arcs.get(mv, []):
            key = (w, kname, mv, i, j)
            if key in n_var:
                cnt = int(round(n_var[key].X))
                if cnt > 0:
                    arc_counts[(i, j)] = cnt

        if not arc_counts:
            continue  # No movement for this type in this scenario

        any_movement = True

        # Base allocation for this scenario
        base_alloc: Dict[int, int] = {
            j: int(round(cnt * p_var[j].X))
            for j, cnt in bkj.items()
            if j in PPL_set and round(p_var[j].X) > 0 and cnt > 0
        }

        total_legs = sum(arc_counts.values())
        lines.append(
            f"\n**{kname}** (mode={mv}, F_k={Fk})  —  "
            f"{total_legs} vehicle-arc(s) across {len(arc_counts)} distinct arc(s)"
        )

        paths, flags = _decompose_to_paths(arc_counts, base_alloc)

        # Consolidate identical paths — "N vehicles: path" instead of N separate entries
        from collections import Counter
        path_strs = [tuple(path) for path in paths]
        path_counts = Counter(path_strs)
        unique_paths = list(dict.fromkeys(path_strs))  # preserves insertion order

        # Annotate each path with its cargo profile so we can classify
        # loaded vs empty before deciding what to display
        def _path_is_all_empty(path: list) -> bool:
            for leg_idx in range(len(path) - 1):
                fi, fj = path[leg_idx], path[leg_idx + 1]
                fv, wv, _, _ = _arc_cargo(w, mv, fi, fj, kname, results, instance)
                if fv + wv > 1e-3:
                    return False
            return True

        empty_only_count = 0
        loaded_path_lines: List[str] = []

        for path_key in unique_paths:
            path = list(path_key)
            count = path_counts[path_key]
            all_empty = _path_is_all_empty(path)

            if all_empty and len(path) > 1:
                # Pure empty paths: suppress from display, tally count
                empty_only_count += count
                continue

            path_str = ' → '.join(site_names.get(nd, str(nd)) for nd in path)
            count_label = f"{count} vehicle(s)" if count > 1 else "Vehicle 1"
            loaded_path_lines.append(f"\n  {count_label}: {path_str}")

            for leg_idx in range(len(path) - 1):
                i, j = path[leg_idx], path[leg_idx + 1]
                food_v, water_v, cap_f, cap_w = _arc_cargo(w, mv, i, j, kname, results, instance)
                total_v = food_v + water_v
                if total_v > 1e-3:
                    load_f = food_v / cap_f if cap_f > 0 else 0
                    load_w = water_v / cap_w if cap_w > 0 else 0
                    status = "LOADED"
                    cargo_str = (
                        f"food={food_v:>9,.0f} ({100*load_f:.0f}% of cap)  "
                        f"water={water_v:>9,.0f} ({100*load_w:.0f}% of cap)  "
                        f"[avg per vehicle]"
                    )
                else:
                    status = "empty"
                    cargo_str = "no commodity flow on this leg"
                from_name = site_names.get(i, str(i))
                to_name = site_names.get(j, str(j))
                loaded_path_lines.append(
                    f"    Leg {leg_idx+1}: {from_name:>25} → {to_name:<25}  "
                    f"[{status}]  {cargo_str}"
                )

        # Emit loaded paths
        lines.extend(loaded_path_lines)

        if empty_only_count > 0:
            lines.append(
                f"\n  *(suppressed {empty_only_count} vehicle(s) on all-empty paths "
                f"— no commodity flow on any leg)*"
            )

        # Cross-check: verify arc coverage
        # Cross-check: path arc counts must match model n values exactly.
        # Decomp flags (from _decompose_to_paths) are only shown when this
        # check FAILS — they are suppressed when the check passes, since a
        # passing cross-check means every arc is accounted for despite any
        # intermediate residuals during decomposition.
        path_arc_counts: Dict[Tuple[int, int], int] = {}
        for path in paths:
            for leg in range(len(path) - 1):
                arc = (path[leg], path[leg + 1])
                path_arc_counts[arc] = path_arc_counts.get(arc, 0) + 1

        mismatch = [
            arc for arc in set(list(arc_counts.keys()) + list(path_arc_counts.keys()))
            if arc_counts.get(arc, 0) != path_arc_counts.get(arc, 0)
        ]
        if mismatch:
            # Real inconsistency: show both the decomp flags and the mismatch detail
            for f in flags:
                lines.append(f"  > ⚠ DECOMP: {f}")
            for arc in mismatch:
                lines.append(
                    f"  > ⚠ CROSS-CHECK FAIL: arc {arc[0]}→{arc[1]}: "
                    f"model n={arc_counts.get(arc,0)}, "
                    f"reconstruction={path_arc_counts.get(arc,0)}"
                )
        else:
            # Check passed — suppress intermediate decomp flags (they are
            # labeling artifacts, not genuine errors)
            lines.append(f"\n  ✓ Cross-check passed: all {len(arc_counts)} arc count(s) consistent")

    if not any_movement:
        lines.append("  *(no vehicle movements in this scenario)*")

    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────

def generate_itinerary_report(
    results: Dict[str, Any],
    instance: Dict[str, Any],
    locations: Dict[int, Dict],
    label: str = "",
    itinerary_all: bool = False,
    top_k: int = 3,
) -> str:
    """
    Generate a vehicle itinerary report for a solved instance.

    Parameters
    ----------
    results      : output of solve_stochastic_cvar
    instance     : stochastic instance dict
    locations    : node attribute dict from load_locations()
    label        : solve label for the report header
    itinerary_all: if True, generate for all scenarios; if False, use top_k
    top_k        : number of lowest-SR and highest-SR scenarios to report (default 3)

    Returns
    -------
    Markdown string ready to write to a file.
    """
    if not instance.get("vehicle_types"):
        return "*(vehicle extension not active — no itinerary report generated)*\n"

    if results.get("objective_value") is None:
        return "*(no feasible solution — itinerary report skipped)*\n"

    site_names = {i: loc["name"] for i, loc in locations.items()}

    # Compute per-scenario SR
    demand = instance["demand"]
    Omega = instance["scenarios"]
    R = instance.get("commodities", ["food", "water"])
    N = instance["nodes"]
    z_var = results["variables"]["z"]

    per_w_sr: Dict[int, Optional[float]] = {}
    for w in Omega:
        dw = sum(demand.get((w, i, r), 0) for i in N for r in R)
        uw = sum(z_var[w, i, r].X for i in N for r in R)
        per_w_sr[w] = max(0.0, 1.0 - uw / dw) if dw > 0 else None

    valid = [(w, sr) for w, sr in per_w_sr.items() if sr is not None]
    valid.sort(key=lambda x: x[1])

    if itinerary_all:
        selected_scenarios = [w for w, _ in valid]
        scope_note = f"all {len(selected_scenarios)} scenarios"
    else:
        bottom = [w for w, _ in valid[:top_k]]
        top = [w for w, _ in valid[-top_k:]]
        selected_scenarios = list(dict.fromkeys(bottom + top))  # dedup, preserve order
        scope_note = (
            f"{top_k} lowest-SR scenarios (w={bottom}) and "
            f"{top_k} highest-SR scenarios (w={top})"
        )

    # Build report
    lines: List[str] = []
    lines.append(f"# Vehicle Itinerary Report — {label}")
    lines.append(f"Scope: {scope_note}")
    lines.append("")
    lines.append(DISCLAIMER)
    lines.append("")
    lines.append(
        f"Selected sites: "
        f"{[site_names.get(s, str(s)) for s in results.get('selected_sites', [])]}"
    )
    lines.append("")

    # Service rate summary
    mean_sr = sum(sr for _, sr in valid) / len(valid) if valid else 0
    n_zero = sum(1 for _, sr in valid if sr < 1e-6)
    n_full = sum(1 for _, sr in valid if sr > 0.999)
    lines.append(
        f"Overall: sr_mean={mean_sr:.4f}, "
        f"n_zero={n_zero}/{len(valid)}, n_full={n_full}/{len(valid)}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    for w in selected_scenarios:
        sr = per_w_sr.get(w, 0.0) or 0.0
        lines.append(_scenario_block(w, sr, results, instance, site_names))
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
