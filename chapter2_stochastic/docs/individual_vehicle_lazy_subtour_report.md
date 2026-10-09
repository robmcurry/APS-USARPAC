# Individual Vehicle Indexing + Lazy Subtour Elimination: Tractability Investigation

**Repo:** `aps_usarpac/model/model.py` · **Branch:** `individual-vehicle-indexing`
**Network:** 50 nodes, 1,502 air arcs · **Real fleet:** C-17×12, C-130J×16 · **Solver:** Gurobi (academic license)

A working log of implementing per-vehicle-instance indexing for air-mode fleets (C-17, C-130J) with Gurobi lazy-constraint subtour elimination, and the resulting tractability wall at real fleet sizes on the full 50-node network — four mitigations attempted, all converging on the same diagnosis.

---

## Summary for review

**What works:** the individual-vehicle formulation (per-instance binaries `n_ind[w,k,l,i,j]`) and its lazy-callback subtour elimination are implemented, correct, and validated — confirmed on a toy network with a genuine non-home cycle (3 phantom double-booked instances with no mitigation → 0 with the callback active), and confirmed cheap (callback overhead never exceeded 1% of wall-clock in any run, including the 3-hour one).

**What doesn't work yet:** at real fleet sizes (C-17=12, C-130J=16 → 28 individually-indexed instances) on the real 50-node/1,502-air-arc network, the extensive-form MIP cannot find *any* feasible solution in 45 minutes, and only reaches depth 5 of the branch-and-bound tree (49 nodes, best bound essentially frozen) after a full 3 hours. A parallel run at the toy-tested fleet size (F<sub>k</sub>=1) finds 4 solutions and 12.09% gap in the same 45 minutes — so the wall is fleet-size-specific, not a general property of the network or the callback mechanism.

**Two mitigations tried, both ineffective:** scalar symmetry-breaking constraints (ordering same-home-node instances by total arc count) and Gurobi's `MIPFocus=1`. Neither meaningfully moved the best bound.

> **⚠ UPDATE (Run 6, added after initial circulation): β=0.5 was independently compounding the difficulty.** All five runs above (and the diagnosis drawn from them) used β=0.5 — a diagnostic override, not the real locked value β=0.90. A sixth run repeating Run 1's exact plain-baseline configuration (no symmetry-break, no MIPFocus) with **only β changed to 0.90** found **2,731 branch-and-bound nodes at depth 108**, versus Run 1's **1 node at depth 0** — in the identical 45-minute budget. See §3.5 below. This does not overturn the size/scale findings (§5), but it means the "intractable" conclusion in §6 was drawn entirely under the hardest tail-weighting tested, and needs qualification: the real production configuration may behave qualitatively differently.

> **⚠ UPDATE #2 (Runs 7–8): the β=0.90 question raised above is now answered, and the answer is negative.** Run 6 was repeated from a fresh terminal (Run 7, §3.6) and reproduced almost exactly (731 lazy cuts and 9 MIPSOL calls in both, node count within 4%), confirming Run 6 wasn't a fluke. It was then extended to the full 3-hour budget (Run 8, §3.7): **13,206 nodes explored, still zero incumbents, best bound moved only ~0.024% across the entire 3 hours.** The node-exploration rate is *not* a clean accelerating curve as Run 6's 45-minute window suggested — the 3-hour run shows a fast initial climb, then a ~52-minute near-stall, then an uneven recovery of bursts and slow stretches. Extending the time budget further is not a productive lever on its own; the reformulation directions in §7 are the more promising path. §6's Finding #1 (weak root relaxation) is now confirmed to hold at the real β=0.90 value, not just β=0.5.

---

## 1. Pipeline & what changed

```
pacific_cities.csv / network/nodes.csv
        |
        v
network_builder.py  (50-node graph, modal arcs)
        |
        v
scenario_generator.py  (N scenarios, seed)
        |
        v
input_builder.py  build_stochastic_instance() + build_vehicle_params()
        |
        v
model.py  solve_stochastic_cvar()
        |
        +-- vehicle_formulation="aggregate" (unchanged)
        |     -> n[w,k,m,i,j] integer vehicle-count (locked baseline)
        |     -> model.optimize()  (unchanged, Method=2)
        |
        +-- vehicle_formulation="individual" (this branch)
              -> n_ind[w,k,l,air,i,j] per-instance binary
              -> _build_subtour_callback()  (Gurobi LazyConstraints, cbLazy SEC cuts)
              -> VehicleSymmetryBreak  (scalar ordering, new)
              -> model.optimize(callback), MIPFocus=1 (new)
```

### Code changes this session (`model/model.py`)

- **`_build_subtour_callback()`** — new module-level function. On every Gurobi `MIPSOL`, batches `cbGetSolution` over all `n_ind`, does forward BFS reachability from each vehicle instance's fixed home node, groups any unreached selected arcs into weakly-connected components, and adds a direct DFJ-style subtour-elimination cut per component:
  `sum(n_ind[w,k,l,air,a,b] for a,b in S) <= |S| - 1`
- **Default flip:** `_debug_skip_departure_single_node` default changed `False → True` — the static `DepartureSingleNode`/`DepartureNodeLink` constraints (5) are now off by default; the lazy callback is the sole subtour-elimination mechanism going forward. Passing `False` restores the static constraint for A/B comparison; the callback still attaches independently either way.
- **`VehicleSymmetryBreak`** (new constraint family) — for each (type, home node) group with ≥2 interchangeable instances, orders them by total outbound air-arc count: `total_outbound[w,k,l] <= total_outbound[w,k,l+1]`. Provably preserves at least one optimum (any solution can be relabeled within an interchangeable group to satisfy the order).
- **`MIPFocus=1`** — set only when `vehicle_formulation="individual"`, alongside the existing `LazyConstraints=1`. Does not touch the aggregate branch.
- All four changes scoped strictly to `vehicle_formulation="individual"`; **aggregate regression: ALL PASS** re-confirmed after every change (objective, sites, gap, and all 19 variable/constraint family counts match the locked `baseline_aggregate_toy.json` baseline exactly, every time).

### Code changes for Runs 7–8 (`model/model.py`)

- **`_debug_node_log_interval_sec`** (new, default `None`) — when set, `_build_subtour_callback`'s returned callback also handles `GRB.Callback.MIP` (Gurobi's periodic B&B polling point, independent of `MIPSOL`) and appends `{elapsed_sec, node_count, best_bound, sol_count}` to `stats["node_log"]` roughly every N seconds, so a long run's node-exploration rate can be inspected over time rather than only at completion. Zero effect when unset.
- **`_debug_incumbent_flag_path`** (new, default `None`) — when set, writes a one-line marker file the instant a MIPSOL candidate survives its own callback call with zero lazy cuts added (i.e. a genuine accepted incumbent), so an external process can detect the event without waiting for `model.optimize()` to return. **Caught and fixed a bug in the first version of this**, which flagged on the *first MIPSOL invocation seen* rather than the first one that wasn't itself cut — see the Run 8 write-up (§3.7) for how this was caught and corrected mid-investigation.
- **Graceful `SIGTERM` handling** — when the lazy callback is active, `solve_stochastic_cvar` now installs a `SIGTERM` handler around `model.optimize()` that calls `model.terminate()` instead of letting Python's default handler kill the process outright. Lets an external memory-pressure watchdog (`scripts/memory_watchdog.py`, new) stop a run cleanly and still get usable partial results/status, instead of an uncontrolled kill.
- All three changes scoped to the individual-formulation lazy-callback path only; **aggregate regression: ALL PASS** and the toy individual smoke test re-confirmed clean after each.

---

## 2. Toy-scale correctness validation

All on the 4-node/5-scenario toy instance (`toy_vehicle_test.py`), before any real-network spend:

| Test | Config | Result | Verdict |
|---|---|---|---|
| Smoke test | F_k=1, callback on | OPTIMAL, 0.43% gap, 339 vars / 672 constrs | clean |
| fk2 departure test (a) | F_k=2, static constr. 5 ON + callback | obj=899,868,645 · 0 phantom | clean |
| fk2 departure test (b) | F_k=2, static constr. 5 OFF, DAG network | obj=870,431,613 · 0 phantom | DAG — no cycle to test |
| Cyclic test (no mitigation) | F_k=2, added 2↔3 cycle, no constraint, no callback | **3 phantom/disconnected instances** | bug confirmed |
| Cyclic test (callback) | F_k=2, same cyclic net., callback only | obj=866,362,177 · **0 phantom** (1 cut, 3 MIPSOL calls) | fixed |
| Feasibility probe | force n_ind onto phantom pair, constr. 5 off | Gurobi status: FEASIBLE | proves constr. 3 alone insufficient |
| Symmetry-break check | F_k=2, cyclic net. + VehicleSymmetryBreak | 10 new constrs, obj Δ<0.001%, **0 phantom** | clean |

The cyclic test is the load-bearing proof: it added a return arc (3,2) opposite the existing (2,3), giving one non-home 2↔3 cycle. With no subtour-elimination mechanism, the solver freely double-booked 3 vehicle instances onto that cycle for zero home-departure cost — e.g. instance `w=0, k=C-130J, l=1` showed selected arcs `{(1,4), (2,3), (3,2)}` where only `(1,4)` was reachable from home node 1. The lazy callback eliminates this with a single cut.

---

## 3. Real-network experiment log

50 nodes, 1,502 air arcs, N=10 scenarios, seed=32, β=0.5 (diagnostic override — see note below) unless stated.

### Run 1 — Baseline lazy callback · real fleet (F_k=12/16) · 45-min cap

First real-scale test of the individual formulation. Problem size alone was the first surprise: **493,323 variables (420,582 binary)** from `n_ind` and 125,222 constraints — two to three orders of magnitude past anything toy-tested.

| Status | SolCount | Nodes | Best bound | Wall-clock | Peak RSS | Lazy cuts | MIPSOL calls | Callback time |
|---|---|---|---|---|---|---|---|---|
| TIME_LIMIT | 0 | 1 | 1.7643e9 | 2703.1s | 1953 MB | 1163 | 12 | 2.26s / 2700s |

### Run 2 — Fleet-size isolation diagnostic · F_k=1/1 · 45-min cap

Same network, same N, same everything — only C-17/C-130J fleet size dropped to the toy-tested minimum (1 each → 2 instances instead of 28). Tests whether the wall is network density or fleet-size combinatorics.

| Status | SolCount | Nodes | Gap | Best bound | Wall-clock | Peak RSS | Subtour check |
|---|---|---|---|---|---|---|---|
| TIME_LIMIT | **4** | **306,544** | 12.09% | 2.2638e9 | 3363.1s (overshot) | 3953 MB | 0 phantom / 4 |

**Conclusion:** the formulation and callback scale fine — the wall is fleet-size-specific, not network density or a callback defect.

### Run 3 — + VehicleSymmetryBreak · real fleet (F_k=12/16) · 45-min cap

Hypothesis: permutation symmetry among 12 interchangeable C-17s / 16 interchangeable C-130Js (many sharing a home node) is blowing up the B&B tree. Added scalar ordering constraints scoped per (type, home-node) group (200 new constraints, confirmed via row-count delta).

| Status | SolCount | Nodes | Best bound | Wall-clock | Peak RSS | Lazy cuts | MIPSOL calls |
|---|---|---|---|---|---|---|---|
| TIME_LIMIT | 0 | 1 | 1.7640e9 | 2703.2s | 1737 MB | 884 | 8 |

**Result:** best bound essentially identical to Run 1 (Δ<0.02%). Symmetry-breaking constraints correctly wired in but had no measurable effect.

### Run 4 — + MIPFocus=1 · real fleet (F_k=12/16) · 45-min cap

Solver-level tuning to prioritize finding *any* incumbent over proving optimality, in place of the symmetry-break constraints.

| Status | SolCount | Nodes | Best bound | Wall-clock | Peak RSS | Lazy cuts | MIPSOL calls |
|---|---|---|---|---|---|---|---|
| TIME_LIMIT | 0 | 1 (2 children created in final seconds) | 1.7635e9 | 2703.2s | 1618 MB | 399 | 3 |

**Result:** first sign of life — branching began, but only in the last instant of the 45-minute budget.

### Run 5 — + MIPFocus=1, extended · real fleet (F_k=12/16) · 3-hour cap, memory-monitored

One-off exploratory run (4× budget) to see whether Run 4's late branching continues given more time. First attempt self-aborted safely at 526s on system-wide memory pressure (unrelated to this job — see §5); relaunched after confirming headroom. Full run completed cleanly, no aborts.

| Status | SolCount | Nodes | Best bound | Wall-clock | Peak RSS | Lazy cuts | MIPSOL calls |
|---|---|---|---|---|---|---|---|
| TIME_LIMIT | 0 | **49** (depth 5) | 1.7636e9 | 10,803.2s (3.00 hr) | 5,207 MB | 399 (unchanged) | 3 (unchanged) |

**Result:** 2.7 additional hours of branching bought 5 levels of tree depth and zero incumbents; best bound moved <0.01% past Run 4. Extrapolated convergence rate is impractical (see §4 and §6) — **under β=0.5**. See Run 6 immediately below for the same test at the real β value.

### Run 6 — β=0.90 (real value), otherwise identical to Run 1 · 45-min cap

> **This is the most consequential single result in this investigation.** Isolated single-variable test: exact Run 1 configuration (plain lazy-callback baseline, `_debug_skip_symmetry_break=True`, `_debug_mip_focus=0` — i.e. neither of the two later mitigations active), F_k=12/16, N=10, seed=32 — **with only β changed from 0.5 to 0.90.**

| | Run 1 (β=0.5) | Run 6 (β=0.90) |
|---|---|---|
| Status | TIME_LIMIT, SolCount=0 | TIME_LIMIT, SolCount=0 |
| **Nodes explored** | **1** (never left root) | **2,731** |
| Max depth | 0 | **108** |
| Best bound | 1.7643e9 | 5.0278e9 |
| Wall-clock | 2703.1s | 2703.0s |
| Peak RSS | 1,953 MB | 4,751 MB |
| Lazy cuts | 1,163 | 731 |
| MIPSOL calls | 12 | 9 |
| It/Node (late-run) | n/a (never branched) | fell from ~142,723 to ~2,380 |

**Root-relaxation phase is beta-independent.** Both runs spend ~700–750s grinding through root cutting planes with IntInf in the 5,000+ range — essentially identical in character. The divergence happens the instant branching starts: at β=0.5 the first branch (`0 2` at 2,690s) was the *last* thing that happened in the whole 45 minutes; at β=0.90 the first branch happened at 751s and the tree kept growing for the remaining ~32.5 minutes.

**Node-exploration rate within Run 6: accelerating, not plateauing.** Splitting the branching phase into two clean halves (to avoid noise from Gurobi's uneven log-print cadence):

| Window | Duration | Nodes gained | Rate |
|---|---|---|---|
| 751s → 1,771s (1st half) | ~17.0 min | 781 | 45.9 nodes/min |
| 1,771s → 2,700.2s (2nd half) | ~15.5 min | 1,950 (to official final count 2,731) | **125.8 nodes/min** |

The second half ran at ~2.7× the first half's rate — a real acceleration, not an artifact of a single noisy sample. This means extending the time budget is likely to keep producing nodes at a similar-or-better clip; the healthy branching itself shows no sign of stalling.

**One important caveat:** node/depth growth accelerating does not mean the search is proportionally closer to an incumbent. The IntInf floor (integer infeasibility count at the best node) barely moved — mostly 5,000–5,400 throughout, dipping to only 4,813 by the end (~11% reduction over the full branching phase). The tree is being explored faster and faster, but the feasibility-distance signal isn't (yet) showing the same acceleration. **Still no incumbent was found even at β=0.90 within 45 minutes.**

**Implication:** every finding in §6 (Diagnosis) below was drawn entirely under β=0.5, the hardest tail-weighting tested. Whether the same weak-root-relaxation diagnosis holds at β=0.90 given more time — or whether β=0.90 eventually produces a feasible solution and a usable gap on a similar timescale to the F_k=1 result — is now the open question a longer β=0.90 run (not yet performed) would answer.

### Run 7 — β=0.90 repeat (fresh terminal) · 45-min cap

Run 6 was launched from a PyCharm terminal that was later closed; before extending the time budget, it was relaunched from a fresh terminal session to confirm it reproduces rather than chasing a one-off.

| | Run 6 (original) | Run 7 (repeat) |
|---|---|---|
| Status | TIME_LIMIT, SolCount=0 | TIME_LIMIT, SolCount=0 |
| Nodes explored | 2,731 | 2,846 |
| Best bound | 5.0278e9 | 5.02777e9 |
| Lazy cuts | 731 | **731 (identical)** |
| MIPSOL calls | 9 | **9 (identical)** |
| Peak RSS | 4,751 MB | 7,027 MB |
| Wall-clock | 2703.0s | 2703.1s |

Identical lazy-cut and MIPSOL counts, near-identical best bound — a faithful reproduction (node-count variance of ~4% is normal run-to-run noise in Gurobi's parallel B&B thread scheduling, not a different outcome). Peak RSS ran meaningfully higher this time (7.0 vs 4.75 GB) with no other behavioral difference; still far under any safety threshold.

Node-exploration rate, sampled every 5 minutes this time (new instrumentation — `_debug_node_log_interval_sec`): nothing happens for the first ~13 minutes (root cutting planes), then branching starts and continues at a noisy but roughly increasing pace through the 45-minute cap, consistent with Run 6's finding. One instrumentation caveat surfaced here: the 5-minute sampler only fires inside Gurobi's periodic `MIP` callback poll, so a burst of nodes in the final ~90 seconds before the cap (1,756 → 2,846 nodes) wasn't resolved at finer granularity — real, but its exact shape within that window is unknown.

### Run 8 — β=0.90 · 3-hour extension

Motivated directly by Run 6/7's finding that the node rate was still accelerating, not plateauing, at the 45-minute cutoff. Same configuration as Run 7 (F_k=12/16, N=10, seed=32, lazy callback, no symmetry-break, no MIPFocus override, β=0.90), TimeLimit extended to 10,800s (3 hours). Launched as a detached background process (survives terminal closure) with an independent external memory watchdog (`scripts/memory_watchdog.py`) sampling system-wide pressure + process RSS every 15s, empowered to send a graceful `SIGTERM` (routed to `model.terminate()`) if pressure escalated. It never needed to — pressure stayed "normal" for the entire run.

**A methodology note, reported transparently:** partway through this run, a marker file appeared claiming an incumbent had been found at 178s (obj=5.0459e9). Cross-checking the live Gurobi log showed the "Incumbent" column was still `-` (SolCount=0) more than 25 minutes later — impossible if that had been a real accepted incumbent. The bug: the flag-write logic fired on the *first MIPSOL callback invocation seen*, but a MIPSOL candidate that gets cut via `cbLazy` in that same callback call is rejected by Gurobi, not accepted — it never becomes the incumbent. Fixed in `model.py` (flag now only fires when a callback call adds zero cuts) and reconfirmed against the aggregate regression + toy smoke test; the false-positive file from this run was set aside rather than trusted. No genuine incumbent flag fired for the remainder of the run.

**Final result:**

| | Run 6 (45 min) | Run 7 (45 min) | **Run 8 (3 hr)** |
|---|---|---|---|
| Status | TIME_LIMIT, SolCount=0 | TIME_LIMIT, SolCount=0 | **TIME_LIMIT, SolCount=0** |
| Nodes explored | 2,731 | 2,846 | **13,206** |
| Best bound | 5.0278e9 | 5.02777e9 | **5.02870e9** |
| Lazy cuts | 731 | 731 | **1,253** |
| MIPSOL calls | 9 | 9 | **16** |
| Peak RSS | 4,751 MB | 7,027 MB | **9,461 MB** |
| Incumbent found | No | No | **No** |

**The node-exploration rate is bursty, not cleanly accelerating** — the "accelerating, not plateauing" read from Run 6/7 held only over the short window those runs covered:

1. **0–12 min:** 0 nodes (root cutting planes).
2. **12–58 min:** fast acceleration, 0 → 8,770 nodes, peaking around ~500 nodes/min.
3. **~58–110 min (52 minutes):** a long near-stall — only 8,770 → 9,429 nodes, **~12.7 nodes/min**, a ~40× slowdown from the peak rate. CPU stayed active throughout (this is expensive individual branch nodes, not a hang).
4. **110–180 min:** uneven recovery — bursts of 60–90 nodes/min alternating with slow stretches of 8–16 nodes/min, ending at 13,206 total nodes.

**What 4× the time budget bought:** 4.8× more nodes explored, but the best bound moved only **~0.024% in total across the full 3 hours** (~0.018% of that beyond the 45-minute mark already captured by Run 6/7). This directly answers the question Run 6 left open (§7, "Extend the β=0.90 test"): extending the time budget further is not a productive lever on its own, at either β value. Finding #1 in §6 (weak root/LP relaxation relative to the true integer hull) is now confirmed to hold at the real β=0.90 configuration, not just the β=0.5 diagnostic runs — see the updated scope note in §6.

### Note on β=0.5 (Runs 1–5)

All five runs above use β=0.5 (CVaR tail = 5 of 10 scenarios) rather than the locked production value β=0.90 (tail = 1 scenario at N=10), specifically to generate genuine routing pressure across half the instance rather than concentrating all risk-aversion weight on a single scenario. **These results are not comparable to locked β=0.90 baselines** and are a lower bound on what β=0.90 or larger N would show, not a direct estimate. Staged runs at β=0.90 (N=25, N=50 — "Step B/C") were never reached; Step A itself never cleared the review gate.

---

## 4. Best bound & branch-and-bound progress (Run 5, 3-hour)

The clearest single picture of the stall: root relaxation converges in ~60s, then 3 hours of work moves the bound by ~0.4% total.

| Elapsed | Best bound | Nodes explored | Note |
|---|---|---|---|
| 60s | 1.75685e9 | 0 | root relaxation complete |
| 529s | 1.7594e9 | 0 | root cutting planes |
| 974s | 1.7631e9 | 0 | root cutting planes |
| 1910s | 1.7634e9 | 0 | root cutting planes, still node 0 |
| 2,690s (45 min) | 1.7635e9 | 0 (2 children created) | branching begins — this is where the 45-min cap cuts off |
| 3,643s | 1.7635e9 | 1 | depth 1 |
| 4,325s | 1.7635e9 | 3 | depth 2 |
| 5,770s | 1.7635e9 | 7 | depth 3 |
| 7,052s | 1.7635e9 | 15 | depth 4 |
| 10,482s | 1.7635e9 | 27 | depth 5 |
| 10,800s (3.00 hr) | 1.7636e9 | 39 (49 total explored) | time limit reached, still no incumbent |

**RSS over the 3-hour run:** grew from ~2 GB (after root relaxation) to a peak of **5,207 MB**, fluctuating in the 3–5 GB band for the back half of the run — well under the 6 GB safety cap. System memory pressure stayed at "normal" for all 720 samples (15s interval) of this run.

---

## 5. Problem size & resource comparison

| Config | Total vars | Binary vars | n_ind count | Constraints | Peak RSS |
|---|---|---|---|---|---|
| F_k=1/1 (toy-tested minimum) | 102,543 | 30,062 | 30,040 | 111,182 | 3,953 MB |
| **F_k=12/16 (real fleet)** | **493,323** | **420,582** | **420,560** | **125,222–125,422** | **1,618–5,207 MB** |
| Ratio (real / F_k=1) | 4.8× | 14.0× | 14.0× | 1.1× | — |

The binary-variable count scales linearly with total fleet size (14× more instances → 14× more `n_ind` binaries) while the constraint count barely moves (1.1×) — the growth is entirely in the combinatorial *choice* space, not the constraint structure. This is consistent with the F_k=1 vs. F_k=12/16 outcome gap being a genuine combinatorial-scale effect rather than a numerically harder LP.

### System memory context (Run 5 relaunch)

- Total system RAM: 18.0 GB
- Free before relaunch: 2.66 GB
- Peak process RSS: 5,207 MB
- Samples at normal pressure: 720 / 720

First 3-hour attempt was safely self-terminated by an external monitoring wrapper (15s sampling of `kern.memorystatus_vm_pressure_level` + process RSS, independent of the Gurobi process) after sustained system-wide pressure escalation at 526s — the solve's own RSS at that moment was a modest 369 MB, well under any threshold, confirming the trigger was other concurrently-running applications, not this job. The design choice was a graceful `SIGTERM` via an external watchdog rather than relying on macOS's own jetsam killer, which had previously caused an uncontrolled kill on this machine. After closing other applications, the relaunch ran the full 3 hours with zero pressure events.

---

## 6. Diagnosis

> **Scope note:** the four findings below are drawn from Runs 1–5, all of which used β=0.5. Run 6 (§3.5) shows β itself is a major independent factor — at β=0.90, node exploration goes from 1 node/depth 0 to 2,731 nodes/depth 108 in the same 45 minutes. Runs 7–8 (§3.6–3.7) extended this to a full 3-hour budget: 13,206 nodes, still zero incumbents, and only ~0.024% total best-bound movement — with the node rate turning out to be bursty (fast climb, ~52-min near-stall, uneven recovery) rather than cleanly accelerating. **Finding #1 (weak root relaxation) is now confirmed to hold at β=0.90 over a full 3-hour run, not just β=0.5** — that part is structural to the formulation, not an artifact of the diagnostic β override. Findings #2–4 (callback is cheap, weak symmetry-breaking doesn't help, MIPFocus reveals-but-doesn't-break the wall) were only tested at β=0.5 and have not yet been re-examined at β=0.90.

Four independent lines of evidence, gathered across five real-network runs (all at β=0.5 — see scope note above), all point to the same root cause:

1. **The root LP relaxation is weak, not the branching or the callback.** IntInf (integer infeasibility count) sits in the 4,000–6,500 range through nearly all of every run, including the 3-hour one, and improves only marginally with hundreds of cutting planes (Gomory, MIR, StrongCG, Flow cover, Zero half, RLT). The bound moves ~0.4% total across 45 minutes of pure cutting-plane work at the root, and a further <0.01% across 2.7 more hours of actual branching.

2. **The callback is not the bottleneck.** In every run, callback time is ≤1% of wall-clock (2.26s / 2700s in Run 1; 27.87s / 3362s even in the F_k=1 run with many more MIPSOL hits; 16.10s / 10800s in Run 5). MIPSOL invocations stayed in the single-to-low-double digits even over 3 hours (3 calls in Run 5) — the solver simply isn't generating integer-feasible candidates often enough for the callback to matter.

3. **Symmetry-breaking (scalar version) didn't help.** Run 3's 200 additional constraints, correctly derived and verified to preserve optimality, produced a best bound statistically indistinguishable from Run 1. Two plausible explanations, not mutually exclusive: Gurobi's own presolve already captures a comparable amount of this symmetry, or a single scalar inequality per adjacent pair is too weak relative to the true combinatorial structure to matter at this scale. A full lexicographic ordering on the entire arc-incidence vector was not tried — it is a strictly stronger (and strictly more expensive to formulate) cut than what was tested here.

4. **MIPFocus=1 revealed the shape of the wall without breaking it.** It was the only variant across five runs to make the solver leave the root node at all. But the resulting branching is extraordinarily expensive — 100,000–1,400,000 simplex iterations per node — and 2.7 hours only reached depth 5. At that rate, reaching a depth sufficient to discover feasible integer solutions, let alone a usable optimality gap, is many hours to days away, not a modest budget increase.

Put together: the individual-vehicle formulation, the lazy-callback subtour elimination, and the overall pipeline are all functioning exactly as designed — the F_k=1 result (306,544 nodes, 4 solutions, 12.09% gap, all in 45 minutes) proves that conclusively. What's missing is *formulation strength* at the real fleet scale: something about how 28 individually-indexed, largely-interchangeable vehicle instances relate to a 1,502-arc near-complete graph is producing a root relaxation that's far looser than the true integer hull, and neither of the two mitigations tried here (weak symmetry-breaking, solver-focus tuning) closes that gap.

---

## 7. Open directions for discussion

Framed as discussion points for review, not as a settled recommendation — ranked roughly by how directly each targets the diagnosis above.

### ~~Extend the β=0.90 test before pursuing formulation changes~~ — DONE (§3.7), answer is negative
*Targeted: determining whether a reformulation is even necessary*

**Resolved by Run 8.** A 3-hour β=0.90 extension explored 13,206 nodes (4.8× Run 6/7's count) but moved the best bound only ~0.024% in total and never found an incumbent. The node rate is not a clean accelerating curve either — it includes a ~52-minute near-stall. Extending the time budget further is not a productive lever on its own, at either β value. The column-generation/pattern/Benders directions below are necessary, not optional, if this formulation is to reach a usable gap at real fleet scale.

### Column generation / Dantzig–Wolfe on a per-vehicle-instance pricing problem
*Targets: root relaxation strength*

Each individual vehicle instance's route is a resource-constrained path (distance budget D_k, home-node start) — exactly the structure column generation is built for. A master problem selecting route "columns" per instance, with a shortest-path-like pricing subproblem enforcing the distance budget, would sidestep the compact-formulation's weak LP relaxation entirely; subtour elimination becomes free (a column is a path by construction, not a set of arcs that can form a cycle). This is the standard approach for vehicle-routing-with-fleet-constraints problems of this shape in the OR literature, and is the most direct answer to finding #1 above.

### Pattern/group formulation instead of full individual identity
*Targets: whether individual indexing is even necessary*

Since instances within a (type, home-node) group are fully interchangeable in the objective, the model may not need labeled identities at all — only a count of how many instances from each group follow each feasible route pattern. This is essentially a bin-packing-over-patterns reformulation: closer in size to the aggregate formulation (which already solves fast) while still correctly enforcing per-instance distance budgets and turnaround rules, because those rules only depend on which pattern a vehicle follows, not which specific instance follows it. Worth scoping whether the dissertation's requirement for "individual" indexing is really about vehicle *identity*, or about correctly capturing per-vehicle resource constraints that a pattern formulation could capture just as well at far lower cost.

### Full lexicographic ordering (not yet tried)
*Targets: symmetry, more aggressively*

Run 3 tested only a single scalar inequality per adjacent same-home pair. A full lexicographic ordering on each instance's entire arc-incidence vector (fixing an arc order and requiring instance l's vector to be lexicographically ≤ instance l+1's) is a materially stronger cut, standard in the orbitope/symmetry-breaking literature for exactly this "identical items assigned to identical slots" structure. More expensive to formulate (one binary comparison chain per arc position, per pair, per scenario) and was explicitly deferred pending validation of the cheaper version first — which is now known not to be enough on its own.

### Arc-set pruning before individual indexing
*Targets: root relaxation strength, cheaply*

1,502 air arcs is close to a complete graph over 50 nodes (max 2,450). Many of these arcs are almost certainly dominated on cost/distance grounds for any given origin — a k-nearest-neighbor or cost-percentile pre-filter (keep, say, each node's cheapest 10–15 outbound arcs) could shrink `n_ind`'s key space by an order of magnitude before any MIP-level fix is even attempted, without necessarily excluding the true optimum if genuinely long detours are already economically dominated in this cost structure. Cheapest of the ideas here to test, and orthogonal to all the others — worth trying first, or in combination.

### Warm start derived from the aggregate formulation
*Targets: getting a feasible solution, even if not optimal*

The aggregate formulation already solves fast (locked baseline: 274 vars, sub-second). Splitting its `n[w,k,m,i,j]` integer counts into a naive per-instance assignment (e.g. round-robin instances onto the aggregate's chosen arcs) would give Gurobi's MIP start mechanism a complete, feasible incumbent immediately — converting this from a "prove any solution exists" search into an "improve this solution" search, which behaves very differently under `MIPFocus=1`. Untried; flagged as the natural next experiment if the mitigations above are judged too large a scope change for now.

### Benders decomposition on site selection vs. routing
*Targets: decomposing the problem structurally*

The existing lazy-callback mechanism is already conceptually a Benders-style cut (infeasibility in the subproblem → cut added to the master). A fuller Benders split — first-stage `p[i]` site selection as master, all vehicle routing (including subtour elimination) as subproblem generating both feasibility and optimality cuts — is a heavier restructuring than anything tried here, but may be the most scalable long-term answer if column generation proves too large a rewrite for the current timeline.

---

## Appendix: reproducibility

| Artifact | Path |
|---|---|
| Core implementation | `model/model.py` |
| Toy instance builder (F_k override) | `toy_vehicle_test.py` |
| Smoke / departure / cyclic toy tests | `scripts/toy_individual_{smoke,fk2_departure,cyclic}_test.py` |
| Staged real-network runner (Steps A/B/C + F_k=1 diagnostic) | `scripts/lazy_subtour_staged_run.py` |
| Aggregate regression guard | `scripts/check_aggregate_regression.py` |
| Run 1 log | `output/lazy_subtour_stepA_N10_seed32.log` (pre-symmetry-break copy) |
| Run 2 log (F_k=1) | `output/lazy_subtour_stepA_fk1diag_N10_seed32.log` |
| Run 5 log + memory trace (3hr) | `output/_probe_stepA_mipfocus_3hr.log`, `_probe_stepA_mipfocus_3hr_memlog.csv` |
| Run 6 log (β=0.90 isolation) | `output/lazy_subtour_stepA_beta090_N10_seed32.log` |
| Run 7 runner + log (β=0.90 repeat, 45 min) | `scripts/lazy_subtour_run6_repeat.py`, `output/lazy_subtour_run6_repeat_N10_seed32_beta090.log`, `output/lazy_subtour_run6_repeat_summary.txt` |
| Run 8 runner + log (β=0.90, 3 hr) | `scripts/lazy_subtour_run6_3hr.py`, `output/lazy_subtour_run6_3hr_N10_seed32_beta090.log`, `output/lazy_subtour_run6_3hr_summary.txt` |
| External memory watchdog (Run 8) | `scripts/memory_watchdog.py`, `output/_run6_3hr_memwatchdog.csv` |
| Locked aggregate baseline | `output/baseline_aggregate_toy.json` |

All real-network runs: N=10 scenarios, seed=32, real 50-node network via `network/network_builder.py` + `scenarios/scenario_generator.py` + `model/input_builder.py`, `mip_gap` from config default (1%). Vehicle fleet sizes and basing logic from `config/model_parameters.yaml` `vehicles:` block, unchanged except where explicitly overridden (F_k=1 diagnostic).

**Debug toggles added for Run 6** (`model/model.py`, both default to preserving current/post-Run-5 behavior): `_debug_skip_symmetry_break: bool = False` (pass `True` to omit the `VehicleSymmetryBreak` constraint family) and `_debug_mip_focus: int = 1` (pass `0` to restore Gurobi's default balanced focus instead of feasibility-first). Together with the pre-existing `_debug_skip_departure_single_node`, these let any prior run's exact configuration be reproduced for single-variable A/B comparison.

**Debug toggles added for Runs 7–8** (`model/model.py`, both default `None` / no effect): `_debug_node_log_interval_sec: float = None` (periodic node-count/best-bound/sol-count logging inside the lazy callback, at the given interval) and `_debug_incumbent_flag_path: str = None` (writes a marker file the instant a MIPSOL candidate is accepted with zero lazy cuts added — i.e. a genuine incumbent, not just any integer-feasible candidate Gurobi happened to find). `solve_stochastic_cvar` also now routes `SIGTERM` to `model.terminate()` whenever the lazy callback is active, so `scripts/memory_watchdog.py` (or any external monitor) can stop a run cleanly.
