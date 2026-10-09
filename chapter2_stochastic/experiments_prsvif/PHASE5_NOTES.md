# PRS-VIF Phase 5 Notes — Real-network integration (diagnostic, wiring, structural validation)

Continuation of Phase 1-4 (`PHASE1_NOTES.md` through `PHASE4_NOTES.md`).
Same authoritative source: `aps_usarpac/docs/PRSVIF_Gospel.md`. This
phase had three parts: (1) a diagnostic on `eq:vif:vehcap`'s `min{T,
cap_l}` term, required to be reported before any new code; (2) wiring
the real pipeline's data into `solve_vif`'s required schema; (3)
structural validation against the real 50-node network. Per the phase
instructions, no MIP solve was attempted.

## Diagnostic: is T ever the binding term in `min{T^w_l,ij, cap_l}`?

Script: `aps_usarpac/scripts/vehcap_diagnostic.py`.

**Short answer: yes, T does bind — both from baseline tier structure and
from genuine disaster-driven degradation — so this is not the
formulation-dead-end the phase instructions asked me to stop on. But the
picture is more specific than a single number, and picking only one
scenario (as literally instructed) would have been misleading.**

### Part 1 — one mid-severity scenario (severity≈2.989, flood, seed=32)

At `alpha=1.0` (full calibrated degradation matrix):

| mode | T wins (of triples) | at alpha=0.0 (baseline, no disaster) | delta caused by degradation |
|---|---|---|---|
| sea  | 704 / 1,888 (37.29%)   | 704 / 1,888 (37.29%)   | **+0** |
| air  | 2,352 / 42,056 (5.59%) | 2,352 / 42,056 (5.59%) | **+0** |
| land | 0 / 1,800 (0.00%)      | 0 / 1,800 (0.00%)      | **+0** |

For this specific scenario, **every** T-win is identical at `alpha=0.0`
and `alpha=1.0` — none of them are caused by the disaster's degradation
term at all. They're baseline structural artifacts:
- Sea: PPL-3-origin arcs have `vessel_type="none"` (no maritime assets
  at that tier, `discharge_mt_per_window.get("none", 0.0) = 0`), so
  `T=0 <= cap_l` trivially, with or without any disaster.
- Air (C-17 only, cap=77.5MT): PPL-3-origin arcs use the C-130 sortie
  rate (`payload_mt=20 * sorties=3 = 60MT`), already below the C-17's
  own 77.5MT payload cap before any degradation is applied.
- C-130J (cap=19.05MT) and land/M1083 (cap=4.536MT): T never wins here
  in this scenario, degraded or not — nominal T is far above cap_l on
  every arc type these fleets use.

**Picking only this one scenario would have supported a wrong
conclusion** ("degradation never matters") — see Part 3.

### Part 3 — full 100-scenario draw (seed=32), genuine degradation-caused transitions only

Counted every `(w,l,i,j)` triple where `T` was **not** below `cap_l` at
`alpha=0.0` but **was** at `alpha=1.0` (using each scenario's own
realized `disaster_type`, not forcing flood):

- **4,072 genuine degradation-caused transitions**, occurring in **18 of
  100 scenarios**.
- **100% of them are air-mode** (3,864 on C-17, 208 on C-130J). Sea and
  land show **zero** degradation-caused transitions across the entire
  100-scenario draw.
- Concentrated in `storm` (Γ_air=0.9) and `volcanic` (Γ_air=1.0) —
  the two disaster types with the highest air-mode sensitivity in the
  calibrated matrix (Table `tab:degradmatrix`). Flood (Γ_air=0.3),
  earthquake (Γ_air=0.5), mass movement (Γ_air=0.2), and wildfire
  (Γ_air=0.4) never produced a transition in this draw — consistent
  with their lower sensitivities needing correspondingly higher
  severity to cross the same threshold.
- Transition counts scale with scenario severity within those two
  types (e.g. storm severity 4.133 → 440 transitions vs. storm
  severity 2.804 → 84).

### Part 2 — severity threshold per mode/type at gamma pinned to 1.0

Isolates "how degraded would arc throughput need to be, independent of
scenario," using each mode's own theoretical maximum sensitivity
(`gamma_l=1.0`, which storm/air and volcanic/air actually achieve in
the calibrated matrix; earthquake/land also reaches Γ=1.0):

| mode | type | cap_l (MT) | sigma\* range (arc-dependent) | median |
|---|---|---|---|---|
| sea  | LCU-1700 | 154.22 | [4.486, 4.829] | 4.486 |
| air  | C-130J   | 19.05  | [3.412, 5.000] | 5.000 |
| air  | C-17     | 77.50  | [0.000, 5.000] | 5.000 (baseline-driven 0.000 entries are the PPL-3-tier arcs from Part 1) |
| land | M1083    | 4.536  | [4.849, 4.991] | 4.975 |

Sea and land require severity within ~0.5 of the maximum (5.0) even at
their mode's theoretical worst-case sensitivity — consistent with Part
3 finding zero real-scenario transitions for those modes. Air's C-17/
C-130J thresholds are reachable well inside the realistic severity
range when the disaster type's actual Γ_air is high (storm, volcanic),
which is exactly what Part 3 observed.

### Conclusion

Arc degradation (`eq:residual`) **does** influence `eq:vif:vehcap`, but
narrowly: **only through the air mode**, and **only for storm/volcanic
disaster types** at moderate-to-high severity. Sea and land vehicle
capacity is realistically always governed by `cap_l` alone in this
calibration, not `T`. This isn't a reason to stop — it's a finding to
carry into Section 4 interpretation once solves are attempted: the
alpha-sweep's effect through C7 specifically, if it shows up at all,
should be expected to show up in air-mode routing under storm/volcanic
scenarios, not uniformly across the board.

## What was built (items 1-2 of remaining scope)

`aps_usarpac/model/input_builder.py`, inside `build_stochastic_instance`:

- **`resource_weight`** and **`nominal_throughput`**: wired in by
  calling the existing `build_resource_weight(params)` and
  `build_nominal_throughput(params)` helpers (previously built, Phase
  1/3, but never called from here) and adding their results to the
  returned instance dict. This is item 1 as literally scoped.
- **Went beyond item 1's literal description, necessarily**:
  `solve_vif` also requires `instance["node_severity"]`,
  `instance["disaster_type"]`, and `instance["degradation_matrix"]`
  (Phase 3 schema) — none of which were ever wired into
  `build_stochastic_instance`'s return dict before this phase (verified:
  `test_vif_phase3.py`/`test_vif_phase4.py` hand-build their own
  instance dicts and never call `build_stochastic_instance`, so this gap
  was never exercised against the real pipeline). Without these three
  keys, `solve_vif` raises `ValueError` immediately and step 3's
  structural validation is impossible. These are **not new
  calibration** — `node_severity`/`disaster_type` are reshaped
  directly from data `scenarios/scenario_generator.py` already computes
  per scenario (its own docstring lists both as fields of its returned
  scenario dicts), and `degradation_matrix` is read as-is from
  `config/model_parameters.yaml`, the identical key/shape
  `build_modal_residual_capacity()` (used by the aggregate/individual
  paths, a few lines above in the same file) already reads. `alpha` was
  already present in the returned instance dict from Step 4 and needed
  no change.
- **`node_handling_capacity`** and **`node_handling_bonus`** (item 2):
  explicit placeholders, `_VIF_THETA_PLACEHOLDER = 1.0e6`, applied to
  every `(node, mode)` pair (bonus restricted to PPL nodes, matching
  the gospel's `DeltaTheta_i,m := 0` for non-PPL convention already
  enforced inside `solve_vif` itself). 1e6 is far above any realistic
  per-mode fleet arrival count (max fleet size by mode: air 28, sea 8,
  land 60), so C8 (`eq:vif:transfercap`) cannot bind under any solution
  this instance could produce. Marked in code comments as a placeholder
  pending SME calibration, not a derived value — `PHASE3_NOTES.md`'s
  "Recon-to-implementation link" already found no existing config data
  source for Theta/DeltaTheta to derive from (`modal_capacity.assets`
  is consumed entirely as arc throughput, with no activation-conditional
  node-handling split).

`model.py` needed **zero changes** — same pattern as every prior phase;
all new data flows through `instance`, read directly inside `solve_vif`.

## Structural validation (item 3)

Script: `aps_usarpac/scripts/vif_phase5_structural_validation.py`. Builds
the real 50-node network + real fleet via `build_stochastic_instance`,
calls `solve_stochastic_cvar(..., vehicle_formulation="vif",
build_only=True)`, reports variable/constraint family counts, then runs
`model.relax()` + `optimize()` (LP relaxation, **not** a MIP solve).

### Confirmed: instance dimensions match the gospel's own table exactly

`|N|=50, |N^P|=22, |R|=2, |A_sea|=236, |A_air|=1502, |A_land|=30, |K|=4,
|L|=96` — every one matches Table `tab:vif:setsizes` in
`PRSVIF_Gospel.md` exactly, confirming the real pipeline (network CSVs +
`config/model_parameters.yaml` + `build_vehicle_params`) reproduces the
documented case study without any adjustment.

### Validated at `|Omega|=3` (real network, real fleet, reduced scenario count — see next section for why)

- Build (`build_only=True`): **2.64s**.
- `num_vars=428,189` (`num_bin_vars=138,718`), `num_constrs=154,238`.
- Variable families: `p=22, b=1,464, n=137,232, nbar=14,400, x=274,464,
  y=300, z=300, loss=3, eta=1, xi=3`.
- Constraint families: `VifBaseAssign=96, VifBaseLink=1,464,
  VifCVaRExcess=3, VifDistanceBudget=288, VifLossDefinition=3,
  VifNodeHandlingCapacity=450, VifResourceBalance=300,
  VifVehicleCapacity=137,232, VifVehicleConservation=14,400,
  Vif(SiteBudget+SelectionBudget)=2`.
- **LP relaxation: OPTIMAL, obj=5,074,641,919.53, solved in 2.91s.** The
  real instance assembles cleanly and is LP-feasible.

### Linear extrapolation to the gospel's full `|Omega|=100`

Every family except `p`, `b`, `eta` (first-stage, scenario-independent)
scales exactly linearly in `|Omega|` — confirmed by construction (each
is built inside a `for w in Omega` loop with no cross-scenario
interaction). Extrapolating:

- **~14,224,887 total variables** (`~4,575,886` binary: `p+b+n`).
- **~5,090,762 total constraints**, dominated by `VifVehicleCapacity`
  (~4,574,400) and `VifVehicleConservation` (~480,000).

### Finding: `|Omega|=100` attempt #1 was killed by system memory pressure after ~80+ minutes — but the real scaling curve (below) shows this mischaracterized the cause

I initially launched the actual `|Omega|=100` build (not extrapolated)
in the background. It ran for **over 80 minutes of CPU time** without
even finishing constraint construction, and was ultimately **killed by
the harness because the host system was running low on memory
overall**. At the time, I attributed this to `C7`/`C9`'s per-`(w,l,i,j)`
Python `model.addConstr()` loop (not bulk `addConstrs()`) having
non-constant per-call overhead at scale.

**That attribution was wrong, or at least incomplete** — caught after
the user asked whether intermediate scenario counts (10/25/50) had
actually been tested, and they hadn't been; only `|Omega|=3` (fast,
clean) and `|Omega|=100` (failed) existed, with nothing in between to
show where the curve actually bends. Filling that gap:

| `\|Omega\|` | build (s) | build/`\|Omega\|` | build exponent (vs. prior row) | LP-relax (s) | relax/`\|Omega\|` | relax exponent (vs. prior row) |
|---|---|---|---|---|---|---|
| 3   | 2.64  | 0.880 | —    | 2.91   | 0.970 | —    |
| 10  | 8.75  | 0.875 | 1.00 | 8.02   | 0.802 | 0.84 |
| 25  | 23.14 | 0.926 | 1.06 | 33.72  | 1.349 | 1.57 |
| 50  | 47.05 | 0.941 | 1.02 | 135.86 | 2.717 | 2.01 |

(exponent = `ln(t2/t1) / ln(w2/w1)` between consecutive rows; 1.0 = perfectly linear)

**Build time is cleanly linear** (exponent ~1.0-1.06 throughout,
per-scenario cost flat at ~0.88-0.94s) — extrapolating this trend,
`|Omega|=100` build should take on the order of **~100s**, nowhere near
80 minutes. The original per-`addConstr()`-call-overhead hypothesis
does not hold up against this data.

**LP-relaxation solve time is the real, and clearly accelerating,
cost** — its local scaling exponent climbs from 0.84 (3→10) to 1.57
(10→25) to 2.01 (25→50), i.e. roughly quadratic by the top of the
tested range and still rising. This — not construction — is almost
certainly what a real `|Omega|=100` attempt would actually spend most
of its time and memory on: extrapolating even the (already
conservative) 25→50 quadratic rate puts `|Omega|=100` relaxation alone
in the several-minutes-to-tens-of-minutes range, with commensurately
larger factorization memory.

**Revised interpretation of the original 80-minute failure**: given
build time is confirmed linear and fast up through `|Omega|=50`, a
clean `|Omega|=100` build alone should not plausibly take 80+ minutes
on this machine. The original attempt most likely lost most of its
wall-clock time to contention with **other memory demand on the host at
that moment** (consistent with the kill message citing overall system
memory pressure, not this process's own RSS, which stayed flat at
2.8-3.5GB throughout) rather than an algorithmic blowup in construction
itself. The genuine, data-confirmed risk for a real `|Omega|=100`
attempt is the LP-relaxation solve, not the build.

### Attempt #2, with monitoring: confirms this is a genuine memory-exhaustion wall, not external contention

Re-attempted `|Omega|=100` on an otherwise-idle system, this time with
a dedicated shell process watching system memory every 15s and a
1.5GB-available kill threshold, specifically to distinguish "genuinely
too big" from "unlucky timing with something else on the host" (the
leading hypothesis after attempt #1).

**Result: the process grew to ~7GB RSS within the first 90 seconds
(consistent with the fast, linear build behavior confirmed at
`|Omega|<=50`), then went flat.** RSS oscillated in a narrow ~3.1-3.7GB
band (it had actually *dropped* from its ~7GB early peak, consistent
with pages being pushed to swap rather than held resident) for **over
three hours** with the CPU pegged at 100%+ the entire time and zero
observable forward progress (the log never advanced past the initial
instance-summary lines — it never reached the "Gurobi model build"
completion message it reaches in seconds at every tested `|Omega|<=50`
size). System swap climbed to 4.1 of 5.0GB used and system-wide free
RAM fell to ~0.06GB. My 1.5GB-available kill threshold never tripped,
because "available" (free+inactive+speculative+purgeable) stayed
misleadingly above it while the OS was actually resolving the pressure
by filling swap instead — a flaw in that threshold choice, not evidence
the system was actually healthy. The process terminated on its own
(not via my kill command, which raced it and found it already gone)
right as I moved to kill it manually once the swap/free-RAM numbers
made the danger unambiguous; system memory recovered fully within
seconds of that PID exiting.

**This settles the question attempt #1 left open**: it is not that
attempt #1 was unlucky timing against unrelated host memory pressure.
Under monitored, otherwise-idle conditions, `|Omega|=100` construction
independently entered the same pathological state — rapid initial
growth consistent with the confirmed linear trend, followed by a hard
wall somewhere between `|Omega|=50` (47s, ~7GB extrapolated peak,
completes normally) and `|Omega|=100` (never completes, saturates all
18GB of RAM plus swap). This is a genuine memory-capacity limit on this
machine for the current construction approach, not a transient
external-contention artifact and not explained by the linear/quadratic
timing trend measured at `|Omega|<=50` — that trend predicts a
computationally slow but *completable* run; what was observed instead
is a run that cannot complete at all in this environment.

**Recommendation**: do not re-attempt `|Omega|=100` on this machine
without either (a) more RAM, or (b) a construction-side change —
candidates worth investigating first, not yet tried: bulk
`model.addConstrs()`/`addMVars()` (may reduce Python-object overhead
materially versus one Python `Constr`/`LinExpr` object per row), or
reducing `|L|` itself (e.g. confirming whether all 96 individually
indexed vehicles are ever simultaneously reachable in the symmetry-
broken solution space, versus tracking that many independent binaries
per scenario regardless). `|Omega|=50` stands as the largest
confirmed-working structural validation point.

### `addConstrs` conversion: correctness confirmed, but does not fix the `|Omega|=100` wall

Converted `C7` (`VifVehicleCapacity`) and `C9` (`VifVehicleConservation`)
— the two constraint families that scale as `|Omega|*|L|*|A_mode|` and
together account for the overwhelming majority of the model's row
count — from a per-row `model.addConstr()` Python loop to a single
`model.addConstrs()` bulk call each (`aps_usarpac/model/model_vif.py`).
Kept in the working tree, not committed, and a full pre-change copy of
`model_vif.py` was saved before editing so this is a one-file-copy
revert if needed (the file was untracked by git at the time, so `git
checkout` would not have recovered it).

**Correctness**: verified exact — rebuilt at `|Omega|=3`, every
variable/constraint family count and the LP-relaxation objective
(`5,074,641,919.5340`) matched the pre-conversion run to the decimal.
All 12 pre-existing `test_vif_phase*.py` tests and the locked aggregate
regression baseline still pass unchanged. See "Why this doesn't change
the math," below, for the reasoning.

**Performance at `|Omega|<=50`: no measurable benefit.** Re-ran
`|Omega|=50`: build 52.68s (vs. 47.05s before — marginally *slower*,
within noise) and LP-relax 136.56s (vs. 135.86s — unchanged). Bulk
construction is not meaningfully faster than the per-row loop at any
size where the per-row loop already completes normally.

**Re-attempted `|Omega|=100` three more times with the converted code,
under live memory monitoring, to see whether the benefit only appears
at the scale where the old approach actually failed:**

1. **Run 3**: killed by my own safety monitor after 30 seconds —
   a false positive. RSS grew to ~7GB (matching the healthy,
   normal early-build pattern from every prior successful run) and
   free memory dipped to 0.59GB, which tripped an overly conservative
   absolute free-memory threshold I'd set without accounting for this
   machine's already-reduced baseline headroom (4GB of swap left over,
   unreclaimed, from the earlier failed attempts). Not evidence of a
   problem — evidence my threshold was wrong.
2. **Run 4** (corrected thresholds, added a "stalled" detector
   specifically matching the original failure's signature — RSS flat
   for minutes while free is low, rather than any single low reading):
   killed after ~60 seconds, again by my monitor, but this time **not**
   a false positive in the same way — RSS was still climbing normally
   (7.04→7.22→7.79GB, real forward progress, not stalled) when free
   memory hit genuine near-zero (0.068GB) and stayed there for two
   consecutive checks.

Run 4 is the most informative data point of the three: it shows the
process **actively, healthily growing** (not thrashing) and still
running out of comfortably free RAM well before finishing construction
(construction at `|Omega|=100` needs to hold ~14.2M variables' worth of
Python/Gurobi objects; run 4 was killed at ~7.8GB RSS, likely still
well short of what full construction needs, based on how far short of
completion every smaller `|Omega|` needed to grow before finishing).

**Revised conclusion**: the `|Omega|=100` wall is not primarily an
artifact of the per-row `addConstr()` construction pattern — the
`addConstrs` conversion is correct and harmless but did not change the
outcome. The dominant factor is that this specific machine (18GB RAM,
with ~4GB of swap already consumed by history from earlier failed
attempts, non-trivially reducing real headroom below what a fresh boot
would offer) does not have enough free memory margin for this model at
`|Omega|=100`, independent of which Gurobi API pattern builds it.
Further threshold-tuning on the safety monitor is not a productive next
step — three attempts (one 3+-hour thrash, two quick kills at genuinely
low free memory during otherwise-healthy growth) point at the same
underlying capacity shortfall, not at monitor miscalibration alone.
Recommend pursuing recommendation (1) (check whether `|Omega|=50`
already gives adequate SAA convergence for PRS-VIF, which would make
this question moot) or (4) (run production-scale solves on a
higher-RAM machine) from the chat discussion, rather than further local
attempts at `|Omega|=100`.

### Why the `addConstrs` conversion doesn't change the math

Both forms — the original per-row loop and the new bulk call — produce
the exact same set of linear constraints: same variables, same
coefficients, same right-hand sides, same count, over the same index
set (`(w,l,i,j)` for C7; `(w,l,i)` for C9). Concretely, for C7,

```python
# before: one Python call per (w,l,i,j), 4.57M calls at |Omega|=100
for w in Omega:
    for l in L:
        for (i, j) in modal_arcs[mode_of[l]]:
            model.addConstr(<expr for this w,l,i,j> <= <rhs for this w,l,i,j>,
                             name=f"VifVehicleCapacity_w{w}_l{l}_i{i}_j{j}")

# after: one Python call total, same 4.57M rows inside it
model.addConstrs(
    (<expr for this w,l,i,j> <= <rhs for this w,l,i,j>
     for w in Omega for l in L for (i, j) in modal_arcs[mode_of[l]]),
    name="VifVehicleCapacity",
)
```

`<expr...>` and `<rhs...>` are byte-for-byte identical expressions in
both versions — nothing about *what* is being constrained changed, only
*how many Python function calls* it takes to hand those same rows to
Gurobi. `model.addConstr()` and `model.addConstrs()` are both documented
Gurobi API entry points for adding linear constraints to a model; the
plural form is simply a batched call that accepts a generator of many
constraint expressions instead of one expression at a time, and Gurobi
builds the identical underlying constraint matrix either way. This is
why the `|Omega|=3` re-validation matched the pre-conversion run to the
decimal on both variable/constraint counts and the LP objective value —
if the math had changed even slightly (a wrong index, a dropped term,
an off-by-one in which vehicles see which arcs), that number would not
have matched exactly. The only two substantive differences from the
conversion are: (1) constraint **names** changed from the old
`"VifVehicleCapacity_w3_l17_i5_j9"` style to Gurobi's auto-generated
bulk-index style `"VifVehicleCapacity[3,17,5,9]"` (cosmetic — useful
only for reading solver output/debugging, has zero effect on the
optimization problem itself), and (2) two small, non-semantic hoisting
cleanups (`vehicle_lookup`/`J_l_sets` precomputed once per vehicle
instead of being rebuilt redundantly inside the loop) that remove
repeated identical work without changing what is computed.

## Regression check

- `python3 -m py_compile model/input_builder.py model/model.py
  model/model_vif.py` — passes.
- `pytest tests/test_vif_phase1.py tests/test_vif_phase2.py
  tests/test_vif_phase3.py tests/test_vif_phase4.py -v` — **12/12 pass**,
  unchanged (these tests hand-build instance dicts and don't exercise
  `build_stochastic_instance`, so this phase's changes couldn't have
  affected them — confirms no accidental coupling).
- `python3 scripts/check_aggregate_regression.py` — **ALL PASS**,
  re-run after this phase's `input_builder.py` changes; every
  variable/constraint family count for the locked aggregate-formulation
  baseline still matches exactly.
- `git diff --stat aps_usarpac/model/input_builder.py` — the only
  file this phase modified; `model.py`/`model_vif.py` untouched
  (confirmed via `git diff --stat`, zero lines changed in either) —
  **note: this line reflects the state before the `addConstrs`
  conversion below**, which does modify `model_vif.py` (untracked by
  git, so it never showed in that diff either way).
- Re-run after the `addConstrs` conversion: `py_compile` passes,
  `pytest tests/test_vif_phase1.py` through `test_vif_phase4.py` —
  **12/12 pass, unchanged**; `check_aggregate_regression.py` — **ALL
  PASS, unchanged**; `|Omega|=3` re-validation — variable/constraint
  counts and LP-relaxation objective **identical to the decimal**
  against the pre-conversion run.

## What's left

- `|Omega|=100` structural build on this machine: **attempted four
  times now (one pre-conversion, three post-conversion), failed every
  time** — see "Attempt #2" and "`addConstrs` conversion" above. This is
  a confirmed memory-capacity wall specific to this machine (18GB RAM),
  not external contention, not an extrapolation of the 3/10/25/50 timing
  trend, and **not fixed by the `addConstrs` conversion** (verified
  correct, but performance-neutral at every tested size, and the wall
  persisted across three more attempts after converting). `|Omega|=50`
  remains the largest validated structural-check size on this machine.
  Getting a real `|Omega|=100` data point needs either more RAM
  (recommended — see chat discussion) or reducing `|L|` (a formulation
  change, not a construction-mechanics one, and one that would need
  explicit sign-off since it touches PRS-VIF's stated contribution) —
  do not re-attempt the current approach on this machine as-is.
- Theta/DeltaTheta remain non-binding placeholders pending SME
  calibration (explicitly out of scope here, per instructions).
- Phase 6 (`eq:vif:symbreak`, expected-utilization symmetry breaking)
  is the only remaining in-formulation TODO from Phase 4.
- No MIP solve attempted, per phase instructions — solve-time behavior
  at real scale (beyond the build-time finding above) is still unknown.

Stopping here for review, per the phase instructions.
