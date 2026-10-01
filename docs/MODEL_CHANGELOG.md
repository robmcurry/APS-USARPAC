# Model Changelog — pending written-document updates

**Purpose.** The model is changing faster than it is worth rewriting the paper.
This log records every change that will eventually require an edit to a written
document, so the documents can be updated **once**, in one pass, when the model
settles — instead of being churned on every change.

**This is not a code changelog.** `git log` is authoritative for code. Every
entry here answers a different question: *when we go update the paper, what
specifically has to change, and to what?*

**How to use this later.** Hand over (a) the current written documents and
(b) this file. Each entry names the affected section/table/equation and what it
should now say. Work top to bottom, then clear the "Documents known to be
stale" section at the bottom.

**Written artifacts in scope**
| Artifact | Location | What it is |
|---|---|---|
| `main.tex` | `~/Downloads/main.tex` | Chapter 1 as a standalone paper; contains the PRS-VIF model |
| `staged_chapter.tex` | `~/Downloads/staged_chapter.tex` | Add-in chapter: type-indexed distance-state model + staged method |
| `staged_approach_summary.pdf` | `~/Downloads/` | Preliminary summary + Stage 1A/1B results |
| `prepositioning_leadership_slide.pptx` | `~/Downloads/` | LOE/phase briefing slide |
| `pacific_network_slide.pptx` | `~/Downloads/` | Network + fleet + results infographic |

---

## 2026-09-30 — Node handling capacity (Θ) calibrated

**Change.** Replaced the flat `_VIF_THETA_PLACEHOLDER = 1.0e4` (applied
uniformly to every node and mode, ~200× any plausible fleet size, therefore
never binding) with a real calibration: baseline Θ from each node's own S/A/L/R
infrastructure rating, plus a PPL-activation bonus ΔΘ on top.

**Rationale.** The constraint existed in the formulation but was inert in every
result produced to date. The paper's claim that "siting buys throughput as well
as inventory" was not actually being exercised by any number.

**Document impact.**
- `main.tex` §3.4 already describes Θ and ΔΘ correctly in principle — the
  paper was right, the *code* was the placeholder. No change to the
  formulation text.
- **New content needed**: the actual calibration values and their derivation
  (air = assets × sorties/window per tier; sea = assets × discharge ÷ LCU-1700
  payload; land = road+rail MT/day × window ÷ M1083 payload), plus the
  activation-bonus fractions and why they are what they are.
- **Assumption to state explicitly**: bonus fractions are 15% (air), 15% (sea),
  25% (land). Land is highest because Army logistics units — trucks, drivers,
  convoy security, traffic control — are the organic competency a PPL siting
  decision most directly brings; sea/air infrastructure (berths, runways) is
  fixed and largely unaffected by Army presence. **These percentages are a
  reasoned judgment, not derived from a historical or doctrinal study.** They
  must be labelled as pending SME validation, same status as Γ and φ_r.
- **Known gap to disclose**: the sea bonus is computed from T-AKR/T-AKE
  discharge tonnage, treating that tonnage as a property of the *port* rather
  than the ship. That is an interpretive choice, not a like-for-like number.

---

## 2026-09-30 — Vehicle fleet expanded from 4 to 8 types

**Change.** Added T-AKR, T-AKE, EPF (sea) and PLS (land). Sea went from one
vehicle type to four; land from one to two. Air unchanged (C-17, C-130J).

**Rationale.** Sea and land were each represented by a single vehicle type,
which gave the model no modal tradeoffs to make. Also resolved a pre-existing
inconsistency: the Θ config already referenced TAKR/TAKE vessel types that had
no corresponding `vehicles:` entry.

**Document impact.**
- `main.tex` §3.2.2 "The Vehicle Fleet" currently says *"sea and land by one
  representative type each, the LCU-1700 landing craft and the M1083 medium
  tactical vehicle."* **This sentence is now wrong** and must be rewritten.
- `main.tex` Appendix A `tab:vehicleparams` needs **four new rows**.
- Lit-review `tab:litmatrix` row for "This paper" — "Fleet representation:
  Individual, routed" still describes PRS-VIF; revisit when the VIF/distance-state
  decision is made.
- **New content**: each platform's role and the tradeoff it creates —
  T-AKR bulk/major-port-only, T-AKE mid-size, **EPF the austere-port-capable
  flexible option** (lowest rating floor, no PPL-tier restriction), PLS
  heavier-but-needs-better-roads against M1083.
- **Sourcing to cite**: Navy fact sheets (T-AKR, T-AKE), Spearhead-class
  specs (EPF), militarytoday.com (PLS M1075A1). Speeds and payloads are
  sourced; **fleet sizes, basing tiers, and min-rating thresholds are planning
  assumptions** and must be labelled as such.

---

## 2026-10-01 — Per-type single-leg range cap (`max_leg_km`) — NEW FORMULATION ELEMENT

**Change.** Added an optional per-vehicle-type `max_leg_km`. Any physical arc
longer than a type's real unrefueled range is excluded from that type's
distance-expanded network, independently of the shared per-mode arc-existence
ceiling. Applied to EPF (2,222 km), C-130J (4,426 km), C-17 (5,371 km).

**Rationale.** The shared arc-existence ceilings (sea 4,500 km, air 6,000 km)
let vehicles be assigned single legs far beyond what they can physically travel
at the payload the model gives them. Measured effect:

| Type | Cap | Arcs excluded |
|---|---|---|
| EPF | 2,222 km | 160 of 236 sea arcs (**68%**) |
| C-130J | 4,426 km | 478 of 1,502 air arcs (**32%**) |
| C-17 | 5,371 km | 166 of 1,502 air arcs (11%) |

**Document impact — this is the most significant entry in this log.**
- **`staged_chapter.tex` needs a formulation change, not just prose.** The
  chapter currently defines arcs available to type *k* as $A_k := A_{m(k)}$ —
  i.e. every arc of that type's mode. That is **no longer true**. Needs either
  a restricted set definition (e.g. $A_k := \{(i,j) \in A_{m(k)} :
  \text{dist}_{ij} \le \text{maxleg}_k\}$) or an explicit note in the
  distance-state network construction section.
- **New parameter** to add to the chapter's Parameters list: $\text{maxleg}_k$,
  the real single-leg unrefueled range of type *k*.
- **Clarify in text**: the cap is checked against *raw physical distance*, not
  distance + turnaround penalty $\psi_k$, because $\psi_k$ is a time proxy, not
  distance actually travelled.
- **Relationship to $D_k$ to explain**: $D_k$ is a *cumulative* 3-day budget
  across multiple legs and assumes refuelling between them; $\text{maxleg}_k$
  bounds any *single* leg. These are different constraints and the chapter
  should say so — currently it only has the former.
- EPF's config comment and the leadership/network slides describe EPF as the
  flexible option; **its modelled utilisation roughly halved** after this fix
  (60% → 33% of scenarios active). Any narrative claiming EPF flexibility
  needs to be checked against post-fix numbers.

---

## 2026-10-01 — Duty-factor convention documented; PLS speed corrected

**Change.** `cruise_speed_km_day` is not max speed × 24 h — it bakes in an
implicit operational duty factor (air ~0.5–0.6, land 0.50, sea ~0.8–1.0). The
four vehicles added 2026-09-30 were specified at a duty factor of 1.00, which
was inconsistent with the original four. PLS corrected 2,400 → 1,200 km/day.

**Rationale.** PLS at 2,400 km/day vs M1083 at 1,116 implied PLS was 2.15×
faster, when the real machines are within 6% of each other (62 vs 58 mph).
Ships deliberately left at 1.00 — they genuinely do steam 24/7 on watch
rotations.

**Document impact.**
- **`main.tex` §3.2.3 is currently misleading.** It says *"Each vehicle of type
  k with effective cruise speed $v_k$ has a budget $D_k = \kappa v_k$."* The
  word "effective" is carrying an undocumented modelling assumption. The
  duty-factor convention **must be stated explicitly** — including that it
  differs by mode and why (crew duty limits and maintenance for air, driver
  rest and convoy pacing for land, continuous watch rotations for sea).
- Appendix A vehicle table: PLS speed value changed.
- **Quantified effect to report if these results are used**: PLS distance
  flown −41%, M1083 +18%, as land work shifted back to the platform that had
  been wrongly passed over.

---

## 2026-10-01 — Payload-range tradeoff: documented as an assumption, NOT implemented

**Change.** None to the model. Decision recorded: each vehicle keeps **one**
(payload, range) point — `payload_kg` is max payload and `max_leg_km` is the
real range *at that max payload*.

**Rationale.** Real aircraft trade payload against range continuously (a C-17
carries 170,900 lb for ~2,900 nm but 130,000 lb for ~5,200 nm — nearly double
the reach for 24% less load). The model represents none of that curve. The
chosen single point is deliberately the **conservative corner**: it assumes
every sortie flies max-loaded, so it understates reach rather than overstating
it — the safe direction for a risk-averse posture model.

**Document impact.**
- **`staged_chapter.tex` needs an explicit limitations/assumptions statement.**
  This is exactly the kind of simplification a committee will probe. State it
  plainly, state the direction of the bias, and state that it may miss postures
  which a lightly loaded long-range sortie would make viable.
- Record the deferred alternatives so the choice looks considered rather than
  overlooked: (a) piecewise payload-range curve — needs the vehicle-arc variable
  to carry a payload-regime index, a formulation change; (b) split each aircraft
  into heavy/short and light/long pseudo-types sharing one airframe pool — needs
  a constraint coupling $b$ across the pair, since each type currently has an
  independent $\sum_j b_{kj} = F_k$.

---

## 2026-10-01 — `inventory_disruption.cutoff_severity` documented as inert

**Change.** Documentation only. The distance-state model never reads this config
value; it hardcodes node availability at severity ≥ 1.0, which is 5× more
aggressive than the configured 5.0.

**Document impact.**
- **None to the paper** — `eq:availability` in `main.tex` already specifies the
  threshold of 1, so the code matches the paper and the *config* is the orphan.
- Worth a one-line implementation note if an implementation appendix is ever
  written, so nobody tunes a knob that does nothing.

---

## 2026-09-29/30 — Computational changes (low documentation priority)

- Stage 2 (per-scenario routing) parallelised via `ProcessPoolExecutor`;
  ~4× wall-clock speedup. The chapter already claims Stage 2 is
  "embarrassingly parallel" — this makes that claim true in the
  implementation, where previously it was sequential.
- `NodefileStart = 2.0` added to all solver params as an OOM safeguard.
- Missing top-level `network/` package restored (nothing in the branch could
  import without it).
- **Document impact**: only relevant to a computational-results or
  implementation section. If runtimes are reported anywhere, note they are
  post-parallelisation.

---

## Open items — resolved neither in code nor in documentation

These came out of the 2026-10-01 real-world audit and are **not yet addressed**.
Each needs either a model change or an explicit limitation statement.

1. **Sea arc ceiling derived from a vessel that is not in the fleet.**
   `ship_km_per_day: 1500` → 4,500 km ceiling, but actual effective speeds are
   408–1,556 km/day. Consequence: LCU-1700 cannot reach **80.5%** of sea arcs
   even spending its entire 3-day budget on one leg; T-AKE 55.1%; T-AKR 31.4%.
   The mechanics handle this correctly, but three of four sea platforms are
   confined to a minority of the network they nominally belong to.
2. **The 72-hour horizon may be too tight for the sealift just added.**
   T-AKR spends **33%** of its entire 3-day budget on a single 24 h port
   turnaround; T-AKE 25%. These ships realistically get ~1 delivery per
   planning window — worth confronting given APS-afloat doctrine is built on
   exactly these ships.
3. **T-AKR payload is a deadweight-style figure for a RoRo.** LMSR capacity is
   normally expressed as ~393,000 sq ft of vehicle deck, not bulk tonnage,
   because it carries rolling stock. 13,200 MT of water = 880,440 person-days
   may overstate what a RoRo usefully delivers in palletised relief.
4. **M1083's real unrefueled range is 483 km** against a 3,348 km 3-day budget,
   and the longest land arc is 2,608 km — a ~56-hour continuous drive at
   modelled speed, through potentially disaster-degraded roads, inside a
   72-hour window.

---

## Documents known to be stale

- **`staged_approach_summary.pdf`** — all results tables predate the audit.
  Baseline objective shown as 23.40B; current is **23.62B** (+0.91%).
  Per-vehicle utilisation figures all predate the `max_leg_km` and PLS fixes.
  Regenerate from `scratchpad/build_summary_pdf.py` after the model settles.
- **`pacific_network_slide.pptx`** — results panel and vehicle fleet profile
  predate the audit; PLS payload/speed and EPF utilisation shown are stale.
- **`main.tex`** — §1.4 "Research Contribution" and §1.6 "Chapter Organization"
  still describe PRS-VIF as the contribution; `tab:litmatrix` likewise. Deferred
  deliberately pending the VIF-vs-distance-state decision, **not** an oversight.
- **`staged_chapter.tex`** — predates `max_leg_km` entirely (see above), and
  contains no payload-range or duty-factor assumption statement.
