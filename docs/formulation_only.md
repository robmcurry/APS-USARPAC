# Methods {#sec:methods}

## The Theater Sustainment Decision Maker

Theater logistics posture decisions are made by the USARPAC Commanding
General, with recommendations developed by the 8th Theater Sustainment
Command in coordination with the USARPAC G4 and supporting planning
staffs. Planners determine: (i) which network nodes serve as
Prepositioned Logistics Locations (PPLs), (ii) the quantity and
composition of resources stored at each PPL, (iii) the regional
distribution of those resources, and (iv) the allocation of
transportation assets by class across the network.

At decision time, disaster location, severity, infrastructure
degradation, and resulting demand are all uncertain. The objective is a
sustainment posture that balances efficiency under expected conditions
with robustness against severe but plausible disruptions. The model
formalizes this as a two-stage stochastic program: the first stage
selects PPL sites before uncertainty is realized; the second stage
allocates inventory, vehicles, and commodity flows in recourse to each
realized disaster scenario. A Conditional Value-at-Risk (CVaR) objective
at confidence level $\beta$ evaluates the first-stage decision by its
performance across the worst-performing tail of scenarios rather than by
its average, reflecting the asymmetry between the cost of
under-preparing for a catastrophic event and the cost of over-preparing
for a mild one.

## Network Construction {#sec:network}

The network comprises $|N| = 50$ nodes spanning the USARPAC area of
responsibility: major population centers, commercial ports, and military
logistics facilities across East Asia, Southeast Asia, Oceania, and the
central Pacific. Each node carries four infrastructure ratings on a
conditions-based 1--3 scale (null where the modality is absent): sea
port capability ($S$), airfield capability ($A$), land distribution
capability ($L$), and rail connectivity ($R$). These ratings are the
single source from which arc eligibility, PPL tier, and vehicle basing
eligibility are all derived.

All 50 nodes are eligible disaster epicenters; epicenter sampling is
conditioned only on the country-level frequency weighting of
Section [3.4.2](#sec:epicenterweights){reference-type="ref"
reference="sec:epicenterweights"}, and no node is excluded on the basis
of its function within the network. Of the 50 nodes, $|N^P| = 22$ are
PPL candidates, requiring a US access agreement in force or reasonably
available, sea or air capability sufficient to receive prepositioned
stock, and land capability to distribute it without major augmentation.
Candidates are classified into three tiers --- PPL-1 Strategic Hubs,
PPL-2 Operational Nodes, and PPL-3 Contingency Sites --- and tier
determines both the inventory ceiling $\bar{q}_{ir}$
(Section [3.10](#sec:inventory){reference-type="ref"
reference="sec:inventory"}) and vehicle basing eligibility
(Section [3.3](#sec:vehicles){reference-type="ref"
reference="sec:vehicles"}).

### Modal Arc Construction {#sec:modalarcs}

The model constructs three parallel directed arc layers,
$A_{\text{sea}}$, $A_{\text{air}}$, and $A_{\text{land}}$, reflecting
the distinct infrastructure requirements, throughput rates, and
disruption mechanisms of maritime, air, and terrestrial movement. A node
pair may be connected on one, several, or no modes. Arc existence
combines a capability threshold at both endpoints with a distance limit
consistent with the 0--72 hour response window:

- **Maritime** ($A_{\text{sea}}$): $S \geq 2$ at both endpoints,
  great-circle distance $\leq 4{,}500$ km, consistent with sustainment
  vessel transit speeds within the window.

- **Air** ($A_{\text{air}}$): $A \geq 2$ at both endpoints,distance
  $\leq 6{,}000$ km, consistent with strategic and theater airlift
  range.

- **Land** ($A_{\text{land}}$): a manually specified set of 15
  contiguous node pairs (30 directed arcs) representing documented
  overland corridors (the Japanese home-island chain, the eastern
  Australian corridor, and the Southeast Asian mainland corridor), since
  overland feasibility depends on a continuous road or rail corridor
  rather than proximity.

This yields $|A_{\text{sea}}| = 236$, $|A_{\text{air}}| = 1{,}502$, and
$|A_{\text{land}}| = 30$ directed arcs; air dominates connectivity due
to its lower capability threshold and longer permissible range.

At nodes with more than one mode, commodity may transfer between modes
subject to a node-specific handling capacity $\kappa_{i,m_1m_2}$ and a
network-wide cost multiplier $\theta_{m_1m_2}$
(Section [3.8](#sec:formulation){reference-type="ref"
reference="sec:formulation"}). Eleven nodes possess all three modes and
support the full transfer set.

### Unreachable Epicenter Nodes {#sec:unreachable}

Three nodes --- Shanghai, Guangzhou, and Chengdu --- were added as
demand-generating epicenter nodes so that China's empirical disaster
frequency weight of 23.4 percent
(Section [3.4.2](#sec:epicenterweights){reference-type="ref"
reference="sec:epicenterweights"}) is represented; without them the
China weight would fall through to a uniform draw, materially
understating disaster likelihood in the most frequently struck country
in the record. These nodes carry $S = A = 1$, below the arc-eligibility
threshold on either mode, and no land corridor: they are
epicenter-eligible but unreachable. This is a deliberate scope decision
--- USARPAC's prepositioning mission does not contemplate direct
delivery into mainland China. A disaster centered on one of these nodes
still drives realistic severity decay and downstream demand at reachable
neighbors through the spatial mechanism of
Section [3.4.5](#sec:severitydecay){reference-type="ref"
reference="sec:severitydecay"}, while demand at the unreachable node
itself is set to zero by the reachability rule of
Section [3.5](#sec:demand){reference-type="ref" reference="sec:demand"}.

## Vehicle Fleet {#sec:vehicles}

Arc capacity is generated by a heterogeneous fleet of vehicle types
rather than treated as a generic arc property, allowing the model to
track movement by type. Air is served by two types --- the C-17
Globemaster III (strategic) and C-130J Super Hercules (tactical),
reflecting materially different payload and runway requirements --- and
sea and land by one representative type each (LCU-1700 landing craft;
M1083 medium tactical vehicle). Vehicle types are indexed $k \in K_m$
within each mode.

### Basing Eligibility and Fleet Distribution {#sec:vehiclebasing}

Each type may be based only at PPL nodes meeting a type-specific
capability rating drawn from the same $S$/$A$/$L$ framework that governs
arc eligibility: the C-17 requires $A = 3$, the C-130J $A \geq 2$, the
LCU-1700 $S \geq 2$, and the M1083 $L \geq 1$. In addition to the rating
threshold, basing for the three scarce, high-value types --- the C-17,
C-130J, and LCU-1700 --- is restricted to PPL-1 Strategic Hubs only;
PPL-2 and PPL-3 sites receive no allocation of these types regardless of
rating. Only the M1083, whose fleet is large relative to its eligible
node count, is distributed across the full PPL-1/2/3 eligibility set in
a 3:2:1 tier-weighted proportion. The tier-proportional allocation uses
floor division, with any rounding remainder assigned to the
highest-tier, lowest-node-index eligible site. This restriction exists
because proportional distribution of a small fleet across a large
eligible set degenerates under integer rounding --- every node's floor
share is zero and the entire fleet collapses onto the single remainder
node --- and because concentrating scarce strategic assets at top-tier
hubs is independently defensible operationally.

The resulting basing parameter $b_{k,j}$ is fixed and identical across
scenarios, with each scenario treated as an independent realization in
which the fleet begins at its base distribution. Basing is coupled to
PPL activation: a node's allocation materializes only if the node is
selected, entering the formulation as $b_{k,j} \cdot p_j$
(Equation [\[eq:vehcons\]](#eq:vehcons){reference-type="ref"
reference="eq:vehcons"}), so an unselected node hosts no fleet. Fleet
sizes $F_k$ are fixed exogenous planning assumptions; sizing the fleet
to a target CVaR level is identified as future work.

### Time Representation: Distance Budgets and Turnaround {#sec:timeproxy}

The model has no internal clock. The 72-hour response window is
represented through distance budgets: a vehicle of type $k$ with
effective cruise speed $v_k$ (km/day) has a total budget $D_k = 3 v_k$
over the window. Every leg consumes distance, and every intermediate
stop --- arrival followed by departure within the same scenario ---
additionally consumes a turnaround-equivalent distance
$\psi_k = (\text{turnaround}_m / 24) \times v_k$, where turnaround time
is specified at the mode level. The budget is tracked fleet-wide
(Equation [\[eq:distbudget\]](#eq:distbudget){reference-type="ref"
reference="eq:distbudget"}) rather than per vehicle: the formulation
bounds total fleet distance consumption without tracking any individual
vehicle's remaining range, and therefore cannot distinguish several
vehicles each flying one leg from one vehicle flying several. This
trade-off is accepted for a tractable, symmetry-safe formulation that
avoids per-vehicle path variables; exact itinerary tracking is
identified as future refinement.

### Vehicle Specifications and Capacity Derivation {#sec:vehiclespecs}

Vehicle capacities derive from public payload specifications converted
to person-days using two humanitarian reference rates: the WFP standard
emergency food ration of 540 g per person per day dry weight, and the
Sphere minimum of 15 L (approximately 15 kg) of water per person per
day. Water is roughly 28 times heavier than food per person-day, so
water binds vehicle weight capacity long before food. Capacity
$\text{cap}_{k,r}$ is therefore computed independently per resource;
food and water capacities each assume a single-commodity load, which
modestly overstates throughput on mixed sorties. Tables
[1](#tab:vehiclespecs){reference-type="ref"
reference="tab:vehiclespecs"} and
[2](#tab:vehicleparams){reference-type="ref"
reference="tab:vehicleparams"} report the specifications and adopted
parameters. Fleet sizes are stated planning assumptions at generous
squadron-equivalent allocations; effective cruise speeds adjust
published speeds for realistic daily operating tempo (14 h/day air,
reflecting crew duty limits; 20 h/day sea, reflecting shift-rotation
crewing; 12 h/day land, reflecting driver duty limits); turnaround is 2
h air, 6 h sea, 1 h land.

::: {#tab:vehiclespecs}
  Type                    Mode                Max Payload                       Cruise Speed
  ----------------------- ------ ------------------------ ----------------------------------
  C-17 Globemaster III    Air      170,900 lb (77,500 kg)   $\approx$`<!-- -->`{=html}450 kn
  C-130J Super Hercules   Air       42,000 lb (19,051 kg)   $\approx$`<!-- -->`{=html}350 kn
  LCU-1700 class          Sea              170 short tons                    11 kn sustained
  M1083 (5-ton FMTV)      Land       10,000 lb (4,536 kg)                             58 mph

  : Vehicle type specifications (public sources).
:::

::: {#tab:vehicleparams}
  Type         Fleet $F_k$   $v_k$ (km/day)   Turnaround   Food cap (pd)   Water cap (pd)
  ---------- ------------- ---------------- ------------ --------------- ----------------
  C-17                  12           11,662         2 hr         143,519            5,167
  C-130J                16            7,840         2 hr          35,280            1,270
  LCU-1700               8              408         6 hr         285,594           10,281
  M1083                 60            1,116         1 hr           8,400              302

  : Adopted vehicle fleet, speed, turnaround, and per-resource capacity
  parameters.
:::

Two limitations are noted: published C-130J payload varies by variant
(the standard J-model factsheet figure is used), and no large oceangoing
vessel is yet modeled alongside the LCU-1700; Maritime Prepositioning
Ships based at Guam and Saipan are a natural second sea type once a
specific hull class with clean published figures is selected.

## Scenario Generation and Probability Calibration {#sec:scenariogen}

Each scenario $\omega \in \Omega$ is a disaster realization defined by
an epicenter node, a disaster type $\tau(\omega) \in \Theta$, an
epicenter severity, and the node-level severity, demand, and arc
degradation derived from them. Two stochastic inputs are empirically
calibrated: epicenter location probability and severity distribution.

### Data Source

Calibration uses the Emergency Events Database (EM-DAT) maintained by
CRED [@emdat], filtered to natural disasters in the USARPAC region
(Eastern Asia, South-eastern Asia, Melanesia, Micronesia, Polynesia,
Australia and New Zealand) and to the HA/DR-relevant types flood, storm,
earthquake, mass movement (wet), volcanic activity, and wildfire: 2,666
events over 2000--2026. EM-DAT is the standard open-access disaster
database in the humanitarian logistics literature; its methodology and
limitations are documented in Guha-Sapir and Below [@guhasapir2002].
Reporting bias in low-capacity Pacific island nations likely
underweights the most vulnerable nodes marginally; subnational
(DesInventar), physical-hazard (IBTrACS, ShakeMap), and commercial
(NatCatSERVICE) alternatives were evaluated and rejected on coverage,
translation, or licensing grounds.

### Epicenter Location Weights {#sec:epicenterweights}

Epicenter countries are drawn from a multinomial with weights
$$\begin{equation}
w_c = \frac{n_c}{\sum_{c' \in C} n_{c'}},
\label{eq:countryweights}
\end{equation}$$ where $n_c$ is the recorded event count for country
$c$. Table [3](#tab:weights){reference-type="ref"
reference="tab:weights"} reports the top fifteen weights; China (23.4%),
Indonesia (15.4%), and the Philippines (14.3%) account for over half of
sampled epicenters. Within the selected country, the epicenter node is
drawn uniformly across that country's network nodes --- a tractability
simplification noted for future refinement.

::: {#tab:weights}
  Country             ISO     Event Count   Weight
  ------------------- ----- ------------- --------
  China               CHN             624    0.234
  Indonesia           IDN             411    0.154
  Philippines         PHL             381    0.143
  Viet Nam            VNM             197    0.074
  Japan               JPN             165    0.062
  Australia           AUS             124    0.047
  Thailand            THA             120    0.045
  Malaysia            MYS              80    0.030
  Taiwan              TWN              73    0.027
  Republic of Korea   KOR              63    0.024
  Myanmar             MMR              59    0.022
  Papua New Guinea    PNG              53    0.020
  New Zealand         NZL              38    0.014
  Lao PDR             LAO              30    0.011
  Cambodia            KHM              29    0.011

  : Top-fifteen epicenter sampling weights from EM-DAT frequency,
  2000--2026.
:::

### Disaster Type Sampling {#sec:typesampling}

Disaster type $\tau(\omega)$ is drawn conditional on the epicenter's
country, from a country-specific frequency table derived from the same
filtered EM-DAT record; countries without a usable country-specific
vector fall back to the global marginal type distribution. Type draws
use a dedicated random-number stream seeded identically to the main
scenario stream, so that type realizations are reproducible and
statistically independent of the epicenter and severity draws.

### Disaster Severity Distribution

Severity is calibrated to EM-DAT's Total Affected field, the most
consistently reported impact measure and the one semantically aligned
with the demand formulation. Raw values are log-transformed and
normalized, $$\begin{equation}
\tilde{s} = \frac{\log(1+\text{Affected}) - \log(1+\text{Affected})_{\min}} {\log(1+\text{Affected})_{\max} - \log(1+\text{Affected})_{\min}}, \label{eq:severitynorm}
\end{equation}$$ then rescaled to $[1,5]$ via $s = 1 + 4\tilde{s}$.
Because $\tilde{s} \in (0,1)$ strictly, realized severity is strictly
below 5.

Four bounded candidate distributions were fit and compared by the
Kolmogorov--Smirnov statistic $$\begin{equation}
D_n = \sup_x |F_n(x) - F(x)|.
\label{eq:ks}
\end{equation}$$ The Kumaraswamy distribution ($a = 2.417$, $b = 3.747$)
achieved the best parametric fit ($D_n = 0.0448$, versus 0.0564 Beta and
0.2036 truncated Normal, with a nonparametric KDE baseline at 0.0122)
and is used for severity draws; it is defined on the open unit interval
with a closed-form CDF convenient for Monte Carlo sampling
[@kumaraswamy1980generalized; @fletcher1996kumaraswamy]. With
$n = 2{,}391$ observations, formal KS hypothesis tests reject all
parametric fits, a known large-sample artifact [@razali2011power];
selection is therefore based on the statistic as a practical fit
measure.

### Spatial Severity Decay {#sec:severitydecay}

The disaster impact area extends beyond the epicenter. The affected
radius grows linearly with epicenter severity $s$, $$\begin{equation}
\text{radius}(s) = 250 + 250\,s \quad \text{(km)},
\label{eq:radius}
\end{equation}$$ and node-level severity decays linearly with distance
from the epicenter within that radius: $$\begin{equation}
s^\omega_i =
\begin{cases}
s \cdot \left(1 - \dfrac{\text{dist}(i, \text{epi})}{\text{radius}(s)}\right)
& \text{dist}(i,\text{epi}) \leq \text{radius}(s), \\[6pt]
0 & \text{otherwise}.
\end{cases}
\label{eq:decay}
\end{equation}$$ A severity-5 event therefore affects nodes within 1,500
km of the epicenter; a severity-1 event within 500 km.

## Demand Calibration {#sec:demand}

Nodal demand is $$\begin{equation}
d^\omega_{ir} = \phi_r \cdot s^\omega_i \cdot P_i
\quad \text{for } i \in R(N), \qquad
d^\omega_{ir} = 0 \text{ otherwise},
\label{eq:demand}
\end{equation}$$ where $\phi_r$ is the per-capita demand rate for
resource $r$ ($\phi_{\text{food}} = 0.15$, $\phi_{\text{water}} = 0.20$,
drawn from the humanitarian logistics literature and flagged for USARPAC
J4 calibration), $P_i$ is node population (UN World Urbanization
Prospects 2018 and national census data), and $R(N) \subseteq N$ is the
set of nodes reachable by at least one inbound arc across
$A_{\text{sea}} \cup A_{\text{air}} \cup A_{\text{land}}$ --- the rule
that zeroes demand at the unreachable nodes of
Section [3.2.2](#sec:unreachable){reference-type="ref"
reference="sec:unreachable"}. Demand is scenario-dependent: demand surge
and arc degradation are driven by the same severity realization and are
allowed to co-vary, which is precisely the joint tail behavior the CVaR
objective is designed to capture. (An earlier single-mode version held
demand at its scenario average to isolate arc-capacity uncertainty; that
design is not used here.)

## Arc Capacity and Degradation {#sec:arcdeg}

### Nominal Arc Capacity {#sec:nominalcapacity}

Nominal capacity $U_{m,ij,r}$ is derived, per mode and per commodity,
from the infrastructure tier of the arc's endpoints: each tier is
assigned a representative daily asset throughput (vessel discharge rates
for sea, sortie throughput for air, road/rail metric-ton-per-day ratings
for land), expressed in metric tons per day over the response window and
converted to person-days using a single conversion basis of 1,852
person-days per metric ton for food and 66.7 for water. These
conversions follow directly from the WFP 540 g/person/day ration and the
Sphere 15 L/person/day minimum ($1{,}000/0.54$ and $1{,}000/15$
respectively) --- the same basis used for vehicle capacities in
Section [3.3.3](#sec:vehiclespecs){reference-type="ref"
reference="sec:vehiclespecs"}, so arc and vehicle capacities are
denominated consistently.

### The Degradation Matrix

Disasters reduce throughput as a function of both mode and disaster
type: an earthquake damages roads differently than shipping lanes, and a
flood degrades an airfield differently than a port. For each mode $m$
and type $\tau$, a baseline sensitivity $\Gamma_{m\tau}$ is specified
(Table [4](#tab:degradmatrix){reference-type="ref"
reference="tab:degradmatrix"}). Land is most sensitive to earthquake
($\Gamma_{\text{land,eq}} = 1.0$); maritime movement is comparatively
insensitive across types; air falls between, more sensitive to storm
($0.9$) than flood ($0.3$). All six disaster types are live in every
model run, since baseline scenarios sample the full type distribution of
Section [3.4.3](#sec:typesampling){reference-type="ref"
reference="sec:typesampling"}.

::: {#tab:degradmatrix}
  Mode     Flood   Storm   Earthquake   Volcanic   Mass Mvmt.   Wildfire
  ------ ------- ------- ------------ ---------- ------------ ----------
  Sea        0.2     0.7          0.3         --           --         --
  Air        0.3     0.9          0.5        1.0           --         --
  Land       0.9     0.5          1.0         --           --         --

  : Baseline degradation sensitivity $\Gamma_{m\tau}$ by mode and
  disaster type.
:::

Degradation applies uniformly to every vehicle type on a mode; a
disaster does not yet shift *which* types can use a node (e.g., a flood
reducing an airfield from C-17-capable to C-130-only). Representing that
as a discrete capability-tier transition is a candidate refinement, not
implemented here.

### Degradation Scaling and Residual Capacity

The matrix is scaled by the sweep parameter $\alpha \in [0,1]$:
$$\begin{equation}
\gamma_m(\tau) = \Gamma_{m\tau} \cdot \alpha.
\label{eq:gammascale}
\end{equation}$$ At $\alpha = 0$ no arc loses capacity; at $\alpha = 1$
the full matrix applies. The primary analysis examines
$\alpha \in \{0.0, 0.25, 0.50, 0.75, 1.0\}$; a mode-isolation design
sweeps a single mode's $\alpha_m \in \{0, 0.5, 1.0\}$ with the other two
held at 1.0, and a type-dominance design forces all scenarios to a
single type while epicenter and severity vary. Residual capacity is
$$\begin{equation}
u^\omega_{m,ij,r} = U_{m,ij,r} \cdot
\max\!\left(0,\ 1 - \gamma_m(\tau(\omega)) \cdot
\frac{\max(s^\omega_i, s^\omega_j)}{5}\right),
\label{eq:residual}
\end{equation}$$ degrading by the worse-off endpoint --- throughput at a
node is constrained by that node's condition regardless of the far end
--- with severity normalized by its maximum of 5.

## Sets, Parameters, and Decision Variables {#sec:notation}

### Sets

- $N$: nodes, $|N| = 50$; $N^P \subseteq N$: PPL candidates,
  $|N^P| = 22$

- $R = \{\text{food}, \text{water}\}$: resource types

- $M = \{\text{sea}, \text{air}, \text{land}\}$:
  modes;$A_m \subseteq N \times N$: directed arcs on mode $m$

- $T_i \subseteq M \times M$: feasible transfer mode-pairs at node $i$

- $\Theta$: disaster types (flood, storm, earthquake, volcanic, mass
  movement, wildfire)

- $\Omega$: scenarios, $|\Omega| = 100$

- $K_m$: vehicle types on mode $m$; $J_k \subseteq N^P$: basing-eligible
  nodes for type $k$
  (Section [3.3.1](#sec:vehiclebasing){reference-type="ref"
  reference="sec:vehiclebasing"})

### Parameters

- $d^\omega_{ir}$: demand
  (Eq. [\[eq:demand\]](#eq:demand){reference-type="ref"
  reference="eq:demand"}); $\phi_r$: per-capita demand rate; $P_i$:
  population

- $\bar{q}_{ir}$: inventory ceiling at $i$ if activated; $\rho$:
  safety-stock fraction; $a^\omega_{ir} \in \{0,1\}$: availability
  factor, 0 if $s^\omega_i$ reaches the cutoff severity

- $U_{m,ij,r}$, $u^\omega_{m,ij,r}$: nominal and residual arc capacity
  (Eqs. [\[eq:residual\]](#eq:residual){reference-type="ref"
  reference="eq:residual"}); $c_{m,ij}$: per-unit transport cost;
  $\text{dist}_{ij}$: arc distance

- $\kappa_{i,m_1m_2}$: node-specific transfer capacity;
  $\theta_{m_1m_2}$: network-wide transfer cost multiplier

- $s^\omega_i$: severity
  (Eq. [\[eq:decay\]](#eq:decay){reference-type="ref"
  reference="eq:decay"}); $\Gamma_{m\tau}$, $\alpha$, $\gamma_m(\tau)$:
  degradation
  (Eqs. [\[eq:gammascale\]](#eq:gammascale){reference-type="ref"
  reference="eq:gammascale"})

- $\delta_{ir}$: unmet-demand penalty; $P_{\max}$, $f_i$, $B$: site
  count cap, activation cost, selection budget; $\beta$: CVaR level;
  $\bar{Q}_r$: theater inventory ceiling (see
  Eq. [\[eq:invceiling\]](#eq:invceiling){reference-type="ref"
  reference="eq:invceiling"} discussion)

- $F_k$, $v_k$, $D_k = 3v_k$: fleet size, effective speed distance
  budget; $\psi_k = (\text{turnaround}_m/24) \times v_k$: turnaround
  penalty; $\text{cap}_{k,r}$: per-resource vehicle capacity; $b_{k,j}$:
  initial basing; $\varepsilon$: vehicle-movement tie-breaker

### Decision Variables

- First stage: $p_i \in \{0,1\}$, 1 if node $i$ is selected as a PPL

- $x^\omega_{m,ijr} \geq 0$: commodity flow;
  $\tau^\omega_{i,m_1m_2,r} \geq 0$: intermodal transfer;
  $z^\omega_{ir} \geq 0$: unmet demand; $y^\omega_{ir} \geq 0$:
  inventory released

- $n^\omega_{k,m,ij} \geq 0$, integer: type-$k$ vehicles traversing arc
  $(i,j)$. Indexing by type rather than individual identity keeps the
  formulation symmetry-safe: same-type vehicles are interchangeable, so
  the model is not exposed to combinatorially equivalent solutions
  differing only in which unit was assigned where.

- Auxiliary: $\eta$ (Value-at-Risk threshold), $\xi^\omega \geq 0$ (CVaR
  excess loss), $L^\omega \geq 0$ (scenario loss)

## Model Formulation {#sec:formulation}

The second-stage problem is embedded directly into a single-level
extensive form that simultaneously determines first-stage siting and
second-stage recourse across all scenarios. For each scenario, the loss
is $$\begin{equation}
L^\omega = \sum_{i\in N}\sum_{r\in R}\delta_{ir}z^\omega_{ir}
+ \sum_{m\in M}\sum_{(i,j)\in A_m}\sum_{r\in R}c_{m,ij}\,x^\omega_{m,ijr}
+ \sum_{i\in N}\sum_{(m_1,m_2)\in T_i}\sum_{r\in R}\theta_{m_1m_2}\,\tau^\omega_{i,m_1m_2,r}
+ \varepsilon \sum_{k}\sum_{m\in M}\sum_{(i,j)\in A_m} n^\omega_{k,m,ij}.
\label{eq:loss}
\end{equation}$$ The four terms are the unmet-demand penalty, transport
cost across all modal layers, intermodal transfer cost, and a
vehicle-movement term with $\varepsilon = 0.04$. The fourth term is a
lexicographic tie-breaker: negligible against $\delta_{ir} = 500$ and
without influence on delivery decisions, it exists so the solver
strictly prefers solutions with fewer vehicle-arc traversals and no
empty repositioning legs when delivery outcomes are identical,
eliminating a degenerate class of cost-free circular vehicle movement
that is otherwise indistinguishable from optimal under standard MIP gap
tolerances.

**Objective.** $$\begin{equation}
\min_{p,x,\tau,n,y,z,\eta,\xi} \quad
\eta + \frac{1}{1-\beta} \sum_{\omega \in \Omega} \pi^\omega \xi^\omega
\label{eq:objective}
\end{equation}$$

*First-stage constraints.* Site selection is bounded by a maximum count,
a cost-weighted budget, and a theater-wide inventory ceiling:
$$\begin{align}
\sum_{i \in N^P} p_i &\leq P_{\max} \label{eq:pmax} \\
\sum_{i \in N^P} f_i\, p_i &\leq B \label{eq:budget} \\
\sum_{i \in N^P} \bar{q}_{ir}\, p_i &\leq \bar{Q}_r \quad \forall r \in R \label{eq:invceiling} \\
p_i &\in \{0,1\} \quad \forall i \in N^P \label{eq:binary}
\end{align}$$ Under current parameters ($f_i = 1$ for all candidates,
$B = 12$, $P_{\max} = 3$),
Equation [\[eq:budget\]](#eq:budget){reference-type="ref"
reference="eq:budget"} can never bind.
Equation [\[eq:invceiling\]](#eq:invceiling){reference-type="ref"
reference="eq:invceiling"} is *not implemented* in the current model ---
no corresponding constraint or $\bar{Q}_r$ parameter exists in the
codebase. It is retained in the written formulation because
$P_{\max} = 3$ would keep it non-binding even if implemented;
implementation becomes necessary when $P_{\max}$ is expanded to the 5-
and 7-site analyses identified as future work.

*CVaR constraints.* $$\begin{align}
\xi^\omega &\geq L^\omega - \eta \quad \forall \omega \in \Omega \label{eq:cvar1} \\
\xi^\omega &\geq 0 \quad \forall \omega \in \Omega \label{eq:cvar2}
\end{align}$$

*Inventory release.* $$\begin{align}
y^\omega_{ir} &\leq (1-\rho)\,\bar{q}_{ir}\,a^\omega_{ir}\,p_i     \quad \forall i \in N^P, r \in R, \omega \in \Omega \label{eq:release} \\
y^\omega_{ir} &= 0 \quad \forall i \in N \setminus N^P, r \in R, \omega \in \Omega \label{eq:norelease}
\end{align}$$

*Aggregate flow balance.* The conservation law of the model:
$$\begin{align}
\sum_{m}\sum_{j:(j,i)\in A_m} x^\omega_{m,jir} + y^\omega_{ir} + z^\omega_{ir} &= d^\omega_{ir} + \sum_{m}\sum_{j:(i,j)\in A_m} x^\omega_{m,ijr} \quad \forall i \in N^P, r, \omega \label{eq:balppl} \\
\sum_{m}\sum_{j:(j,i)\in A_m} x^\omega_{m,jir} + z^\omega_{ir} &= d^\omega_{ir} + \sum_{m}\sum_{j:(i,j)\in A_m} x^\omega_{m,ijr} \quad \forall i \in N \setminus N^P, r, \omega \label{eq:balnon}
\end{align}$$

*Per-mode outbound feasibility with transfer.* Outbound flow on a mode
must be backed by inbound flow on that mode, transfers into it, or (at
PPLs) released inventory: $$\begin{align}
\sum_{j:(i,j)\in A_m} x^\omega_{m,ijr} + \sum_{m_2:(m,m_2)\in T_i} \tau^\omega_{i,mm_2,r} \;\leq\;&
\sum_{j:(j,i)\in A_m} x^\omega_{m,jir} + \sum_{m_1:(m_1,m)\in T_i} \tau^\omega_{i,m_1m,r} + \mathbb{1}[i \in N^P]\, y^\omega_{ir} \notag \\
&\quad \forall i \in N,\ m \in M,\ r \in R,\ \omega \in \Omega \label{eq:permode}
\end{align}$$

*Transfer backing.* Transfers out of a mode cannot exceed what arrived
on that mode by arc plus released inventory: $$\begin{align}
\sum_{m_2:(m,m_2)\in T_i} \tau^\omega_{i,mm_2,r} \;\leq\; \sum_{j:(j,i)\in A_m} x^\omega_{m,jir} + \mathbb{1}[i \in N^P]\,y^\omega_{ir} \quad \forall i, m, r, \omega \label{eq:transferbacking}
\end{align}$$ This constraint is necessary in addition to
Equation [\[eq:permode\]](#eq:permode){reference-type="ref"
reference="eq:permode"}: without it, a closed transfer cycle across
three or more modes at one node (sea$\to$air$\to$land$\to$sea) can
mutually justify itself, each transfer's outbound leg backed by another
transfer's inbound leg with no arc-delivered or released commodity ever
entering the loop --- a self-sustaining, physically baseless cycle.
Transfer backing requires every transferred unit to trace to an arc
delivery or inventory release.

*Capacity constraints.* $$\begin{align}
x^\omega_{m,ijr} &\leq u^\omega_{m,ij,r} \quad \forall m, (i,j) \in A_m, r, \omega \label{eq:arccap} \\
\sum_{r \in R} \tau^\omega_{i,m_1m_2,r} &\leq \kappa_{i,m_1m_2} \quad \forall i, (m_1,m_2) \in T_i, \omega \label{eq:transfercap}
\end{align}$$

*Vehicle conservation.* A vehicle may depart a node only if present
there --- through activated basing or prior arrival within the scenario:
$$\begin{align}
\sum_{i:(i,j)\in A_m} n^\omega_{k,m,ij} + b_{k,j}\,p_j \;\geq\;
\sum_{j':(j,j')\in A_m} n^\omega_{k,m,jj'}
\quad \forall k \in K_m, j \in N, m, \omega
\label{eq:vehcons}
\end{align}$$ No separate fleet-cardinality constraint is imposed. An
earlier formulation capped total type-$k$ arc traversals at $F_k$ per
scenario; this mischaracterized fleet usage by treating every traversal
as permanently consuming one vehicle, restricting a fleet of $F_k$
vehicles to exactly $F_k$ total legs regardless of leg length and
blocking physically feasible multi-leg routing. Since
Equation [\[eq:vehcons\]](#eq:vehcons){reference-type="ref"
reference="eq:vehcons"} already prevents using vehicles that are neither
based nor arrived, and
Equation [\[eq:distbudget\]](#eq:distbudget){reference-type="ref"
reference="eq:distbudget"} bounds total feasible travel by the fleet's
aggregate range, the explicit traversal-count constraint was removed as
both incorrect and redundant.

*Vehicle-capacity-constrained flow.* Flow on an arc is bounded by the
per-resource capacity of the vehicles committed to it: $$\begin{align}
x^\omega_{m,ijr} \;\leq\; \sum_{k \in K_m} \text{cap}_{k,r}\, n^\omega_{k,m,ij} \quad \forall m, (i,j) \in A_m, r, \omega \label{eq:vehcap}
\end{align}$$

*Fleet-wide distance budget.* Total distance consumed by type-$k$
vehicles, plus turnaround penalties on through-flow, cannot exceed the
fleet's aggregate budget: $$\begin{align}
\sum_{m}\sum_{(i,j)\in A_m} \text{dist}_{ij}\, n^\omega_{k,m,ij} + \psi_k \sum_{j \in N} h^\omega_{k,j} \;\leq\; F_k\, D_k \quad \forall k, \omega \label{eq:distbudget}
\end{align}$$ where $h^\omega_{k,j}$ is type-$k$ outbound vehicle flow
at node $j$, exempted only at nodes that are both basing-eligible and
*selected*: the exemption applies where $b_{k,j}\, p_j > 0$, not merely
where $b_{k,j} > 0$. Because $p_j$ is a first-stage binary, the product
of outbound flow and $p_j$ is linearized with a McCormick envelope (an
auxiliary variable $g^\omega_{k,j}$ and three bounding constraint
families), ensuring the turnaround waiver applies only at nodes the
model has actually activated. Charging the penalty on all non-exempt
outbound flow is a conservative linear approximation of exact
intermediate-stop counting.

*Nonnegativity and integrality.* $$\begin{align}
z^\omega_{ir},\, y^\omega_{ir} &\geq 0 \label{eq:nonneg1} \\
x^\omega_{m,ijr} &\geq 0 \label{eq:nonneg2} \\
\tau^\omega_{i,m_1m_2,r} &\geq 0 \label{eq:nonneg3} \\
n^\omega_{k,m,ij} &\geq 0,\ \text{integer} \label{eq:integrality}
\end{align}$$

## Parameter Values {#sec:paramvalues}

Table [5](#tab:paramvalues){reference-type="ref"
reference="tab:paramvalues"} consolidates the numeric values of every
model parameter not already reported in the vehicle and degradation
tables.

::: {#tab:paramvalues}
  Parameter                                   Value                                 Notes
  ------------------------------------------- ------------------------------------- ------------------------------------------------------------------------------------------------------
  $\beta$                                     0.9                                   CVaR confidence level
  $\delta_{ir}$                               500                                   Uniform across nodes and resources
  $P_{\max}$                                  3                                     Expansion to 5 and 7 planned
  $B$                                         12                                    Non-binding: $f_i = 1\ \forall i$, $B > P_{\max}$
  $f_i$                                       1                                     All hub types
  $c_{m,ij}$                                  $\text{dist}_{ij} \times 10^{-4}$     Uniform across modes
  $\theta_{m_1m_2}$                           15.0 / 2.5 / 4.0 / 3.5 / 6.0 / 10.0   sea$\to$air / sea$\to$land / air$\to$land / land$\to$sea / land$\to$air / air$\to$sea
  $\varepsilon$                               0.04                                  Vehicle-movement tie-breaker (Eq. [\[eq:loss\]](#eq:loss){reference-type="ref" reference="eq:loss"})
  $\phi_{\text{food}}, \phi_{\text{water}}$   0.15, 0.20                            Per-capita demand rates
  $\rho$                                      0.20                                  Safety-stock fraction
  Availability cutoff                         5.0                                   $a^\omega_{ir} = 0$ if $s^\omega_i \geq 5.0$; inactive under current calibration (see below)
  pd per MT                                   1,852 (food), 66.7 (water)            Sphere/WFP basis; shared by arc and vehicle capacities
  $\pi^\omega$                                $1/|\Omega|$                          Equal SAA weights

  : Model parameter values.
:::

The inventory availability mechanism never fires under the current
calibration: realized severity is strictly below 5 by construction
(Section [3.4](#sec:scenariogen){reference-type="ref"
reference="sec:scenariogen"}), so $a^\omega_{ir} \equiv 1$ in every
scenario. The mechanism is retained as headroom for calibrations in
which extreme severities can disable a supply origin, and is documented
here as inactive rather than presented as a live effect.

## Inventory Parameters {#sec:inventory}

Inventory capacity is assigned by tier: PPL-1 holds
$\bar{q}_{ir} = 1{,}300{,}000$ person-days per resource, PPL-2
$600{,}000$, and PPL-3 $150{,}000$, with safety-stock fraction
$\rho = 0.20$ retained at every activated site. These values were
recalibrated downward from an earlier parameterization (5,000,000 /
1,500,000 / 300,000 with $\rho = 0.25$) under which inventory satisfied
demand in nearly every scenario regardless of network condition,
producing service rates near 1.0 across the full $\alpha$ sweep and
leaving the model outside the regime in which network uncertainty can
affect outcomes. The reduced values place the model where genuine unmet
demand occurs across a meaningful share of scenarios. They remain
planning parameters subject to calibration against actual APS-4 theater
set composition with USARPAC J4 input.

## Computational Implementation {#sec:computation}

The model is implemented in Python with Gurobi 11.0.3, solved as a
Sample Average Approximation with $|\Omega| = 100$ scenarios (seed 32)
and equal scenario weights $\pi^\omega = 1/|\Omega|$. Equal weighting is
the standard SAA convention and is distinct from the non-uniform,
EM-DAT-calibrated probabilities used to *draw* the scenarios: the
calibration determines which scenarios appear in the sample and how
often, and the sample average is then an unbiased estimator of the
expectation under the calibrated distribution; weighting realized
scenarios unequally on top of non-uniform sampling would double-count
the calibration.

With the vehicle extension active, the extensive form includes the
integer vehicle-count family $n^\omega_{k,m,ij}$ and four additional
constraint families (vehicle conservation, vehicle-capacity flow,
distance budget with McCormick linearization, and transfer backing) in
every scenario. Solve times at a 1 percent MIP gap are on the order of
five minutes per instance on single-machine hardware, compared with
7--12 seconds for the earlier formulation without vehicle constraints;
final reporting runs use a 0.01 percent gap, at roughly 30 minutes per
instance, which eliminates residual low-cost circular flows that survive
within a looser gap tolerance. A time limit of 3,600 seconds per solve
is imposed as a safeguard.

**Convergence analysis.**

**Codebase.** The implementation is a layered Python package with four
stages: network construction (`network/`), scenario generation
(`scenarios/`), instance assembly and MIP formulation (`model/`), and
sensitivity analysis (`analysis/`). All parameters are maintained in a
single consolidated configuration file (`config/model_parameters.yaml`)
read by one loader, so capability thresholds, degradation values,
vehicle specifications, and calibration constants can be adjusted
without code changes.

