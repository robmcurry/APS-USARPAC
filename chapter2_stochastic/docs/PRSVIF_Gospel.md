# Introduction

## Setting the Theater as an Irrevocable Decision

Setting the theater is the broad range of activities continuously conducted to establish the conditions for executing operations across an area of responsibility . For the Army this means posturing forces and capabilities and establishing the footprints and agreements that grant access to ports, terminals, airfields, and other strategic sites . Each geographic combatant commander (GCC) sets the theater within their assigned area and delegates that responsibility to the Army Service Component Command, which serves as the theater army and holds the greatest Service capacity for the task . The theater army oversees area-wide distribution, recovery, and redistribution of supplies and equipment, and executes much of this through its theater sustainment command . U.S. Army Pacific (USARPAC) serves as the theater army for U.S. Pacific Command (USPACOM).

What those activities leave in place is theater posture, and prepositioning is its sustainment component: stocks and transport held forward so that response does not wait on movement from outside the theater. This posture is not maintained for armed conflict alone. Most of the world’s natural disaster activity occurs within the USPACOM area of responsibility , and when an event occurs, the theater must respond. Humanitarian assistance and disaster relief (HA/DR) therefore draw on strategic commitments made years in advance.

Resources are finite so perfect posture cannot be established everywhere at once. Each commitment becomes a choice which PPLs hold stock, how much each holds, and where transport is based. These are not preparations for an operation. They are the first and most irrevocable decisions of one. Once they are made, what has been committed cannot be recovered without penalty . No commander can redirect an access agreement or a constructed facility when a disaster arrives. The posture is fixed, and its quality is determined before any consequence is observed.

## Planning Under Uncertainty

Prepositioning and posture decisions are difficult, but not simply because the environment is complex. The difficulty is timing. Planners cannot observe the metrics used to judge a posture decision at the moment they choose it: where the disaster or conflict will occur, how severe it will be, which ports, roads, and airfields will survive, and the residual resource demand signal. These unknowns are also correlated. The event that drives demand at a node is the same event that degrades the infrastructure serving it, so a posture is tested hardest precisely where it has been weakened. Commercial logistics measures performance in cost. Humanitarian response has no equivalent. Doctrine distinguishes measures of performance, task accomplishment, from measures of effectiveness, achieving an objective or a desired operational effect . HA/DR posture is judged on effectiveness — the share of post-disaster demand the theater satisfies — which is not measured monetarily and depends on an unobservable future state of the environment. Prepositioning decisions must therefore be judged against a criterion chosen a priori rather than observed, and the choice matters. A posture optimized to minimize average unmet demand and one optimized against the worst outcomes need may look different.

Doctrine directs planners to engage uncertainty rather than avoid it. Sustainment planners must understand, balance, and take prudent risk while developing mitigation alternatives as part of the sustainment plan . They analyze, simulate, and assess courses of action (COAs) at prescribed steps within the joint planning process (JPP) . Alternatively, doctrine does not prescribe how to represent risk, how to measure alternatives against it, or what constitutes prudence in a particular decision. Those judgments rely on professional experience, which is often sound but difficult to examine, repeat, or defend in the way a stated criterion can. The shortcoming lies not in the planning process but in the tools available within it.

The cost of that gap is not inefficiency. It is demand that goes unmet in the days after an event, among populations that have already lost the capacity to meet it themselves, at a moment when the posture can no longer be changed.

## Analytical Gap in Campaign-Level Sustainment Planning

Our previous work addressed a narrow version of this problem, combining a disaster scenario generation engine with a deterministic mixed-integer program (MIP) to determine prepositioning locations, routing, redundancy, and post-response posture . Solving a MIP across the full set of generated scenarios revealed which prepositioning locations persist across plans and which operational capabilities bind. Our analytical framework also quantified tradeoffs that logisticians have long navigated intuitively.

Each generated plan is optimal for a different assumed future, so committing to any one of them means explicitly planning for the disaster it was built for, and the formulation does not weight the most severe outcomes. The broader literature provides examples of both capabilities, but never in the same model and never at a scale commensurate with the Pacific theater. The study of prepositioning under uncertainty is mature: two-stage stochastic formulations are well established, risk-averse variants using conditional value-at-risk (CVAaR) are available, and location-routing models integrate siting with vehicle assignment . Each was developed on problems small enough that siting and fleet decisions could be treated separately, which USPACOM distances do not permit. Section 2 examines the success and gaps of these methods for handling uncertainty. Consequently, the question theater sustainment planners actually ask remains open: which prepositioning strategies, implemented before an event, are resilient to network degradation and demand surge occurring together, and how does that set change based on a commander’s risk profile?

## Research Contribution

We develop PRS-VIF, a vehicle-indexed formulation of the prepositioning and routing strategy problem and a risk-aware two-stage stochastic program that determines theater posture under a conditional value-at-risk objective. Our formulation makes three contributions to the prepositioning literature. First, it determines PPL activation and the basing of individually indexed, heterogeneous vehicles jointly as first-stage decisions. This ensures transportation capacity is generated by a fleet whose position is chosen rather than assumed as a property of the network. Second, it represents capacity degradation as a continuous function of both the affected transportation mode and the disaster type, rather than a binary condition of the arc. Third, it calibrates epicenter location, disaster type, and severity empirically from authoritative historical data rather than assumed distributions. We demonstrate the effectiveness of our model on a fifty-node USPACOM network.

We align stochastic modeling with the JPP rather than proposing an alternative to it. When considering various courses of action to determine posture, planners compare alternatives under uncertainty. This is the stage of the JPP our scenario-based optimization model supports directly. Our scenarios supply the range of conditions for testing alternatives, and the model reports how each posture performs across them in a form the planning process already consumes. The intent is not to replace planner judgment but to provide a basis that can be examined, repeated, and defended.

## Chapter Organization

The remainder of this paper is organized as follows. Section 2 reviews joint doctrine as a structured decision system and the operations research literature on prepositioning under uncertainty, closing with the gap this work addresses. Section 3 identifies the decision maker and the decisions in scope, develops the theater network, scenario generation and calibration, and the disaster impact model, then presents the PRS-VIF formulation. Section 4 reports and interprets computational results on our synthetic case study in the Pacific Theater. Section 5 concludes. Appendix A presents an aggregate type-indexed formulation retained as a tractability benchmark.

# Literature Review

In section 2.1, we examine the doctrinal framework related to sustainment decision-making and the gaps in prescribed analytical methods. Section 2.2 then reviews the operations research literature on prepositioning under uncertainty.

## Joint Doctrine and the Analytical Disconnect

Section [1.2](#sec:planning) established that doctrine directs planners to take prudent risk without specifying how to quantify it. Joint publications govern HA/DR planning through the JPP and its supporting coordination mechanisms , and each stage yields products through which decisions are framed and communicated. These artifacts record identified assumptions and risks, but lack the form necessary to make quantitative comparisons. Operational assessment doctrine provides the means of judging execution using measures of performance (MOPs) and effectiveness (MOEs), but still lacks methods for quantitatively evaluating decisions under uncertainty .

Planning outputs are executed and refined through the Battle Rhythm, the routine cycle of command and staff activities that synchronizes current and future operations . During these cycles, logisticians rely on running estimates and demand updates to maintain awareness of current capability, capacity, and throughput. Air, sea, and terrestrial transport operations are coordinated within discrete planning intervals, but theater prepositioning precede these cycles. Resource siting and transportation basing are committed in advance of any specific event and are not revisited. The Battle Rhythm simply governs operational alignment of assets already in place from intial prepositioning. Therefore, the relevant question is whether the committed posture can deliver effectively within the response window, not how distribution is sequenced across successive intervals. We represent that window as a single planning horizon rather than as a sequence of periods, enforcing vehicle range and turnaround through distance budgets. This matches the level at which the decision is made. Strategic posture determines what capability exists in theater. Sequencing is an execution problem the Battle Rhythm resolves once posture is fixed.

The disconnect runs in both directions. Doctrine directs planners to take prudent risk without prescribing how to quantify it, and the analytical literature quantifies risk without reference to the process. Altay and Green  first characterized it as a problem of engagement, with operations research contributing little relative to the scale of the problem. Today, the models exist, but evidence of implementation is limited. A systematic review of prepositioning studies finds that quantitative models are underutilized in practice, and that the field offers little evidence of their application to real decisions . Reviews of the broader disaster operations literature reach a similar conclusion . Doctrine does not specify how uncertainty should be analyzed, and the analytical literature has developed largely without reference to the process that would consume its results. Our research addresses both halves of that disconnect.

## Prepositioning Under Uncertainty

Next, we review the optimization literature on prepositioning and relief distribution. Table [\[tab:litmatrix\]](#tab:litmatrix) summarizes the models discussed below according to the decisions each represents, its treatment of risk, and how it represents transportation capacity and network capacity degredation.

| Study                    | Uncertainty                              | Risk measure            | Prepositioning decision                  | Fleet representation | Fleet decision   | Modes                | Capacity degradation             |
| :----------------------- | :--------------------------------------- | :---------------------- | :--------------------------------------- | :------------------- | :--------------- | :------------------- | :------------------------------- |
| **This paper (PRS-VIF)** | Demand, network capacity                 | CVaR                    | Sites                                    | Individual, routed   | Basing           | Three                | Continuous, by mode and disaster |
| Barbarosoğlu & Arda      | Demand, arc capacity, supply             | Expected value          | None (flow only)                         | None                 | None             | Multiple             | Random arc capacity              |
| Balcik & Beamon          | Demand                                   | Expected value          | Sites and stock                          | None                 | None             | One                  | None                             |
| Ukkusuri & Yushimito     | Link failure                             | Path reliability        | Sites                                    | None (paths)         | None             | One                  | Link failure                     |
| Rawls & Turnquist        | Demand, network, stock                   | Expected value          | Sites and stock                          | None                 | None             | One                  | Arc availability                 |
| Döyen et al.             | Demand                                   | Expected value          | Sites and stock, two echelon<sup>a</sup> | None                 | None             | One                  | None                             |
| Noyan                    | Demand, transport capacity, stock damage | Mean-CVaR               | Sites and stock                          | None                 | None             | One                  | Arc capacity                     |
| Alem et al.              | Demand, arc availability, supply, budget | CVaR, semidev., minimax | Stock only                               | Counts per arc       | Sizing           | Multiple<sup>b</sup> | Binary, by vehicle type          |
| Zhong et al.             | Demand                                   | CVaR with regret        | Sites                                    | Routed, homogeneous  | Depot assignment | One                  | None                             |
| Wei et al.               | None                                     | None                    | Depots                                   | Routed, homogeneous  | Number used      | One                  | None                             |
| Ertem et al.             | None                                     | None                    | Hubs given                               | None                 | None             | Three                | None                             |
| Zhang & Chen             | Correlated demand                        | Distributionally robust | Sites and stock                          | None                 | None             | One                  | None                             |
| Ren et al.               | Road capacity                            | Expected value          | None                                     | Counts, four types   | None             | One                  | Road capacity                    |
| Taghvaei & Rabbani       | Demand, hub and arc disruption           | \(p\)-hub center        | Hubs and fortification                   | None                 | None             | Multiple             | Hubs and arcs                    |

<sup>a</sup>Post-disaster rescue centers are sited in the second stage, placing binary variables in the recourse problem.  
<sup>b</sup>Vehicle types include trucks, boats, and helicopters, which imply distinct modes, though modal arc layers are not represented explicitly.

Research modeling disaster logistics as an optimization problem is relatively new as much of the work below published after Altay and Green’s call to action. Barbarosoğlu and Arda  were among the first to formulate disaster response as a two-stage stochastic program. Their work modeled transportation of first-aid commodities over an urban network as a multi-commodity, multi-modal flow problem where arc capacity, supply, and demand are realized together as a scenario. Their analysis focused on movement plans rather than prepositioning. The two stages are defined by the iterative information disclosure during a response, not preposition commitments.

Balcik and Beamon  brought the prepositioning decision into the optimization problem. Their formulation determines both the number and location of distribution centers and the quantity of each relief item at each location through a variant of the maximal covering location model subject to pre- and post-disaster budget constraints. Rawls and Turnquist  combined the two lines, embedding siting and stocking decisions in the first stage of a two-stage stochastic program whose recourse distributes commodities over a network with scenario-dependent link capacity. Their formulation provides a structural template for much of the subsequent work, including our own. Döyen et al.  extended this across echelons, siting pre-disaster rescue centers in the first stage and post-disaster centers in the second. This places binary variables in the recourse problem and complicates decomposition accordingly.

Two features of their work persist in much of the proceeding literature. Transportation capacity is a property of the arc rather than any vehicle, and performance is measured based on expected cost. The second is a poor fit for theater posture. A posture is committed years in advance, endures once set, and it is judged on effectiveness rather than on cost. Expected cost combines probability and consequence into a single average, which can understate the importance of rare but severe scenarios. To consider such scenarios, Value-at-Risk (VaR) offers a simple alternative by setting a loss threshold at a chosen confidence level, but it says nothing about loss severity beyond the threshold. Artzner et al.  also show that VaR can fail subadditivity and is therefore not generally a coherent risk measure. CVaR addresses both problems by considering losses in the tail beyond the VaR threshold. Rockafellar and Uryasev  show that CVaR has useful convexity properties and can be written in a linear form for finite scenario sets. Krokhmal et al.  extend this approach to cases where CVaR is implemented as a constraint rather than as the objective.

Noyan  brought this to disaster prepositioning directly, extending the model of Rawls and Turnquist by augmenting its expected-cost criterion with a mean-CVaR objective and developing single-cut and multicut Benders decomposition algorithms to solve it. Risk-averse prepositioning is therefore not an open methodological question, and we adopt the criterion rather than extend it. However, Noyan’s model does not contain a set of heterogenous vehicles. Instead, the recourse problem is a multi-commodity network flow in which transportation capacity enters as a property of the arc, where no vehicle, mode, or basing decision appears in the formulation. Much of the subsequent work focuses on risk treatments and uncertainty while retaining the same basic pre-disaster design and post-disaster recourse structure. Noyan et al.  impose multivariate risk constraints benchmarking performance measures against a reference, Elçi and Noyan  embed relief network design in a chance-constrained two-stage program, Wang et al.  carry a mean-CVaR criterion into a distributionally-robust setting, and Zhang and Chen  address correlated demand through a moments-based distributionally robust location-inventory model. Finally, Bertsimas and Sim  offer the robust alternative through a budget of uncertainty. In each of these formulations risk is a modeling choice rather not an input, and none report the effect of risk variability on posture decisions. We prefer a tail-sensitive measure because the planner’s degree of risk aversion is itself a parameter of interest.

Additional research made transportation decisions more explicit. Ukkusuri and Yushimito  brought network reliability into the prepositioning problem by locating supplies based on the most reliable paths through a network where links may fail. Later location-routing models treated facility location and vehicle routing together. Wei et al.  determine depot locations and vehicle routes at the same time under soft time windows, but their problem is deterministic and solved after the disaster is known. Zhong et al.  add risk aversion to this setting by choosing which distribution centers to open, assigning vehicles to them, and routing those vehicles using a conditional value-at-risk with regret criterion. Their model still assumes a single transportation mode, and demand is the only source of uncertainty while network conditions remain deterministic.

Alem et al.  is the closest formulation to ours. Their two-stage stochastic network flow model includes multiple vehicle types, sizes the fleet before the disaster, and operates over multiple time periods. It also allows several risk measures, including CVaR, semideviation, and minimax regret. Alem et al. also generates transportation capacity through the available fleet rather than treating it as a property of the network. Two differences remain. First, vehicles are represented as counts by type, arc, and time period rather than as individual assets. So, the model does not track the continuity of a specific vehicle across successive movements. Second, the first-stage fleet decision determines how many vehicles of each type to contract, but not where those vehicles are based. The warehouse and relief-center locations are fixed in advance, and the model chooses how much stock to place at those nodes rather than which nodes to open.

Multi-modal relief distribution has developed seperately. Ertem et al.  model containerized movement across road, rail, and sea through intermodal transfer hubs and conclude that intermodal routing is more robust to disruption than any single mode alone. Taghvaei and Rabbani  embed intermodal transportation in a two-stage stochastic hub network with fortification and post-disaster arc restoration under a \(p\)-hub center objective. These models bring transportation modes and network resilience into the relief problem, but they still stop short of making vehicle basing and vehicle tracking decisions jointly. Ertem et al. carry a limited multimodal fleet, but vehicles remain aggregate flows rather than individually identified assets, while Taghvaei and Rabbani model intermodal capacity through the hub-and-spoke network rather than through an explicitly based fleet.

Modeling network degradation has received less attention in the recent literature. Many formulations treat a link as either available or unavailable rather than allowing capacity to decline by degree. Kamyabniya et al.  identify unrealistic assumptions about road and network conditions as a persistent weakness in the literature, noting that only 3 of 106 reviewed studies considered infrastructure resistance and that perfect information about road and traffic conditions remains common. Ren et al.  are a recent exception. Their multi-stage stochastic model allows road capacity to fall continuously across earthquake-collapse scenarios and coordinates four transportation modes, but the problem remains focused on post-disaster scheduling and minimizes expected cost. Alem et al.  also make arc availability depend on vehicle type and scenario, but only as a binary indicator of whether a vehicle type can use a given arc. These studies show progress toward more realistic network disruption, but they do not combine disaster-specific degradation, pre-disaster fleet posture, and risk-sensitive planning in a single formulation. Kamyabniya et al.  further recommend models tailored to the type, evolution, size, and effects of specific disasters rather than relying on overly general formulations.

Applied work in military and defense settings has developed alongside this literature. Apte and Yoho  optimize the allocation of U.S. Navy ships to humanitarian missions, and Apte  surveys strategic prepositioning as a model for humanitarian readiness. Boone et al.  formulate scheduled service network design for Marine Corps expeditionary logistics, routing heterogeneous connectors under range and capacity constraints, and Murphy  evaluates Marine Corps response options in USPACOM. Strinsky  treats multi-commodity flow in a degraded environment through chance constraints and a two-stage model with randomly realized disruption, which is risk-averse but opposition-driven rather than disaster-driven and carries no prepositioning decision. While valuable, none of these address the unique aspects of setting the Army theater posture.

## Research Gap

Together this body of work illustrates both the progress made and the gaps that remain. Three limitations remain. First, where infrastructure degradation is represented, it is a property of the network rather than of the interaction between disaster type and mode. Second, the fleet is represented at a coarser resolution than the posture decision requires. Risk-averse formulations size fleets without locating them ; routing formulations locate individual vehicles but assume a homogeneous fleet, a single mode, and a deterministic network . We are aware of no formulation in which individually indexed, heterogeneous vehicles operate across multiple modes and their basing is determined jointly with PPL selection in the first stage. Third, practical uptake remains limited, as cited above.

For Army theater sustainment in the USPACOM, these limitations compound rather than accumulate. The disaster response literature is almost entirely urban or regional in scale, comprising networks of thirty to fifty nodes spanning hundreds of kilometers within which a single mode reaches the entire network and intermodal movement functions as a resilience option. At theater scale no single mode reaches the entire network, so multi-modal movement is a structural necessity and the mode-specific vulnerability of each layer becomes a first-order determinant of whether relief arrives. Strategic lift is scarce relative to the demand a major event generates, which makes the fleet and its prepositioned location the more frequently binding constraint rather than the fleet’s capacity. Finally, Army planners do not seek to minimize expected cost, like many humanitarian organizations. Military staff executes a doctrinally specified planning process, which imposes requirements on the form analytical results must take if they are to be used at all.

What is therefore missing is a formulation that: a) determines prepositioning siting and fleet basing jointly before the disaster is realized; b) models a tail-sensitive objective, a multi-modal network whose capacity degrades as a function of both the mode and the type of disaster; and is calibrated from the empirical disaster record and expressed through the artifacts by which theater posture decisions are actually made. This research develops and demonstrates that formulation.

# Methods

## The Theater Sustainment Decision Maker

The USARPAC Commanding General determines theater logistics posture decisions are made by , based on recommendations from the 8th Theater Sustainment Command (8th TSC) and in coordination with the USARPAC staff. Three decisions define a posture plan: which sites serve as a preposition location (PPL), resource inventory level at each PPL, and how transportation assets are based. The PRS-VIF address the first and third. Resource inventory is a parameter determined by the tier-based categorization of each PPL since storage capacity is a function of infrastructure. Our model determines the activated PPL sites rather than how much inventory to hold because we assume each PPL will store the maximum allowable amount of each resource. The deterministic predecessor to our model constrained the regional balance of stock remaining after a response . The PRS-VIF does not. Safety stock is tracked at every node, but theater reserve is established from the beginning by holding back a fixed fraction \(\rho\) of inventory from release at every activated PPL. Safety stock is therefore distributed with the posture itself rather than by a post-disaster balancing requirement. PRS-VIF provides the fourth decision vehicle-by-vehicle rather than by class.

Posture is determined before a disaster occurs (Section [1.2](#sec:planning)). Thus, we implement a two-stage stochastic program to model in which the first stage activates PPL sites and determines where transport vehicles are based. The second stage routes those vehicles and resources based on the residual demand signal across the network. The CVaR objective at confidence level \(\beta\) scores a posture by its performance in the worst scenarios rather than a sample average approximation, based on the liklihood of the given set of scenarios.

## Network and Fleet

In this work and computational case study, our network comprises \(|N| = 50\) nodes spanning the USPACOM area of responsibility: major population centers, commercial ports, and military logistics facilities across East Asia, Southeast Asia, Oceania, and the central Pacific. Each node \(i \in N\) has a capability rating \(\lambda_{i,m} \in \{1,2,3\}\) for each mode \(m \in M\) — sea port, airfield, and land distribution. These ratings determine arc eligibility, PPL tier, and vehicle basing eligibility, as described later.

All 50 nodes are eligible disaster epicenters. The subset \(N^P\) includes 22 PPL candidates, requiring a US access agreement in force, sea or air capability sufficient to receive prepositioned stock, and land capability to distribute it without major augmentation. Candidates are classified into three tiers — PPL-1 Strategic Hubs, PPL-2 Operational Nodes, and PPL-3 Contingency Sites that determine the maximum inventory and vehicle basing eligibility. Three nodes, Shanghai, Guangzhou, and Chengdu, carry \(\lambda_{i,\text{sea}} = \lambda_{i,\text{air}} = 1\) and no land corridor, making them epicenter-eligible but unreachable. They are included so that China’s 23.4% share of the historical natural disaster occurrences (Section [\[sec:epicenterweights\]](#sec:epicenterweights)) is represented. USPACOM’s prepositioning mission does not include direct delivery to mainland China. If a disaster does occur at one of these three nodes, the demand at that node remains at zero while neighboring nodes outside of China may be affected.

### Modal Arcs

Our model includes three arc layers: \(A_{\text{sea}}\), \(A_{\text{air}}\), and \(A_{\text{land}}\). Each layer contains unique infrastructure requirements, throughput rates, and disruption mechanisms of air, maritime, and terrestrial movement. Node pairs may be connected on one, several, or no modes. Arc existence combines a capability threshold at both endpoints with a distance limit consistent with the 72-hour response window:

  - **Maritime** (\(A_{\text{sea}}\)): \(\lambda_{i,\text{sea}} \geq 2\) at both endpoints, great-circle distance \(\leq 4{,}500\) km.

  - **Air** (\(A_{\text{air}}\)): \(\lambda_{i,\text{air}} \geq 2\) at both endpoints, distance \(\leq 6{,}000\) km.

  - **Land** (\(A_{\text{land}}\)): a manually specified set of 15 contiguous node pairs (30 directed arcs) representing documented overland corridors, since overland feasibility depends on a continuous road or rail corridor rather than proximity.

This yields \(|A_{\text{sea}}| = 236\), \(|A_{\text{air}}| = 1{,}502\), and \(|A_{\text{land}}| = 30\) directed arcs.

### The Vehicle Fleet

We use a heterogeneous fleet to model resource movement. The arc capacity is vehicle-dependent as opposed to a generic assigned throughput. Air is served by two types, the C-17 Globemaster III and the C-130J Super Hercules, reflecting materially different payload and runway requirements; sea and land by one representative type each, the LCU-1700 landing craft and the M1083 medium tactical vehicle. Set \(K\) contains all specific vehicle types, with \(K_m \subseteq K\) the types operating on mode \(m\). Specifications and adopted parameters are reported in Appendix [6](#app:params).

First-stage decisions include where all vehicles are based. Each type may be based only at PPL sites meeting a type-specific capability rating, drawn from the same \(\lambda\) that determines arc eligibility: the C-17 requires \(\lambda_{j,\text{air}} = 3\), the C-130J requires \(\lambda_{j,\text{air}} \geq 2\), the LCU-1700 requires \(\lambda_{j,\text{sea}} \geq 2\), and the M1083 requires \(\lambda_{j,\text{land}} \geq 1\) \(\forall j \in N\). The three scarce, high-value types are further restricted to PPL-1 Strategic Hubs, reflecting the practice of concentrating strategic lift at established theater hubs; while the M1083 may be based across all three tiers. These rules define the eligible set \(J_k\) containing all nodes which vehicle type \(k\) can be based. Fleet sizes \(F_k\) are fixed planning assumptions.

We measure vehicle capacity in metric tons. Payloads convert to person-days through two humanitarian reference rates: the World Food Program (WFP) emergency ration of 540 g per person per day and the Sphere minimum of 15 L of water per person per day. Water is roughly 28 times heavier than food per person-day, so water binds vehicle weight capacity long before food does. The per-unit weights \(w_r\) for resource \(r\) convert person-day quantities to weight wherever a capacity limit applies, so mixed loads are represented exactly rather than approximated.

### Time Representation

Our model does not explicitly model the passage of time. Instead, we represent time as a response window based on a distance budget. Each vehicle of type \(k \in K\) with effective cruise speed \(v_k\) has a budget \(D_k = \kappa v_k\) over a horizon of \(\kappa = 3\) days. Every leg consumes its arc distance plus a turnaround penalty \(\psi_k = (\tau_m/24)\,v_k\) where \(\tau_m\) is the service time in hours at a stop on mode \(m\). A vehicle arriving at an intermediate stop must offload and reset before departing, and \(\psi_k\) charges that time as distance the vehicle could otherwise have covered.The budget constrains each vehicle separately rather than the fleet as a whole. Our formulation distinguishes one vehicle traveling several legs from several vehicles travelling on each to ensure that this strategic plan remains operationally feasible when a disaster occurs.

## Scenario Generation and Demand

Each scenario \(\omega \in \Omega\) is a single natural disaster event having a disaster type \(\nu(\omega) \in \mathcal{T}\), epicenter location, and disaster severity all calibrated from a historical record. Residual node-specific severity, demand, and network degradation follow from these calibrations.

Our calibrations are based on the Emergency Events Database (EM-DAT) maintained by CRED  We filter to include only natural disasters in the USPACOM subregions and six HA/DR-relevant types: flood, storm, earthquake, mass movement (wet), volcanic activity, and wildfire. The filter yields 2,666 events from 2000 to 2026, with 2026 a partial year. Flood (1,050) and storm (1,003) account for 77 percent of the record, followed by earthquake (307), mass movement (181), volcanic activity (74), and wildfire (51). EM-DAT is the standard open-access disaster database in the humanitarian logistics literature, and its methodology and limitations are documented by Guha-Sapir and Below .

Epicenter countries are drawn from a multinomial distribution with weights
\[w_c = \frac{n_c}{\sum_{c \in C} n_{c}},
\label{eq:countryweights}\]
where \(n_c\) is the recorded event count for country \(c\). Table [1](#tab:weights) reports the top fifteen; China (23.4%), Indonesia (15.4%), and the Philippines (14.3%) account for over half of sampled epicenters. Within the selected country the epicenter node is drawn uniformly for tractability.

<div id="tab:weights">

| Country           | Event Count | Weight |
| :---------------- | ----------: | -----: |
| China             |         624 |  0.234 |
| Indonesia         |         411 |  0.154 |
| Philippines       |         381 |  0.143 |
| Viet Nam          |         197 |  0.074 |
| Japan             |         165 |  0.062 |
| Australia         |         124 |  0.047 |
| Thailand          |         120 |  0.045 |
| Malaysia          |          80 |  0.030 |
| Taiwan            |          73 |  0.027 |
| Republic of Korea |          63 |  0.024 |
| Myanmar           |          59 |  0.022 |
| Papua New Guinea  |          53 |  0.020 |
| New Zealand       |          38 |  0.014 |
| Lao PDR           |          30 |  0.011 |
| Cambodia          |          29 |  0.011 |

Top-fifteen epicenter sampling weights from EM-DAT frequency, 2000–2026.

</div>

Disaster type is then drawn conditional on the epicenter’s country, from a country-specific frequency table derived from the same filtered record. Countries without a usable historical data revert to global expected values among all disasters in all countries.

### Severity and Spatial Decay

Disaster severity \(\sigma\) is based on EM-DAT’s Total Affected field, the most consistently reported impact measure and the one aligned with the demand formulation. Total Affected spans several orders of magnitude, so raw values are log-transformed and then min-max normalized,
\[\tilde{\sigma} = \frac{\log(1+\text{Affected}) - \log(1+\text{Affected})_{\min}}{\log(1+\text{Affected})_{\max} - \log(1+\text{Affected})_{\min}},
\label{eq:severitynorm}\]
where \(\tilde{\sigma}\) places each recorded event within the observed range of log impact and is bounded on the unit interval. Rescaling by \(\sigma = 1 + 4\tilde{\sigma}\) maps that range onto \([1,5]\), the common scale on which the affected radius, the degradation ratio, and the availability threshold are all defined. A severity of 1 corresponds to the least impactful event in the record and 5 to the most.

We fit three bounded parametric families and ranked them by the Kolmogorov–Smirnov statistic \(D_n\). The Kumaraswamy distribution (\(a = 2.417\), \(b = 3.747\)) fit best, at \(D_n = 0.0448\) against \(0.0564\) for Beta and \(0.2036\) for the truncated Normal, and we use it for severity draws. A kernel density estimate fits better still (\(D_n = 0.0122\)) but has no closed-form inverse CDF and so cannot support reproducible sampling. Because all parameters are estimated from the same sample, we use \(D_n\) to compare fits rather than as a hypothesis test .

The impact area extends beyond the epicenter. The affected radius grows linearly with epicenter severity \(\sigma\),
\[\text{radius}(\sigma) = 250 + 250\,\sigma \quad \text{(km)},
\label{eq:radius}\]
where the intercept sets a minimum footprint of 250 km, so that even the least severe recorded event affects a neighborhood rather than a single node, and the slope adds a further 250 km per unit of severity. These constants are a planning-scale calibration and are candidates for subject-matter validation.

Node-level severity then decays linearly with distance from the epicenter within that radius:
\[\sigma^\omega_i =
\begin{cases}
\sigma  \left(1 - \dfrac{\text{dist}(i, \text{epi})}{\text{radius}(\sigma)}\right) & \text{dist}(i,\text{epi}) \leq \text{radius}(\sigma), \\[6pt]
0 & \text{otherwise}.
\end{cases}
\label{eq:decay}\]
For example, a severity-5 event therefore affects nodes within 1,500 km of the epicenter, while a severity-1 event affects nodes within 500 km.

### Demand

Nodal demand is
\[d^\omega_{ir} = \phi_r \sigma^\omega_i  \zeta_i \quad \text{for } i \in N^R, \forall\, r \in R \qquad d^\omega_{ir} = 0 \text{ otherwise},
\label{eq:demand}\]
where \(\phi_r\) is the per-capita demand rate for resource \(r\) (\(\phi_{\text{food}} = 0.15\), \(\phi_{\text{water}} = 0.20\), drawn from the humanitarian logistics literature and flagged for USARPAC G4 calibration), \(\zeta_i\) is node population , and \(N^R \subseteq N\) is the set of nodes reachable by at least one inbound arc across \(A_{\text{sea}} \cup A_{\text{air}} \cup A_{\text{land}}\). This rule zeroes demand at all nodes in mainland China.

## Disaster Impact on Network Capacity

The nominal throughput \(T_{m,ij}\) on arc \((i,j) \in A\) for mode \(m \in M\) follows per mode from the infrastructure tier of the arc’s endpoints — vessel discharge rates for sea, sortie throughput for air, road ratings for land — expressed in metric tons over the planning horizon. Each individual vehicle \(\ell\) inherits the throughput according to its mode, with \(T_{\ell,ij} := T_{m,ij}\,  \forall\ \ell \in L_m\), \(m \in M\). Nominal node handling capacity \(\Theta_{i,m}\) is the number of mode-\(m\) vehicle arrivals node \(i\) can process over the horizon, derived likewise from its capability rating. Activation adds handling capacity \(\Delta\Theta_{i,m}\), reflecting the materials handling equipment, personnel, and throughput augmentation that accompany an established PPL; \(\Delta\Theta_{i,m} := 0\) for \(i \notin N^P\). Siting therefore buys throughput as well as inventory.

Disaster-induced degradation depends on both mode and disaster type. For example, an earthquake damages roads differently than shipping lanes and volcanic ash closes airspace rather than ports. Table [2](#tab:degradmatrix) gives a baseline sensitivity \(\Gamma_{m\nu}\) for all eighteen mode-type pairs, Table [2](#tab:degradmatrix) gives a baseline sensitivity \(\Gamma_{m\nu}\) for all eighteen mode-type pairs. The matrix is a starting calibration pending subject-matter validation, so we scale it uniformly by a sweep parameter \(\alpha \in [0,1]\) to examine posture behavior across a range of disruption intensity without respecifying the matrix itself:
\[\gamma_m(\nu) = \Gamma_{m\nu}\, \alpha,
\label{eq:gammascale}\]
written \(\gamma_\ell\) for \(\ell \in L_m\). At \(\alpha = 0\) the network is undegraded; at \(\alpha = 1\) the full calibrated matrix applies.

<div id="tab:degradmatrix">

| Mode | Flood | Storm | Earthquake | Volcanic | Mass Mvmt. | Wildfire |
| :--- | :---: | :---: | :--------: | :------: | :--------: | :------: |
| Sea  |  0.2  |  0.7  |    0.3     |   0.2    |    0.1     |   0.1    |
| Air  |  0.3  |  0.9  |    0.5     |   1.0    |    0.2     |   0.4    |
| Land |  0.9  |  0.5  |    1.0     |   0.6    |    0.8     |   0.3    |

Baseline degradation sensitivity \(\Gamma_{m\nu}\) by mode and disaster type.

</div>

Respective arc and node capacities are then
\[T^\omega_{\ell,ij} = T_{\ell,ij} \cdot \max\!\left\{0,\ 1 - \gamma_\ell(\nu(\omega))  \frac{\max\{\sigma^\omega_i, \sigma^\omega_j\}}{5}\right\} \qquad \forall \ell \in L,\ (i,j) \in A_\ell,\ \omega \in \Omega,
\label{eq:residual}\]
\[\Theta^\omega_{i,m} = \left(\Theta_{i,m} + \Delta\Theta_{i,m}\,p_i\right) \max\!\left\{0,\ 1 - \gamma_m(\nu(\omega))\frac{\sigma^\omega_i}{5}\right\} \qquad \forall i \in N,\ m \in M,\ \omega \in \Omega.
\label{eq:thetadegradation}\]

In this formulation, an arcs degradation is a function of its most affected endpoint, since both terminals constrain movement regardless of direction. Alternatively, a node degrades by local severity alone. Degradation applies uniformly across vehicle types so a disaster does change a node’s ability to send or receive a specific vehicle type.

All prepositioned resources and based vehicles are assumed destroyed or inaccessible when severity reaches a minimum threshold:
\[a^\omega_i =
\begin{cases}
0 & \text{if } \sigma^\omega_i \geq 1, \\
1 & \text{otherwise},
\end{cases}
\qquad \forall i \in N,\ \omega \in \Omega.
\label{eq:availability}\]

Because severity decays to zero at the edge of the affected radius a node loses availability within \(\text{radius}(\sigma)\,(1 - 1/\sigma)\) of the epicenter: 1,200 km for a severity-5 event, roughly 670 km at severity 3. With that, a PPL inside the impact area cannot transport resources when recieveing substantial relief for aid.

## Risk-Aware Vehicle-Indexed Model

A solution to the PRS-VIF activates PPLs and bases vehicles before a disaster is realized, then routes and allocates in recourse.

#### Sets

  - \(N\): Set of all nodes.

  - \(N^P \subseteq N\): Set of candidate prepositioning locations (PPLs).

  - \(M\): Set of transportation modes.

  - \(A_m \subseteq N \times N\), \(\forall m \in M\): Set of directed arcs available on mode \(m\).

  - \(R\): Set of resource types.

  - \(K\): Set of vehicle types.

  - \(L\): Set of individual vehicles, tracked separately and indexed lexicographically by type.

  - \(L_k \subseteq L\), \(\forall k \in K\): Set of vehicles of type \(k\).

  - \(L_m \subseteq L\), \(\forall m \in M\): Set of vehicles operating on mode \(m\).

  - \(J_k \subseteq N^P\), \(\forall k \in K\): Set of nodes at which vehicles of type \(k\) may be based.

  - \(A_\ell \subseteq N \times N\), \(\forall \ell \in L\): Set of directed arcs available to vehicle \(\ell\).

  - \(\Omega\): Set of disaster scenarios comprising the sample average approximation sample.

#### Parameters

Resource quantities are in person-day units; \(w_r\) converts to metric tons at each capacity limit.

  - \(d^\omega_{ir}\), \(\forall i \in N,\ r \in R,\ \omega \in \Omega\): Demand for resource \(r\) at node \(i\) in scenario \(\omega\) (person-day units).

  - \(\bar q_{ir}\), \(\forall i \in N,\ r \in R\): Maximum inventory of resource \(r\) at node \(i\) (person-day units).

  - \(\rho\): Safety-stock fraction held in storage.

  - \(w_r\), \(\forall r \in R\): Weight of one unit of resource \(r\) (metric tons per person-day unit).

  - \(\text{dist}_{ij\ell}\), \(\forall (i,j) \in A_\ell,\ \ell \in L\): Length of arc \((i,j)\) (km) for vehicle \(\ell\).

  - \(T_{\ell,ij}\), \(\forall \ell \in L,\ (i,j) \in A_\ell\): Nominal throughput of arc \((i,j)\) for vehicle \(\ell\) (metric tons).

  - \(\Theta_{i,m}\), \(\forall i \in N,\ m \in M\): Baseline handling capacity of node \(i\) for mode \(m\) (vehicle arrivals per planning horizon).

  - \(F_k = |L_k|\), \(\forall k \in K\): Fleet size of vehicle type \(k\).

  - \(v_k\), \(\forall k \in K\): Effective cruise speed of vehicle type \(k\) (km/day).

  - \(D_{\ell} = \kappa\, v_\ell\), \(\forall \ell \in L\): Distance budget over the planning horizon (km).

  - \(\psi_{\ell} = (\tau_{\ell}/24)\, v_{\ell}\), \(\forall \ell \in l\): Turnaround penalty for vehicle \(\ell\), expressed as the distance the vehicle could have travelled during the service duration \(\tau_{\ell}\) (hours) incurred at each stop.

  - \(\text{cap}_k\), \(\forall k \in K\): Payload capacity of vehicle type \(k\) (metric tons).

  - \(\nu(\omega)\), \(\forall \omega \in \Omega\): Disaster type in scenario \(\omega\).

  - \(\sigma^\omega\), \(\forall \omega \in \Omega\): Epicenter severity in scenario \(\omega\), drawn from the fitted Kumaraswamy distribution and valued in \((1,5)\).

  - \(\sigma^\omega_i\), \(\forall i \in N,\ \omega \in \Omega\): Severity at node \(i\) in scenario \(\omega\), valued in \([0,5)\), with \(\sigma^\omega_i = 0\) at nodes outside the affected radius.

  - \(\Gamma_{m,\nu}\), \(\forall m \in M\): Susceptibility of mode \(m\) to disaster type \(\nu\).

  - \(\alpha\): Global degradation scaling factor, with \(\gamma_m(\nu) := \Gamma_{m,\nu}\,\alpha\) written \(\gamma_\ell\) for \(\ell \in L_m\).

  - \(c_{\ell,ij}\), \(\forall \ell \in L,\ (i,j) \in A_\ell\): Transportation cost per unit of resource moved by vehicle \(\ell\) on arc \((i,j)\).

  - \(\delta_{ir}\): Penalty per unit of unmet demand \(\forall\,  r \in R\), \(i \in N\).

  - \(f_i\), \(\forall i \in N^P\): Fixed activation cost of PPL \(i\).

  - \(B\): Activation budget.

  - \(P_{\max}\): Maximum number of PPLs that may be activated.

  - \(\varepsilon\): Small tie-breaking cost per vehicle movement.

  - \(\beta\): CVaR confidence level.

  - \(\pi^\omega\), \(\forall \omega \in \Omega\): Probability of scenario \(\omega\).

Vehicle characteristics are listed in Table [4](#tab:vehicleparams) and referenced per vehicle as \(D_\ell\), \(\psi_\ell\), and \(\text{cap}_\ell\).

#### Variables

First-stage variables are common to all scenarios.

  - \(p_i \in \{0,1\}\), \(\forall i \in N^P\): 1 if node \(i\) is activated as a PPL, and 0 otherwise.

  - \(b_{\ell,j} \in \{0,1\}\), \(\forall \ell \in L,\ j \in J_k\): 1 if vehicle \(\ell\) is based at node \(j\), and 0 otherwise; \(b_{\ell,j} := 0\) for \(j \in N \setminus J_k\).

The remainder are recourse decisions after realizing scenario \(\omega\).

  - \(n^\omega_{\ell,ij} \in \{0,1\}\), \(\forall \ell \in L,\ (i,j) \in A_\ell,\ \omega \in \Omega\): 1 if vehicle \(\ell\) traverses arc \((i,j)\), and 0 otherwise.

  - \(\bar n^\omega_{\ell,i} \geq 0\), \(\forall \ell \in L,\ i \in N,\ \omega \in \Omega\): Equals 1 if vehicle \(\ell\) terminates its route at node \(i\) and 0 otherwise;

  - \(x^\omega_{\ell,ijr} \geq 0\), \(\forall \ell \in L,\ (i,j) \in A_\ell,\ r \in R,\ \omega \in \Omega\): Quantity of resource \(r\) transported by vehicle \(\ell\) on arc \((i,j)\).

  - \(y^\omega_{ir} \geq 0\), \(\forall i \in N,\ r \in R,\ \omega \in \Omega\): Quantity of resource \(r\) retained at node \(i\) once local demand and onward shipment have been met.

  - \(\Lambda^\omega \geq 0\), \(\forall \omega \in \Omega\): Total loss incurred in scenario \(\omega\).

  - \(\xi^\omega \geq 0\), \(\forall \omega \in \Omega\): Excess of the scenario loss over \(\eta\).

  - \(\eta \in \mathbb{R}\): Value-at-Risk threshold.

Residual arc throughput \(T^\omega_{\ell,ij}\), residual handling capacity \(\Theta^\omega_{i,m}\), and node availability \(a^\omega_i\) follow from the realized scenario by Equations [\[eq:residual\]](#eq:residual)–[\[eq:availability\]](#eq:availability).

#### Objective Function

The scenario loss
\[\Lambda^\omega = \sum_{i \in N}\sum_{r \in R} \delta_{ir}\, z^\omega_{ir}
+ \sum_{\ell \in L}\sum_{(i,j)\in A_\ell}\sum_{r \in R} c_{\ell,ij}\, x^\omega_{\ell,ijr}
+ \varepsilon \sum_{\ell \in L}\sum_{(i,j)\in A_\ell} n^\omega_{\ell,ij}
\qquad \forall \omega \in \Omega
\label{eq:vif:loss}\]
sums penalized unmet demand, transport cost, and a tie-breaking term on vehicle movements. The objective
\[\min \quad \eta + \frac{1}{1-\beta} \sum_{\omega \in \Omega} \pi^\omega\, \xi^\omega
\label{eq:vif:objective}\]
minimizes the conditional value-at-risk of this loss at level \(\beta\) rather than its expectation. Because \(\beta\) is a planning input rather than a modeling constant, the commander’s risk posture enters the model directly.

#### Risk Measure Constraints

\[\begin{aligned}
\text{s.t.} \quad \xi^\omega &\geq \Lambda^\omega - \eta &&\forall \omega \in \Omega, \label{eq:vif:cvar1} \\
\xi^\omega &\geq 0 &&\forall \omega \in \Omega. \label{eq:vif:cvar2}\end{aligned}\]
Constraints [\[eq:vif:cvar1\]](#eq:vif:cvar1) and [\[eq:vif:cvar2\]](#eq:vif:cvar2) are the Rockafellar–Uryasev linearization. With the objective they force \(\xi^\omega = \max\{0,\ \Lambda^\omega - \eta\}\), so \(\eta\) takes the \(\beta\)-quantile of the loss distribution and the second objective term is the expected loss among the worst \((1-\beta)\) fraction of scenarios.

#### PPL Activation and Basing Constraints

\[\begin{aligned}
\sum_{i \in N^P} p_i &\leq P_{\max}, \label{eq:vif:pmax} \\
\sum_{i \in N^P} f_i\, p_i &\leq B, \label{eq:vif:budget} \\
\sum_{j\in J_k} b_{\ell,j} &= 1 &&\forall k \in K,\ \ell \in L_k, \label{eq:vif:baseassign} \\
b_{\ell,j} &\leq p_j &&\forall k \in K,\ \ell \in L_k,\ j \in J_k. \label{eq:vif:baselink}\end{aligned}\]
Constraints [\[eq:vif:pmax\]](#eq:vif:pmax) and [\[eq:vif:budget\]](#eq:vif:budget) cap the number of activated PPLs and the total activation cost, while constraints [\[eq:vif:baseassign\]](#eq:vif:baseassign) and  [\[eq:vif:baselink\]](#eq:vif:baselink) assign every vehicle to exactly one base and permit basing only at an activated node, respectively.

#### Resource Balance Constraints

\[\sum_{\ell\in L}\sum_{j:(j,i)\in A_\ell} x^\omega_{\ell,jir}
+ (1-\rho)\,\bar q_{ir}\, a^\omega_i\, p_i
+ z^\omega_{ir}
= d^\omega_{ir} + y^\omega_{ir}
+ \sum_{\ell\in L}\sum_{j:(i,j)\in A_\ell} x^\omega_{\ell,ijr}
\qquad \forall i \in N,\ r \in R,\ \omega \in \Omega.
\label{eq:vif:balance}\]
Constraints [\[eq:vif:balance\]](#eq:vif:balance) conserve each resource at every node and scenario: inbound flow, released stock, and unmet demand balance realized demand, retained stock, and outbound flow. Stock is released only from an activated node that remains available, while assuming \(\rho\,\bar q_{ir}\) number of resource \(r\) units remain held in safety stock at \(i\).

#### Intermodal Transfer

Transload volume is bounded by the receiving mode’s residual handling capacity \(\Theta^\omega_{i,m}\) (Section [3.4](#sec:arcdeg)), and each stop consumes turnaround distance from the vehicle’s budget (Section [3.2.3](#sec:timeproxy)). Routing through an intermediate node is therefore not free.

We assume the time required to transfer modes is captured in turnaround penalty.

#### Transport and Handling Capacity Constraints

\[\begin{aligned}
\sum_{r\in R} w_r\, x^\omega_{\ell,ijr} &\leq \min\{T^\omega_{\ell,ij},\ \text{cap}_\ell\}\, n^\omega_{\ell,ij}
&&\forall \ell \in L,\ (i,j) \in A_\ell,\ \omega \in \Omega, \label{eq:vif:vehcap}\\
\sum_{\ell\in L_m}\sum_{j:(j,i)\in A_\ell} n^\omega_{\ell,ji} &\leq \Theta^\omega_{i,m}
&&\forall i\in N,\ m\in M,\ \omega\in\Omega. \label{eq:vif:transfercap}\end{aligned}\]
Constraints [\[eq:vif:vehcap\]](#eq:vif:vehcap) limit the weight transported on an arc according to the minimum between the residual arc throughput and vehicle payload and permit a resource transport only when the vehicle traverses the arc. Constraints [\[eq:vif:transfercap\]](#eq:vif:transfercap) limit vehicle arrivals at each node and mode according to the residual handling capacity.

#### Vehicle Routing Constraints

\[\begin{aligned}
\sum_{j:(i,j)\in A_\ell} n^\omega_{\ell,ij} + \bar n^\omega_{\ell,i} - \sum_{j:(j,i)\in A_\ell} n^\omega_{\ell,ji}
&= a^\omega_i\, b_{\ell,i}
&&\forall \ell \in L,\ i \in N,\ \omega \in \Omega, \label{eq:vif:conservation}\\
\sum_{(i,j)\in A_\ell} n^\omega_{\ell,ij} \left(\text{dist}_{ij\ell} + \psi_\ell\right)
&\leq D_\ell
&&\forall \ell \in L,\ \omega \in \Omega. \label{eq:vif:distbudget}\end{aligned}\]
\[n^\omega_{\ell,ij} \leq \sum_{\substack{(i',j')\in A_\ell\\ i'\notin S,\ j'\in S}} n^\omega_{\ell,i'j'}
+ \sum_{i'\in S} a^\omega_{i'}\, b_{\ell,i'}
\quad \forall \ell\in L,\ \omega\in\Omega,\ S\subseteq N,\ |S|\geq 2,\ (i,j)\in A_\ell \text{ with } i,j\in S.
\label{eq:vif:sec}\]
Constraints [\[eq:vif:conservation\]](#eq:vif:conservation) enforce the flow balance of each vehicle. They ensure each vehicle, if utilized, originates at its base and terminates exactly where its route ends, possibly at the base itself. In this, a vehicle based at an unavailable node cannot depart unless arriving from another node. Constraints [\[eq:vif:distbudget\]](#eq:vif:distbudget) bound total distance traveled, inclusive of turnaround, by each vehicle reach over the horizon \(\mathcal{T}\). Finally, constraints [\[eq:vif:sec\]](#eq:vif:sec) eliminate subtours by requiring any arc used within a subset \(S\) to be supported by an arc entering \(S\) or by an available vehicle based inside it. These constraints are exponential in \(|N|\) and are not enumerated. Violated inequalities are separated as lazy constraints during branch and bound.

#### Variable Domain Constraints

\[\begin{aligned}
p_i &\in \{0,1\} &&\forall i \in N^P, \label{eq:vif:dom1}\\
b_{\ell,j} &\in \{0,1\} &&\forall \ell \in L,\ j \in J_k, \label{eq:vif:dom2} \\
n^\omega_{\ell,ij} &\in \{0,1\} &&\forall \ell \in L,\ (i,j) \in A_\ell,\ \omega \in \Omega, \label{eq:vif:dom3} \\
\bar n^\omega_{\ell,i} &\geq 0 &&\forall \ell \in L,\ i \in N,\ \omega \in \Omega, \label{eq:vif:dom4} \\
x^\omega_{\ell,ijr} &\geq 0 &&\forall \ell \in L,\ (i,j) \in A_\ell,\ r \in R,\ \omega \in \Omega, \label{eq:vif:dom5} \\
y^\omega_{ir} &\geq 0 &&\forall i \in N,\ r\in R,\ \omega \in \Omega, \label{eq:vif:dom6}\\
0 \leq z^\omega_{ir} &\leq d^\omega_{ir} &&\forall i \in N,\ r \in R,\ \omega \in \Omega, \label{eq:vif:dom7} \\
\eta &\in \mathbb{R}. \label{eq:vif:dom8}\end{aligned}\]
Constraints [\[eq:vif:dom1\]](#eq:vif:dom1)–[\[eq:vif:dom8\]](#eq:vif:dom8) impose the binary and nonnegativity restrictions. The termination variables \(\bar n^\omega_{\ell,i}\) are declared continuous rather than binary. Given the integrality of \(n^\omega_{\ell,ij}\) and \(b_{\ell,j}\), constraints [\[eq:vif:conservation\]](#eq:vif:conservation) force them integral at any feasible solution.

## Computational Considerations

Vehicles of a type are identical, so any permutation within a type may yield an alternative optimal solution that would unnecessarily enlarges the branch-and-bound tree without producing a distinct plan. We break this symmetry by requiring expected utilization to be nonincreasing in the vehicle index:
\[\sum_{\omega\in\Omega} \pi^\omega \sum_{(i,j)\in A_\ell} n^\omega_{\ell,ij}
\;\geq\;
\sum_{\omega\in\Omega} \pi^\omega \sum_{(i,j)\in A_{\ell+1}} n^\omega_{\ell+1,ij}
\quad \forall k \in K,\ \ell \in L_k \text{ with } \ell+1 \in L_k,
\label{eq:vif:symbreak}\]
so the vehicles used in any solution form a prefix of each type. Ordering on expected rather than scenario-specific utilization is necessary for validity, since basing is first-stage and shared across scenarios. Unlike the preceding constraints, [\[eq:vif:symbreak\]](#eq:vif:symbreak) do remove feasible solutions; they remove symmetrical solutions.

# Results and Analysis

## Planned Sensitivity Analysis Agenda

1.  **Inventory tier scale.** Sweep tier capacities at 1\(\times\) (current), 2\(\times\), 4\(\times\), and an explicitly labeled supply-unconstrained bounding case at 10\(\times\), to determine whether inventory or fleet/arc throughput is the binding constraint on service rate, and to identify the scale at which the constraint transitions from one to the other.

2.  **\(P_{\max}\) (maximum PPL count).** Sweep \(P_{\max} \in \{3, 5, 7\}\) to test whether additional PPLs materially improve service rate and whether the PPL-selection stability observed at \(P_{\max}=3\) (Tokyo, Seoul, Guam selected across all tested conditions) persists as more PPL slots become available.

3.  **Selection budget and PPL-type mix.** With \(P_{\max}\) fixed, vary the total selection budget \(B\) and per-tier activation cost to determine at what budget level the constraint becomes binding and begins to force selection toward lower-tier (PPL-2/3) sites rather than PPL-1 hubs; not active at the current \(B=12\).

4.  **Fleet size.** Sweep fleet size at 50%, 100%, 150%, and 200% of current values per vehicle type, to reconfirm under the current (reconciled-capacity, \(\beta=0.90\)) model specification whether fleet throughput remains slack, consistent with the earlier finding that doubling fleet size did not improve mean service rate.

5.  **Safety stock fraction \(\rho\).** Sweep \(\rho \in \{0.10, 0.20, 0.30\}\) independently of inventory tier scale, to isolate the effect of the safety-stock assumption from the effect of total capacity, which have previously only been varied together.

6.  **Multi-seed robustness across the full primary sweep.** Extend the multi-seed convergence check beyond the single tested condition to the full \(\alpha\) sweep (and, as bandwidth allows, to items 1–4 and 6 above), to confirm PPL selection and objective stability are not artifacts of a single random scenario draw at any tested condition.

7.  **Demand rate \(\phi_r\).** Test alternative values of \(\phi_{\text{food}}\) and \(\phi_{\text{water}}\) against the current literature-drawn values (0.15, 0.20), which are explicitly identified elsewhere in this chapter as requiring USARPAC G4 validation before operational use.

8.  **Degradation matrix \(\Gamma_{m\tau}\).** Test alternative baseline degradation sensitivity values against the current matrix (Table [2](#tab:degradmatrix)), which is likewise identified as a starting calibration pending subject-matter expert input.

9.  **Vehicle basing restriction.** Test relaxing the PPL-1-only basing restriction for the C-17, C-130J, and LCU-1700 (Section [\[sec:vehiclebasing\]](#sec:vehiclebasing)) to include PPL-2 sites, to determine whether the degenerate-concentration fix itself is materially influencing PPL selection or service-rate outcomes, independent of the underlying network and demand structure.

Item 6 (multi-seed robustness) functions as a resolution parameter on the preceding items rather than a standalone dimension: once a sensitivity result of interest is identified among items 1–4, it will be re-examined across multiple random seeds before being reported as a stable finding.

## Risk Posture Effects Under CVaR

## Operational Meaning of Delivery-Time Phases

# Conclusion

# Model Parameters

This appendix reports the numeric values of every parameter used in
PRS-VIF. Vehicle specifications and derived capacities are given in
Section [6.1](#app:params:fleet); remaining model parameters and instance
dimensions follow.

## Vehicle Fleet

Vehicle capacities derive from public payload specifications
(Table [3](#tab:vehiclespecs)) converted to person-days through two
humanitarian reference rates: the WFP standard emergency food ration of
540 g per person per day dry weight, and the Sphere minimum of 15 L
(approximately 15 kg) of water per person per day.
Table [4](#tab:vehicleparams) reports the adopted parameters, including
the distance budget \(D_k = \kappa v_k\) and the turnaround penalty
\(\psi_k = (\tau_m/24)\,v_k\) used in the formulation.

<div id="tab:vehiclespecs">

| Type                  | Mode |            Max Payload |      Cruise Speed |
| :-------------------- | :--- | ---------------------: | ----------------: |
| C-17 Globemaster III  | Air  | 170,900 lb (77,500 kg) | \(\approx\)450 kn |
| C-130J Super Hercules | Air  |  42,000 lb (19,051 kg) | \(\approx\)350 kn |
| LCU-1700 class        | Sea  |         170 short tons |   11 kn sustained |
| M1083 (5-ton FMTV)    | Land |   10,000 lb (4,536 kg) |            58 mph |

Vehicle type specifications (public sources).

</div>

<div id="tab:vehicleparams">

|          |         |          |            |         |            |                  |          |           |
| :------- | ------: | -------: | ---------: | ------: | ---------: | ---------------: | -------: | --------: |
| Type     | \(F_k\) |  \(v_k\) | \(\tau_m\) | \(D_k\) | \(\psi_k\) | \(\text{cap}_k\) | Food cap | Water cap |
|          |         | (km/day) |       (hr) |    (km) |       (km) |             (MT) |     (pd) |      (pd) |
| C-17     |      12 |   11,662 |          2 |  34,986 |        972 |           77.500 |  143,519 |     5,167 |
| C-130J   |      16 |    7,840 |          2 |  23,520 |        653 |           19.051 |   35,280 |     1,270 |
| LCU-1700 |       8 |      408 |          6 |   1,224 |        102 |          154.221 |  285,594 |    10,281 |
| M1083    |      60 |    1,116 |          1 |   3,348 |         47 |            4.536 |    8,400 |       302 |

Adopted vehicle fleet, speed, reach, turnaround, and capacity parameters.

</div>

Notes: Fleet sizes \(F_k\) are planning assumptions at squadron-equivalent
allocations. Effective cruise speeds \(v_k\) adjust published speeds for
daily operating tempo: 14 h/day air, reflecting crew duty limits; 20 h/day
sea, reflecting shift-rotation crewing; and 12 h/day land, reflecting
driver duty limits. Distance budgets use a planning horizon of
\(\kappa = 3\) days. Person-day capacities are derived from \(\text{cap}_k\)
and the per-unit weights \(w_r\) and are reported for interpretation only;
the formulation applies the metric-ton capacity directly.

Two limitations are noted. Published C-130J payload varies by variant, and
the standard J-model factsheet figure is used here. No large oceangoing
vessel is modeled alongside the LCU-1700; Maritime Prepositioning Ships
based at Guam and Saipan are a natural second sea type once a specific
hull class with clean published figures is selected.

## Model Parameter Values

Table [5](#tab:paramvalues) consolidates the numeric values of every model parameter not reported in Table [4](#tab:vehicleparams) or Table [2](#tab:degradmatrix).

<div id="tab:paramvalues">

| Parameter                                   | Value                                      | Notes                                                                                            |
| :------------------------------------------ | :----------------------------------------- | :----------------------------------------------------------------------------------------------- |
| \(\beta\)                                   | 0.9                                        | CVaR confidence level                                                                            |
| \(\delta_{ir}\)                             | 500                                        | Uniform across nodes and resources                                                               |
| \(P_{\max}\)                                | 3                                          | Sensitivity sweep reported in Section [4](#sec:results)                                          |
| \(B\)                                       | 12                                         | Non-binding: \(f_i = 1\ \forall i\), \(B > P_{\max}\)                                            |
| \(f_i\)                                     | 1                                          | All hub types                                                                                    |
| \(c_{\ell,ij}\)                             | \(\text{dist}_{ij} \times 10^{-4}\)        | Uniform across modes                                                                             |
| \(\varepsilon\)                             | 0.04                                       | Vehicle-movement tie-breaker (Eq. [\[eq:vif:loss\]](#eq:vif:loss))                               |
| \(\phi_{\text{food}}, \phi_{\text{water}}\) | 0.15, 0.20                                 | Per-capita demand rates                                                                          |
| \(\rho\)                                    | 0.20                                       | Safety-stock fraction                                                                            |
| \(\kappa\)                                  | 3                                          | Planning horizon (days), the 72-hour response window                                             |
| Availability cutoff                         | 1.0                                        | \(a^\omega_i = 0\) if \(\sigma^\omega_i \geq 1.0\) (Eq. [\[eq:availability\]](#eq:availability)) |
| \(w_{\text{food}}, w_{\text{water}}\)       | \(5.4\times10^{-4}\), \(1.5\times10^{-2}\) | MT per person-day                                                                                |
| pd per MT                                   | 1,852 (food), 66.7 (water)                 | Sphere/WFP basis; reciprocal of \(w_r\)                                                          |
| \(\pi^\omega\)                              | \(1/|\Omega|\)                             | Equal SAA weights                                                                                |

Model parameter values.

</div>

Inventory capacity is assigned by tier: PPL-1 holds \(\bar{q}_{ir} = 1{,}300{,}000\) person-days per resource, PPL-2 \(600{,}000\), and PPL-3 \(150{,}000\), with safety-stock fraction \(\rho = 0.20\) retained at every activated site. Tier ceilings are calibrated so that inventory does not trivially dominate: at higher ceilings, service rates approach 1.0 across the full \(\alpha\) sweep and network condition ceases to bind.

## Instance Dimensions

<div id="tab:vif:setsizes">

| Set                           |  Size |
| :---------------------------- | ----: |
| \(|N|\) (all nodes)           |    50 |
| \(|N^P|\) (PPL candidates)    |    22 |
| \(|R|\) (resources)           |     2 |
| \(|M|\) (modes)               |     3 |
| \(|A_{\text{sea}}|\)          |   236 |
| \(|A_{\text{air}}|\)          | 1,502 |
| \(|A_{\text{land}}|\)         |    30 |
| \(|K|\) (vehicle types)       |     4 |
| \(|L|\) (individual vehicles) |    96 |
| \(|\Omega|\) (scenarios)      |   100 |

Instance dimensions.

</div>

# Aggregate Type-Indexed Formulation

The formulation presented here indexes vehicles by type rather than by
individual unit. It shares the network construction, fleet composition,
scenario generation, demand calibration, and degradation model of
Sections [3.2](#sec:network) through [3.4](#sec:arcdeg), and differs only in
how vehicle movement is represented. Because same-type vehicles are
interchangeable under this indexing, the formulation is symmetry-free by
construction and substantially smaller than PRS-VIF; it forgoes the
per-vehicle itinerary resolution described in Section [3.2.3](#sec:timeproxy),
and it treats vehicle basing as an exogenous allocation rather than a
decision. It is retained as a tractability benchmark for PRS-VIF.

Notation in this appendix is self-contained and is not yet reconciled with
the main text; symbols are redefined here where they differ.

## Sets

  - \(N\): nodes, \(|N| = 50\); \(N^P \subseteq N\): PPL candidates, \(|N^P| = 22\)

  - \(R = \{\text{food}, \text{water}\}\): resource types

  - \(M = \{\text{sea}, \text{air}, \text{land}\}\): modes; \(A_m \subseteq N \times N\): directed arcs on mode \(m\)

  - \(T_i \subseteq M \times M\): feasible transfer mode-pairs at node \(i\)

  - \(\Theta\): disaster types (flood, storm, earthquake, volcanic, mass movement, wildfire)

  - \(\Omega\): scenarios, \(|\Omega| = 100\)

  - \(K_m\): vehicle types on mode \(m\); \(J_k \subseteq N^P\): basing-eligible nodes for type \(k\) (Section [3.2.2](#sec:vehicles))

## Parameters

  - \(d^\omega_{ir}\): demand (Eq. [\[eq:demand\]](#eq:demand));

  - \(\phi_r\): per-capita demand rate;

  - \(P_i\): population

  - \(\bar{q}_{ir}\): inventory ceiling at \(i\) if activated;

  - \(\rho\): safety-stock fraction;

  - \(a^\omega_{ir} \in \{0,1\}\): availability factor, 0 if
    \(s^\omega_i\) reaches the cutoff severity

  - \(U_{m,ij,r}\), \(u^\omega_{m,ij,r}\): nominal and residual arc capacity (Eqs. [\[eq:residual\]](#eq:residual));

  - \(c_{m,ij}\): per-unit transport cost;
    \(\text{dist}_{ij}\): arc distance

  - \(\kappa_{i,m_1m_2}\): node-specific transfer capacity;
    \(\theta_{m_1m_2}\): network-wide transfer cost multiplier

  - \(s^\omega_i\): severity (Eq. [\[eq:decay\]](#eq:decay));
    \(\Gamma_{m\tau}\), \(\alpha\), \(\gamma_m(\tau)\): degradation (Eqs. [\[eq:gammascale\]](#eq:gammascale))

  - \(\delta_{ir}\): unmet-demand penalty;
    \(P_{\max}\), \(f_i\), \(B\): site count cap, activation cost, selection budget; \(\beta\): CVaR level;
    \(\bar{Q}_r\): theater inventory ceiling (see Eq. [\[eq:invceiling\]](#eq:invceiling) discussion)

  - \(F_k\), \(v_k\), \(D_k = 3v_k\): fleet size, effective speed distance budget;
    \(\psi_k = (\text{turnaround}_m/24) \times v_k\): turnaround penalty; \(\text{cap}_{k,r}\): per-resource vehicle capacity;
    \(b_{k,j}\): initial basing;
    \(\varepsilon\): vehicle-movement tie-breaker

## Decision Variables

  - First stage: \(p_i \in \{0,1\}\), 1 if node \(i\) is selected as a PPL

  - \(x^\omega_{m,ijr} \geq 0\): commodity flow;
    \(\tau^\omega_{i,m_1m_2,r} \geq 0\): intermodal transfer;
    \(z^\omega_{ir} \geq 0\): unmet demand;
    \(y^\omega_{ir} \geq 0\): inventory released

  - \(n^\omega_{k,m,ij} \geq 0\), integer: type-\(k\) vehicles traversing arc \((i,j)\).

  - Auxiliary: \(\eta\) (Value-at-Risk threshold),
    \(\xi^\omega \geq 0\) (CVaR excess loss),
    \(L^\omega \geq 0\) (scenario loss)

## Objective and Scenario Loss

\[L^\omega = \sum_{i\in N}\sum_{r\in R}\delta_{ir}z^\omega_{ir}
+ \sum_{m\in M}\sum_{(i,j)\in A_m}\sum_{r\in R}c_{m,ij}\,x^\omega_{m,ijr}
+ \sum_{i\in N}\sum_{(m_1,m_2)\in T_i}\sum_{r\in R}\theta_{m_1m_2}\,\tau^\omega_{i,m_1m_2,r}
+ \varepsilon \sum_{k}\sum_{m\in M}\sum_{(i,j)\in A_m} n^\omega_{k,m,ij}.
\label{eq:loss}\]

\[\min_{p,x,\tau,n,y,z,\eta,\xi} \quad
\eta + \frac{1}{1-\beta} \sum_{\omega \in \Omega} \pi^\omega \xi^\omega
\label{eq:objective}\]

## Constraints

#### First-stage selection.

\[\begin{aligned}
\sum_{i \in N^P} p_i &\leq P_{\max} \label{eq:pmax} \\
\sum_{i \in N^P} f_i\, p_i &\leq B \label{eq:budget} \\
\sum_{i \in N^P} \bar{q}_{ir}\, p_i &\leq \bar{Q}_r \quad \forall r \in R \label{eq:invceiling} \\
p_i &\in \{0,1\} \quad \forall i \in N^P \label{eq:binary}\end{aligned}\]

#### CVaR linearization (Rockafellar–Uryasev).

\[\begin{aligned}
\xi^\omega &\geq L^\omega - \eta \quad \forall \omega \in \Omega \label{eq:cvar1} \\
\xi^\omega &\geq 0 \quad \forall \omega \in \Omega \label{eq:cvar2}\end{aligned}\]

#### Inventory release.

\[\begin{aligned}
y^\omega_{ir} &\leq (1-\rho)\,\bar{q}_{ir}\,a^\omega_{ir}\,p_i     \quad \forall i \in N^P, r \in R, \omega \in \Omega \label{eq:release} \\
y^\omega_{ir} &= 0 \quad \forall i \in N \setminus N^P, r \in R, \omega \in \Omega \label{eq:norelease}\end{aligned}\]

#### Aggregate flow balance.

\[\begin{aligned}
\sum_{m}\sum_{j:(j,i)\in A_m} x^\omega_{m,jir} + y^\omega_{ir} + z^\omega_{ir} &= d^\omega_{ir} + \sum_{m}\sum_{j:(i,j)\in A_m} x^\omega_{m,ijr} \quad \forall i \in N^P, r, \omega \label{eq:balppl} \\
\sum_{m}\sum_{j:(j,i)\in A_m} x^\omega_{m,jir} + z^\omega_{ir} &= d^\omega_{ir} + \sum_{m}\sum_{j:(i,j)\in A_m} x^\omega_{m,ijr} \quad \forall i \in N \setminus N^P, r, \omega \label{eq:balnon}\end{aligned}\]

#### Per-mode outbound feasibility with transfer.

\[\begin{aligned}
\sum_{j:(i,j)\in A_m} x^\omega_{m,ijr} + \sum_{m_2:(m,m_2)\in T_i} \tau^\omega_{i,mm_2,r} \;\leq\;&
\sum_{j:(j,i)\in A_m} x^\omega_{m,jir} + \sum_{m_1:(m_1,m)\in T_i} \tau^\omega_{i,m_1m,r} + \mathbbm{1}[i \in N^P]\, y^\omega_{ir} \notag \\
&\quad \forall i \in N,\ m \in M,\ r \in R,\ \omega \in \Omega \label{eq:permode}\end{aligned}\]

#### Transfer backing.

\[\begin{aligned}
\sum_{m_2:(m,m_2)\in T_i} \tau^\omega_{i,mm_2,r} \;\leq\; \sum_{j:(j,i)\in A_m} x^\omega_{m,jir} + \mathbbm{1}[i \in N^P]\,y^\omega_{ir} \quad \forall i, m, r, \omega \label{eq:transferbacking}\end{aligned}\]

#### Capacity.

\[\begin{aligned}
x^\omega_{m,ijr} &\leq u^\omega_{m,ij,r} \quad \forall m, (i,j) \in A_m, r, \omega \label{eq:arccap} \\
\sum_{r \in R} \tau^\omega_{i,m_1m_2,r} &\leq \kappa_{i,m_1m_2} \quad \forall i, (m_1,m_2) \in T_i, \omega \label{eq:transfercap}\end{aligned}\]

#### Vehicle conservation.

\[\begin{aligned}
\sum_{i:(i,j)\in A_m} n^\omega_{k,m,ij} + b_{k,j}\,p_j \;\geq\;
\sum_{j':(j,j')\in A_m} n^\omega_{k,m,jj'}
\quad \forall k \in K_m, j \in N, m, \omega
\label{eq:vehcons}\end{aligned}\]

#### Vehicle-capacity-constrained flow.

\[\begin{aligned}
x^\omega_{m,ijr} \;\leq\; \sum_{k \in K_m} \text{cap}_{k,r}\, n^\omega_{k,m,ij} \quad \forall m, (i,j) \in A_m, r, \omega \label{eq:vehcap}\end{aligned}\]

#### Fleet-wide distance budget.

\[\begin{aligned}
\sum_{m}\sum_{(i,j)\in A_m} \text{dist}_{ij}\, n^\omega_{k,m,ij} + \psi_k \sum_{j \in N} h^\omega_{k,j} \;\leq\; F_k\, D_k \quad \forall k, \omega \label{eq:distbudget}\end{aligned}\]

#### Nonnegativity and integrality.

\[\begin{aligned}
z^\omega_{ir},\, y^\omega_{ir} &\geq 0 \label{eq:nonneg1} \\
x^\omega_{m,ijr} &\geq 0 \label{eq:nonneg2} \\
\tau^\omega_{i,m_1m_2,r} &\geq 0 \label{eq:nonneg3} \\
n^\omega_{k,m,ij} &\geq 0,\ \text{integer} \label{eq:integrality}\end{aligned}\]

## Model Size

### Set Sizes

<div id="tab:setsizes">

| Set                                                              |    Size |
| :--------------------------------------------------------------- | ------: |
| \(|N|\) (all nodes)                                              |      50 |
| \(|N^P|\) (PPL candidates)                                       |      22 |
| \(|A_{\text{sea}}|\)                                             |     236 |
| \(|A_{\text{air}}|\)                                             |   1,502 |
| \(|A_{\text{land}}|\)                                            |      30 |
| \(|\Omega|\) (scenarios)                                         |     100 |
| \(|K_{\text{sea}}|,\ |K_{\text{air}}|,\ |K_{\text{land}}|\)      | 1, 2, 1 |
| \(\sum_i |T_i|\) (active transfer mode-pairs, summed over nodes) |     106 |

Set sizes for the reported instance (seed 32, \(\alpha=1.0\)).

</div>

### Decision Variables

Table [8](#tab:varsizes) reports each variable count alongside the formula
that produces it.

<div id="tab:varsizes">

| Type                             | Symbol              |       Count | Scales as                                 |
| :------------------------------- | :------------------ | ----------: | :---------------------------------------- |
| Site selection                   | \(p\)               |          22 | \(|N^P|\)                                 |
| Modal flow                       | \(x\)               |     353,600 | \(\left(\sum_m|A_m|\right)|R||\Omega|\)   |
| Unmet demand                     | \(z\)               |      10,000 | \(|N||R||\Omega|\)                        |
| Inventory release                | \(y\)               |      10,000 | \(|N||R||\Omega|\)                        |
| Intermodal transfer              | \(\tau\)            |      21,200 | \(\left(\sum_i|T_i|\right)|R||\Omega|\)   |
| Vehicle count                    | \(n\)               |     327,000 | \(\left(\sum_k|A_{m(k)}|\right)|\Omega|\) |
| CVaR threshold                   | \(\eta\)            |           1 | scalar                                    |
| CVaR excess loss                 | \(\xi\)             |         100 | \(|\Omega|\)                              |
| Loss (code-only)                 | \(L\)               |         100 | \(|\Omega|\)                              |
| Turnaround auxiliary (code-only) | \(g_{\text{turn}}\) |       3,400 | McCormick linearization                   |
| **Total**                        |                     | **725,423** | matches `model.NumVars`                   |

Decision variable counts.

</div>

\(x\) and \(n\) together account for 680,600 of 725,423 variables (93.8
percent), and both are indexed by scenario. The vehicle-count \(n\) is
entirely integer. The release variable \(y\) is defined over all of \(N\) (50
nodes), not only \(N^P\) (22).

### Constraints

Table [9](#tab:constraintsizes) reports the same treatment for constraint
types, bucketed by the model’s internal naming convention and reconciled
against `model.NumConstrs`.

<div id="tab:constraintsizes">

| Type                                      |       Count | Formulation reference                                        |
| :---------------------------------------- | ----------: | :----------------------------------------------------------- |
| Site count / selection budget             |           2 | Eq. [\[eq:pmax\]](#eq:pmax), [\[eq:budget\]](#eq:budget)     |
| Inventory release bound                   |       4,400 | Eq. [\[eq:release\]](#eq:release)                            |
| Non-PPL release zero                      |       5,600 | Eq. [\[eq:norelease\]](#eq:norelease)                        |
| Flow balance (PPL + non-PPL)              |      10,000 | Eq. [\[eq:balppl\]](#eq:balppl), [\[eq:balnon\]](#eq:balnon) |
| Modal outbound feasibility                |      30,000 | Eq. [\[eq:permode\]](#eq:permode)                            |
| Arc capacity                              |     353,600 | Eq. [\[eq:arccap\]](#eq:arccap)                              |
| Transfer backing                          |      14,600 | Eq. [\[eq:transferbacking\]](#eq:transferbacking)            |
| Transfer capacity                         |      10,600 | Eq. [\[eq:transfercap\]](#eq:transfercap)                    |
| Vehicle conservation                      |      20,000 | Eq. [\[eq:vehcons\]](#eq:vehcons)                            |
| Vehicle-capacity flow                     |     353,600 | Eq. [\[eq:vehcap\]](#eq:vehcap)                              |
| Turnaround exemption (McCormick, 3 types) |      10,200 | linearization of Eq. [\[eq:distbudget\]](#eq:distbudget)     |
| Distance budget                           |         400 | Eq. [\[eq:distbudget\]](#eq:distbudget)                      |
| Loss definition, CVaR excess              |         200 | Eq. [\[eq:loss\]](#eq:loss), [\[eq:cvar1\]](#eq:cvar1)       |
| **Total**                                 | **813,202** | matches `model.NumConstrs`                                   |

Constraint counts.

</div>

Arc capacity and vehicle-capacity flow represent 87 percent of all
constraints, each matching the shape of the \(x\). The three-constraint
McCormick linearization of the turnaround exemption (10,200 constraints
total) is the explicit, countable cost of keeping
Equation [\[eq:distbudget\]](#eq:distbudget) linear despite the exemption’s dependence on
the product of a first-stage binary \(p_j\) and second-stage flow.
