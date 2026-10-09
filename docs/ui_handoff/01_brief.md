# 01 — Product brief (DRAFT)

## What this is
A prototype interface for exploring the output of a prepositioning optimization model for US Army Pacific logistics. The model decides where to pre-position sustainment capability (ports, airfields, handling capacity) and how the available airlift, sealift, and ground lift would deliver supplies after a disaster, while hedging against the worst scenarios.

## What this is not
- Not an operational tool. Planners will not run it to make real decisions.
- Not a live solver front end (first version). The solver needs a Gurobi license and runs for minutes to hours.

## Why build it
The research team needs planners to react to the model's outputs, so it can learn what a planner would need to trust and use them. The immediate occasion is observing the Keen Sword exercise, where the team is an observer only. The interface exists to elicit requirements, not to meet them.

## Primary persona: Strategic planner
A staff officer in a theater logistics section (for example USARPAC G4). Knows the theater, the platforms, and the exercise cycle. Does not know optimization. Wants to know: where should we put things, what does it buy us, and how bad is the bad day?

Vocabulary to use in the UI: sites, lift, delivery, unmet demand, bad-day performance. Avoid: CVaR, scenario tree, MIP, Θ, distance state. Where a technical term is unavoidable, pair it with a plain label.

## Secondary persona: Researcher
Sees the same screens plus the technical detail: solver gap, run provenance, parameters.

## Outcomes we want
1. A planner can say where the model recommends pre-positioning, and see why, in under two minutes.
2. A planner can compare two risk settings and see what changes.
3. Every number on screen can be traced to a named run with its parameters.

## Constraints
- Data is produced offline by Python scripts and delivered as files (JSON, CSV).
- Network is built from `pacific_cities.csv` (41 cities across the Pacific); the data contract (doc 03, not yet written) will fix the exact node set.
- Eight vehicle types: C-17, C-130J, LCU-1700, T-AKR, T-AKE, EPF, M1083, PLS.
- Unclassified data only. No CUI in this repository.

## Open questions for the research team
- Audience devices (laptop browser only, or also tablet)?
- Offline use at exercise venues?
- Who hosts it, if anyone beyond a laptop?
