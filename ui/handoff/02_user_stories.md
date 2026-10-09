# 02 — User stories (DRAFT)

Format: "As a ..., I want ..., so that ...", with Given/When/Then acceptance criteria. Priority: M = must for first version, S = should, C = could.

## Epic A — See the recommendation

**A1 (M) Recommended sites on a map**
As a strategic planner, I want to see the recommended pre-positioning sites on a map of the Pacific, so that I can judge them against what I know about the theater.
- Given a completed run, when I open it, then the map shows every selected site, visually distinct from unselected nodes.
- Given the map is shown, when I select a site, then I see its name, type (port, airfield, both), and handling capacity.

**A2 (M) Why this site**
As a planner, I want a plain-language summary of what each selected site contributes, so that I can explain it to my boss.
- Given a selected site, when I open its detail, then I see demand it serves, which vehicles deliver through it, and how much handling capacity it uses.

## Epic B — Compare risk settings

**B1 (M) Choose a risk setting**
As a planner, I want to choose how cautious the plan should be, so that I can see the cost of hedging against a bad day.
- Given runs at several risk settings exist, when I choose one, then the map and summary update to that run.
- Given a setting is chosen, then its plain-language label and its numeric value are both shown.

**B2 (S) Side-by-side comparison**
As a planner, I want two runs side by side, so that I can see what changes when I get more cautious.
- Given two runs, when I compare them, then sites added or dropped are highlighted, and average and bad-day performance are shown for both.

## Epic C — Understand the fleet

**C1 (M) Vehicle utilization**
As a planner, I want to see how heavily each vehicle type is used, so that I can tell which platforms are the constraint.
- Given a run, when I open the fleet view, then each of the 8 types shows quantity available, quantity used, and share of cargo delivered.

**C2 (C) Binding indicator**
As a planner, I want a flag on vehicle types that are fully used, so that I notice shortages.

## Epic D — Understand the bad day

**D1 (M) Unmet demand by place**
As a planner, I want to see where demand goes unmet in the worst cases, so that I can see where the plan is thin.
- Given a run, when I switch to bad-day view, then nodes are shaded by unmet demand in the worst scenarios, with a legend.

**D2 (S) Scenario browser**
As a planner, I want to step through individual disaster scenarios, so that I can see the epicenter and what the plan did.
- Given a scenario list, when I select one, then the map shows the epicenter, affected nodes, and deliveries.

## Epic E — Trust and traceability

**E1 (M) Run provenance**
As a researcher, I want every screen to show which run it came from and with what parameters, so that no number is unattributed.
- Given any view, then a visible footer or panel shows run id, date, risk setting, seed, scenario count, and solver gap.

**E2 (M) Warnings are visible**
As a researcher, I want solver warnings (for example, too few scenarios to resolve the tail) shown with the run, so that weak results are not read as strong.

**E3 (S) Export**
As a planner, I want to export the current view as an image and the underlying numbers as CSV, so that I can use them in a briefing.

## Epic F — Capture planner feedback (supports the Keen Sword goal)

**F1 (S) Notes on a view**
As a researcher, I want to record a planner's comment tied to a specific view and run, so that requirements come out of real reactions.
- Given any view, when I add a note, then it is saved with run id, view name, and timestamp.

## Out of scope for the first version
Live solving, user accounts, editing model inputs, multi-user collaboration.
