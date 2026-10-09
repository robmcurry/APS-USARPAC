# APS-USARPAC

Optimization models for prepositioning US Army sustainment capability in the Indo-Pacific, supporting a PhD dissertation in Industrial Engineering at the University of Arkansas. All models need a Gurobi license.

| Folder | What it is | Status |
|---|---|---|
| `chapter1_deterministic/` | Chapter 1: deterministic multi-period prepositioning and routing model, with sensitivity experiments | Complete |
| `chapter2_stochastic/` | Chapter 2: type-indexed, distance-state CVaR model and the staged matheuristic | Active |
| `ui/` | Handoff package for the UI build team: brief, user stories, data contract | Active |
| `archive/` | Legacy models, the original course project, and loose files kept for the record | Frozen |

Start with the README in the folder you need. Each chapter folder is self-contained: run its code from inside that folder.

Branches: `main` is the integration branch. `staged_approach` carries the Chapter 2 work. `individual-vehicle-indexing` carries the earlier PRS-VIF line.
