# lineage/ — frozen reference, do not edit

`aps_usarpac_aggregate_2026-07/` is the 10 July 2026 snapshot of the aggregate-fleet, two-stage stochastic CVaR pipeline (commit `c131561`), the origin of the Chapter 2 code. Run it from inside that folder.

**Equivalence check (2026-10-09).** On the 4-node toy instance, the old `solve_stochastic_cvar` and the current `model.model.solve_stochastic_cvar(vehicle_formulation="aggregate")` both returned objective 744,048,140.88 at gap 0, the same sites (1 and 2), and the same model size (274 variables, 577 constraints). On the 50-node network with the old config, both input builders produced identical scenarios and identical instances, apart from added keys in the current builder (`node_handling_capacity`, `node_handling_bonus`, `nominal_throughput`, `degradation_matrix`, `node_severity`, `disaster_type`, `resource_weight`) and a `cap_tons` field on each vehicle type. No existing value changed.
