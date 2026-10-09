"""
scenario_generator.py

Scenario generator for the stochastic course project. This is the first
stage of the pipeline (network/nodes.csv -> scenarios -> stochastic instance
-> gurobi model). It produces the exogenous random draws (epicenter, severity,
node-level severity) that downstream code in model/input_builder.py
turns into demand, inventory availability, and residual arc capacity.
"""

import json
import os
import random
from typing import Dict, List

import networkx as nx
import pandas as pd
from geopy.distance import geodesic

from config.loader import load_parameters


def _get_coords(location: Dict) -> tuple[float, float]:
    """
    Return (lat, lon) coordinates from a location record.

    location: a dict from the locations table, may use either lowercase
    keys (lat/lon/coords) or the raw csv column names (Latitude/Longitude)

    returns a (lat, lon) float tuple
    """
    if "coords" in location and location["coords"] is not None:
        coords = tuple(location["coords"])
        return float(coords[0]), float(coords[1])

    lat = location.get("lat", location.get("Latitude"))
    lon = location.get("lon", location.get("Longitude"))
    return float(lat), float(lon)


def _normalize_dict_keys(d: Dict) -> Dict[str, float]:
    """
    Convert tuple or scalar keys to strings so nested dictionaries can be exported to CSV.

    d: a dict whose keys may be ints, tuples, etc

    returns the same dict with every key converted to its str() form
    """
    normalized = {}
    for k, v in d.items():
        normalized[str(k)] = v
    return normalized


# the six standardized disaster type labels used throughout the type-sampling
# config blocks (degradation_matrix, type_sampling.fallback_distribution) and
# the EM-DAT-derived frequency table loaded by load_type_frequencies()
DISASTER_TYPE_LABELS = ["flood", "storm", "earthquake", "volcanic", "mass_movement", "wildfire"]


def load_type_frequencies(csv_path: str = None) -> Dict[str, Dict[str, float]]:
    """
    Load country-conditional disaster type frequency vectors computed from
    EM-DAT (scenarios/data/country_type_frequencies.csv).

    This file is a static, committed project artifact, not a run-generated
    output: it is a deterministic derivation of calibration/emdat_data.xlsx
    that does not change between scenario-generation runs. It is produced by
    calibration/step1_country_type_frequencies.py, which should be re-run by
    hand (and the result re-committed) only if the underlying EM-DAT extract
    changes. It intentionally does not live under output/, which is treated
    as disposable/regenerable-per-run and is not tracked in git.

    csv_path: path to the frequency table - defaults to
    project_root/aps_usarpac/scenarios/data/country_type_frequencies.csv,
    resolved relative to this file so it works regardless of cwd

    returns a dict keyed by country_iso, each value a dict mapping the six
    standardized disaster type labels to that country's conditional
    probability of each type
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "data", "country_type_frequencies.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Country-type frequency table not found at {csv_path}.\n"
            "This is a committed static artifact, not a generated run output, so it "
            "should not normally go missing. To regenerate it, run from the project "
            "root:\n"
            "    python calibration/step1_country_type_frequencies.py\n"
            "(requires calibration/emdat_data.xlsx). If you don't want country-"
            "conditional type sampling, set type_sampling.enabled: false in "
            "config/model_parameters.yaml instead."
        )

    df = pd.read_csv(csv_path)
    type_frequencies = {}
    for _, row in df.iterrows():
        type_frequencies[row["country_iso"]] = {label: float(row[label]) for label in DISASTER_TYPE_LABELS}
    return type_frequencies


def sample_disaster_type(country_iso: str, type_frequencies: Dict[str, Dict[str, float]], fallback: Dict[str, float], rng: random.Random) -> str:
    """
    Draw a standardized disaster type for a scenario's epicenter country.

    country_iso: iso3 code of the scenario's epicenter country
    type_frequencies: dict from load_type_frequencies(), keyed by country_iso
    fallback: dict mapping the six standardized type labels to global marginal
    probabilities (type_sampling.fallback_distribution), used when
    country_iso has no entry in type_frequencies
    rng: random.Random instance used for this draw

    returns one of the six standardized disaster type labels
    """
    distribution = type_frequencies.get(country_iso, fallback)
    types = list(distribution.keys())
    weights = list(distribution.values())
    return rng.choices(types, weights=weights, k=1)[0]


def generate_scenarios(
    G: nx.Graph,
    locations: Dict[int, Dict],
    num_scenarios: int = None,
    seed: int = None,
    save_path: str = None,
    forced_type: str = None,
) -> List[Dict]:
    """
    Generate disaster scenarios for the stochastic course project.

    This simulator now produces only the exogenous scenario realization needed
    to model arc-capacity uncertainty later in the input builder.

    G: networkx graph of the node network - not actually used here, kept in
    the signature for interface consistency with callers
    locations: dict mapping node id to its attributes (lat, lon, country_iso, etc)
    num_scenarios: how many scenarios to draw - falls back to params["num_scenarios"]
    seed: random seed for this draw - falls back to params["seed"], pass an
    explicit value to get an independent sample for convergence testing
    save_path: where to write the scenario csv - defaults to output/stoch_scenarios.csv
    forced_type: if not None, all scenarios use this disaster type instead of
        sampling from the EM-DAT frequency table. Must be one of:
        "flood", "storm", "earthquake", "volcanic", "mass_movement", "wildfire".
        Epicenter and severity sampling remain unchanged. Used for the
        Secondary C (type-dominance) sensitivity analysis.

    Scenario structure returned (one dict per scenario):
        - scenario_id
        - epicenter
        - severity
        - affected_nodes
        - node_severity
        - affected_radius_km
        - probability
        - disaster_type
        - country_iso

    Notes:
        - Demand is intentionally NOT generated here.
        - Storage capacity is intentionally NOT generated here.
        - Residual arc capacity is intentionally NOT generated here.
        - Those fixed and model-dependent quantities will be built later in
          model/input_builder.py so that arc-capacity uncertainty can be
          isolated cleanly through the degradation matrix.
    """
    del G  # graph topology is not needed at the scenario-generation stage

    params = load_parameters()

    # seed setup - use the explicit seed if given, otherwise fall back to the
    # config seed, so runs are reproducible unless the caller wants a fresh draw
    if seed is None:
        seed = params.get("seed", 42)
    random.seed(seed)
    print(f"[Stoch Simulator] Using seed: {seed}")

    if num_scenarios is None:
        num_scenarios = params.get("num_scenarios", 100)

    # disaster type sampling setup
    # type_frequencies/fallback drive sample_disaster_type() below. The draws
    # use their own random.Random instance (seeded the same as the run) rather
    # than the shared global `random` module, so that adding this new draw
    # does not shift the epicenter/severity sequence already drawn from the
    # global stream - preserving backward-compatible reproducibility for
    # existing seeds while still being deterministic per seed.
    type_sampling_cfg = params.get("type_sampling", {})
    type_sampling_enabled = type_sampling_cfg.get("enabled", True)
    type_fallback = type_sampling_cfg.get("fallback_distribution", {})
    type_frequencies = load_type_frequencies() if type_sampling_enabled else {}
    type_rng = random.Random(seed)

    node_ids = list(locations.keys())
    scenarios: List[Dict] = []

    for s_id in range(num_scenarios):
        # epicenter sampling block
        # pick the disaster's home country first using the configured weights
        # (epicenter_weights), then pick uniformly among that country's nodes
        # this two-step sampling lets us calibrate country-level disaster
        # frequency from emdat without needing per-node weights
        country_weights = params.get("epicenter_weights", {})
        if country_weights:
            node_country = {i: locations[i].get("country_iso", locations[i].get("Country ISO", "")) for i in node_ids}
            countries = list(country_weights.keys())
            weights = [country_weights[c] for c in countries]
            selected_country = random.choices(countries, weights=weights, k=1)[0]
            country_nodes = [i for i in node_ids if node_country[i] == selected_country]
            # if the selected country has no nodes in the network, fall back
            # to a uniform draw over all nodes so the scenario still generates
            epicenter = random.choice(country_nodes) if country_nodes else random.choice(node_ids)
        else:
            epicenter = random.choice(node_ids)

        # disaster type sampling block
        # draw a standardized disaster type conditioned on the epicenter's
        # country, using the EM-DAT-derived frequency table with a fallback
        # to the global marginal distribution for countries not in the table.
        # If forced_type is set (Secondary C analysis), bypass sampling entirely
        # so that all 100 scenarios share the same type while epicenter and
        # severity still vary — isolating the effect of threat environment.
        country_iso = locations[epicenter].get("country_iso", locations[epicenter].get("Country ISO", ""))
        if forced_type is not None:
            disaster_type = forced_type
        elif type_sampling_enabled:
            disaster_type = sample_disaster_type(country_iso, type_frequencies, type_fallback, type_rng)
        else:
            disaster_type = None

        # severity sampling block
        # draw severity from a kumaraswamy distribution via inverse cdf sampling
        # kumaraswamy was chosen because it is bounded on [0,1] like a beta
        # distribution but has a closed-form inverse cdf, so we can sample with
        # a single uniform draw u without numerical inversion
        kumaraswamy_a = float(params["default_disaster"].get("kumaraswamy_a", 2.417))
        kumaraswamy_b = float(params["default_disaster"].get("kumaraswamy_b", 3.747))
        u = random.random()
        x_unit = (1.0 - (1.0 - u) ** (1.0 / kumaraswamy_b)) ** (1.0 / kumaraswamy_a)
        # map the [0,1] kumaraswamy draw onto the 1-5 severity scale
        severity = 1.0 + 4.0 * x_unit

        # affected radius calculation
        # the affected radius is the operational footprint of the disaster -
        # nodes within this distance of the epicenter feel some severity,
        # nodes outside it are untouched - radius grows with severity so
        # bigger disasters affect a wider area
        base_radius_km = params["default_disaster"]["affected_radius_km"]["base"]
        multiplier_km = params["default_disaster"]["affected_radius_km"]["multiplier"]
        affected_radius_km = base_radius_km + severity * multiplier_km

        epicenter_coords = _get_coords(locations[epicenter])

        affected_nodes = []
        node_severity = {}

        # node severity loop
        # for every node compute its distance to the epicenter and apply a
        # linear decay - severity is full strength at the epicenter and falls
        # off to zero at the edge of the affected radius, representing the
        # idea that damage intensity fades with distance from the disaster
        for i in node_ids:
            node_coords = _get_coords(locations[i])
            dist_km = geodesic(epicenter_coords, node_coords).kilometers

            if dist_km <= affected_radius_km:
                local_severity = max(0.0, severity * (1.0 - dist_km / affected_radius_km))
                affected_nodes.append(i)
            else:
                local_severity = 0.0

            node_severity[i] = local_severity

        # scenario dict assembly
        # bundle the exogenous draw into a single record - downstream code
        # uses node_severity to drive demand, inventory availability, and
        # residual arc capacity, and probability for the cvar weighting
        scenario = {
            "scenario_id": s_id,            # index of this scenario, used as the Omega set in the model
            "epicenter": epicenter,         # node id where the disaster originated
            "severity": severity,           # epicenter severity on the 1-5 scale
            "affected_nodes": affected_nodes,  # node ids within the affected radius
            "node_severity": node_severity,    # per-node severity after distance decay
            "affected_radius_km": affected_radius_km,  # footprint radius for this scenario
            "probability": 1.0 / num_scenarios,  # equal-weight probability used in the cvar objective
            "disaster_type": disaster_type,  # standardized type label, drives gamma[mode][type] degradation
            "country_iso": country_iso,      # epicenter country, for diagnostics
        }
        scenarios.append(scenario)

    # csv save block
    # write the scenario set to disk so it can be inspected or reused without
    # re-running the random draw - useful for debugging and for verifying
    # which epicenters/severities a given seed produced
    # default output path is project_root/output/stoch_scenarios.csv, computed
    # relative to this file (scenarios/) so it resolves the same regardless of cwd
    default_output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    output_file = save_path if save_path is not None else os.path.join(default_output_dir, "stoch_scenarios.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    scenario_rows = []
    for scenario in scenarios:
        epicenter_id = scenario["epicenter"]
        epicenter_name = locations.get(epicenter_id, {}).get(
            "name",
            locations.get(epicenter_id, {}).get("Name", "Unknown"),
        )
        row = {
            "scenario_id": scenario["scenario_id"],
            "epicenter": epicenter_id,
            "epicenter_name": epicenter_name,
            "severity": scenario["severity"],
            "affected_radius_km": scenario["affected_radius_km"],
            "affected_nodes": json.dumps(scenario["affected_nodes"]),
            "node_severity": json.dumps(_normalize_dict_keys(scenario["node_severity"])),
            "probability": scenario["probability"],
            "disaster_type": scenario["disaster_type"],
            "country_iso": scenario["country_iso"],
        }
        scenario_rows.append(row)

    df = pd.DataFrame(scenario_rows)
    df.to_csv(output_file, index=False)
    print(f"[Stoch Simulator] Scenarios saved to {output_file}")

    return scenarios
