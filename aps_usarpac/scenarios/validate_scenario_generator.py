"""
validate_scenario_generator.py

Independent statistical validation of scenarios/scenario_generator.py against
its own calibrated inputs (epicenter_weights, the Kumaraswamy severity
distribution, and the country-conditional disaster-type frequency table).

This does not re-implement the sampling logic — it calls the real
generate_scenarios() (and the real load_type_frequencies() /
sample_disaster_type()) with a large scenario count and checks that the
empirical output matches what each calibrated parameter says it should
produce. It exists to answer one question with numbers instead of code
review: "is the program actually drawing from the distributions it claims
to draw from?"

Run from the aps_usarpac directory:
    python -m scenarios.validate_scenario_generator

Exits non-zero if any check fails so it can be used as a regression guard,
not just a one-off report.
"""

import os
import random
import sys

import numpy as np
import pandas as pd
from scipy import stats

from config.loader import load_parameters
from scenarios.scenario_generator import (
    generate_scenarios,
    load_type_frequencies,
    sample_disaster_type,
    DISASTER_TYPE_LABELS,
)

# large, separate from the production seed=32/N=100 run used for actual
# results, so this validation never gets confused with a reportable result
VALIDATION_SEED = 987654321
N_SCENARIOS = 20000

FAILURES = []


def check(label: str, condition: bool, detail: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}: {detail}")
    if not condition:
        FAILURES.append(label)


def load_locations() -> dict:
    nodes = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "network", "nodes.csv"))
    locations = {}
    for _, row in nodes.iterrows():
        d = row.to_dict()
        d["pop"] = row["Population"]
        d["country_iso"] = row["Country ISO"]
        locations[int(row["Node ID"])] = d
    return locations


def kumaraswamy_cdf(x, a, b):
    return 1 - (1 - x ** a) ** b


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    params = load_parameters()
    locations = load_locations()
    node_countries = {i: locations[i]["country_iso"] for i in locations}

    section("0. Generating validation sample "
            f"(N={N_SCENARIOS}, seed={VALIDATION_SEED}, separate from production seed=32)")
    scenarios = generate_scenarios(
        G=None, locations=locations,
        num_scenarios=N_SCENARIOS, seed=VALIDATION_SEED,
        save_path=os.path.join(os.path.dirname(__file__), "..", "output", "_validation_scenarios.csv"),
    )
    print(f"  Generated {len(scenarios)} scenarios.")

    # ------------------------------------------------------------------
    section("1. Kumaraswamy severity distribution")
    # ------------------------------------------------------------------
    ka = float(params["default_disaster"]["kumaraswamy_a"])
    kb = float(params["default_disaster"]["kumaraswamy_b"])
    print(f"  Configured parameters: a={ka}, b={kb} (from load_parameters() -> config/model_parameters.yaml)")

    severities = np.array([s["severity"] for s in scenarios])
    x_unit = (severities - 1.0) / 4.0  # invert the [1,5] rescaling back to (0,1)

    check(
        "severity bounds",
        bool(np.all(severities > 1.0) and np.all(severities < 5.0)),
        f"min={severities.min():.4f}, max={severities.max():.4f} (must lie strictly in (1,5))",
    )

    ks_stat, ks_p = stats.kstest(x_unit, kumaraswamy_cdf, args=(ka, kb))
    # with N=20000 well-specified draws, KS stat should be tiny (sampling
    # noise only) — a large value here would mean the inverse-CDF formula
    # in scenario_generator.py does not actually implement Kumaraswamy(a,b)
    check(
        "empirical draws vs. theoretical Kumaraswamy CDF (KS test)",
        ks_stat < 0.02,
        f"KS statistic={ks_stat:.5f}, p={ks_p:.4f} (threshold 0.02; this tests sampling "
        f"correctness, not the model's fit to real EM-DAT data — that fit, KS=0.0448, "
        f"was already validated during calibration)",
    )

    theoretical_mean = kb * stats.beta(1 + 1 / ka, kb).mean() if False else None
    # closed form: E[X] = b * B(1+1/a, b) ; compute via scipy's beta function
    from scipy.special import beta as beta_fn
    theo_mean_unit = kb * beta_fn(1 + 1 / ka, kb)
    emp_mean_unit = x_unit.mean()
    check(
        "empirical mean vs. theoretical Kumaraswamy mean",
        abs(emp_mean_unit - theo_mean_unit) < 0.01,
        f"empirical={emp_mean_unit:.5f}, theoretical={theo_mean_unit:.5f} "
        f"(mode ~2.95 on the [1,5] severity scale per calibration notes)",
    )

    # ------------------------------------------------------------------
    section("2. Epicenter country sampling vs. configured epicenter_weights")
    # ------------------------------------------------------------------
    epicenter_weights = params["epicenter_weights"]
    weight_countries = set(epicenter_weights.keys())
    node_country_set = set(node_countries.values())

    check(
        "every epicenter_weights country has >=1 network node (no dead weight)",
        weight_countries.issubset(node_country_set),
        f"missing: {sorted(weight_countries - node_country_set) or 'none'}",
    )
    check(
        "every network-node country has an epicenter_weights entry",
        node_country_set.issubset(weight_countries),
        f"missing: {sorted(node_country_set - weight_countries) or 'none'}",
    )

    empirical_country = pd.Series(
        [node_countries[s["epicenter"]] for s in scenarios]
    ).value_counts(normalize=True)

    countries_sorted = sorted(epicenter_weights.keys())
    expected = np.array([epicenter_weights[c] for c in countries_sorted])
    expected = expected / expected.sum()
    observed_frac = np.array([empirical_country.get(c, 0.0) for c in countries_sorted])
    max_dev = float(np.max(np.abs(observed_frac - expected)))

    observed_counts = observed_frac * N_SCENARIOS
    expected_counts = expected * N_SCENARIOS
    chi2, chi2_p = stats.chisquare(observed_counts, f_exp=expected_counts)

    check(
        "empirical epicenter-country distribution vs. configured weights (max abs deviation)",
        max_dev < 0.01,
        f"max |observed-expected| = {max_dev:.5f} across {len(countries_sorted)} countries "
        f"(chi2={chi2:.2f}, p={chi2_p:.4f}, N={N_SCENARIOS})",
    )

    top3 = ["CHN", "IDN", "PHL"]
    print("  Top-3 by configured weight vs. empirical:")
    for c in top3:
        print(f"    {c}: configured={epicenter_weights[c]:.4f}  empirical={empirical_country.get(c, 0.0):.4f}")

    # ------------------------------------------------------------------
    section("3. Disaster-type sampling vs. country_type_frequencies.csv / fallback_distribution")
    # ------------------------------------------------------------------
    type_freq_path = os.path.join(os.path.dirname(__file__), "data", "country_type_frequencies.csv")
    type_frequencies = load_type_frequencies(type_freq_path)
    fallback = params["type_sampling"]["fallback_distribution"]

    in_table = sorted(node_country_set & set(type_frequencies.keys()))
    fallback_countries = sorted(node_country_set - set(type_frequencies.keys()))
    print(f"  Network countries with a table entry: {len(in_table)} -> {in_table}")
    print(f"  Network countries using fallback (not in table): {fallback_countries}")

    # 3a. direct sampling test of sample_disaster_type() for a table country
    #     and a fallback country, independent of the full generate_scenarios() run
    rng = random.Random(VALIDATION_SEED)
    N_DIRECT = 20000

    table_test_country = in_table[0]
    draws = [sample_disaster_type(table_test_country, type_frequencies, fallback, rng) for _ in range(N_DIRECT)]
    empirical = pd.Series(draws).value_counts(normalize=True).reindex(DISASTER_TYPE_LABELS, fill_value=0.0)
    expected_row = pd.Series(type_frequencies[table_test_country]).reindex(DISASTER_TYPE_LABELS, fill_value=0.0)
    max_dev_country = float(np.max(np.abs(empirical.values - expected_row.values)))
    check(
        f"sample_disaster_type('{table_test_country}') matches its table row",
        max_dev_country < 0.02,
        f"max abs deviation={max_dev_country:.5f} over N={N_DIRECT} draws\n"
        f"    table:      {expected_row.round(4).to_dict()}\n"
        f"    empirical:  {empirical.round(4).to_dict()}",
    )

    fallback_sum = float(sum(fallback.values()))
    check(
        "type_sampling.fallback_distribution values sum to 1.0 (as a literal probability table)",
        abs(fallback_sum - 1.0) < 1e-6,
        f"sum={fallback_sum:.4f}. random.choices() normalizes weights internally, so sampling "
        f"is NOT broken by this (see next check), but the config values are not literal "
        f"probabilities as their names/comments imply — anyone reading the config file "
        f"and expecting these six numbers to sum to 1 will be misled. Likely leftover "
        f"probability mass from EM-DAT disaster types outside the six modeled here "
        f"(e.g. drought, extreme temperature) that never got renormalized away.",
    )

    if fallback_countries:
        fb_test_country = fallback_countries[0]
        draws_fb = [sample_disaster_type(fb_test_country, type_frequencies, fallback, rng) for _ in range(N_DIRECT)]
        empirical_fb = pd.Series(draws_fb).value_counts(normalize=True).reindex(DISASTER_TYPE_LABELS, fill_value=0.0)
        # random.choices() normalizes weights internally, so the *actual* sampled
        # distribution is fallback/sum(fallback), not the raw (non-unit-sum) config
        # values — compare against the renormalized version to test sampling
        # correctness in isolation from the sum-to-1 issue flagged above.
        expected_fb_raw = pd.Series(fallback).reindex(DISASTER_TYPE_LABELS, fill_value=0.0)
        expected_fb = expected_fb_raw / expected_fb_raw.sum()
        max_dev_fb = float(np.max(np.abs(empirical_fb.values - expected_fb.values)))
        check(
            f"sample_disaster_type('{fb_test_country}') [not in table] falls back to "
            f"global marginal, correctly renormalized by random.choices()",
            max_dev_fb < 0.02,
            f"max abs deviation={max_dev_fb:.5f} over N={N_DIRECT} draws (compared against "
            f"fallback values renormalized to sum to 1, matching random.choices' actual behavior)\n"
            f"    fallback (renormalized): {expected_fb.round(4).to_dict()}\n"
            f"    empirical:               {empirical_fb.round(4).to_dict()}",
        )

    # 3b. end-to-end: aggregate disaster_type frequency actually produced by
    #     generate_scenarios(), for the single largest-weight country, vs its table row
    dom_country = table_test_country
    dom_scenarios = [s for s in scenarios if node_countries[s["epicenter"]] == dom_country]
    if len(dom_scenarios) >= 200:
        emp_e2e = pd.Series(
            [s["disaster_type"] for s in dom_scenarios]
        ).value_counts(normalize=True).reindex(DISASTER_TYPE_LABELS, fill_value=0.0)
        max_dev_e2e = float(np.max(np.abs(emp_e2e.values - expected_row.values)))
        check(
            f"generate_scenarios() end-to-end type mix for '{dom_country}' matches its table row",
            max_dev_e2e < 0.05,
            f"max abs deviation={max_dev_e2e:.5f} over {len(dom_scenarios)} scenarios with epicenter in {dom_country}",
        )
    else:
        print(f"  [SKIP] too few draws with epicenter in {dom_country} ({len(dom_scenarios)}) for a stable e2e check")

    # ------------------------------------------------------------------
    section("4. Affected radius / node-severity decay formula (direct, non-statistical)")
    # ------------------------------------------------------------------
    base_r = float(params["default_disaster"]["affected_radius_km"]["base"])
    mult_r = float(params["default_disaster"]["affected_radius_km"]["multiplier"])

    sample = scenarios[0]
    expected_radius = base_r + sample["severity"] * mult_r
    check(
        "affected_radius_km == base + severity*multiplier",
        abs(sample["affected_radius_km"] - expected_radius) < 1e-6,
        f"got {sample['affected_radius_km']:.3f}, expected {expected_radius:.3f}",
    )

    epicenter_severity_ok = True
    decay_monotonic_ok = True
    zero_outside_ok = True
    from geopy.distance import geodesic
    for s in scenarios[:200]:  # spot-check a subset for speed
        epi = s["epicenter"]
        epi_coords = (locations[epi]["Latitude"], locations[epi]["Longitude"])
        radius = s["affected_radius_km"]
        prev_dist = -1
        for i, sev in s["node_severity"].items():
            node_coords = (locations[i]["Latitude"], locations[i]["Longitude"])
            dist = geodesic(epi_coords, node_coords).km
            if dist > radius + 1e-6:
                if sev != 0.0:
                    zero_outside_ok = False
            else:
                expected_sev = max(0.0, s["severity"] * (1.0 - dist / radius))
                if abs(sev - expected_sev) > 1e-6:
                    decay_monotonic_ok = False
        if abs(s["node_severity"].get(epi, -1) - s["severity"]) > 1e-6:
            epicenter_severity_ok = False

    check("epicenter node_severity == scenario severity (zero distance decay)", epicenter_severity_ok,
          "checked first 200 scenarios")
    check("node_severity matches linear decay formula severity*(1-dist/radius) for all affected nodes",
          decay_monotonic_ok, "checked first 200 scenarios")
    check("node_severity == 0 for every node beyond affected_radius_km", zero_outside_ok,
          "checked first 200 scenarios")

    # ------------------------------------------------------------------
    section("5. Reproducibility (same seed -> identical output)")
    # ------------------------------------------------------------------
    scenarios_repeat = generate_scenarios(
        G=None, locations=locations,
        num_scenarios=500, seed=32,
        save_path=os.path.join(os.path.dirname(__file__), "..", "output", "_validation_repeat_a.csv"),
    )
    scenarios_repeat2 = generate_scenarios(
        G=None, locations=locations,
        num_scenarios=500, seed=32,
        save_path=os.path.join(os.path.dirname(__file__), "..", "output", "_validation_repeat_b.csv"),
    )
    identical = all(
        a["epicenter"] == b["epicenter"]
        and a["disaster_type"] == b["disaster_type"]
        and abs(a["severity"] - b["severity"]) < 1e-12
        for a, b in zip(scenarios_repeat, scenarios_repeat2)
    )
    check("seed=32 reproduces byte-identical epicenter/type/severity draws across two runs", identical,
          f"compared {len(scenarios_repeat)} scenarios")

    # ------------------------------------------------------------------
    section("SUMMARY")
    # ------------------------------------------------------------------
    if FAILURES:
        print(f"  {len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print("  All checks PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
