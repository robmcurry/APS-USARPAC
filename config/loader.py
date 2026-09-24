"""
loader.py

Small helper module that loads model_parameters.yaml into a plain
dict so the rest of the pipeline (scenario generator, input builder, model) can
read configuration values without each module needing its own yaml/path
handling.
"""

import yaml
import os

def load_parameters(config_path: str = None):
    """
    Load the stochastic model parameter file.

    config_path: optional path to a yaml file - if not given, defaults to
    model_parameters.yaml in this same directory

    returns the parsed yaml contents as a nested dict (seed, num_scenarios,
    default_disaster, commodities, beta, P_max, vehicles, etc.)
    """
    if config_path is None:
        # Default path relative to project structure
        config_path = os.path.join(os.path.dirname(__file__), "model_parameters.yaml")

    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
    return params
