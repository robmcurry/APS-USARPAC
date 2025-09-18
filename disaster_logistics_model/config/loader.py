# loader.py
import yaml
import os

def load_parameters(config_path: str = None):
    if config_path is None:
        # Default path relative to project structure
        config_path = os.path.join(os.path.dirname(__file__), "model_parameters.yaml")

    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
        print(params)
    return params