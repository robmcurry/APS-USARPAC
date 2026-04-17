from Experiments.common import run_single_experiment

if __name__ == "__main__":
    run_single_experiment(
        experiment_code="E2",
        experiment_label="Low Fragility",
        gamma=0.2,
        beta=0.90,
        p_max=5,
        budget=None,   # use baseline budget from YAML
        tau=4,
        rho=0.3,
        num_scenarios=10,
        base_output_dir="output",
        verbose=False,
    )