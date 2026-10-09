from Experiments.common import run_single_experiment

if __name__ == "__main__":
    run_single_experiment(
        experiment_code="E5",
        experiment_label="High Risk",
        gamma=0.5,
        beta=0.95,
        p_max=5,
        budget=None,   # replace with B0 once calibrated
        tau=4,
        rho=0.3,
        num_scenarios=10,
        base_output_dir="output",
        verbose=False,
    )