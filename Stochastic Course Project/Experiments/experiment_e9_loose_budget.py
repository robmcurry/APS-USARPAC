from Experiments.common import run_single_experiment

if __name__ == "__main__":
    run_single_experiment(
        experiment_code="E9",
        experiment_label="Loose Budget",
        gamma=0.5,
        beta=0.90,
        p_max=5,
        budget=15,   # replace with B0 once calibrated
        tau=4,
        rho=0.3,
        num_scenarios=10,
        base_output_dir="output",
        verbose=False,
    )