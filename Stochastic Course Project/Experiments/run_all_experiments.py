from Experiments.common import run_single_experiment

if __name__ == "__main__":

    experiments = [
        # --- Baseline ---
        dict(
            experiment_code="E1",
            experiment_label="Baseline",
            gamma=0.5,
            beta=0.90,
            p_max=5,
            budget=None,
            tau=4,
            rho=0.3,
        ),

        # --- Fragility ---
        dict(
            experiment_code="E2",
            experiment_label="Low Fragility",
            gamma=0.2,
            beta=0.90,
            p_max=5,
            budget=None,
            tau=4,
            rho=0.3,
        ),
        dict(
            experiment_code="E3",
            experiment_label="High Fragility",
            gamma=0.8,
            beta=0.90,
            p_max=5,
            budget=None,
            tau=4,
            rho=0.3,
        ),

        # --- Risk ---
        dict(
            experiment_code="E4",
            experiment_label="Low Risk",
            gamma=0.5,
            beta=0.70,
            p_max=5,
            budget=None,
            tau=4,
            rho=0.3,
        ),
        dict(
            experiment_code="E5",
            experiment_label="High Risk",
            gamma=0.5,
            beta=0.95,
            p_max=5,
            budget=None,
            tau=4,
            rho=0.3,
        ),

        # --- Capacity ---
        dict(
            experiment_code="E6",
            experiment_label="Low Capacity",
            gamma=0.5,
            beta=0.90,
            p_max=3,
            budget=None,
            tau=4,
            rho=0.3,
        ),
        dict(
            experiment_code="E7",
            experiment_label="High Capacity",
            gamma=0.5,
            beta=0.90,
            p_max=7,
            budget=None,
            tau=4,
            rho=0.3,
        ),

        # --- Budget ---
        dict(
            experiment_code="E8",
            experiment_label="Tight Budget",
            gamma=0.5,
            beta=0.90,
            p_max=5,
            budget=9,
            tau=4,
            rho=0.3,
        ),
        dict(
            experiment_code="E9",
            experiment_label="Loose Budget",
            gamma=0.5,
            beta=0.90,
            p_max=5,
            budget=15,
            tau=4,
            rho=0.3,
        ),
    ]

    for exp in experiments:
        print("\n" + "=" * 70)
        print(f"Running {exp['experiment_code']}: {exp['experiment_label']}")
        print("=" * 70)

        run_single_experiment(
            experiment_code=exp["experiment_code"],
            experiment_label=exp["experiment_label"],
            gamma=exp["gamma"],
            beta=exp["beta"],
            p_max=exp["p_max"],
            budget=exp["budget"],
            tau=exp["tau"],
            rho=exp["rho"],
            num_scenarios=10,
            base_output_dir="output",
            verbose=False,
        )

    print("\nAll experiments complete.")