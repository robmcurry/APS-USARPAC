import pandas as pd
import os
import re

def compile_results(input_dir="output", output_file="output/Data/Base100.csv"):
    records = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.startswith("batch_summary") or not file.endswith(".csv"):
                continue

            filepath = os.path.join(root, file)

            # Try to extract parameters from filename
            num_APS = L = base_redundancy = None

            if "Redun" in file:
                match = re.search(r"Redun_(\d+)", file)
                if match:
                    base_redundancy = int(match.group(1))
                    L = 3
                    num_APS = 5  # adjust if your experiment used a different value

            elif "L_" in file:
                match = re.search(r"L_(\d+)", file)
                if match:
                    L = int(match.group(1))
                    base_redundancy = 3
                    num_APS = 5  # adjust if needed

            elif "NumAPS" in file:
                match = re.search(r"NumAPS_(\d+)", file)
                if match:
                    num_APS = int(match.group(1))
                    L = 3
                    base_redundancy = 3

            else:
                print(f"Skipped file: {file} (couldn't parse parameters)")
                continue

            # Load the batch file
            df = pd.read_csv(filepath)

            for _, row in df.iterrows():
                records.append({
                    "scenario_id": row.get("scenario_id"),
                    "num_APS": num_APS,
                    "L": L,
                    "base_redundancy": base_redundancy,
                    "objective": row.get("objective"),
                    "total_flow": row.get("total_flow"),
                    "shortfall_cost": row.get("shortfall_cost", None),
                    "deficit_cost": row.get("deficit_cost", None),
                    "status": row.get("status", None),
                    "severity": row.get("severity", None),
                    "epicenter": row.get("epicenter", None)
                })

    # Create and save master DataFrame
    compiled_df = pd.DataFrame(records)
    compiled_df.sort_values(["num_APS", "L", "base_redundancy", "scenario_id"], inplace=True)
    compiled_df.to_csv(output_file, index=False)
    print(f"Compiled results saved to {output_file}")

if __name__ == "__main__":
    compile_results()