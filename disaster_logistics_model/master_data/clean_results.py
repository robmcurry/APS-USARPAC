import pandas as pd
import os

def remove_duplicate_rows(csv_path: str):
    """
    Remove fully duplicate rows from the CSV and save a cleaned version.
    Returns the path to the cleaned file.
    """
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    # Remove duplicate rows ignoring computational time differences
    cols_to_compare = [col for col in df.columns if col != "computational_time"]

    before = len(df)
    df_clean = df.drop_duplicates(subset=cols_to_compare)
    after = len(df_clean)

    cleaned_path = os.path.join(
        os.path.dirname(csv_path),
        "CLEANED_" + os.path.basename(csv_path)
    )

    df_clean.to_csv(cleaned_path, index=False)

    print(f"Removed {before - after} duplicate rows.")
    print(f"Clean file saved to:\n{cleaned_path}")

    return cleaned_path

if __name__ == "__main__":
    csv_path = "master_data/Master_All Scenario_Def_1_100_1000_Final.csv"
    remove_duplicate_rows(csv_path)