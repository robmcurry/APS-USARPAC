import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- CONFIGURATION ----------
USE_FOLDER = False  # Set to False to use a single file
FOLDER_PATH = './output/Data'  # Replace with the path to your data folder
MAIN_FILE_NAME = 'Base100.csv'  # File to use if USE_FOLDER is False
OUTPUT_DIR = "./output/Master Plots"

# ---------- SETUP ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_files(folder_path, use_folder, main_file):
    if use_folder:
        return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    else:
        return [os.path.join(folder_path, main_file)]

def generate_plots(df, tag):
    sns.set(style="whitegrid")

    # Boxplot: Objective by num_APS
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='num_APS', y='objective')
    plt.title('Objective by num_APS')
    plt.savefig(f'{OUTPUT_DIR}/boxplot_objective_num_APS_{tag}.png')
    plt.close()

    # Scatterplot: Severity vs Objective by num_APS
    if 'severity' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x='severity', y='objective', hue='num_APS')
        plt.title('Objective vs Severity by num_APS')
        plt.savefig(f'{OUTPUT_DIR}/scatter_severity_objective_{tag}.png')
        plt.close()

    # Heatmap: Objective by num_APS and L
    if 'L' in df.columns:
        heatmap_data = df.groupby(['num_APS', 'L'])['objective'].mean().unstack()
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="coolwarm")
        plt.title('Mean Objective by num_APS and L')
        plt.savefig(f'{OUTPUT_DIR}/heatmap_objective_APS_L_{tag}.png')
        plt.close()

    # Bar chart: Objective by Epicenter
    if 'epicenter' in df.columns:
        epicenter_means = df.groupby('epicenter')['objective'].mean().sort_values()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=epicenter_means.values, y=epicenter_means.index)
        plt.title('Average Objective by Epicenter')
        plt.xlabel('Mean Objective')
        plt.savefig(f'{OUTPUT_DIR}/bar_objective_epicenter_{tag}.png')
        plt.close()

    # Lineplot: Objective by Scenario ID
    if 'scenario_id' in df.columns:
        plt.figure(figsize=(12, 5))
        sns.lineplot(data=df.sort_values('scenario_id'), x='scenario_id', y='objective', hue='num_APS', marker='o')
        plt.title('Objective by Scenario ID')
        plt.savefig(f'{OUTPUT_DIR}/lineplot_objective_scenario_{tag}.png')
        plt.close()

def main():
    files = load_files(FOLDER_PATH, USE_FOLDER, MAIN_FILE_NAME)
    try:
        combined_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        tag = "combined"
        generate_plots(combined_df, tag)
        print(f"Generated plots for combined data")
    except Exception as e:
        print(f"Error processing combined data: {e}")

if __name__ == "__main__":
    main()