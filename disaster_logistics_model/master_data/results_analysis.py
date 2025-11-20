import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#
# Folder where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load node names
CITY_CSV = os.path.join(os.path.dirname(BASE_DIR), "data", "pacific_cities.csv")
if os.path.exists(CITY_CSV):
    city_df = pd.read_csv(CITY_CSV)

    # Basic name mapping
    NODE_TO_NAME = dict(zip(city_df["Node ID"], city_df["Node Name"]))

    # 1. Reverse mapping: name -> node ID
    NAME_TO_NODE = dict(zip(city_df["Node Name"], city_df["Node ID"]))

    # 2. Geolocation mapping: node -> (lat, lon)
    if "Latitude" in city_df.columns and "Longitude" in city_df.columns:
        NODE_TO_GEO = {
            row["Node ID"]: (row["Latitude"], row["Longitude"])
            for _, row in city_df.iterrows()
        }
    else:
        NODE_TO_GEO = {}

    # 3. Region mapping: node -> region code (expects column "Region")
    if "Region" in city_df.columns:
        NODE_TO_REGION = dict(zip(city_df["Node ID"], city_df["Region"]))
    else:
        NODE_TO_REGION = {}

else:
    NODE_TO_NAME = {}
    NAME_TO_NODE = {}
    NODE_TO_GEO = {}
    NODE_TO_REGION = {}

# ============================================================
# 1. PATHS / DIRECTORIES
# ============================================================


# Input CSV – same folder
CSV_PATH = os.path.join(BASE_DIR, "CLEANED_Master_All Scenario_Def_1_100_1000_Final.csv")

# Output folders (will be created if they don't exist)
OUTPUT_DIR = os.path.join(BASE_DIR, "aps_siting_outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")

for d in [OUTPUT_DIR, FIG_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# 2. LOADING & BASIC CLEANING
# ============================================================

def load_data():
    """
    Load the master CSV, parse aps_locations, and filter to feasible scenarios.
    Feasible: status == "2"
    Infeasible: status == "infeasible" (kept separate for later use if needed).
    """
    print(f"Loading: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Parse aps_locations from string -> Python list
    if "aps_locations" in df.columns:
        df["aps_locations"] = df["aps_locations"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
        )

    # Separate feasible vs infeasible
    if "status" in df.columns:
        feasible = df[df["status"] == "2"].copy()
        infeasible = df[df["status"] == "infeasible"].copy()
    else:
        feasible = df.copy()
        infeasible = df.iloc[0:0].copy()  # empty

    print(f"Total rows:     {len(df)}")
    print(f"Feasible rows:  {len(feasible)}")
    print(f"Infeasible rows:{len(infeasible)}\n")

    # Save a snapshot of full dataset for reference
    df.to_csv(os.path.join(TABLE_DIR, "full_dataset_copy.csv"), index=False)

    return df, feasible, infeasible


# ============================================================
# 3. APS SITING – GLOBAL FREQUENCY
# ============================================================

def aps_global_frequency(feasible_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how often each node is selected as an APS across all feasible scenarios.
    Returns a DataFrame indexed by node with columns:
      - count
      - share_of_feasible_scenarios
    """
    if "aps_locations" not in feasible_df.columns:
        print("No aps_locations column; cannot compute APS global frequency.")
        return pd.DataFrame()

    freq = {}
    scenario_count = 0

    for aps_list in feasible_df["aps_locations"]:
        if isinstance(aps_list, list):
            scenario_count += 1
            for node in aps_list:
                freq[node] = freq.get(node, 0) + 1

    if scenario_count == 0:
        print("No APS lists found in feasible scenarios.")
        return pd.DataFrame()

    freq_df = pd.DataFrame.from_dict(freq, orient="index", columns=["count"])
    freq_df.index.name = "node"
    freq_df["share_of_feasible_scenarios"] = freq_df["count"] / scenario_count
    freq_df = freq_df.sort_values("count", ascending=False)

    # Map node numbers to names if available
    freq_df["node_name"] = freq_df.index.map(lambda x: NODE_TO_NAME.get(int(x), str(x)))
    freq_df = freq_df.set_index("node_name")

    # Save table
    out_path = os.path.join(TABLE_DIR, "aps_global_frequency.csv")
    freq_df.to_csv(out_path)

    # Publication-ready LaTeX table
    tex_out = out_path.replace(".csv", ".tex")
    with open(tex_out, "w") as f_tex:
        f_tex.write(freq_df.to_latex(index=True, float_format="%.3f"))

    print(f"Saved APS global frequency table -> {out_path}")

    # Plot top 20 nodes by frequency
    plt.figure(figsize=(10, 6))
    freq_df.head(20)["count"].plot(kind="bar")
    plt.title("APS Selection – Top 20 Nodes (Feasible Scenarios Only)")
    plt.xlabel("Node")
    plt.ylabel("Selection Count")
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "aps_top20_global_frequency.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved APS global frequency figure -> {fig_path}")

    return freq_df


# ============================================================
# 4. APS SITING – BY SEVERITY BINS
# ============================================================

def make_severity_bins(df: pd.DataFrame, col: str = "severity") -> pd.DataFrame:
    """
    Add a 'severity_bin' column using fixed bins: 1–2, 2–3, 3–4, 4–5, 5–6.
    This matches your preference for binned severity analysis.
    """
    if col not in df.columns:
        return df

    # Adjust bin edges to integer 1–5 boundaries
    bins = [1, 2, 3, 4, 5]
    labels = ["1–2", "2–3", "3–4", "4–5"]

    df = df.copy()
    df["severity_bin"] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return df


def aps_by_severity_bin(feasible_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each severity bin, compute how often each node appears as an APS.
    Returns a pivot table: index = severity_bin, columns = node, values = count.
    """
    if "aps_locations" not in feasible_df.columns or "severity" not in feasible_df.columns:
        print("Missing aps_locations or severity; skipping APS-by-severity-bin analysis.")
        return pd.DataFrame()

    df = make_severity_bins(feasible_df)

    records = []
    for _, row in df.iterrows():
        aps_list = row["aps_locations"]
        sev_bin = row["severity_bin"]
        if pd.isna(sev_bin) or not isinstance(aps_list, list):
            continue
        for node in aps_list:
            records.append((sev_bin, node))

    if not records:
        print("No APS records after severity binning.")
        return pd.DataFrame()

    sev_node_df = pd.DataFrame(records, columns=["severity_bin", "node"])
    pivot = (
        sev_node_df
        .groupby(["severity_bin", "node"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    # Rename columns (nodes) with actual names
    pivot = pivot.rename(columns=lambda x: NODE_TO_NAME.get(int(x), str(x)))

    # ---- Add two summary rows: total data points per bin, and infeasible per bin ----
    try:
        full_df = pd.read_csv(CSV_PATH)
        full_df = make_severity_bins(full_df)

        # Total scenarios per severity bin
        total_per_bin = full_df.groupby("severity_bin").size()

        # Infeasible scenarios per severity bin
        infeasible_per_bin = (
            full_df[full_df["status"] == "infeasible"]
            .groupby("severity_bin")
            .size()
        )

        # Create rows matching the pivot's columns
        total_row = pd.Series(
            {col: total_per_bin.get(bin_label, 0) for col in pivot.columns},
            name="TOTAL_SCENARIOS"
        )

        infeasible_row = pd.Series(
            {col: infeasible_per_bin.get(bin_label, 0) for col in pivot.columns},
            name="INFEASIBLE_SCENARIOS"
        )

        # Insert rows at the top
        pivot = pd.concat([total_row.to_frame().T, infeasible_row.to_frame().T, pivot])
    except Exception as e:
        print("Warning: Could not compute summary rows:", e)

    out_path = os.path.join(TABLE_DIR, "aps_by_severity_bin.csv")
    pivot.to_csv(out_path)

    tex_out = out_path.replace(".csv", ".tex")
    with open(tex_out, "w") as f_tex:
        f_tex.write(pivot.T.to_latex(index=True, float_format="%.3f"))

    print(f"Saved APS-by-severity-bin table -> {out_path}")

    # Heatmap of APS frequency by severity bin
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="viridis")
    plt.title("APS Selection Frequency by Severity Bin and Node (Feasible Only)")
    plt.xlabel("Node")
    plt.ylabel("Severity Bin")
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "aps_by_severity_bin_heatmap.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved APS-by-severity-bin heatmap -> {fig_path}")

    return pivot


# ============================================================
# 5. APS SITING – PAIRWISE CO-SELECTION
# ============================================================

def aps_pairwise_coselection(feasible_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a pairwise co-selection matrix:
    entry (i, j) = number of feasible scenarios where nodes i and j both appear in aps_locations.
    Diagonal (i, i) = number of feasible scenarios where node i is used as APS at all.
    """
    if "aps_locations" not in feasible_df.columns:
        print("No aps_locations column; skipping pairwise co-selection.")
        return pd.DataFrame()

    # Collect all unique APS nodes across feasible scenarios
    all_nodes = set()
    for aps_list in feasible_df["aps_locations"]:
        if isinstance(aps_list, list):
            all_nodes.update(aps_list)

    if not all_nodes:
        print("No APS nodes found in feasible solutions.")
        return pd.DataFrame()

    all_nodes = sorted(all_nodes)
    pair_df = pd.DataFrame(0, index=all_nodes, columns=all_nodes, dtype=int)

    # Fill co-selection counts
    for aps_list in feasible_df["aps_locations"]:
        if not isinstance(aps_list, list):
            continue
        unique_nodes = sorted(set(aps_list))
        for i, ni in enumerate(unique_nodes):
            # diagonal: node selected in this scenario
            pair_df.loc[ni, ni] += 1
            for nj in unique_nodes[i + 1:]:
                pair_df.loc[ni, nj] += 1
                pair_df.loc[nj, ni] += 1

    # Rename both index and columns
    pair_df = pair_df.rename(index=lambda x: NODE_TO_NAME.get(int(x), str(x)))
    pair_df = pair_df.rename(columns=lambda x: NODE_TO_NAME.get(int(x), str(x)))

    out_path = os.path.join(TABLE_DIR, "aps_pairwise_coselection.csv")
    pair_df.to_csv(out_path)

    tex_out = out_path.replace(".csv", ".tex")
    with open(tex_out, "w") as f_tex:
        f_tex.write(pair_df.to_latex(index=True, float_format="%.3f"))

    print(f"Saved APS pairwise co-selection table -> {out_path}")

    # Heatmap (this can get big; still useful for pattern hunting)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pair_df, cmap="magma")
    plt.title("APS Pairwise Co-selection (Feasible Scenarios Only)")
    plt.xlabel("Node")
    plt.ylabel("Node")
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "aps_pairwise_coselection_heatmap.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved APS pairwise co-selection heatmap -> {fig_path}")

    return pair_df


# ============================================================
# 6. DRIVER – RUN ALL APS SITING ANALYSIS
# ============================================================

def run_aps_siting_analysis():
    df, feasible, infeasible = load_data()

    # Save infeasible scenario list separately for later diagnosis
    if len(infeasible) > 0:
        bad_path = os.path.join(TABLE_DIR, "infeasible_scenarios.csv")
        infeasible.to_csv(bad_path, index=False)
        print(f"Saved infeasible scenarios -> {bad_path}")

    # 1. Global APS frequency
    freq_df = aps_global_frequency(feasible)

    # 2. APS by severity bin
    sev_pivot = aps_by_severity_bin(feasible)

    # 3. APS pairwise co-selection
    pair_df = aps_pairwise_coselection(feasible)

    # 4. Simple markdown report for siting question
    report_path = os.path.join(OUTPUT_DIR, "aps_siting_report.md")
    with open(report_path, "w") as f:
        f.write("# APS Siting Analysis – Where Do We Put Stuff?\n\n")
        f.write("This report is based **only on feasible scenarios** (`status == '2'`).\n\n")

        f.write("## 1. Global APS Frequency\n")
        if freq_df is not None and not freq_df.empty:
            f.write(freq_df.head(20).to_markdown())
            f.write("\n\n")
        else:
            f.write("No APS frequency data available.\n\n")

        f.write("## 2. APS by Severity Bins\n")
        if sev_pivot is not None and not sev_pivot.empty:
            f.write("APS frequency by severity bin (first few rows):\n\n")
            f.write(sev_pivot.head().to_markdown())
            f.write("\n\n")
        else:
            f.write("No severity-binned APS data available.\n\n")

        f.write("## 3. APS Pairwise Co-selection\n")
        if pair_df is not None and not pair_df.empty:
            f.write("Pairwise co-selection matrix saved as CSV; see tables folder.\n\n")
        else:
            f.write("No pairwise co-selection data available.\n\n")

        f.write("## 4. Infeasible Scenarios\n")
        f.write(f"Number of infeasible scenarios: {len(infeasible)}\n\n")
        if len(infeasible) > 0:
            f.write("Full list in `infeasible_scenarios.csv`.\n")

        f.write("\n\n## Figures\n")
        f.write("All figures saved under `aps_siting_outputs/figures`.\n")

    print("\nAPS siting analysis complete!")
    print(f"- Report:  {report_path}")
    print(f"- Figures: {FIG_DIR}")
    print(f"- Tables:  {TABLE_DIR}")


# ============================================================
# 7. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_aps_siting_analysis()