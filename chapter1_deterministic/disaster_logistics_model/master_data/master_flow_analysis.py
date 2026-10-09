# ============================================================
# master_flow_analysis.py
#
# Purpose:
#   Post-process scenario-level flow outputs to identify
#   dominant supply routes by:
#     (1) frequency of use
#     (2) total flow volume
#
# Outputs:
#   - Clean black-and-white PDF figures
#   - Aggregated CSV summaries
#
# All artifacts saved to:
#   aps_flow_outputs/
# ============================================================

import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_CSV = os.path.join(
    BASE_DIR,
    "Master_All Scenario_Def_1_100_1000_Final_duplicates removed.csv"
)

CITY_CSV = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "pacific_cities.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "aps_flow_outputs")
TOP_K = 20

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

city_df = pd.read_csv(CITY_CSV)

NODE_TO_CITY = dict(
    zip(city_df["Node ID"], city_df["Node Name"])
)

# Global plotting style (match APS figures)
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "font.size": 11,
    "font.family": "serif"
})

BAR_COLOR = "0.3"

# ----------------------------
# Load data
# ----------------------------

df = pd.read_csv(INPUT_CSV)

# Keep only rows with valid flow summaries
df = df[df["flow_summary"].notna()].copy()

# Parse flow_summary safely
def parse_flow_dict(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return {}

df["flow_dict"] = df["flow_summary"].apply(parse_flow_dict)

# ----------------------------
# Explode flows to arc-level table
# ----------------------------

records = []

for _, row in df.iterrows():
    for (i, j), flow in row["flow_dict"].items():
        from_city = NODE_TO_CITY.get(i, f"Node {i}")
        to_city = NODE_TO_CITY.get(j, f"Node {j}")

        records.append({
            "from_node": i,
            "to_node": j,
            "from_city": from_city,
            "to_city": to_city,
            "flow": flow,
            "route": f"{from_city} ({i}) → {to_city} ({j})",
            "route_nodes": f"{i} → {j}"
        })

flow_df = pd.DataFrame(records)

# Save exploded flow table
flow_df.to_csv(
    os.path.join(OUTPUT_DIR, "arc_level_flows.csv"),
    index=False
)

# ----------------------------
# Route usage frequency
# ----------------------------

route_frequency = (
    flow_df
    .groupby("route")
    .size()
    .reset_index(name="frequency")
    .sort_values("frequency", ascending=False)
)

route_frequency.to_csv(
    os.path.join(OUTPUT_DIR, "route_frequency.csv"),
    index=False
)

top_freq = (
    route_frequency
    .head(TOP_K)
    .sort_values("frequency")
)

plt.figure()
plt.barh(top_freq["route"], top_freq["frequency"], color=BAR_COLOR)
plt.xlabel("Number of Scenarios Used")
plt.ylabel("Route")
plt.title("Most Frequently Used Supply Routes")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "route_frequency_topK.pdf")
)
plt.close()

# ----------------------------
# Route total flow volume
# ----------------------------

route_volume = (
    flow_df
    .groupby("route")["flow"]
    .sum()
    .reset_index(name="total_flow")
    .sort_values("total_flow", ascending=False)
)

route_volume.to_csv(
    os.path.join(OUTPUT_DIR, "route_total_flow.csv"),
    index=False
)

top_volume = (
    route_volume
    .head(TOP_K)
    .sort_values("total_flow")
)

plt.figure()
plt.barh(top_volume["route"], top_volume["total_flow"], color=BAR_COLOR)
plt.xlabel("Total Flow Volume")
plt.ylabel("Route")
plt.title("Highest Volume Supply Routes")

ax = plt.gca()
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
ax.ticklabel_format(style='plain', axis='x')

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "route_total_flow_topK.pdf")
)
plt.close()

# ----------------------------
# Frequency vs Volume scatter
# ----------------------------

route_stats = (
    route_frequency
    .merge(route_volume, on="route")
)

route_stats.to_csv(
    os.path.join(OUTPUT_DIR, "route_frequency_vs_volume.csv"),
    index=False
)

plt.figure()
plt.scatter(
    route_stats["frequency"],
    route_stats["total_flow"],
    color=BAR_COLOR,
    s=15
)
plt.xlabel("Route Usage Frequency")
plt.ylabel("Total Flow Volume")
plt.title("Route Frequency vs Total Flow")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "route_frequency_vs_volume.pdf")
)
plt.close()

print("Flow analysis complete. Outputs saved to:", OUTPUT_DIR)