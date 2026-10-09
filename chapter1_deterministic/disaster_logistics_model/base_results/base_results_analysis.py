import matplotlib
matplotlib.use("TkAgg")
import os
import ast
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif"
})

# ---------------------------------------------------------------------
# Load Base Data
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
DATA_FILE = os.path.join(BASE_DIR, "CLEANED_base_All Scenario_Def_1_100_1000_Final.csv")
CITY_FILE = os.path.join(os.path.dirname(BASE_DIR), "data", "pacific_cities.csv")

df = pd.read_csv(DATA_FILE)

# Ensure aps_locations parsed as list
def parse_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(str(x))
    except:
        return []

df["aps_locations"] = df["aps_locations"].apply(parse_list)

# ---------------------------------------------------------------------
# Load City Mapping
# ---------------------------------------------------------------------
city_df = pd.read_csv(CITY_FILE)
node_to_city = dict(zip(city_df["Node ID"], city_df["Node Name"]))

# ---------------------------------------------------------------------
# Compute APS frequency counts
# ---------------------------------------------------------------------
all_nodes = []
for lst in df["aps_locations"]:
    all_nodes.extend(lst)

aps_counts = pd.Series(all_nodes).value_counts().sort_values(ascending=False)
aps_counts.name = "count"
aps_counts = aps_counts.reset_index().rename(columns={"index": "node"})

TOP_N = 20  # change this to 5, 20, etc.
aps_counts = aps_counts.head(TOP_N)

# Add city names
aps_counts["city_label"] = aps_counts["node"].apply(
    lambda n: f"{node_to_city.get(n, 'Unknown')} ({n})"
)

# ---------------------------------------------------------------------
# Plot Bar Chart
# ---------------------------------------------------------------------
plt.figure(figsize=(10, 8))
plt.barh(aps_counts["city_label"], aps_counts["count"], color="0.3")
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.subplots_adjust(left=0.35)

for i, v in enumerate(aps_counts["count"]):
    plt.text(v + 0.5, i, str(v), va="center", fontsize=10)

plt.title("APS Site Selection Frequency (100 Base Scenarios)")
plt.xlabel("Number of Selections")
plt.ylabel("Top 20 Candidate Locations (Node ID")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "aps_frequency.pdf"), format="pdf", bbox_inches="tight")
plt.close()
plt.show()