from pathlib import Path
from urllib.request import urlretrieve

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
MASTER_SUMMARY_PATH = OUTPUT_DIR / "all_experiments_summary.csv"
PLOTTING_SUMMARY_PATH = OUTPUT_DIR / "all_experiments_plotting_summary.csv"
FIGURES_DIR = PROJECT_ROOT / "visualizations" / "figures"
PRESENTATION_FIGURES_DIR = PROJECT_ROOT / "visualizations" / "presentation_figures"

LOCATIONS_CSV_PATH = PROJECT_ROOT / "pacific_cities.csv"
BASEMAP_DIR = PROJECT_ROOT / "visualizations" / "basemap_data"
NATURAL_EARTH_COUNTRIES_URL = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
def load_world_basemap(basemap_dir: Path) -> gpd.GeoDataFrame:
    """
    Load a low-resolution world country basemap from Natural Earth.
    The zipped shapefile is downloaded once and then reused locally.
    """
    basemap_dir.mkdir(parents=True, exist_ok=True)
    zip_path = basemap_dir / "ne_110m_admin_0_countries.zip"

    if not zip_path.exists():
        urlretrieve(NATURAL_EARTH_COUNTRIES_URL, zip_path)

    world = gpd.read_file(zip_path)
    return world


PLOTTING_COLUMNS = [
    "experiment_code",
    "experiment_label",
    "gamma",
    "beta",
    "P_max",
    "budget_used",
    "num_selected_sites",
    "objective_value",
    "total_unmet",
    "avg_scenario_loss",
    "total_transport_cost",
]


FRAGILITY_CODES = ["E2", "E1", "E3"]
CAPACITY_BUDGET_CODES = ["E1", "E6", "E7", "E8", "E9"]
RISK_CODES = ["E4", "E1", "E5"]

COLOR_STEEL_BLUE = "#5E7486"
COLOR_RUST = "#9A2F24"
COLOR_GREEN = "#2F6B35"
COLOR_CHARCOAL = "#2F2F2F"


def add_bar_labels(
    ax,
    heights,
    label_values=None,
    decimals: int = 1,
) -> None:
    """
    Add numeric labels above bars.

    Parameters
    ----------
    ax : matplotlib axis
        Axis containing the bars.
    heights : sequence of float
        The actual plotted bar heights.
    label_values : sequence of float, optional
        Raw values to display in the labels. If omitted, uses heights.
    decimals : int
        Number of decimal places to show.
    """
    if label_values is None:
        label_values = heights

    ymax = max(heights) if len(heights) > 0 else 0.0
    offset = 0.02 * ymax if ymax > 0 else 0.0

    for i, (height, label_value) in enumerate(zip(heights, label_values)):
        ax.text(
            i,
            height + offset,
            f"{label_value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def load_all_experiment_summaries(output_dir: Path) -> pd.DataFrame:
    """
    Read every experiment-level summary.csv file from the output directory
    and combine them into a single dataframe.
    """
    summary_files = sorted(output_dir.glob("E*/summary.csv"))

    if not summary_files:
        raise FileNotFoundError(
            f"No summary.csv files found under {output_dir}"
        )

    frames = []
    for summary_file in summary_files:
        df = pd.read_csv(summary_file)
        df["source_folder"] = summary_file.parent.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if "experiment_code" in combined.columns:
        combined["experiment_num"] = (
            combined["experiment_code"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )
        combined = combined.sort_values(
            by=["experiment_num", "experiment_code"],
            kind="stable",
        ).drop(columns=["experiment_num"])

    return combined


def load_selected_scenario_losses(output_dir: Path, experiment_codes: list[str]) -> pd.DataFrame:
    """
    Read scenario_losses.csv for the selected experiments and combine them into
    a single dataframe.
    """
    frames = []

    for experiment_code in experiment_codes:
        matches = sorted(output_dir.glob(f"{experiment_code}_*/scenario_losses.csv"))
        if not matches:
            raise FileNotFoundError(
                f"No scenario_losses.csv found for experiment code {experiment_code}"
            )

        scenario_file = matches[0]
        df = pd.read_csv(scenario_file)
        df["source_folder"] = scenario_file.parent.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["experiment_code"] = pd.Categorical(
        combined["experiment_code"], categories=experiment_codes, ordered=True
    )
    combined = combined.sort_values(["experiment_code", "scenario_id"]).reset_index(drop=True)

    return combined


def load_locations(csv_path: Path) -> pd.DataFrame:
    """
    Load node location metadata from pacific_cities.csv.
    """
    locations_df = pd.read_csv(csv_path)
    locations_df = locations_df.rename(
        columns={
            "Node ID": "node_id",
            "Node Name": "node_name",
            "Latitude": "lat",
            "Longitude": "lon",
            "Country": "country",
            "Region": "region",
        }
    )
    return locations_df


def load_all_selected_sites(output_dir: Path) -> pd.DataFrame:
    """
    Read every experiment-level selected_sites.csv file from the output directory
    and combine them into a single dataframe.
    """
    selected_site_files = sorted(output_dir.glob("E*/selected_sites.csv"))

    if not selected_site_files:
        raise FileNotFoundError(
            f"No selected_sites.csv files found under {output_dir}"
        )

    frames = []
    for selected_file in selected_site_files:
        df = pd.read_csv(selected_file)
        df["source_folder"] = selected_file.parent.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def prepare_plotting_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a smaller dataframe containing the core fields needed for
    experiment-level plotting and comparison.
    """
    missing_cols = [col for col in PLOTTING_COLUMNS if col not in summary_df.columns]
    if missing_cols:
        raise KeyError(
            "The combined summary is missing required plotting columns: "
            f"{missing_cols}"
        )

    plot_df = summary_df[PLOTTING_COLUMNS].copy()
    plot_df["experiment_num"] = (
        plot_df["experiment_code"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(int)
    )
    plot_df["display_label"] = (
        plot_df["experiment_code"].astype(str)
        + "\n"
        + plot_df["experiment_label"].astype(str)
    )

    plot_df = plot_df.sort_values(
        by=["experiment_num", "experiment_code"],
        kind="stable",
    ).reset_index(drop=True)

    return plot_df


def subset_by_codes(plot_df: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """
    Return a plotting subset ordered by the provided experiment codes.
    """
    subset = plot_df[plot_df["experiment_code"].isin(codes)].copy()
    subset["experiment_code"] = pd.Categorical(
        subset["experiment_code"], categories=codes, ordered=True
    )
    subset = subset.sort_values("experiment_code").reset_index(drop=True)
    return subset


def save_plotting_summary(plot_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the smaller plotting dataframe for reuse.
    """
    plot_df.to_csv(output_path, index=False)


def plot_fragility_comparison(plot_df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Plot total unmet demand and objective value across fragility settings.
    """
    fragility_df = subset_by_codes(plot_df, FRAGILITY_CODES)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        fragility_df["gamma"],
        fragility_df["total_unmet"] / 1e9,
        marker="o",
    )
    ax.set_title("Total Unmet Demand vs Network Fragility")
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Total Unmet Demand (Billions)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "fragility_total_unmet.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        fragility_df["gamma"],
        fragility_df["objective_value"] / 1e6,
        marker="o",
    )
    ax.set_title("Objective Value vs Network Fragility")
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Objective Value (Millions)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "fragility_objective_value.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_budget_capacity_comparison(plot_df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Compare baseline, capacity, and budget experiments on total unmet demand.
    """
    compare_df = subset_by_codes(plot_df, CAPACITY_BUDGET_CODES)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    unmet_heights = (compare_df["total_unmet"] / 1e9).tolist()
    unmet_labels = unmet_heights
    ax1.bar(compare_df["display_label"], unmet_heights)
    ax1.set_title("Capacity and Budget Comparison")
    ax1.set_ylabel("Total Unmet Demand (Billions)")
    add_bar_labels(ax1, unmet_heights, label_values=unmet_labels, decimals=1)

    site_heights = compare_df["num_selected_sites"].tolist()
    ax2.bar(compare_df["display_label"], site_heights)
    ax2.set_xlabel("Experiment")
    ax2.set_ylabel("Selected Sites")
    add_bar_labels(ax2, site_heights, label_values=site_heights, decimals=0)

    ax2.tick_params(axis="x", rotation=0)
    fig.subplots_adjust(hspace=0.25, bottom=0.12)
    fig.savefig(figures_dir / "capacity_budget_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_selected_sites(plot_df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Plot the number of selected sites for all experiments.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    site_heights = plot_df["num_selected_sites"].tolist()
    ax.bar(plot_df["display_label"], site_heights)
    ax.set_title("Number of Selected Prepositioned Sites by Experiment")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Selected Sites")
    ax.tick_params(axis="x", rotation=0)
    add_bar_labels(ax, site_heights, label_values=site_heights, decimals=0)
    fig.tight_layout()
    fig.savefig(figures_dir / "num_selected_sites_by_experiment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_selected_site_frequency_map(
    output_dir: Path,
    locations_csv_path: Path,
    figures_dir: Path,
) -> None:
    """
    Plot all candidate nodes and highlight how frequently each node is selected
    across experiments.
    """
    locations_df = load_locations(locations_csv_path)
    selected_sites_df = load_all_selected_sites(output_dir)

    selection_counts = (
        selected_sites_df.groupby(["node_id", "node_name"], as_index=False)
        .agg(selection_count=("experiment_code", "nunique"))
    )

    map_df = locations_df.merge(selection_counts, on=["node_id", "node_name"], how="left")
    map_df["selection_count"] = map_df["selection_count"].fillna(0)

    world = load_world_basemap(BASEMAP_DIR)

    candidate_gdf = gpd.GeoDataFrame(
        map_df,
        geometry=gpd.points_from_xy(map_df["lon"], map_df["lat"]),
        crs="EPSG:4326",
    )
    selected_gdf = candidate_gdf[candidate_gdf["selection_count"] > 0].copy()

    fig, ax = plt.subplots(figsize=(12, 8))

    world.plot(ax=ax, color="whitesmoke", edgecolor="lightgray", linewidth=0.5)

    candidate_gdf.plot(
        ax=ax,
        color="gray",
        markersize=10,
        alpha=0.6,
        label="Candidate nodes",
    )

    selected_plot = ax.scatter(
        selected_gdf["lon"],
        selected_gdf["lat"],
        s=80 + 25 * selected_gdf["selection_count"],
        c=selected_gdf["selection_count"],
        cmap="viridis",
        edgecolors="black",
        linewidths=0.5,
        label="Selected nodes",
        zorder=3,
    )

    for _, row in selected_gdf.iterrows():
        ax.annotate(
            f"{int(row['node_id'])}: {row['node_name']}",
            (row["lon"], row["lat"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            zorder=4,
        )

    cbar = fig.colorbar(selected_plot, ax=ax)
    cbar.set_label("Number of experiments selected")

    ax.set_xlim(95, 185)
    ax.set_ylim(-45, 50)
    ax.set_title("Selected Site Frequency Across Experiments")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(figures_dir / "selected_site_frequency_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_risk_comparison(plot_df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Compare risk-posture experiments on objective value and total unmet demand.
    """
    risk_df = subset_by_codes(plot_df, RISK_CODES)

    fig, ax = plt.subplots(figsize=(8, 5))
    objective_heights = (risk_df["objective_value"] / 1e6).tolist()
    ax.bar(risk_df["display_label"], objective_heights)
    ax.set_title("Objective Value by Risk Setting")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Objective Value (Millions)")
    ax.tick_params(axis="x", rotation=0)
    add_bar_labels(ax, objective_heights, label_values=objective_heights, decimals=1)
    fig.tight_layout()
    fig.savefig(figures_dir / "risk_objective_value.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    unmet_heights = (risk_df["total_unmet"] / 1e9).tolist()
    ax.bar(risk_df["display_label"], unmet_heights)
    ax.set_title("Total Unmet Demand by Risk Setting")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Total Unmet Demand (Billions)")
    ax.tick_params(axis="x", rotation=0)
    add_bar_labels(ax, unmet_heights, label_values=unmet_heights, decimals=1)
    fig.tight_layout()
    fig.savefig(figures_dir / "risk_total_unmet.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_risk_scenario_loss_distribution(output_dir: Path, figures_dir: Path) -> None:
    """
    Plot a boxplot of scenario loss distributions for the risk experiments.
    """
    risk_loss_df = load_selected_scenario_losses(output_dir, RISK_CODES)

    grouped = []
    labels = []
    for experiment_code in RISK_CODES:
        subset = risk_loss_df[risk_loss_df["experiment_code"] == experiment_code].copy()
        grouped.append((subset["loss"] / 1e6).tolist())
        label = subset["experiment_label"].iloc[0]
        labels.append(f"{experiment_code}\n{label}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(grouped, labels=labels, showfliers=False)
    ax.set_title("Scenario Loss Distribution by Risk Setting")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Scenario Loss (Millions)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "risk_scenario_loss_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def style_presentation_axis(ax) -> None:
    """
    Apply a clean presentation-oriented axis style for Keynote-ready figures.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=11)


def plot_presentation_fragility_objective(plot_df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Create a Keynote-ready figure showing objective value as a function of
    network fragility for E2, E1, and E3.
    """
    fragility_df = subset_by_codes(plot_df, FRAGILITY_CODES)
    values = fragility_df["objective_value"] / 1e6

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        fragility_df["gamma"],
        values,
        marker="o",
        linewidth=3,
        markersize=8,
        color=COLOR_RUST,
    )

    for gamma_value, objective_value in zip(fragility_df["gamma"], values):
        ax.text(
            gamma_value,
            objective_value + 0.35,
            f"{objective_value:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=COLOR_RUST,
        )

    ax.set_title("Objective Value vs. Network Fragility", fontsize=16, fontweight="bold")
    ax.set_xlabel("Network Fragility (gamma)", fontsize=13)
    ax.set_ylabel("Objective Value (Millions)", fontsize=13)
    ax.set_xticks(fragility_df["gamma"])
    ax.set_ylim(values.min() - 2, values.max() + 3)
    style_presentation_axis(ax)

    fig.tight_layout()
    fig.savefig(figures_dir / "presentation_fragility_objective_value.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "presentation_fragility_objective_value.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_presentation_fragility_transport_cost(plot_df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Create a Keynote-ready supporting bar chart showing transportation cost
    across fragility experiments.
    """
    fragility_df = subset_by_codes(plot_df, FRAGILITY_CODES)
    labels = ["Low\ngamma=0.2", "Baseline\ngamma=0.5", "High\ngamma=0.8"]
    heights = (fragility_df["total_transport_cost"] / 1e9).tolist()

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar(labels, heights, color=COLOR_RUST, alpha=0.9)
    add_bar_labels(ax, heights, label_values=heights, decimals=2)

    ax.set_title("Transportation Cost", fontsize=13, fontweight="bold")
    ax.set_ylabel("Billions", fontsize=11)
    style_presentation_axis(ax)

    fig.tight_layout()
    fig.savefig(figures_dir / "presentation_fragility_transport_cost.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "presentation_fragility_transport_cost.pdf", bbox_inches="tight")
    plt.close(fig)


def get_first_existing_column(df: pd.DataFrame, candidate_columns: list[str], column_description: str) -> str:
    """
    Return the first column name that exists in df from candidate_columns.
    Raises a clear error if none are present.
    """
    for column in candidate_columns:
        if column in df.columns:
            return column

    raise KeyError(
        f"Could not find a column for {column_description}. "
        f"Tried: {candidate_columns}. Available columns: {df.columns.tolist()}"
    )


def plot_presentation_risk_posture_slopegraph(plot_df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Create a Keynote-ready apples-to-apples risk posture visual.

    The bars show average scenario loss components in the same loss units:
    unmet demand penalty plus transportation cost. The CVaR objective value is
    overlaid as a point to show how the risk-aware objective compares with the
    average loss components.
    """
    risk_df = subset_by_codes(plot_df, RISK_CODES)
    risk_loss_df = load_selected_scenario_losses(OUTPUT_DIR, RISK_CODES)

    transport_col = get_first_existing_column(
        risk_loss_df,
        ["transportation_cost", "transport_cost", "total_transport_cost"],
        "transportation cost",
    )
    unmet_penalty_col = get_first_existing_column(
        risk_loss_df,
        ["unmet_demand_penalty", "unmet_penalty", "total_unmet_penalty"],
        "unmet demand penalty",
    )

    component_rows = []
    for row in risk_df.itertuples(index=False):
        scenario_subset = risk_loss_df[risk_loss_df["experiment_code"] == row.experiment_code].copy()
        component_rows.append(
            {
                "experiment_code": row.experiment_code,
                "experiment_label": row.experiment_label,
                "beta": row.beta,
                "avg_transport_cost_m": scenario_subset[transport_col].mean() / 1e6,
                "avg_unmet_penalty_m": scenario_subset[unmet_penalty_col].mean() / 1e6,
                "avg_loss_m": scenario_subset["loss"].mean() / 1e6,
                "cvar_objective_m": row.objective_value / 1e6,
            }
        )

    component_df = pd.DataFrame(component_rows)
    x_positions = list(range(len(component_df)))
    x_labels = [
        f"{row.experiment_code}\n{row.experiment_label}\nbeta={row.beta:.2f}"
        for row in component_df.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    unmet_bars = ax.bar(
        x_positions,
        component_df["avg_unmet_penalty_m"],
        color=COLOR_RUST,
        alpha=0.9,
        label="Avg. unmet demand penalty",
    )
    transport_bars = ax.bar(
        x_positions,
        component_df["avg_transport_cost_m"],
        bottom=component_df["avg_unmet_penalty_m"],
        color=COLOR_STEEL_BLUE,
        alpha=0.9,
        label="Avg. transportation cost",
    )

    cvar_points = ax.plot(
        x_positions,
        component_df["cvar_objective_m"],
        marker="D",
        linestyle="None",
        markersize=8,
        color=COLOR_GREEN,
        label="CVaR objective value",
        zorder=5,
    )

    for idx, (x_value, avg_loss) in enumerate(zip(x_positions, component_df["avg_loss_m"])):
        x_offset = -0.10 if idx == 0 else 0.0
        y_offset = 2.0 if idx == 0 else 3.0

        ax.text(
            x_value + x_offset,
            avg_loss + y_offset,
            f"Avg loss\n{avg_loss:.1f}M",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLOR_CHARCOAL,
        )

    for idx, (x_value, cvar_value) in enumerate(zip(x_positions, component_df["cvar_objective_m"])):
        x_offset = 0.16 if idx == 0 else 0.13
        y_offset = 3.5 if idx == 0 else 0.0

        ax.text(
            x_value + x_offset,
            cvar_value + y_offset,
            f"CVaR\n{cvar_value:.1f}M",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=COLOR_GREEN,
        )

    ax.set_title("Risk Posture Changes Loss Components", fontsize=16, fontweight="bold")
    ax.set_xlabel("Risk Posture", fontsize=13)
    ax.set_ylabel("Loss Components (Millions)", fontsize=13)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style_presentation_axis(ax)

    max_y = max(component_df["avg_loss_m"].max(), component_df["cvar_objective_m"].max())
    ax.set_ylim(0, max_y * 1.25)

    fig.tight_layout()
    fig.savefig(figures_dir / "presentation_risk_posture_slopegraph.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "presentation_risk_posture_slopegraph.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_slide10_presentation_visuals(plot_df: pd.DataFrame, output_dir: Path, figures_dir: Path) -> None:
    """
    Generate the presentation-ready products for Slide 10 in a separate
    directory so existing report figures remain unchanged.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_presentation_fragility_objective(plot_df, figures_dir)
    plot_presentation_fragility_transport_cost(plot_df, figures_dir)
    plot_presentation_risk_posture_slopegraph(plot_df, figures_dir)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    combined_summary = load_all_experiment_summaries(OUTPUT_DIR)
    combined_summary.to_csv(MASTER_SUMMARY_PATH, index=False)

    plot_df = prepare_plotting_summary(combined_summary)
    save_plotting_summary(plot_df, PLOTTING_SUMMARY_PATH)

    plot_budget_capacity_comparison(plot_df, FIGURES_DIR)
    plot_fragility_comparison(plot_df, FIGURES_DIR)
    plot_selected_sites(plot_df, FIGURES_DIR)
    plot_selected_site_frequency_map(OUTPUT_DIR, LOCATIONS_CSV_PATH, FIGURES_DIR)
    plot_risk_comparison(plot_df, FIGURES_DIR)
    plot_risk_scenario_loss_distribution(OUTPUT_DIR, FIGURES_DIR)
    plot_slide10_presentation_visuals(plot_df, OUTPUT_DIR, PRESENTATION_FIGURES_DIR)

    print(f"Combined summary saved to: {MASTER_SUMMARY_PATH}")
    print(f"Plotting summary saved to: {PLOTTING_SUMMARY_PATH}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Presentation figures saved to: {PRESENTATION_FIGURES_DIR}")

    print("\nPlotting summary preview:")
    print(plot_df)

    print("\nPlotting columns:")
    print(plot_df.columns.tolist())


if __name__ == "__main__":
    main()