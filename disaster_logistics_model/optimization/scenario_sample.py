import pandas as pd
import folium
import random
from folium.plugins import MarkerCluster

# Load data
batch_df = pd.read_csv("../output/batch_summary.csv")
cities_df = pd.read_csv("../data/pacific_cities.csv")
# Ensure 'epicenter' and 'Node ID' are strings for matching
batch_df["epicenter"] = batch_df["epicenter"].astype(str)
cities_df["Node ID"] = cities_df["Node ID"].astype(str)

# Merge to get city location and names
merged_df = batch_df.merge(cities_df, left_on="epicenter", right_on="Node ID", how="left")

# Drop rows with missing lat/long
merged_df = merged_df.dropna(subset=["Latitude", "Longitude"])

# Group by scenario type and select one example with high objective value from each
selected = (
    merged_df.groupby("scenario_type", group_keys=False)
    .apply(lambda g: g.loc[g["objective"].idxmax()])
    .reset_index(drop=True)
)

# Filter to ensure diverse locations (spatial distance check not implemented here)
selected = selected.sort_values("scenario_type")

# Create folium map centered on average location
map_center = [selected["Latitude"].mean(), selected["Longitude"].mean()]
fmap = folium.Map(location=map_center, zoom_start=3, tiles="cartodbpositron")
marker_cluster = MarkerCluster().add_to(fmap)

# Add markers for each selected scenario
for _, row in selected.iterrows():
    popup_text = (
        f"<b>City:</b> {row['Node Name']}<br>"
        f"<b>Scenario ID:</b> {row['scenario_id']}<br>"
        f"<b>Scenario Type:</b> {row['scenario_type']}<br>"
        f"<b>Objective Value:</b> {row['objective']:.2f}<br>"
        f"<b>Impact Radius (km):</b> {row.get('impact_radius_km', 'N/A')}<br>"
        f"<b>Infrastructure Loss Rate:</b> {row.get('infrastructure_loss_rate', 'N/A'):.2f}<br>"
        f"<b>Population:</b> {row.get('Population', 'N/A')}"
    )
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=10,
        color="blue",
        fill=True,
        fill_opacity=0.6,
        popup=folium.Popup(popup_text, max_width=300),
    ).add_to(marker_cluster)

# Save map
fmap.save("../output/maps/scenario_sample_map.html")