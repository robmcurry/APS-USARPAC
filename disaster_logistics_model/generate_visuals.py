# Import necessary libraries and visualization functions
import os
from disaster_logistics_model.visualization.generate_summary_charts import (
    generate_aps_frequency_chart,
    generate_objective_chart,
    generate_aps_selection_map
)

# Set paths for input data and output directory
batch_summary_path = "output/batch_summary.csv"
cities_data_path = "data/pacific_cities.csv"
output_dir = "output"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Step 1: Generate APS Frequency Chart
print("Generating APS Frequency Chart...")
generate_aps_frequency_chart(batch_summary_path, output_dir)
print("APS Frequency Chart saved!")

# Step 2: Generate Objective Value Chart
print("Generating Objective Value Chart...")
generate_objective_chart(batch_summary_path, output_dir)
print("Objective Value Chart saved!")

# Step 3: Generate APS Selection Map
# print("Generating APS Selection Map...")
# generate_aps_selection_map(batch_summary_path, cities_data_path, output_dir)
# print("APS Selection Map saved!")

print("\nAll visuals generated successfully.")