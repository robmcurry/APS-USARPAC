def generate_scenario_type_composite_maps(batch_summary_path, location_data, output_dir):
    import pandas as pd
    import folium
    import ast
    import os
    from collections import defaultdict
    import numpy as np

    df = pd.read_csv(batch_summary_path)
    scenario_types = df["scenario_type"].unique()

    for stype in scenario_types:
        filtered = df[df["scenario_type"] == stype]
        node_scores = defaultdict(list)

        for _, row in filtered.iterrows():
            aps = ast.literal_eval(row["aps_locations"])
            for node in aps:
                node_scores[node].append(row["composite_score"])

        avg_scores = {node: sum(scores) / len(scores) for node, scores in node_scores.items()}
        all_scores = list(avg_scores.values())

        if not all_scores:
            continue

        p33 = np.percentile(all_scores, 33)
        p66 = np.percentile(all_scores, 66)

        avg_lat = sum(loc["coords"][0] for loc in location_data.values()) / len(location_data)
        avg_lon = sum(loc["coords"][1] for loc in location_data.values()) / len(location_data)
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=3, tiles="cartodbpositron")

        for node_id in avg_scores:
            str_node = str(node_id)
            loc = location_data.get(str_node)
            score = avg_scores.get(node_id)
            if loc and score is not None:
                if score <= p33:
                    color = "green"
                elif score <= p66:
                    color = "orange"
                else:
                    color = "red"

                folium.CircleMarker(
                    location=loc["coords"],
                    radius=8,
                    color=color,
                    fill=True,
                    fill_opacity=0.6,
                    popup=f"{loc['name']}<br>Score: {score:.2f}<br>Type: {stype}"
                ).add_to(m)

        legend_html = f"""
         <div style='position: fixed; bottom: 30px; left: 30px; width: 180px; height: 120px; 
                     background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                     padding: 10px;'>
         <b>Composite Score (Type {stype})</b><br>
         <i style='color:green;'>●</i> Low (≤ {p33:.2f})<br>
         <i style='color:orange;'>●</i> Moderate ({p33:.2f} – {p66:.2f})<br>
         <i style='color:red;'>●</i> High (&gt; {p66:.2f})
         </div>
         """
        m.get_root().html.add_child(folium.Element(legend_html))

        filename = f"composite_score_type_{stype}.html"
        os.makedirs(output_dir, exist_ok=True)
        m.save(os.path.join(output_dir, filename))
        print(f"Composite score map for scenario type {stype} saved as {filename}")
import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
from collections import Counter

def generate_aps_frequency_chart(batch_summary_path, output_dir):
    df = pd.read_csv(batch_summary_path)

    all_aps = []
    for aps_str in df["aps_locations"]:
        aps = ast.literal_eval(aps_str)
        all_aps.extend(aps)

    aps_counter = Counter(all_aps)
    sorted_aps = sorted(aps_counter.items(), key=lambda x: x[1], reverse=True)

    labels, values = zip(*sorted_aps)
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color='seagreen')
    plt.xlabel("Node ID")
    plt.ylabel("Frequency Selected as APS")
    plt.title("APS Site Selection Frequency Across Scenarios")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    os.makedirs(output_dir, exist_ok=True)
    chart_path = os.path.join(output_dir, "aps_frequency_chart.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"APS frequency chart saved to {chart_path}")

def generate_objective_chart(batch_summary_path, output_dir):
    df = pd.read_csv(batch_summary_path)

    plt.figure(figsize=(12, 6))
    plt.bar(df["scenario_id"], df["objective"], color="steelblue")
    plt.xlabel("Scenario ID")
    plt.ylabel("Objective Value")
    plt.title("Objective Value by Scenario")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    os.makedirs(output_dir, exist_ok=True)
    chart_path = os.path.join(output_dir, "objective_values_chart.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Objective value chart saved to {chart_path}")

def generate_severity_vs_objective_chart(batch_summary_path, output_dir):
    df = pd.read_csv(batch_summary_path)

    plt.figure(figsize=(8, 6))
    plt.scatter(df['severity'], df['objective'], alpha=0.7)
    plt.title('Severity vs. Objective Value')
    plt.xlabel('Severity')
    plt.ylabel('Objective Value')
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'severity_vs_objective.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Severity vs. objective chart saved to {output_path}")


# 3D chart: severity vs objective vs epicenter_neighbors
def generate_3d_severity_objective_connectivity_chart(batch_summary_path, output_dir):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from mpl_toolkits.mplot3d import Axes3D

    df = pd.read_csv(batch_summary_path)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['severity'], df['objective'], df['epicenter_neighbors'],
               c=df['epicenter_neighbors'], cmap='viridis', alpha=0.8)

    ax.set_xlabel('Severity')
    ax.set_ylabel('Objective Value')
    ax.set_zlabel('Epicenter Connectivity (# neighbors)')
    plt.title('Severity vs Objective vs Epicenter Connectivity')

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '3d_severity_objective_connectivity.png')
    plt.savefig(output_path)
    plt.close()
    print(f"3D plot saved to {output_path}")


# Interactive 3D chart using Plotly
def generate_interactive_3d_severity_objective_connectivity_chart(batch_summary_path, output_dir):
    import pandas as pd
    import plotly.graph_objects as go
    import os

    df = pd.read_csv(batch_summary_path)

    fig = go.Figure(data=[go.Scatter3d(
        x=df['severity'],
        y=df['objective'],
        z=df['epicenter_neighbors'],
        mode='markers',
        marker=dict(
            size=6,
            color=df['objective'],  # Use objective value for color
            colorscale='RdBu',
            colorbar=dict(title='Objective'),
            opacity=0.8
        ),
        text=[f"Scenario {sid}" for sid in df['scenario_id']]
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='Severity',
            yaxis_title='Objective Value',
            zaxis_title='Epicenter Connectivity (# neighbors)'
        ),
        title='Interactive 3D: Severity vs Objective vs Connectivity',
        margin=dict(l=0, r=0, b=0, t=30)
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'interactive_3d_severity_objective_connectivity.html')
    fig.write_html(output_path)
    print(f"Interactive 3D plot saved to {output_path}")
def generate_aps_selection_map(batch_summary_path, location_data, output_path):
    import pandas as pd
    import folium
    import ast
    from collections import Counter
    import os

    df = pd.read_csv(batch_summary_path)

    # Count frequency of each APS location
    all_aps = []
    for aps_str in df["aps_locations"]:
        aps = ast.literal_eval(aps_str)
        all_aps.extend(aps)

    aps_counter = Counter(all_aps)

    # Create folium map centered on average location
    avg_lat = sum(loc["coords"][0] for loc in location_data.values()) / len(location_data)
    avg_lon = sum(loc["coords"][1] for loc in location_data.values()) / len(location_data)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=3, tiles="cartodbpositron")

    # Plot APS frequencies
    for node_id, loc in location_data.items():
        count = aps_counter.get(node_id, 0)
        folium.CircleMarker(
            location=loc["coords"],
            radius=5 + count,
            color="darkred",
            fill=True,
            fill_opacity=0.6,
            popup=f"{loc['name']}<br>Selected {count} times"
        ).add_to(m)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"APS selection map saved to {output_path}")

def generate_epicenter_objective_map(batch_summary_path, location_data, output_path):
    import pandas as pd
    import folium
    import os

    df = pd.read_csv(batch_summary_path)

    avg_lat = sum(loc["coords"][0] for loc in location_data.values()) / len(location_data)
    avg_lon = sum(loc["coords"][1] for loc in location_data.values()) / len(location_data)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=3, tiles="cartodbpositron")

    for _, row in df.iterrows():
        epicenter_id = row["epicenter"]
        obj = row["objective"]
        loc = location_data.get(epicenter_id)
        if loc:
            folium.CircleMarker(
                location=loc["coords"],
                radius=5 + min(obj / 1e9, 15),  # scale radius and cap it
                color="crimson",
                fill=True,
                fill_opacity=0.6,
                popup=f"{loc['name']}<br>Obj: {obj:,.0f}"
            ).add_to(m)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"Epicenter objective map saved to {output_path}")


# Generate composite score map per node
def generate_composite_score_map(batch_summary_path, location_data, output_path):
    import pandas as pd
    import folium
    import ast
    from collections import defaultdict
    import os

    df = pd.read_csv(batch_summary_path)

    # Initialize accumulation dict
    node_scores = defaultdict(list)

    # Sum composite scores per node across scenarios
    for _, row in df.iterrows():
        aps = ast.literal_eval(row["aps_locations"])
        for node in aps:
            node_scores[node].append(row["composite_score"])

    # Calculate average composite score per node
    avg_scores = {node: sum(scores)/len(scores) for node, scores in node_scores.items()}

    # Calculate percentiles for dynamic coloring
    import numpy as np
    all_avg_scores = list(avg_scores.values())
    p33 = np.percentile(all_avg_scores, 33)
    p66 = np.percentile(all_avg_scores, 66)

    # Create map
    avg_lat = sum(loc["coords"][0] for loc in location_data.values()) / len(location_data)
    avg_lon = sum(loc["coords"][1] for loc in location_data.values()) / len(location_data)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=3, tiles="cartodbpositron")

    for node_id, loc in location_data.items():
        score = avg_scores.get(node_id)
        if score is not None:
            # Determine color dynamically based on score distribution
            if score <= p33:
                color = "green"
            elif score <= p66:
                color = "orange"
            else:
                color = "red"

            folium.CircleMarker(
                location=loc["coords"],
                radius=8,
                color=color,
                fill=True,
                fill_opacity=0.6,
                popup=f"{loc['name']}<br>Avg Composite Score: {score:.2f}"
            ).add_to(m)

    legend_html = f"""
     <div style='position: fixed; 
                 bottom: 30px; left: 30px; width: 180px; height: 120px; 
                 background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                 padding: 10px;'>
     <b>Composite Score</b><br>
     <i style='color:green;'>●</i> Low (≤ {p33:.2f})<br>
     <i style='color:orange;'>●</i> Moderate ({p33:.2f} – {p66:.2f})<br>
     <i style='color:red;'>●</i> High (&gt; {p66:.2f})
     </div>
     """
    m.get_root().html.add_child(folium.Element(legend_html))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"Composite strategy score map saved to {output_path}")
def generate_disaster_type_impact_charts(batch_summary_path, output_dir):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    df = pd.read_csv(batch_summary_path)

    # Group by scenario type
    grouped = df.groupby("scenario_type").agg({
        "objective": "mean",
        "composite_score": "mean",
        "severity": "mean"
    }).reset_index()

    # Plot average objective by scenario type
    plt.figure(figsize=(10, 6))
    plt.bar(grouped["scenario_type"], grouped["objective"], color='skyblue')
    plt.xlabel("Scenario Type")
    plt.ylabel("Average Objective Value")
    plt.title("Avg Objective by Scenario Type")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "avg_objective_by_scenario_type.png"))
    plt.close()

    # Plot average severity by scenario type
    plt.figure(figsize=(10, 6))
    plt.bar(grouped["scenario_type"], grouped["severity"], color='salmon')
    plt.xlabel("Scenario Type")
    plt.ylabel("Average Severity")
    plt.title("Avg Severity by Scenario Type")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "avg_severity_by_scenario_type.png"))
    plt.close()

    # Plot average composite score by scenario type
    plt.figure(figsize=(10, 6))
    plt.bar(grouped["scenario_type"], grouped["composite_score"], color='limegreen')
    plt.xlabel("Scenario Type")
    plt.ylabel("Avg Composite Score")
    plt.title("Avg Composite Score by Scenario Type")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "avg_composite_score_by_scenario_type.png"))
    plt.close()

    print("Disaster type impact charts saved to output directory.")
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate summary visualizations from batch results.")
    parser.add_argument("--summary", type=str, default="disaster_logistics_model/output/batch_summary.csv", help="Path to batch summary CSV")
    parser.add_argument("--locations", type=str, default="disaster_logistics_model/data/pacific_cities.csv", help="Path to location data CSV")
    parser.add_argument("--output", type=str, default="disaster_logistics_model/output/", help="Directory to save visualizations")

    args = parser.parse_args()

    import pandas as pd
    location_df = pd.read_csv(args.locations)
    location_data = {
        str(row["Node ID"]): {
            "coords": [row["Latitude"], row["Longitude"]],
            "name": row["Node Name"]
        }
        for _, row in location_df.iterrows()
    }

    generate_aps_frequency_chart(args.summary, args.output)
    generate_objective_chart(args.summary, args.output)
    generate_severity_vs_objective_chart(args.summary, args.output)
    generate_3d_severity_objective_connectivity_chart(args.summary, args.output)
    generate_interactive_3d_severity_objective_connectivity_chart(args.summary, args.output)
    generate_aps_selection_map(args.summary, location_data, os.path.join(args.output, "aps_selection_map.html"))
    generate_epicenter_objective_map(args.summary, location_data, os.path.join(args.output, "epicenter_objective_map.html"))
    generate_composite_score_map(args.summary, location_data, os.path.join(args.output, "composite_score_map.html"))
    generate_disaster_type_impact_charts(args.summary, args.output)
    # generate_scenario_type_composite_maps(args.summary, location_data, args.output)
    generate_combined_scenario_type_map(args.summary, location_data, os.path.join(args.output, "combined_scenario_type_map.html"))


# New function: generate_combined_scenario_type_map
def generate_combined_scenario_type_map(batch_summary_path, location_data, output_path):
    import pandas as pd
    import folium
    import ast
    from collections import defaultdict
    import os

    df = pd.read_csv(batch_summary_path)

    # Assign a color per scenario type
    type_colors = {0: "blue", 1: "purple", 2: "orange"}

    node_colors = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        aps = ast.literal_eval(row["aps_locations"])
        stype = row["scenario_type"]
        for node in aps:
            node_colors[node][stype].append(row["composite_score"])

    avg_lat = sum(loc["coords"][0] for loc in location_data.values()) / len(location_data)
    avg_lon = sum(loc["coords"][1] for loc in location_data.values()) / len(location_data)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=3, tiles="cartodbpositron")

    for node_id, stype_dict in node_colors.items():
        str_node = str(node_id)
        loc = location_data.get(str_node)
        if loc:
            for stype, scores in stype_dict.items():
                avg_score = sum(scores) / len(scores)
                color = type_colors.get(stype, "gray")
                folium.CircleMarker(
                    location=loc["coords"],
                    radius=5 + avg_score / 2e9,
                    color=color,
                    fill=True,
                    fill_opacity=0.5,
                    popup=f"{loc['name']}<br>Scenario Type: {stype}<br>Avg Score: {avg_score:.2f}"
                ).add_to(m)

    legend_html = """
     <div style='position: fixed; bottom: 30px; left: 30px; width: 180px; height: 120px;
                 background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                 padding: 10px;'>
     <b>Scenario Type Color</b><br>
     <i style='color:blue;'>●</i> Type 0<br>
     <i style='color:purple;'>●</i> Type 1<br>
     <i style='color:orange;'>●</i> Type 2
     </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"Combined scenario type map saved to {output_path}")