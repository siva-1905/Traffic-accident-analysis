import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# Load dataset
df = pd.read_csv("traffic_accidents.csv")

# Coordinates
coords = np.radians(
    df[["Latitude", "Longitude"]].values
)

# DBSCAN
# eps = approximately 1.6 km
dbscan = DBSCAN(
    eps=0.00025,
    min_samples=15,
    metric="haversine",
    algorithm="ball_tree"
)

df["Hotspot_Cluster"] = dbscan.fit_predict(coords)

# Number of clusters
clusters = sorted(
    c for c in df["Hotspot_Cluster"].unique()
    if c != -1
)

print(f"Number of hotspots detected: {len(clusters)}")

# --------------------------------------------------
# Create hotspot summary
# --------------------------------------------------

hotspots = []

for cluster_id in clusters:

    cluster_data = df[df["Hotspot_Cluster"] == cluster_id]

    total_accidents = len(cluster_data)

    high_severity = (
        cluster_data["Accident_Severity"] == "High"
    ).sum()

    medium_severity = (
        cluster_data["Accident_Severity"] == "Medium"
    ).sum()

    low_severity = (
        cluster_data["Accident_Severity"] == "Low"
    ).sum()

    latitude = cluster_data["Latitude"].mean()
    longitude = cluster_data["Longitude"].mean()

    # Risk score
    risk_score = (
        total_accidents * 0.5
        + high_severity * 0.35
        + medium_severity * 0.15
    )

    hotspots.append({
        "Hotspot_ID": cluster_id,
        "Latitude": latitude,
        "Longitude": longitude,
        "Total_Accidents": total_accidents,
        "High_Severity": high_severity,
        "Medium_Severity": medium_severity,
        "Low_Severity": low_severity,
        "Risk_Score": risk_score
    })

hotspot_df = pd.DataFrame(hotspots)

# Sort by risk
hotspot_df = hotspot_df.sort_values(
    "Risk_Score",
    ascending=False
).reset_index(drop=True)

# Assign ranking
hotspot_df["Rank"] = hotspot_df.index + 1

# Save
hotspot_df.to_csv(
    "models/hotspot_summary.csv",
    index=False
)

# Save dataset with cluster labels
df.to_csv(
    "traffic_accidents.csv",
    index=False
)

print("\nTop Accident Hotspots")
print("=" * 80)

print(
    hotspot_df.head(15).to_string(index=False)
)

print("\nFiles created:")
print("1. models/hotspot_summary.csv")
print("2. traffic_accidents.csv")