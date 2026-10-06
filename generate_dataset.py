import pandas as pd
import numpy as np

np.random.seed(42)

N = 5000


# =========================================================
# CITY DEFINITIONS
# =========================================================

cities = {
    "Chennai": {
        "lat": 13.0827,
        "lon": 80.2707
    },
    "Bengaluru": {
        "lat": 12.9716,
        "lon": 77.5946
    },
    "Coimbatore": {
        "lat": 11.0168,
        "lon": 76.9558
    },
    "Vellore": {
        "lat": 12.9165,
        "lon": 79.1325
    },
    "Madurai": {
        "lat": 9.9252,
        "lon": 78.1198
    },
    "Tiruchirappalli": {
        "lat": 10.7905,
        "lon": 78.7047
    },
    "Salem": {
        "lat": 11.6643,
        "lon": 78.1460
    }
}


city_names = list(cities.keys())


# =========================================================
# HOTSPOT CENTERS
# =========================================================

# Each city has several accident-prone zones.
# This allows DBSCAN and Folium to find meaningful clusters.

hotspots = {
    "Chennai": [
        (13.0827, 80.2707),
        (13.0569, 80.2425),
        (13.0475, 80.2090)
    ],

    "Bengaluru": [
        (12.9716, 77.5946),
        (12.9352, 77.6245),
        (13.0358, 77.5970)
    ],

    "Coimbatore": [
        (11.0168, 76.9558),
        (11.0250, 76.9700)
    ],

    "Vellore": [
        (12.9165, 79.1325),
        (12.9250, 79.1350)
    ],

    "Madurai": [
        (9.9252, 78.1198),
        (9.9400, 78.1300)
    ],

    "Tiruchirappalli": [
        (10.7905, 78.7047),
        (10.8050, 78.6900)
    ],

    "Salem": [
        (11.6643, 78.1460),
        (11.6750, 78.1550)
    ]
}


# =========================================================
# GENERATE CITY
# =========================================================

city_data = np.random.choice(
    city_names,
    N,
    p=[
        0.25,  # Chennai
        0.22,  # Bengaluru
        0.15,  # Coimbatore
        0.10,  # Vellore
        0.10,  # Madurai
        0.09,  # Trichy
        0.09   # Salem
    ]
)


# =========================================================
# GENERATE DATE AND TIME
# =========================================================

dates = pd.date_range(
    start="2023-01-01",
    end="2025-12-31",
    freq="h"
)

random_dates = np.random.choice(
    dates,
    N
)

random_dates = pd.to_datetime(random_dates)


date_values = random_dates.date
time_values = random_dates.strftime("%H:%M:%S")
day_values = random_dates.day_name()
hour_values = random_dates.hour


# =========================================================
# GENERATE LOCATION WITH HOTSPOTS
# =========================================================

latitudes = []
longitudes = []

for city in city_data:

    city_hotspots = hotspots[city]

    # Most accidents occur near known hotspots
    if np.random.random() < 0.75:

        hotspot = city_hotspots[
            np.random.randint(len(city_hotspots))
        ]

        lat = hotspot[0] + np.random.normal(0, 0.008)
        lon = hotspot[1] + np.random.normal(0, 0.008)

    else:

        lat = cities[city]["lat"] + np.random.normal(0, 0.035)
        lon = cities[city]["lon"] + np.random.normal(0, 0.035)

    latitudes.append(lat)
    longitudes.append(lon)


# =========================================================
# WEATHER
# =========================================================

weather = np.random.choice(
    [
        "Clear",
        "Rain",
        "Fog",
        "Cloudy"
    ],
    N,
    p=[
        0.52,
        0.20,
        0.08,
        0.20
    ]
)


# =========================================================
# ROAD CONDITION
# =========================================================

road_condition = []

for w in weather:

    if w == "Rain":

        road_condition.append(
            np.random.choice(
                ["Wet", "Slippery", "Dry"],
                p=[0.65, 0.25, 0.10]
            )
        )

    elif w == "Fog":

        road_condition.append(
            np.random.choice(
                ["Wet", "Dry"],
                p=[0.35, 0.65]
            )
        )

    else:

        road_condition.append(
            np.random.choice(
                ["Dry", "Wet"],
                p=[0.88, 0.12]
            )
        )


# =========================================================
# LIGHT CONDITION
# =========================================================

light_condition = np.random.choice(
    [
        "Daylight",
        "Night",
        "Dusk"
    ],
    N,
    p=[
        0.55,
        0.35,
        0.10
    ]
)


# =========================================================
# SPEED LIMIT
# =========================================================

speed_limit = np.random.choice(
    [
        30,
        40,
        50,
        60,
        80,
        100
    ],
    N,
    p=[
        0.05,
        0.15,
        0.25,
        0.30,
        0.20,
        0.05
    ]
)


# =========================================================
# TRAFFIC DENSITY
# =========================================================

traffic_density = np.random.choice(
    [
        "Low",
        "Medium",
        "High"
    ],
    N,
    p=[
        0.25,
        0.45,
        0.30
    ]
)


# =========================================================
# VEHICLE COUNT
# =========================================================

vehicle_count = []

for density in traffic_density:

    if density == "Low":

        vehicle_count.append(
            np.random.randint(2, 8)
        )

    elif density == "Medium":

        vehicle_count.append(
            np.random.randint(6, 18)
        )

    else:

        vehicle_count.append(
            np.random.randint(12, 35)
        )


# =========================================================
# ACCIDENT SEVERITY
# =========================================================

severity = []


for i in range(N):

    risk = 0

    # Weather
    if weather[i] == "Rain":
        risk += 2

    elif weather[i] == "Fog":
        risk += 3

    elif weather[i] == "Cloudy":
        risk += 1


    # Road
    if road_condition[i] == "Wet":
        risk += 2

    elif road_condition[i] == "Slippery":
        risk += 3


    # Lighting
    if light_condition[i] == "Night":
        risk += 2

    elif light_condition[i] == "Dusk":
        risk += 1


    # Speed
    if speed_limit[i] >= 80:
        risk += 3

    elif speed_limit[i] >= 60:
        risk += 2

    elif speed_limit[i] >= 50:
        risk += 1


    # Traffic
    if traffic_density[i] == "High":
        risk += 2

    elif traffic_density[i] == "Medium":
        risk += 1


    # Vehicles
    if vehicle_count[i] >= 25:
        risk += 3

    elif vehicle_count[i] >= 15:
        risk += 2

    elif vehicle_count[i] >= 10:
        risk += 1


    # Rush hour effect
    if hour_values[i] in [
        7, 8, 9,
        17, 18, 19, 20
    ]:
        risk += 2


    # Weekend effect
    if day_values[i] in [
        "Saturday",
        "Sunday"
    ]:
        risk += 1


    # Random variation
    risk += np.random.randint(0, 3)


    # Final classification

    if risk >= 10:

        severity.append("High")

    elif risk >= 6:

        severity.append("Medium")

    else:

        severity.append("Low")


# =========================================================
# CREATE DATAFRAME
# =========================================================

df = pd.DataFrame({

    "Accident_ID": range(1, N + 1),

    "Date": date_values,

    "Time": time_values,

    "Day_of_Week": day_values,

    "Hour": hour_values,

    "City": city_data,

    "Latitude": np.round(
        latitudes,
        6
    ),

    "Longitude": np.round(
        longitudes,
        6
    ),

    "Weather": weather,

    "Road_Condition": road_condition,

    "Light_Condition": light_condition,

    "Speed_Limit": speed_limit,

    "Traffic_Density": traffic_density,

    "Vehicle_Count": vehicle_count,

    "Accident_Severity": severity
})


# =========================================================
# SAVE DATA
# =========================================================

df.to_csv(
    "traffic_accidents.csv",
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("=" * 50)
print("TRAFFIC ACCIDENT DATASET CREATED")
print("=" * 50)

print(
    f"Total records: {len(df)}"
)

print(
    f"Number of features: {len(df.columns)}"
)

print("\nSeverity Distribution:")

print(
    df["Accident_Severity"].value_counts()
)

print("\nCity Distribution:")

print(
    df["City"].value_counts()
)

print("\nDataset Preview:")

print(
    df.head()
)

print("\nSaved as:")
print("traffic_accidents.csv")