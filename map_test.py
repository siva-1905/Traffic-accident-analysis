import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide")

st.title("Folium Map Test")

m = folium.Map(
    location=[13.0827, 80.2707],
    zoom_start=12,
    tiles="OpenStreetMap"
)

folium.Marker(
    location=[13.0827, 80.2707],
    popup="Chennai",
    tooltip="Chennai"
).add_to(m)

st_folium(
    m,
    width=1200,
    height=600
)