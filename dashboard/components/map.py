import folium
from streamlit_folium import st_folium
import pandas as pd

def get_congestion_color(level):
    """Return color string based on congestion level"""
    if level < 0.3:
        return "green"
    if level < 0.6:
        return "orange"
    if level < 0.8:
        return "red"
    return "darkred"

def render_traffic_map(segments_data, incidents_data=None):
    """
    Render an interactive Folium map with traffic segments and incident markers.
    """
    # Center of Chicago
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=11, tiles="cartodbpositron")

    # Add Traffic Segments
    if segments_data:
        for seg in segments_data:
            color = get_congestion_color(seg.get('congestion_level', 0))
            popup_text = f"<b>{seg['street']}</b> ({seg['direction']})<br/>Speed: {seg['current_speed']} mph<br/>Congestion: {seg['congestion_level']}"
            
            folium.PolyLine(
                locations=[[seg['start_lat'], seg['start_lon']], [seg['end_lat'], seg['end_lon']]],
                color=color,
                weight=5,
                opacity=0.8,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)

    # Add Incident Markers
    if incidents_data:
        for inc in incidents_data:
            popup_text = f"<b>Crash: {inc['street_name']}</b><br/>Damage: {inc['damage_category']}<br/>Date: {inc['crash_date']}"
            
            folium.CircleMarker(
                location=[inc['latitude'], inc['longitude']],
                radius=6,
                color="darkred",
                fill=True,
                fill_color="red",
                fill_opacity=0.9,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)

    # Render in Streamlit
    st_folium(m, width=None, height=600, use_container_width=True)
