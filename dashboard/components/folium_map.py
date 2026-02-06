import folium
from streamlit_folium import folium_static
import streamlit as st
import pandas as pd

def get_congestion_color(level):
    """Return color string based on congestion level"""
    if level < 0.3:
        return 'green'
    if level < 0.6:
        return 'orange'
    if level < 0.8:
        return 'darkred'
    return 'red'

def render_folium_map(segments_data, incidents_data=None):
    """
    Render an interactive Folium map with traffic segments and incident markers.
    """
    if not segments_data:
        st.warning("No segment data available for the map.")
        return

    # Center map on Chicago
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=11, tiles="cartodbpositron")

    # Add Traffic Segments
    for segment in segments_data:
        try:
            start_coord = [segment['start_lat'], segment['start_lon']]
            end_coord = [segment['end_lat'], segment['end_lon']]
            
            # Draw line
            color = get_congestion_color(segment['congestion_level'])
            
            folium.PolyLine(
                locations=[start_coord, end_coord],
                color=color,
                weight=5,
                opacity=0.8,
                tooltip=f"<b>{segment['street']}</b> ({segment['direction']})<br/>Speed: {segment['current_speed']} mph<br/>Congestion: {segment['congestion_level']:.2f}"
            ).add_to(m)
        except Exception as e:
            continue

    # Add Incident Markers
    if incidents_data:
        for incident in incidents_data:
            try:
                folium.Marker(
                    location=[incident['latitude'], incident['longitude']],
                    popup=f"<b>CRASH</b><br/>{incident['street_name']}<br/>{incident['damage']}",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
            except Exception as e:
                continue

    # Display map in Streamlit
    folium_static(m, width=1200, height=600)
