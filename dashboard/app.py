import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.express as px
import time

from utils.api_client import TrafficAPIClient
from components.folium_map import render_folium_map

# Page configuration
st.set_page_config(
    page_title="Chicago Urban Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium feel
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stMetric {
        background-color: #1e2227;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize API Client
api = TrafficAPIClient()

# Sidebar
st.sidebar.title("🚦 Urban Traffic AI")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Live Traffic Map", "AI Traffic Assistant", "Congestion Forecasting", "Health & Monitoring"]
)

st.sidebar.markdown("---")
st.sidebar.info("Data source: Chicago Data Portal")

# Main Content
if menu == "Overview":
    st.title("🏙️ Chicago Traffic Intelligence")
    st.markdown("Monitor and predict traffic patterns with AI-driven insights.")
    
    # Fetch summary data
    summary = api.get_summary()
    metrics = summary.get("metrics", {})
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{metrics.get('total_records', 0):,}")
    with col2:
        st.metric("Avg Congestion", f"{metrics.get('avg_congestion', 0)*100:.1f}%")
    with col3:
        st.metric("Incidents", metrics.get('active_incidents', 0))
    with col4:
        st.metric("AI Knowledge Base", f"{metrics.get('indexed_documents', 0)} Docs")
        
    st.markdown("---")
    
    # Recent segments table
    st.subheader("📍 Recent Traffic Conditions")
    segments = summary.get("segments", [])
    if segments:
        df_segments = pd.DataFrame(segments)
        # Select key columns for display
        display_cols = ['street', 'direction', 'current_speed', 'expected_speed', 'congestion_level', 'timestamp']
        st.dataframe(df_segments[display_cols], use_container_width=True)
    else:
        st.warning("No traffic data available. Please run data ingestion.")

elif menu == "Live Traffic Map":
    st.title("🗺️ Live Traffic Map")
    st.write("Visualizing real-time segment speeds and incidents across Chicago.")
    
    summary = api.get_summary()
    segments = summary.get("segments", [])
    incidents = summary.get("incidents", [])
    
    if segments:
        render_folium_map(segments, incidents)
    else:
        st.error("No location data found in database. Please run ingestion from System Status.")

elif menu == "AI Traffic Assistant":
    st.title("🤖 AI Traffic Assistant (RAG)")
    st.markdown("Ask natural language questions about Chicago traffic, incidents, and roadwork.")
    
    # Chat message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for src in message["sources"]:
                        st.markdown(f"- {src}")

    # Chat input
    if prompt := st.chat_input("Ex: What happened on Desplaines St?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing traffic data..."):
                response = api.ask_rag(prompt)
                answer = response.get("answer", "I encountered an error.")
                sources = response.get("sources", [])
                
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for src in sources:
                            st.markdown(f"- {src}")
                            
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": sources
                })

elif menu == "Congestion Forecasting":
    st.title("📈 Congestion Forecasting (ML)")
    st.write("Predicting traffic spikes using the XGBoost engine.")
    
    col1, col2 = st.columns([1, 2])
    
    # Load all segments for dropdown
    with st.spinner("Fetching available streets..."):
        all_segments_data = api.get_all_segments()
        if all_segments_data:
            # Create labels: "Street (Direction)" -> segment_id
            segment_options = {
                f"{s['street']} ({s['direction']})": s['segment_id'] 
                for s in all_segments_data
            }
            sorted_labels = sorted(segment_options.keys())
        else:
            segment_options = {}
            sorted_labels = []

    with col1:
        st.subheader("Parameters")
        target_hours = st.slider("Forecast Horizon (Hours)", 1, 48, 24)
        
        if sorted_labels:
            selected_label = st.selectbox("Search Street", options=sorted_labels)
            selected_segment_id = segment_options[selected_label]
        else:
            st.error("Could not load street data.")
            selected_segment_id = None
            
        if st.button("Generate Forecast") and selected_segment_id:
            with st.spinner(f"Running ML model for {selected_label}..."):
                # Prediction for the single point
                target_time = datetime.now() + timedelta(hours=target_hours)
                prediction = api.get_prediction(selected_segment_id, target_time)
                
                # Prediction for the trend
                hours = list(range(1, target_hours + 1))
                scores = []
                for h in hours:
                    p = api.get_prediction(selected_segment_id, datetime.now() + timedelta(hours=h))
                    scores.append(p.get('congestion_level', 0))
                
                # Store in session state for persistence
                st.session_state.last_forecast = {
                    "label": selected_label,
                    "prediction": prediction,
                    "trend": pd.DataFrame({
                        "Hour": hours,
                        "Predicted Congestion": scores
                    })
                }
                st.success("Analysis complete!")

    with col2:
        st.subheader("Analysis Results")
        if "last_forecast" in st.session_state:
            res = st.session_state.last_forecast
            st.markdown(f"**Selected Street**: {res['label']}")
            
            # Show the point prediction
            pred = res['prediction']
            st.metric("Future Congestion", f"{pred.get('congestion_level', 0)*100:.1f}%", 
                      help=pred.get('description', ''))
            
            # Show the trend chart
            fig = px.line(res['trend'], x="Hour", y="Predicted Congestion", 
                          title=f"Congestion Trend: {res['label']}",
                          template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show JSON in expander
            with st.expander("Raw AI Output"):
                st.json(pred)
        else:
            st.info("Select a street and click 'Generate Forecast' to see AI predictions.")

elif menu == "Health & Monitoring":
    st.title("🛡️ System Health & Monitoring")
    st.markdown("Real-time observability and guardrail logs.")
    
    tab1, tab2, tab3 = st.tabs(["System Health", "ML Model Drift", "RAG Metrics"])
    
    with tab1:
        st.subheader("Core Services")
        health = api.get_health()
        
        col1, col2 = st.columns(2)
        with col1:
            status = health.get("status", "unknown")
            color = "green" if status == "healthy" else "red"
            st.markdown(f"**Overall Status**: :{color}[{status.upper()}]")
            st.write(f"**API Version**: {health.get('version', '1.0.0')}")
        
        with col2:
            db_health = health.get("components", {}).get("database", {})
            db_status = db_health.get("status", "unknown")
            db_color = "green" if db_status == "connected" else "red"
            st.markdown(f"**Database**: :{db_color}[{db_status.upper()}]")
            
        st.markdown("---")
        st.subheader("Administrative Actions")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Retrain ML Model"):
                with st.spinner("Retraining XGBoost engine..."):
                    res = api.trigger_training()
                    st.info(res.get("message", "Task triggered."))
        with col_b:
            if st.button("Trigger RAG Ingestion"):
                res = api.trigger_rag_ingestion()
                st.success("Background ingestion started.")

    with tab2:
        st.subheader("Statistical Drift Detection")
        st.write("Comparing current traffic distribution against historical reference data.")
        
        if st.button("Check for Model Drift"):
            with st.spinner("Running Kolmogorov-Smirnov test..."):
                drift = api.get_drift()
                if drift.get("status") == "success":
                    is_drifted = drift.get("is_drifted", False)
                    if is_drifted:
                        st.warning("🚨 Model Drift Detected! The data distribution has changed significantly.")
                    else:
                        st.success("✅ No significant drift detected. Model is stable.")
                    
                    st.metric("P-Value", drift.get("p_value", 0), delta="Drift Threshold < 0.05")
                    st.write(f"**K-S Statistic**: {drift.get('ks_statistic')}")
                    st.write(f"**Sample Sizes**: Reference ({drift.get('ref_count')}), Current ({drift.get('curr_count')})")
                else:
                    st.error(f"Drift check failed: {drift.get('message', 'Unknown error')}")

    with tab3:
        st.subheader("RAG Quality Metrics")
        st.write("Monitoring the accuracy and relevance of AI Assistant responses.")
        
        # In a real app, these would come from the database/prometheus
        # Here we mock for the view or pull from summary
        summary = api.get_summary()
        metrics = summary.get("metrics", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Faithfulness", "0.92", "+5%", help="How well AI answers match source data.")
        with col2:
            st.metric("Off-topic Rejections", "14", help="Queries blocked by relevance guardrail.")
            
        st.info("Prometheus metrics are active at `/metrics` for production scraping.")
