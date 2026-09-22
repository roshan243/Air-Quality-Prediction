import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(page_title="Air Quality Prediction Dashboard", layout="wide")

st.title("🌬️ Air Quality Index (AQI) Predictor")
st.markdown("Enter the environmental pollutant parameters below to predict the current Air Quality Index.")

# ============================================================
# Load Model & Feature Importance
# ============================================================
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler']

@st.cache_resource
def load_feature_importance():
    try:
        with open('feature_importance.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

try:
    model, scaler = load_model()
    feature_importance = load_feature_importance()

    # ============================================================
    # Sidebar — Input Parameters
    # ============================================================
    st.sidebar.header("📊 Input Parameters")

    with st.sidebar:
        pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 300.0, 75.0)
        pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, 120.0)
        no2 = st.slider("NO2 (µg/m³)", 0.0, 200.0, 40.0)
        so2 = st.slider("SO2 (µg/m³)", 0.0, 100.0, 20.0)
        co = st.slider("CO (mg/m³)", 0.0, 10.0, 1.5)
        o3 = st.slider("O3 (µg/m³)", 0.0, 200.0, 35.0)
        temp = st.slider("Temperature (°C)", -10.0, 50.0, 30.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)

        predict_btn = st.button("🚀 Predict AQI", type="primary", use_container_width=True)

    # Compute prediction for real-time preview
    input_data = pd.DataFrame(
        [[pm25, pm10, no2, so2, co, o3, temp, humidity]],
        columns=['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temp_C', 'Humidity']
    )
    scaled_input = scaler.transform(input_data)
    live_prediction = model.predict(scaled_input)[0]
    live_aqi = round(live_prediction, 1)

    # ============================================================
    # AQI Category Helper
    # ============================================================
    def get_aqi_category(aqi_value):
        if aqi_value <= 50:
            return "Good", "🟢", "#00E400"
        elif aqi_value <= 100:
            return "Satisfactory / Moderate", "🟡", "#FFFF00"
        elif aqi_value <= 200:
            return "Moderate / Poor", "🟠", "#FF7E00"
        elif aqi_value <= 300:
            return "Very Poor", "🔴", "#FF0000"
        else:
            return "Severe / Hazardous", "🚨", "#8F3F97"

    category, emoji, color = get_aqi_category(live_aqi)

    # ============================================================
    # MAIN LAYOUT — Tabs
    # ============================================================
    tab1, tab2, tab3 = st.tabs(["📈 AQI Prediction", "📊 Feature Analysis", "📁 Batch Prediction"])

    # ----------------------------------------------
    # TAB 1: AQI Prediction
    # ----------------------------------------------
    with tab1:
        col_left, col_right = st.columns([1, 1.2])

        with col_left:
            st.subheader("🎯 Live Prediction Preview")
            st.metric("Predicted AQI", live_aqi, delta=None)

            # AQI Gauge Chart (Plotly)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=live_aqi,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"<b>AQI: {live_aqi}</b><br><span style='font-size:16px'>{emoji} {category}</span>", 'font': {'size': 18}},
                number={'font': {'size': 40}},
                gauge={
                    'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "darkgray"},
                    'bar': {'color': color, 'thickness': 0.5},
                    'bgcolor': 'white',
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': '#00E400'},
                        {'range': [50, 100], 'color': '#FFFF00'},
                        {'range': [100, 200], 'color': '#FF7E00'},
                        {'range': [200, 300], 'color': '#FF0000'},
                        {'range': [300, 500], 'color': '#8F3F97'},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': live_aqi
                    }
                }
            ))
            fig_gauge.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=60, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Arial"}
            )
            st.plotly_chart(fig_gauge, width='stretch')

        with col_right:
            # Feature Contribution Bar Chart
            st.subheader("📊 Feature Contribution to Prediction")
            
            # Calculate contribution as feature_value * feature_importance (approximate)
            contributions = {}
            feature_values = {
                'PM2.5': pm25, 'PM10': pm10, 'NO2': no2, 'SO2': so2,
                'CO': co, 'O3': o3, 'Temp_C': temp, 'Humidity': humidity
            }
            
            if feature_importance:
                max_imp = max(feature_importance.values())
                for feat, imp in feature_importance.items():
                    norm_val = feature_values.get(feat, 0) / (feature_values.get(feat, 1) or 1)
                    contributions[feat] = round(imp * norm_val * 100, 2)
                
                contrib_df = pd.DataFrame(
                    list(contributions.items()),
                    columns=['Feature', 'Contribution (%)']
                ).sort_values('Contribution (%)', ascending=True)

                fig_contrib = px.bar(
                    contrib_df,
                    x='Contribution (%)',
                    y='Feature',
                    orientation='h',
                    title='Relative Feature Influence',
                    color='Contribution (%)',
                    color_continuous_scale=['#00E400', '#FFFF00', '#FF7E00', '#FF0000'],
                    text='Contribution (%)'
                )
                fig_contrib.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_contrib.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "white"},
                    xaxis_title="Contribution (%)",
                    yaxis_title=""
                )
                st.plotly_chart(fig_contrib, width='stretch')
            else:
                st.info("Feature importance data not available. Run `train_model.py` first.")

        # AQI Category Info Strip
        st.markdown("---")
        cols = st.columns(5)
        aqi_levels = [
            (0, 50, "Good", "#00E400", "🟢"),
            (51, 100, "Satisfactory", "#FFFF00", "🟡"),
            (101, 200, "Moderate/Poor", "#FF7E00", "🟠"),
            (201, 300, "Very Poor", "#FF0000", "🔴"),
            (301, 500, "Severe", "#8F3F97", "🚨"),
        ]
        for i, (lo, hi, label, hex_color, icon) in enumerate(aqi_levels):
            with cols[i]:
                st.markdown(
                    f"<div style='background:{hex_color}; padding:10px; border-radius:5px; "
                    f"text-align:center; color:black; font-weight:bold;'>"
                    f"{icon}<br>{lo}-{hi}<br>{label}</div>",
                    unsafe_allow_html=True
                )

    # ----------------------------------------------
    # TAB 2: Feature Analysis
    # ----------------------------------------------
    with tab2:
        st.subheader("🔍 Model Feature Importance")
        
        if feature_importance:
            imp_df = pd.DataFrame(
                list(feature_importance.items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True)

            col_a, col_b = st.columns([1.5, 1])

            with col_a:
                fig_imp = px.bar(
                    imp_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance (XGBoost)',
                    color='Importance',
                    color_continuous_scale='Viridis',
                    text='Importance'
                )
                fig_imp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig_imp.update_layout(
                    height=450,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "white"},
                    xaxis_title="Importance Score",
                    yaxis_title=""
                )
                st.plotly_chart(fig_imp, width='stretch')

            with col_b:
                st.markdown("##### 📋 Feature Importance Table")
                st.dataframe(
                    imp_df.sort_values('Importance', ascending=False).reset_index(drop=True),
                    width='stretch',
                    hide_index=True
                )
                
                st.markdown("##### 💡 Insights")
                top_feature = imp_df.iloc[-1]['Feature']
                top_importance = imp_df.iloc[-1]['Importance']
                st.info(
                    f"The most influential feature is **{top_feature}** "
                    f"(importance: {top_importance:.3f}). "
                    "This indicates which pollutant has the strongest impact on AQI predictions."
                )
        else:
            st.warning("Feature importance data not found. Please run `train_model.py` to generate it.")

    # ----------------------------------------------
    # TAB 3: Batch Prediction (CSV Upload)
    # ----------------------------------------------
    with tab3:
        st.subheader("📁 Batch Prediction via CSV Upload")
        st.markdown("""
        Upload a CSV file with the following columns to get batch AQI predictions:
        
        **Required columns:** `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`, `Temp_C`, `Humidity`
        
        You can download a sample template below.
        """)

        # Sample template download
        sample_df = pd.DataFrame({
            'PM2.5': [45.0, 120.0],
            'PM10': [85.0, 210.0],
            'NO2': [30.0, 65.0],
            'SO2': [15.0, 35.0],
            'CO': [1.2, 2.8],
            'O3': [25.0, 55.0],
            'Temp_C': [28.0, 34.0],
            'Humidity': [55.0, 70.0]
        })

        def to_csv_download(df, filename="sample_template.csv"):
            csv = df.to_csv(index=False)
            return BytesIO(csv.encode())

        st.download_button(
            label="📥 Download Sample Template CSV",
            data=to_csv_download(sample_df),
            file_name="aqi_sample_template.csv",
            mime="text/csv",
        )

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                required_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temp_C', 'Humidity']
                
                missing_cols = [c for c in required_cols if c not in batch_df.columns]
                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}. Please check the file format.")
                else:
                    # Scale and predict
                    batch_scaled = scaler.transform(batch_df[required_cols])
                    batch_predictions = model.predict(batch_scaled)
                    batch_df['Predicted_AQI'] = np.round(batch_predictions, 1)

                    # Add category
                    def assign_category(val):
                        if val <= 50: return "Good"
                        elif val <= 100: return "Satisfactory / Moderate"
                        elif val <= 200: return "Moderate / Poor"
                        elif val <= 300: return "Very Poor"
                        else: return "Severe / Hazardous"

                    batch_df['Category'] = batch_df['Predicted_AQI'].apply(assign_category)

                    st.success(f"✅ Predictions generated for {len(batch_df)} records!")

                    col_show, col_dl = st.columns([3, 1])
                    with col_show:
                        st.dataframe(batch_df, width='stretch', hide_index=True)
                    
                    with col_dl:
                        # Download results
                        def get_csv_download(df):
                            return df.to_csv(index=False).encode('utf-8')

                        st.download_button(
                            label="📥 Download Results CSV",
                            data=get_csv_download(batch_df),
                            file_name="aqi_predictions.csv",
                            mime="text/csv",
                            width='stretch'
                        )

                    # Mini distribution chart
                    st.subheader("📊 AQI Distribution in Batch")
                    fig_dist = px.histogram(
                        batch_df, x='Predicted_AQI', nbins=20,
                        color='Category',
                        color_discrete_map={
                            'Good': '#00E400',
                            'Satisfactory / Moderate': '#FFFF00',
                            'Moderate / Poor': '#FF7E00',
                            'Very Poor': '#FF0000',
                            'Severe / Hazardous': '#8F3F97'
                        },
                        title='Distribution of Predicted AQI Categories'
                    )
                    fig_dist.update_layout(
                        height=350,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "white"},
                        xaxis_title="AQI Value",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig_dist, width='stretch')

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

except FileNotFoundError:
    st.error("`model.pkl` not found! Please make sure you run `train_model.py` first.")
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
