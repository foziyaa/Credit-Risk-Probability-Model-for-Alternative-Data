# src/dashboard.py

import sys
import os
import streamlit as st
import pandas as pd
import mlflow
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# --- Path Correction ---
_src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_src_dir, '..')))
from src import config

# --- Page Configuration ---
st.set_page_config(page_title="Bati Bank BNPL Credit Risk Assessment", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

# --- Caching Functions ---
@st.cache_resource
def load_model():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    try:
        return mlflow.sklearn.load_model(f"models:/{config.MODEL_REGISTRY_NAME}-RandomForest/latest")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model.named_steps['model'])

@st.cache_data
def load_training_data():
    try:
        return pd.read_csv(config.PROCESSED_DATA_PATH)
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return None

# --- Helper Functions ---
def get_risk_tier_and_recommendation(probability):
    if probability < 0.2: return "Very Low", "✅ Approve - Standard Terms", "success"
    elif probability < 0.4: return "Low", "✅ Approve - Standard Terms", "success"
    elif probability < 0.6: return "Medium", "⚠️ Approve - Consider Lower Limit / Manual Review", "warning"
    elif probability < 0.8: return "High", "❌ Decline - Flag for Manual Review", "error"
    else: return "Very High", "❌ Decline - High Confidence", "error"

def generate_shap_summary(shap_explanation):
    shap_df = pd.DataFrame({'feature': shap_explanation.feature_names, 'value': shap_explanation.data, 'shap_value': shap_explanation.values})
    positive_impact = shap_df[shap_df['shap_value'] > 0].sort_values('shap_value', ascending=False).head(3)
    negative_impact = shap_df[shap_df['shap_value'] < 0].sort_values('shap_value', ascending=True).head(2)
    summary_parts = []
    if not positive_impact.empty:
        pos_features = [f"**{row['feature']}** ({row['value']:.0f})" for _, row in positive_impact.iterrows()]
        summary_parts.append(f"Primary factors increasing risk are: {', '.join(pos_features)}.")
    if not negative_impact.empty:
        neg_features = [f"**{row['feature']}** ({row['value']:.0f})" for _, row in negative_impact.iterrows()]
        summary_parts.append(f"Factors decreasing risk include: {', '.join(neg_features)}.")
    return " ".join(summary_parts) if summary_parts else "Feature impacts are balanced."

# --- Load Assets ---
model_pipeline = load_model()
training_data = load_training_data()
if model_pipeline:
    shap_explainer = get_shap_explainer(model_pipeline)

# --- Sidebar ---
st.sidebar.title("👨‍💻 Customer Profile")
st.sidebar.markdown("Enter customer data for a risk assessment.")
def get_user_input():
    recency = st.sidebar.slider("Recency (Days)", 0, 365, 30, help="Days since last transaction.")
    frequency = st.sidebar.slider("Frequency", 1, 100, 10, help="Total number of transactions.")
    monetary = st.sidebar.number_input("Monetary (Total Spend)", min_value=0.0, value=50000.0, step=1000.0)
    avg_monetary = st.sidebar.number_input("Average Spend", min_value=0.0, value=10000.0, step=500.0)
    std_monetary = st.sidebar.number_input("Spend Variation (Std. Dev.)", min_value=0.0, value=5000.0, step=500.0)
    tenure = st.sidebar.slider("Tenure (Days)", 0, 730, 180, help="Customer lifetime in days.")
    num_unique_products = st.sidebar.slider("Unique Products", 1, 50, 5, help="Number of distinct products.")
    num_unique_providers = st.sidebar.slider("Unique Providers", 1, 10, 2, help="Number of distinct providers.")
    input_data = {'Recency': recency, 'Frequency': frequency, 'Monetary': monetary, 'AvgMonetary': avg_monetary, 
                  'StdMonetary': std_monetary, 'Tenure': tenure, 'NumUniqueProducts': num_unique_products,
                  'NumUniqueProviders': num_unique_providers}
    input_df = pd.DataFrame([input_data])
    input_df.columns = config.FEATURES_TO_KEEP
    return input_df
user_input_df = get_user_input()
st.sidebar.divider()
st.sidebar.info("This dashboard uses a RandomForest model tracked via MLflow and explained with SHAP.")

# --- Main App ---
st.title("🏦 Bati Bank BNPL Credit Risk Dashboard")
st.markdown("A tool for real-time, explainable credit risk assessment.")
tab1, tab2 = st.tabs(["📊 **Customer Risk Profile**", "📈 **Global Model Insights**"])

with tab1:
    st.header("Single Customer Risk Assessment")
    with st.expander("View Current Customer Input Data"):
        st.dataframe(user_input_df)

    if model_pipeline and shap_explainer:
        if st.button("Assess Credit Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing customer profile..."):
                probability = model_pipeline.predict_proba(user_input_df)[0][1]
                risk_tier, recommendation, alert_type = get_risk_tier_and_recommendation(probability)

                st.subheader("Risk Decision Panel")
                if alert_type == "success": st.success(f"**Recommendation: {recommendation}**", icon="✅")
                elif alert_type == "warning": st.warning(f"**Recommendation: {recommendation}**", icon="⚠️")
                else: st.error(f"**Recommendation: {recommendation}**", icon="❌")
                col1, col2 = st.columns(2)
                col1.metric(label="**Risk Tier**", value=risk_tier)
                col2.metric(label="**Probability of High Risk**", value=f"{probability:.2%}")
                st.divider()

                st.subheader("Prediction Explanation (SHAP Analysis)")
                scaler = model_pipeline.named_steps['scaler']
                input_scaled = scaler.transform(user_input_df)
                
                # --- *** CORRECTED SHAP LOGIC FOR SINGLE PREDICTION *** ---
                # Call the explainer like a function to get the Explanation object
                shap_explanation_object = shap_explainer(input_scaled)
                
                # We need the explanation for the first (and only) sample for the "high risk" class (class 1)
                shap_explanation_single = shap_explanation_object[0, :, 1]

                # Generate the plain-English summary
                st.markdown(f"**Summary:** {generate_shap_summary(shap_explanation_single)}")
                
                with st.expander("View Detailed SHAP Contribution Plots"):
                    st.info("The force plot shows how features push the prediction. The waterfall plot breaks it down step-by-step.", icon="ℹ️")
                    st_shap(shap.plots.force(shap_explanation_single), height=150)
                    fig, ax = plt.subplots()
                    shap.plots.waterfall(shap_explanation_single, max_display=10, show=False)
                    st.pyplot(fig)
    else:
        st.error("Model/Explainer not loaded. Check MLflow server.")

with tab2:
    st.header("Understanding the Model's Overall Behavior")
    if training_data is not None:
        st.subheader("Global Feature Importance")
        st.info("Features the model considers most important across all predictions.", icon="💡")
        shap_summary_path = config.BASE_DIR / "reports" / "figures" / "shap_summary_plot.png"
        if shap_summary_path.exists():
            st.image(str(shap_summary_path), use_container_width=True)
        else:
            st.warning("SHAP summary plot not found. Run `src/explain_model.py`.")
        
        st.divider()
        st.subheader("Feature Distribution by Risk Group")
        st.info("Compare a feature's distribution for Low vs. High Risk customers.", icon="💡")
        feature_to_plot = st.selectbox("Select a feature to visualize", options=config.FEATURES_TO_KEEP)
        if feature_to_plot:
            fig, ax = plt.subplots()
            low_risk_data = training_data[training_data[config.TARGET_VARIABLE] == 0]
            high_risk_data = training_data[training_data[config.TARGET_VARIABLE] == 1]
            ax.hist(low_risk_data[feature_to_plot], bins=30, alpha=0.7, label='Low Risk', density=True, color='cornflowerblue')
            ax.hist(high_risk_data[feature_to_plot], bins=30, alpha=0.7, label='High Risk', density=True, color='salmon')
            user_value = user_input_df[feature_to_plot].iloc[0]
            ax.axvline(user_value, color='red', linestyle='--', linewidth=2, label=f'Current Input ({user_value:.2f})')
            ax.set_title(f"Distribution of '{feature_to_plot}' by Risk Group")
            ax.set_xlabel("Feature Value")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(axis='y', alpha=0.5)
            st.pyplot(fig)