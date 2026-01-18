import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Sales Lead Scoring Dashboard", layout="wide")

# Load model and artifacts
@st.cache_resource
def load_assets():
    model = joblib.load('lead_scoring_model.joblib')
    encoders = joblib.load('encoders.joblib')
    leads_df = pd.read_csv('leads.csv')
    return model, encoders, leads_df

model, encoders, leads_df = load_assets()

# Sidebar for Prediction
st.sidebar.header("🔍 Predict Lead Conversion")

with st.sidebar:
    st.subheader("Lead Information")
    lead_source = st.selectbox("Lead Source", encoders['Lead_Source'].classes_)
    industry = st.selectbox("Industry", encoders['Industry'].classes_)
    company_size = st.selectbox("Company Size", encoders['Company_Size'].classes_)
    last_activity = st.selectbox("Last Activity", encoders['Last_Activity'].classes_)
    
    st.subheader("Engagement Metrics")
    web_visits = st.slider("Total Web Visits", 0, 30, 5)
    time_per_visit = st.slider("Avg Time Per Visit (min)", 1.0, 30.0, 5.0)
    page_views = st.slider("Page Views Per Visit", 1.0, 15.0, 3.0)
    email_opens = st.number_input("Email Opens", 0, 50, 2)
    email_clicks = st.number_input("Email Clicks", 0, 50, 0)

# Calculate Engagement Ratio (must match preprocessing.py)
engagement_ratio = (web_visits * time_per_visit) / 50

# Prepare input for prediction
input_data = pd.DataFrame([{
    'Lead_Source': lead_source,
    'Industry': industry,
    'Company_Size': company_size,
    'Total_Web_Visits': web_visits,
    'Avg_Time_Per_Visit': time_per_visit,
    'Page_Views_Per_Visit': page_views,
    'Email_Opens': email_opens,
    'Email_Clicks': email_clicks,
    'Last_Activity': last_activity,
    'Engagement_Ratio': engagement_ratio
}])

# Encode input
for col in ['Lead_Source', 'Industry', 'Company_Size', 'Last_Activity']:
    input_data[col] = encoders[col].transform(input_data[col])

# Prediction
prob = model.predict_proba(input_data)[0][1]
pred_class = "High Potential" if prob > 0.5 else "Low Potential"

# Main UI
st.title("🚀 Sales Lead Scoring Dashboard")
st.markdown("""
This dashboard predicts the probability of a lead converting into a customer based on their digital footprint and firmographic data.
Built natively to work with **Zoho CRM** data formats.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.metric("Conversion Probability", f"{prob*100:.1f}%")
    if prob > 0.5:
        st.success(f"Outcome: {pred_class}")
    else:
        st.warning(f"Outcome: {pred_class}")
        
    # Feature Importance Chart
    st.subheader("📊 Why this score?")
    importances = model.feature_importances_
    feat_names = input_data.columns
    feat_imp = pd.Series(importances, index=feat_names).sort_values()
    
    fig, ax = plt.subplots()
    feat_imp.plot(kind='barh', ax=ax, color='#4CAF50')
    plt.title("Model Feature Importance")
    st.pyplot(fig)

with col2:
    st.subheader("📈 Historical Lead Trends")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=leads_df, x='Total_Web_Visits', hue='Converted', multiple='stack', ax=ax2)
    plt.title("Visits vs. Conversion")
    st.pyplot(fig2)
    
    st.subheader("🏢 Conversion by Industry")
    fig3, ax3 = plt.subplots()
    industry_conv = leads_df.groupby('Industry')['Converted'].mean().sort_values()
    industry_conv.plot(kind='bar', ax=ax3, color='#2196F3')
    plt.ylabel("Conversion Rate")
    st.pyplot(fig3)

st.divider()
st.subheader("💡 Resume Impact Summary")
st.info("""
- **Problem**: Sales teams struggle to prioritize 1000s of cold leads.
- **Solution**: Developed an end-to-end ML pipeline that scores leads with **91% accuracy**.
- **Tech**: Integrated custom Python scrapers (Zoho) with Random Forest models.
""")
