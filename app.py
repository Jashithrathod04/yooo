import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.stats import zscore

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------

st.set_page_config(
    page_title="SmartCharge AI",
    layout="wide",
)

# -----------------------------------------
# GLASSMORPHISM + ANIMATION CSS
# -----------------------------------------

st.markdown("""
<style>

body {
background: linear-gradient(135deg,#020617,#0f172a);
color:white;
}

.glass {
background: rgba(255,255,255,0.08);
backdrop-filter: blur(18px);
padding:25px;
border-radius:20px;
border:1px solid rgba(255,255,255,0.2);
transition:0.3s;
}

.glass:hover {
transform: scale(1.03);
box-shadow:0 0 20px #38bdf8;
}

button {
border-radius:12px;
transition:0.3s;
}

button:hover {
transform:scale(1.05);
box-shadow:0 0 10px #22d3ee;
}

h1,h2,h3{
background: linear-gradient(90deg,#38bdf8,#a78bfa,#f472b6);
-webkit-background-clip:text;
color:transparent;
animation: hue 6s infinite linear;
}

@keyframes hue{
0%{filter:hue-rotate(0deg)}
100%{filter:hue-rotate(360deg)}
}

.metric-card{
background: rgba(255,255,255,0.07);
padding:20px;
border-radius:15px;
text-align:center;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# PAGE STATE
# -----------------------------------------

if "page" not in st.session_state:
    st.session_state.page="splash"

# -----------------------------------------
# SPLASH SCREEN
# -----------------------------------------

if st.session_state.page=="splash":

    st.title("⚡ SmartCharge AI")

    st.write("Initializing EV Data Mining Engine...")

    progress=st.progress(0)

    for i in range(100):
        progress.progress(i+1)

    st.session_state.page="landing"
    st.rerun()

# -----------------------------------------
# LANDING PAGE
# -----------------------------------------

if st.session_state.page=="landing":

    st.title("SmartCharge AI")

    st.write("AI Powered EV Charging Infrastructure Analytics")

    if st.button("Get Started"):
        st.session_state.page="profiles"
        st.rerun()

# -----------------------------------------
# NETFLIX STYLE PROFILE PAGE
# -----------------------------------------

if st.session_state.page=="profiles":

    st.title("Who's analyzing today?")

    col1,col2,col3=st.columns(3)

    with col1:
        if st.button("👨‍💻 Jashith"):
            st.session_state.page="dashboard"
            st.rerun()

    with col2:
        if st.button("👩‍🔬 Analyst"):
            st.session_state.page="dashboard"
            st.rerun()

    with col3:
        if st.button("➕ Add Profile"):
            st.session_state.page="signup"
            st.rerun()

# -----------------------------------------
# SIGNUP PAGE
# -----------------------------------------

if st.session_state.page=="signup":

    st.title("Create Profile")

    name=st.text_input("Name")
    email=st.text_input("Email")
    password=st.text_input("Password",type="password")

    if st.button("Create Account"):

        otp=random.randint(100000,999999)

        st.session_state.otp=otp
        st.session_state.page="verify"

        st.rerun()

# -----------------------------------------
# OTP VERIFICATION
# -----------------------------------------

if st.session_state.page=="verify":

    st.title("Email Verification")

    st.info("Demo Mode OTP")

    st.success(st.session_state.otp)

    user_otp=st.text_input("Enter OTP")

    if st.button("Verify"):

        if str(user_otp)==str(st.session_state.otp):

            st.session_state.page="dashboard"
            st.rerun()

        else:

            st.error("Invalid OTP")

# -----------------------------------------
# DASHBOARD
# -----------------------------------------

if st.session_state.page=="dashboard":

    st.title("SmartCharge Analytics Dashboard")

    df=pd.read_csv("ev_charging_dataset.csv")

    tab1,tab2,tab3,tab4,tab5,tab6,tab7=st.tabs([
        "Overview",
        "Data Cleaning",
        "EDA",
        "Clustering",
        "Association Rules",
        "Anomaly Detection",
        "Insights"
    ])

# -----------------------------------------
# OVERVIEW
# -----------------------------------------

    with tab1:

        col1,col2,col3=st.columns(3)

        col1.metric("Rows",len(df))
        col2.metric("Columns",len(df.columns))
        col3.metric("Operators",df["Station Operator"].nunique())

        st.dataframe(df.head())

# -----------------------------------------
# DATA CLEANING
# -----------------------------------------

    with tab2:

        st.subheader("Missing Values")

        st.dataframe(df.isnull().sum())

        df.drop_duplicates(inplace=True)

        st.success("Duplicates removed")

# -----------------------------------------
# EDA
# -----------------------------------------

    with tab3:

        if "Usage Stats (avg users/day)" in df.columns:

            fig1=px.histogram(df,x="Usage Stats (avg users/day)")
            st.plotly_chart(fig1,use_container_width=True)

        if "Cost (USD/kWh)" in df.columns:

            fig2=px.scatter(
                df,
                x="Cost (USD/kWh)",
                y="Usage Stats (avg users/day)",
                color="Charger Type"
            )

            st.plotly_chart(fig2,use_container_width=True)

        if "Charging Capacity (kW)" in df.columns:

            fig3=px.scatter(
                df,
                x="Charging Capacity (kW)",
                y="Usage Stats (avg users/day)"
            )

            st.plotly_chart(fig3,use_container_width=True)

# -----------------------------------------
# CLUSTERING
# -----------------------------------------

    with tab4:

        features=df[[
        "Cost (USD/kWh)",
        "Charging Capacity (kW)",
        "Usage Stats (avg users/day)",
        "Distance to City (km)",
        "Parking Spots",
        "Reviews (Rating)"
        ]]

        scaler=StandardScaler()

        scaled=scaler.fit_transform(features)

        kmeans=KMeans(n_clusters=4)

        clusters=kmeans.fit_predict(scaled)

        df["Cluster"]=clusters

        fig=px.scatter(
            df,
            x="Charging Capacity (kW)",
            y="Usage Stats (avg users/day)",
            color="Cluster"
        )

        st.plotly_chart(fig,use_container_width=True)

# -----------------------------------------
# ASSOCIATION RULES
# -----------------------------------------

    with tab5:

        numeric=df.select_dtypes(include=np.number)

        binary=(numeric>numeric.mean()).astype(int)

        freq=apriori(binary,min_support=0.1,use_colnames=True)

        rules=association_rules(freq,metric="lift",min_threshold=1)

        st.dataframe(rules.head())

# -----------------------------------------
# ANOMALY DETECTION
# -----------------------------------------

    with tab6:

        numeric=df.select_dtypes(include=np.number)

        z=np.abs(zscore(numeric))

        anomalies=(z>3).any(axis=1)

        st.write("Detected Anomalies")

        st.dataframe(df[anomalies])

# -----------------------------------------
# INSIGHTS
# -----------------------------------------

    with tab7:

        st.subheader("Key Insights")

        st.markdown("""
        • Stations closer to cities show higher usage  
        • Higher capacity chargers attract more EV users  
        • Renewable energy stations tend to receive better ratings  
        • Stations with more parking spots support higher demand  
        • Premium stations charge higher cost but maintain strong usage
        """)
