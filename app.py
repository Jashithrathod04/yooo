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

    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9 = st.tabs([
    "Overview",
    "Data Cleaning",
    "Usage Analysis",
    "Cost & Operator Analysis",
    "Correlation Analysis",
    "Clustering",
    "Association Rules",
    "Anomaly Detection",
    "Insights"
    ])

# -----------------------------------------
# OVERVIEW
# -----------------------------------------

   with tab1:

        st.subheader("Dataset Overview")
    
        col1,col2,col3 = st.columns(3)
    
        col1.metric("Total Stations", len(df))
        col2.metric("Operators", df["Station Operator"].nunique())
        col3.metric("Charger Types", df["Charger Type"].nunique())
    
        st.dataframe(df.head())

# -----------------------------------------
# DATA CLEANING
# -----------------------------------------

    with tab2:

        st.subheader("Missing Values")
    
        st.dataframe(df.isnull().sum())
    
        df.drop_duplicates(subset="Station ID", inplace=True)
    
        st.success("Duplicate stations removed")
    
        st.write("Dataset Shape:", df.shape)

# -----------------------------------------
# EDA
# -----------------------------------------

    with tab3:

        st.subheader("Usage Distribution")
    
        fig = px.histogram(
            df,
            x="Usage Stats (avg users/day)",
            nbins=30,
            title="EV Charging Demand Distribution"
        )
    
        st.plotly_chart(fig,use_container_width=True)
        
        st.subheader("Usage Growth Over Time")
    
        fig2 = px.line(
            df,
            x="Installation Year",
            y="Usage Stats (avg users/day)",
            title="Charging Usage Over Years"
        )
    
        st.plotly_chart(fig2,use_container_width=True)
# -----------------------------------------
# CLUSTERING
# -----------------------------------------

    with tab4:

        st.subheader("Cost by Station Operator")
    
        fig = px.box(
            df,
            x="Station Operator",
            y="Cost (USD/kWh)",
            title="Charging Cost Distribution"
        )
    
        st.plotly_chart(fig,use_container_width=True)

        fig2 = px.scatter(
            df,
            x="Cost (USD/kWh)",
            y="Usage Stats (avg users/day)",
            color="Charger Type",
            title="Cost vs Charging Demand"
        )
    
        st.plotly_chart(fig2,use_container_width=True)

# -----------------------------------------
# ASSOCIATION RULES
# -----------------------------------------

    with tab5:

        st.subheader("Feature Correlation Heatmap")
    
        import seaborn as sns
        import matplotlib.pyplot as plt
    
        corr = df.corr(numeric_only=True)
    
        fig,ax = plt.subplots()
    
        sns.heatmap(corr,
                    annot=True,
                    cmap="coolwarm",
                    ax=ax)
    
        st.pyplot(fig)

# -----------------------------------------
# ANOMALY DETECTION
# -----------------------------------------

    with tab6:

        st.subheader("Charging Station Clusters")
    
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
    
        features = df[[
        "Cost (USD/kWh)",
        "Charging Capacity (kW)",
        "Usage Stats (avg users/day)",
        "Distance to City (km)"
        ]]
    
        scaler = StandardScaler()
    
        scaled = scaler.fit_transform(features)
    
        kmeans = KMeans(n_clusters=4)
    
        df["Cluster"] = kmeans.fit_predict(scaled)
    
        fig = px.scatter(
            df,
            x="Charging Capacity (kW)",
            y="Usage Stats (avg users/day)",
            color="Cluster",
            title="Station Clustering"
        )
    
        st.plotly_chart(fig,use_container_width=True)

# -----------------------------------------
# INSIGHTS
# -----------------------------------------
    with tab7:

        st.subheader("Frequent Feature Associations")
    
        from mlxtend.frequent_patterns import apriori
        from mlxtend.frequent_patterns import association_rules
    
        numeric = df.select_dtypes(include="number")
    
        binary = (numeric > numeric.mean()).astype(int)
    
        freq = apriori(binary, min_support=0.1, use_colnames=True)
    
        rules = association_rules(freq, metric="lift", min_threshold=1)
    
        st.dataframe(rules.head())
    
        fig = px.bar(
            rules,
            x="support",
            y=rules["antecedents"].astype(str),
            title="Association Rules"
        )
    
        st.plotly_chart(fig)





    with tab8:

        st.subheader("Anomaly Detection")
    
        from scipy.stats import zscore
    
        numeric = df.select_dtypes(include="number")
    
        z = np.abs(zscore(numeric))
    
        anomalies = (z > 3).any(axis=1)
    
        df_anomaly = df[anomalies]
    
        st.write("Detected Anomalies")
    
        st.dataframe(df_anomaly)
    
        fig = px.scatter(
            df,
            x="Cost (USD/kWh)",
            y="Usage Stats (avg users/day)",
            color=anomalies,
            title="Anomaly Detection Visualization"
        )
    
        st.plotly_chart(fig)



    with tab9:

        st.subheader("Key Insights")
    
        st.markdown("""
        **Major Findings**
    
        - DC Fast Chargers show the highest usage.
        - Stations near city centers experience greater demand.
        - Operators with lower costs attract more EV users.
        - Some stations show extremely high usage indicating potential infrastructure pressure.
        - Renewable energy powered stations tend to receive better ratings.
        """)          
