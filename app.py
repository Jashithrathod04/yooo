import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.stats import zscore





def go_to(page):
    st.session_state.page = page
    st.rerun()
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

/* MAIN BACKGROUND */

.stApp{
background: linear-gradient(-45deg,#020617,#0f172a,#1e293b,#020617);
background-size:400% 400%;
animation: gradientBG 15s ease infinite;
color:white;
}

@keyframes gradientBG{
0%{background-position:0% 50%}
50%{background-position:100% 50%}
100%{background-position:0% 50%}
}


/* FLOATING PARTICLES */

.stApp::before{
content:"";
position:fixed;
top:0;
left:0;
width:100%;
height:100%;
background-image:
radial-gradient(circle at 20% 20%,rgba(56,189,248,0.15),transparent 25%),
radial-gradient(circle at 80% 40%,rgba(168,85,247,0.15),transparent 25%),
radial-gradient(circle at 60% 80%,rgba(244,114,182,0.15),transparent 25%);
animation: floatBG 20s linear infinite;
z-index:-1;
}

@keyframes floatBG{
0%{transform:translateY(0px)}
50%{transform:translateY(-60px)}
100%{transform:translateY(0px)}
}


/* GLASS CARDS */

.glass{
background: rgba(255,255,255,0.06);
backdrop-filter: blur(18px);
border-radius:18px;
padding:25px;
border:1px solid rgba(255,255,255,0.15);
transition:all 0.35s ease;
box-shadow:0 8px 32px rgba(0,0,0,0.3);
}

.glass:hover{
transform:translateY(-6px) scale(1.02);
box-shadow:0 0 30px rgba(56,189,248,0.6);
}


/* METRIC CARDS */

.metric-card{
background: rgba(255,255,255,0.08);
backdrop-filter: blur(12px);
padding:20px;
border-radius:16px;
border:1px solid rgba(255,255,255,0.2);
text-align:center;
transition:0.3s;
}

.metric-card:hover{
transform:scale(1.05);
box-shadow:0 0 25px #38bdf8;
}


/* BUTTONS */

button{
border-radius:12px;
background: linear-gradient(90deg,#38bdf8,#a78bfa);
border:none;
color:white;
font-weight:600;
transition:0.3s;
}

button:hover{
transform:scale(1.05);
box-shadow:0 0 15px #22d3ee;
}


/* TABS STYLE */

div[data-baseweb="tab"]{
font-size:16px;
padding:10px;
transition:0.3s;
}

div[data-baseweb="tab"]:hover{
background:rgba(255,255,255,0.08);
border-radius:10px;
}


/* ANIMATED HEADINGS */

h1,h2,h3{
background: linear-gradient(90deg,#38bdf8,#a78bfa,#f472b6);
-webkit-background-clip:text;
color:transparent;
animation:hue 6s infinite linear;
}

@keyframes hue{
0%{filter:hue-rotate(0deg)}
100%{filter:hue-rotate(360deg)}
}


/* SCROLLBAR */

::-webkit-scrollbar{
width:8px;
}

::-webkit-scrollbar-track{
background:#020617;
}

::-webkit-scrollbar-thumb{
background:linear-gradient(#38bdf8,#a78bfa);
border-radius:10px;
}

.profile-container{
display:flex;
justify-content:center;
gap:60px;
margin-top:80px;
}

.profile-card{
text-align:center;
cursor:pointer;
transition:0.3s;
}

.profile-card img{
width:150px;
height:150px;
border-radius:10px;
border:3px solid transparent;
}

.profile-card:hover{
transform:scale(1.15);
}

.profile-card:hover img{
border:3px solid white;
}

.profile-name{
margin-top:10px;
font-size:18px;
color:#aaa;
}

.profile-card:hover .profile-name{
color:white;
}

.netflix-title{
text-align:center;
font-size:48px;
margin-top:80px;
color:white;
}























.profile-wrapper{
display:flex;
justify-content:center;
gap:60px;
margin-top:80px;
}

.profile-card{
background:rgba(255,255,255,0.05);
backdrop-filter:blur(16px);
padding:20px;
border-radius:20px;
text-align:center;
width:180px;
cursor:pointer;
transition:0.35s;
border:1px solid rgba(255,255,255,0.15);
}

.profile-card:hover{
transform:scale(1.15);
box-shadow:0 0 25px #38bdf8;
}

.profile-img{
width:140px;
height:140px;
border-radius:12px;
object-fit:cover;
}

.profile-name{
margin-top:10px;
font-size:18px;
color:#aaa;
}

.profile-card:hover .profile-name{
color:white;
}



















.profile-container{
display:flex;
justify-content:center;
gap:60px;
margin-top:80px;
}

.metric-profile{
background:rgba(255,255,255,0.06);
backdrop-filter:blur(16px);
padding:25px;
border-radius:20px;
width:200px;
text-align:center;
border:1px solid rgba(255,255,255,0.2);
transition:0.35s;
cursor:pointer;
}

.metric-profile:hover{
transform:scale(1.12);
box-shadow:0 0 30px #38bdf8;
}

.metric-profile img{
width:140px;
height:140px;
border-radius:12px;
object-fit:cover;
}

.metric-profile p{
margin-top:10px;
font-size:18px;
color:#bbb;
}

.metric-profile:hover p{
color:white;
}

a.cardlink{
text-decoration:none;
}
















.card-button button{
opacity:0;
height:180px;
width:100%;
position:absolute;
top:0;
left:0;
cursor:pointer;
}

.card-wrapper{
position:relative;
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

    st.markdown(
    "<h1 style='text-align:center;font-size:55px;'>Who's analyzing today?</h1>",
    unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)

    # PROFILE 1
    with col1:

        st.markdown('<div class="card-wrapper">', unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-profile">
            <img src="https://i.imgur.com/7yUvePI.png">
            <p>Jashith</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card-button">', unsafe_allow_html=True)

        if st.button("profile1"):
            st.session_state.page="dashboard"
            st.rerun()

        st.markdown('</div></div>', unsafe_allow_html=True)

    # PROFILE 2
    with col2:

        st.markdown('<div class="card-wrapper">', unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-profile">
            <img src="https://i.imgur.com/9XnK9QK.png">
            <p>Analyst</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card-button">', unsafe_allow_html=True)

        if st.button("profile2"):
            st.session_state.page="dashboard"
            st.rerun()

        st.markdown('</div></div>', unsafe_allow_html=True)

    # ADD PROFILE
    with col3:

        st.markdown('<div class="card-wrapper">', unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-profile">
            <img src="https://cdn-icons-png.flaticon.com/512/1828/1828817.png">
            <p>Add Profile</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card-button">', unsafe_allow_html=True)

        if st.button("profile3"):
            st.session_state.page="signup"
            st.rerun()

        st.markdown('</div></div>', unsafe_allow_html=True)
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

# -----------------------------------------
# DASHBOARD
# -----------------------------------------

if st.session_state.page=="dashboard":

    st.title("SmartCharge Analytics Dashboard")

    df = pd.read_csv("ev_charging_dataset.csv")

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
    # USAGE ANALYSIS
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
    # COST ANALYSIS
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
    # CORRELATION
    # -----------------------------------------

    with tab5:

        st.subheader("Feature Correlation Heatmap")

        import seaborn as sns
        import matplotlib.pyplot as plt

        corr = df.corr(numeric_only=True)

        fig,ax = plt.subplots()

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

    # -----------------------------------------
    # CLUSTERING
    # -----------------------------------------

    with tab6:

        st.subheader("Charging Station Clusters")

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
    # ASSOCIATION RULES
    # -----------------------------------------

    with tab7:

        st.subheader("Frequent Feature Associations")

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

    # -----------------------------------------
    # ANOMALY DETECTION
    # -----------------------------------------

    with tab8:

        st.subheader("Anomaly Detection")

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

    # -----------------------------------------
    # INSIGHTS
    # -----------------------------------------

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
