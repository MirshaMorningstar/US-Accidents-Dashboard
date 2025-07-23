# authentication.py (Entry point)
import streamlit as st
import streamlit_authenticator as stauth
import pickle
from pathlib import Path

# -------------- AUTH CONFIG -------------------
names = ["Mirsha Morningstar", "Rameez Akther", "Chandru", "Mekesh"]
usernames = ["Mirsha Morningstar", "Rameez", "Chandru", "Mekesh"]

# Load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

# Set collapsed sidebar
st.set_page_config(page_title="Authentication to Streamlit Application", initial_sidebar_state="collapsed")

# Landing welcome text
placeholder1 = st.empty()
placeholder2 = st.empty()
placeholder1.title("💥🚗💣 Welcome to the US Accidents Analysis & Machine Learning Prediction Platform")
placeholder2.markdown(
    """
    #### 👋 Hello and Welcome!

    This interactive, multi-page dashboard is built to **explore, visualize, and predict accident severity** across the United States using real-world crash data. It combines modern UI aesthetics with robust data science pipelines to deliver:

    - 🔎 **Comprehensive Data Exploration**  
    - 🛠️ **Feature Engineering and Correlation Insights**  
    - 📊 **Dynamic Visualizations (EDA + Inferential)**  
    - 🧠 **ML Model Comparison & Accuracy Benchmarking**  
    - 🎯 **Hyperparameter Tuning & Prediction Interface**

    Whether you're a data enthusiast, domain expert, or a decision-maker, this tool gives you full control to **analyze trends, model risks, and optimize safety strategies**.

    ---

    ### 🚦 Why This Project Stands Out

    * Fully modular **ETL + EDA + ML pipelines**  
    * Real-time **interactive UI with theme customization**  
    * Plug-and-play support for **custom CSV uploads**  
    * Supports 25+ ML models using **LazyPredict & GridSearchCV**  
    * **Session-managed prediction flow** without unnecessary reloads  

    ---
    """
)

# Authenticator setup
authenticator = stauth.Authenticate(
    names, usernames, hashed_passwords,
    "US Accidents Dashboard", "abcdef",
    cookie_expiry_days=1
)

DEBUG = True # flag for development and testing purposes only, remove on production

name, authentication_status, username = authenticator.login("Login your credentials", "main")

# -------------- LOGIN HANDLING -------------------

if authentication_status is False:
    st.error("The provided Username or Password is incorrect ❌")
    st.session_state.logged_in = False
    if DEBUG:
        st.markdown("""##### Use **Username**: `Mirsha Morningstar` and **Password**: `AKM69`""")


elif authentication_status is None:
    st.warning("Kindly enter Username: Mirsha Morningstar and Password: AKM69")
    if DEBUG:
        st.markdown("""##### Use **Username**: `Mirsha Morningstar` and **Password**: `AKM69`""")


elif authentication_status:  # successfully logged in
    st.session_state.logged_in = True
    placeholder1.empty()
    placeholder2.empty()

    # --- Showing sidebar menu after login ---
    intro = st.Page(
        "pages/Introduction/2_About_This_Project.py",
        title="About this Application",
        icon=":material/info:"
    )

    # UI Personalization
    changetheme = st.Page(
        "pages/UI Personalization/3_Change_Theme.py",
        title="Change Application Theme",
        icon=":material/brush:"
    )
    colorpalette = st.Page(
        "pages/UI Personalization/4_Color_Palette_Picker.py",
        title="Color Palette Picker",
        icon=":material/palette:"
    )

    # Data Pipeline
    dataprep = st.Page(
        "pages/Data Pipeline/5_Data_Preprocessing_and_Preparation.py",
        title="Data Preprocessing and Preparation",
        icon=":material/cleaning_services:"
    )
    dataqual = st.Page(
        "pages/Data Pipeline/6_Visualisation_of_Data_Quality.py",
        title="Visualisation of Data Quality",
        icon=":material/assignment_turned_in:"
    )
    featureeng = st.Page(
        "pages/Data Pipeline/7_Feature_Engineering_and_Data_Correlation.py",
        title="Feature Engineering and Data Correlation Analysis",
        icon=":material/insights:"
    )

    # Exploratory Data Analysis
    specexp = st.Page(
        "pages/Exploratory Data Analysis/8_Specific_Exploration_of_Data.py",
        title="Specific Exploration of Data",
        icon=":material/zoom_in:"
    )
    genexp = st.Page(
        "pages/Exploratory Data Analysis/9_General_Exploration_of_Data.py",
        title="General Exploration of Data",
        icon=":material/explore:"
    )
    inf = st.Page(
        "pages/Exploratory Data Analysis/10_Inferential_Visualisations.py",
        title="Inferential Visualisations",
        icon=":material/precision_manufacturing:"
    )
    genvis = st.Page(
        "pages/Exploratory Data Analysis/11_General_Visualisations.py",
        title="General Visualisations",
        icon=":material/insert_chart:"
    )

    # Machine Learning
    ml = st.Page(
        "pages/Machine Learning Models/12_Comparison_of_Classifier_Models.py",
        title="Comparison of ML Classifier Models",
        icon=":material/compare_arrows:"
    )
    hyper = st.Page(
        "pages/Machine Learning Models/13_Optimisation_of_Various_Hyperparameters.py",
        title="Hyperparameter Finetuning and Optimisation Techniques",
        icon=":material/tune:"
    )

    if st.session_state.logged_in:
        pg = st.navigation(
            {
                "About this Project": [intro],
                "UI Personalisation and Color Theory": [changetheme, colorpalette],
                "Data ETL Pipeline": [dataprep, dataqual, featureeng],
                "Data EDA Pipeline": [specexp, genexp, inf, genvis],
                "Machine Learning and Hyperparameter Optimisation": [ml, hyper],
            }
        )
    else:
        pg = st.navigation(
            {"About this Project": [intro]}
        )
    # Run the selected page
    pg.run()
