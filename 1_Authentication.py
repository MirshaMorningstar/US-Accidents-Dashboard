# streamlit_app.py (Main Launcher File)
import streamlit as st
import streamlit_authenticator as stauth
import pickle
from pathlib import Path

# ----------------- AUTH CONFIG ------------------
# Setup login credentials
names = ["Mirsha Morningstar", "Rameez Akther", "Chandru", "Mekesh"]
usernames = ["Mirsha Morningstar", "Rameez", "Chandru", "Mekesh"]

# Load hashed passwords from file
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

# Configure authenticator
authenticator = stauth.Authenticate(
    names, usernames, hashed_passwords,
    "US Accidents Dashboard", "abcdef",
    cookie_expiry_days=1
)

# Login UI
name, authentication_status, username = authenticator.login("Login your credentials", "main")

# ----------------- PAGE NAVIGATION ------------------
# Define your pages using st.Page API
from st_pages import Page, show_pages, add_page_title

if authentication_status is False:
    st.error("The provided Username or Password is incorrect !!!")
elif authentication_status is None:
    st.warning("Kindly enter your Username and Password")
elif authentication_status:
    st.session_state["logged_in"] = True

    # Optional: Hide login content once authenticated
    st.empty()

    # Define pages with grouped sections for maximum clarity
    show_pages([
        # ----- 🔖 Introduction -----
        Page("pages/0_About_This_Project.py", "📌 About This Project"),

        # ----- 🔐 Account -----
        Page("pages/1_Authentication.py", "🔐 Log Out"),

        # ----- 🎨 UI Personalization -----
        Page("pages/2_Change_Theme.py", "🎨 Change Theme"),
        Page("pages/3_Color_Palette_Picker.py", "🌈 Color Palette Picker"),

        # ----- 🧪 Data Pipeline -----
        Page("pages/4_Data_Preprocessing_and_Preparation.py", "🧹 Data Preprocessing & Preparation"),
        Page("pages/5_Visualisation_of_Data_Quality.py", "✅ Data Quality Visualisation"),
        Page("pages/6_Feature_Engineering_and_Data_Correlation.py", "🧠 Feature Engineering & Correlation"),

        # ----- 🔍 Exploratory Data Analysis -----
        Page("pages/7_Specific_Exploration_of_Data.py", "🔬 Specific Data Exploration"),
        Page("pages/8_General_Exploration_of_Data.py", "📊 General Data Exploration"),
        Page("pages/9_Inferential_Visualisations.py", "📐 Inferential Visualisations"),
        Page("pages/10_General_Visualisations.py", "📈 General Visualisations"),

        # ----- 🤖 Machine Learning Models -----
        Page("pages/11_Comparison_of_Classifier_Models.py", "📝 Classifier Model Comparison"),
        Page("pages/12_Optimisation_of_Various_Hyperparameters.py", "⚙️ Hyperparameter Optimisation")
    ])

    add_page_title()  # Adds current page title to top
