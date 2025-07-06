# authentication.py (Entry point)
import streamlit as st
import streamlit_authenticator as stauth
import pickle
from pathlib import Path
from st_pages import Page, navigation

# -------------- AUTH CONFIG -------------------
names = ["Mirsha Morningstar", "Rameez Akther", "Chandru", "Mekesh"]
usernames = ["Mirsha Morningstar", "Rameez", "Chandru", "Mekesh"]

# Load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

# Set collapsed sidebar
st.set_page_config(initial_sidebar_state="collapsed")

# Landing welcome text
placeholder1 = st.empty()
placeholder2 = st.empty()
placeholder1.title("💥🚗💣 US Accidents Dashboard")
placeholder2.markdown(
    """Hello There !!! 👋 A Warm Welcome to our Application Dashboard.
    This interactive Streamlit + Plotly dashboard uncovers deep patterns, insights, 
    and relationships from the US Accidents dataset.
    
    Login below to explore the application.
    """
)

# Authenticator setup
authenticator = stauth.Authenticate(
    names, usernames, hashed_passwords,
    "US Accidents Dashboard", "abcdef",
    cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login("Login your credentials", "main")

# -------------- LOGIN HANDLING -------------------

if authentication_status is False:
    st.error("The provided Username or Password is incorrect ❌")

elif authentication_status is None:
    st.warning("Kindly enter your Username and Password 🔐")

elif authentication_status:
    st.session_state["logged_in"] = True
    placeholder1.empty()
    placeholder2.empty()
    st.success(f"Welcome, {name} ✅")
    st.switch_page("pages/UI Personalization/3_Change_Theme.py")  # Redirect after login
