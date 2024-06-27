import pickle
from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth

from streamlit.components.v1 import html

from st_pages import hide_pages

st.set_page_config(initial_sidebar_state="collapsed")

# User authentication
names = ["Mirsha Morningstar", "Rameez Akther", "Chandru", "Mekesh"]
usernames = ["Mirsha Morningstar", "Rameez", "Chandru", "Mekesh"]

# Load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

hide_pages(["streamlit_app", "user_auth", "re_auth"])

placeholder1 = st.empty()
placeholder2 = st.empty()
placeholder1.title("💥🚗💣US Accidents Dashboard")
placeholder2.markdown(
    """Hello There !!!  A Warm Welcome to our Application Dashboard. This highly interactive and sophisticated dashboard provides insights into intricate patterns, relationships and brings forth the overall knowledge of the " US accidents data ". This Web-Application Dashboard is specially built using Streamlit and Plotly. Feel free to Access it by logging in below."""
)

authenticator = stauth.Authenticate(
    names,
    usernames,
    hashed_passwords,
    "US Accidents Dashboard",
    "abcdef",
    cookie_expiry_days=0,
)

# Modify the login box with placeholder text
def custom_login(authenticator):
    form = st.form(key='login_form')
    username = form.text_input('Username', placeholder='Sample Name')
    password = form.text_input('Password', type='password', placeholder='Sample Password')
    login_button = form.form_submit_button('Login')
    if login_button:
        return authenticator._check_credentials(username, password)

name, authentication_status, username = custom_login(authenticator)

if authentication_status == False:
    hide_pages(["streamlit_app", "user_auth", "re_auth", "EDA_window"])
    st.error("The provided Username or Password is incorrect !!!")

if authentication_status == None:
    hide_pages(["streamlit_app", "user_auth", "re_auth", "EDA_window"])
    st.warning("Kindly enter your Username and Password")

if authentication_status:
    st.session_state["logged_in"] = True
    
    print("Success")
    placeholder1.empty()
    placeholder2.empty()
    hide_pages(["streamlit_app", "user_auth", "re_auth", "EDA_window"])
    st.switch_page(r"pages/2_Change_Theme🎨🖼️🔄.py")
