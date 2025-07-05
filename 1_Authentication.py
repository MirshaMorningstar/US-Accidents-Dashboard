import pickle
from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth

from streamlit.components.v1 import html




st.set_page_config(initial_sidebar_state="expanded")

# User authentication
names = ["Mirsha Morningstar", "Rameez Akther", "Chandru", "Mekesh"]
usernames = ["Mirsha Morningstar","Rameez", "Chandru","Mekesh"]

# Load hashed passwords
file_path = Path(__file__).parent/"hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)



placeholder1 = st.empty()
placeholder2 = st.empty()
placeholder1.title("💥🚗💣US Accidents Dashboard")
placeholder2.markdown(
    """Hello There !!!  A Warm Welcome to our Application Dashboard. This highly interactive and sophisticated dashboard provides insights into intricate patterns, relationships and brings forth the overall knowledge of the " US accidents data ". This Web- Application Dashboard is specially built using Streamlit and Plotly. Feel free to Access it by logging in below."""
) 

st.markdown("### The Username is ' **Mirsha Morningstar** '")
st.markdown("### The Password is ' **AKM69** '")
st.markdown("Kindly Login the Application using the above credentials...")
authenticator = stauth.Authenticate(names,usernames,hashed_passwords,"US Accidents Dashboard","abcdef",cookie_expiry_days=0)

name,authentication_status,username = authenticator.login("Login your credentials", "main")



if authentication_status == False :
    st.error("The provided Username or Password is incorrect !!!")

if authentication_status == None:
    #hide_pages(["streamlit_app","user_auth","re_auth","EDA_window"])
    st.warning("Kindly enter your Username and Password")

if authentication_status:
    st.session_state["logged_in"] = True
    
    print("Success")
    placeholder1.empty()
    placeholder2.empty()
   
    st.switch_page(r"pages/2_Change_Theme🎨🖼️🔄.py")
    
    


