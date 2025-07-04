import numpy as np
import pandas as pd
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from ydata_profiling import ProfileReport
import re
import base64
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="The Exploratory Data Analysis Window",
    page_icon="analysis.png",
    menu_items={
        'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
        'Report a bug': "mailto:a.k.mirsha9@gmail.com",
        'About': "### This is an extremely cool web application built as a part of my Data Science Mini Project on the ' US Accidents Dataset '\n"
    }
)

# 🔐 Session state to persist report
if "eda_report_html" not in st.session_state:
    st.session_state.eda_report_html = None

# Strip <html>, <head>, <body> tags to avoid app-in-app
def strip_html_tags(html_str):
    html_str = re.sub(r"(?is)<(html|head|body).*?>", "", html_str)
    html_str = re.sub(r"(?is)</(html|head|body)>", "", html_str)
    return html_str

# Title and info
st.markdown("# **The Exploratory Data Analysis ( EDA ) Window**")
add_vertical_space(2)
st.markdown('''
:smile: This is the **EDA Window** created in Streamlit using the **pandas-profiling** library. :sparkles:  
**Credit: :crossed_flags:** App built in `Python` + `Streamlit`. Refer our [Documentation](https://drive.google.com/drive/folders/1Amtvj8MfXswe0AVTreA4IjSG2UeOl4Lt?usp=sharing)
---
''')
st.text('''
    1. Input Sample Data Frame
    2. Overview and Alerts
    3. Missing/distinct values, correlation, visual plots of variables
    4. Interactions between attributes
    5. Correlation Heatmap and Table
    6. Missing value matrix and heatmap
    7. Sample of first and last rows
''')
add_vertical_space(2)

# Upload block
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown(
        "[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)")

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

# Main logic
def generate_report(df):
    with st.spinner("Generating profiling report... this may take a few seconds..."):
        sampled_df = df.sample(n=min(500, len(df)), random_state=42)
        profile = ProfileReport(sampled_df, explorative=True, minimal=True)
        full_html = profile.to_html()
        safe_html = strip_html_tags(full_html)
        st.session_state.eda_report_html = safe_html

        # Download button
        b64 = base64.b64encode(full_html.encode()).decode()
        st.markdown(
            f'<a href="data:text/html;base64,{b64}" download="EDA_Report.html">📥 Download Full Report</a>',
            unsafe_allow_html=True
        )

if uploaded_file is not None:
    df = load_csv(uploaded_file)
    st.header('**Input DataFrame**')
    st.dataframe(df, use_container_width=True)
    st.write("---")

    if st.button("🔍 Generate EDA Report"):
        generate_report(df)

elif st.button('Press to use our Example "US Accidents" Dataset...'):
    try:
        df = pd.read_csv("US_Accidents_1000.csv")
        st.header('**Input DataFrame**')
        st.dataframe(df, use_container_width=True)
        st.write("---")
        generate_report(df)
    except FileNotFoundError:
        st.error("The example dataset 'US_Accidents_1000.csv' is missing. Please add it to your root directory.")

# Render report if available
if st.session_state.eda_report_html:
    st.success("✅ Report generated. Rendering below:")
    components.html(st.session_state.eda_report_html, height=900, scrolling=True)
