import numpy as np
import pandas as pd
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components # important for rendering pandas report inside html
# Set page configuration
st.set_page_config(
    initial_sidebar_state="expanded",
    page_title="The Exploratory Data Analysis Window",
    page_icon="analysis.png",  # Make sure this file exists
    menu_items={
        'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
        'Report a bug': "mailto:a.k.mirsha9@gmail.com",
        'About': "### This is an extremely cool web application built as a part of my Data Science Mini Project on the ' US Accidents Dataset '\n"
    },
    layout="wide"
)

# Title
st.markdown("# **The Exploratory Data Analysis ( EDA ) Window**")
add_vertical_space(2)
st.markdown('''
:smile: This is the **EDA Window** created in Streamlit using the **pandas-profiling** library. :sparkles:  
---
''')

st.text('''
    1. Input Sample Data Frame
    2. Overview and Alerts
    3. Visual plots of variables
    4. Descriptive Statistics
    5. Quantile Statistics
    6. Missing Values, Imbalance, Constant Labels
    7. Sample of first and last rows
''')
add_vertical_space(2)

# Sidebar file uploader
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)")

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

# Process uploaded file
if uploaded_file is not None:
    df = load_csv(uploaded_file)
    df = df.sample(n=min(500, len(df)), random_state=42) # Use only 500 rows
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')

    st.header('**Dataset Profiling Report**')
    with st.spinner("Generating profiling report... this may take a few seconds..."):
        pr = ProfileReport(df, explorative=True, minimal=True)
        # Convert report to HTML string
        report_html = pr.to_html()
        st.success("✅ Report generated successfully!")
        # Render using HTML iframe (most stable way)
        components.html(report_html, height=900, scrolling=True)

else:
    st.info('Awaiting for CSV file to be uploaded.')
    add_vertical_space(2)

    if st.button('Press to use our Example "US Accidents" Dataset...'):
        try:
            df = pd.read_csv("US_Accidents_1000.csv")
            df = df.sample(n=min(500, len(df)), random_state=42) # Use only 500 rows
            st.header('**Input DataFrame**')
            st.write(df)
            st.write('---')

            st.header('**Dataset Profiling Report**')
            with st.spinner("Generating profiling report... this may take a few seconds..."):
                pr = ProfileReport(df, explorative=True, minimal=True)
                # Convert report to HTML string
                report_html = pr.to_html()
                st.success("✅ Report generated successfully!")
                # Render using HTML iframe (most stable way)
                components.html(report_html, height=900, scrolling=True)
                
        except FileNotFoundError:
            st.error("The example dataset 'US_Accidents_1000.csv' is missing. Please add it to your root directory.")
