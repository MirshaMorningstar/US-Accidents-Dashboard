import numpy as np
import pandas as pd
import streamlit as st
import streamlit_extras as stex
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.dataframe_explorer import dataframe_explorer
from ydata_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report

st.set_page_config(initial_sidebar_state="collapsed",page_title= " The Exploratory Data Analysis Window",
        menu_items={
         'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
         'Report a bug': "mailto:a.k.mirsha9@gmail.com",
         'About': "### This is an extremely cool web application built as a part of my Data Science Mini Project on the ' US Accidents Dataset '\n"
     },page_icon="analysis.png")

# Web App Title
st.markdown('''
# **The Exploratory Data Analysis ( EDA ) Window**''')
add_vertical_space(2)
st.markdown(''':smile: This is the **EDA Window** created in Streamlit using the **pandas-profiling** library. :sparkles: \n
This Window of our Application Provides the following :balloon: ;       

**Credit: :crossed_flags: ** App built in `Python` + `Streamlit`. Refer our [Documentation](https://drive.google.com/drive/folders/1Amtvj8MfXswe0AVTreA4IjSG2UeOl4Lt?usp=sharing) for more details ... 

---
''')

st.text('''
             \t\t1. Input Sample Data Frame\n
             \t\t2. Overview and Alerts\n
             \t\t3. Missing, distinct values, correlation memory size and visual plots of each variable\n
             \t\t4. Interactions between 2 different attributes\n
             \t\t5. Correlation Heatmap and Table of the entire dataset\n
             \t\t6. Missing value count, matrix and heatmap\n
             \t\t7. A Sample of first and last rows\n''' )
add_vertical_space(2)
        
# Upload CSV data
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Pandas Profiling Report
if uploaded_file is not None:
    @st.cache_data
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    pr.to_file(output_file="output.html")
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('**Dataset Profiling Report**')
    st.html("output.html")
    #st_profile_report(pr)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    add_vertical_space(2)
    if st.button('Press to use our Example "US Accidents" Dataset...'):
        # Example data
        @st.cache_data
        def load_data():
            a = pd.read_csv("US_Accidents_1000.csv")
            return a
        
        df = load_data()
        pr = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        pr.to_file(output_file="output.html")
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Dataset Profiling Report**')
        st.html("output.html")
        #st_profile_report(pr)
