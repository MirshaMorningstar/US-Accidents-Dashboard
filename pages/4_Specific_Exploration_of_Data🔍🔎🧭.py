import numpy as np
import pandas as pd
import sklearn.datasets
import streamlit as st
import streamlit_extras as stex
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.dataframe_explorer import dataframe_explorer
from sklearn import datasets

import streamlit.components.v1 as components
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

st.set_page_config(initial_sidebar_state="collapsed", page_title="The Data Frames Explorer Window",
        menu_items={
         'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
         'Report a bug': "mailto:a.k.mirsha9@gmail.com",
         'About': "### This is an extremely cool web application built as a part of my Data Science Mini Project on the ' US Accidents Dataset '\n"
     },
     page_icon="exploratory-analysis (1).png")

st.title("Welcome to the Data Frame Explorer Window")

st.markdown("##### This Window provides you with a highly interactive and customisable method to explore our US Accidents dataset")

add_vertical_space(2)

st.markdown("###### Feel free to explore and filter out our dataset...")
add_vertical_space(2)


df = pd.read_csv("US_Accident23_1000.csv")
filtered_df = dataframe_explorer(df, case=False)
st.dataframe(filtered_df, use_container_width=True)

      
        