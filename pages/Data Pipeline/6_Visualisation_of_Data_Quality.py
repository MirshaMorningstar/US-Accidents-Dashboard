import streamlit as st
import streamlit_extras as stex
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.dataframe_explorer import dataframe_explorer
import plotly.express as px
import warnings
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np

warnings.filterwarnings('ignore')


st.set_page_config(initial_sidebar_state="expanded",page_title= " Data Quality Visualisation ",
        menu_items={
         'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
         'Report a bug': "mailto:a.k.mirsha9@gmail.com",
         'About': "### This is an extremely cool web application built as a part of my Data Science Mini Project on the ' US Accidents Dataset '\n"
     },page_icon="analysis.png",layout="wide")



st.markdown('''
# **The Data Quality Visualisation Window**''')
add_vertical_space(2)
st.markdown(''':smile: This Window is specifically generated to create the data quality visual report in terms of identifying missing values, irregular cardinality and outliers :balloon: ''' )

add_vertical_space(3)

c1,c2,c3 = st.columns(3)
with c2:
    fl = st.file_uploader(":file_folder: Upload your file",type=(["csv","txt","xlsx","xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding = "ISO-8859-1")
else:
    df = pd.read_csv("US_Accidents_1000.csv", encoding = "ISO-8859-1")

add_vertical_space(3)

st.markdown("## Missing Values")

col1,col2 = st.columns([0.75,0.25])



with col1:
    st.markdown('#### Missing Values Heatmap')
    miss_heatmap = px.imshow(df.isnull(),title="Missing Values Heatmap",labels={'x':'Attributes','y':'Frquency counts'})
    
    st.plotly_chart(miss_heatmap,use_container_width=True)

with col2:
    st.markdown("#### Details")
    with st.expander("Missing Values by Attribute",expanded=True):
        st.write(pd.DataFrame({
            'Attribute Name': [x for x in df.columns],
            'Missing Values': [df[x].isnull().sum() for x in df.columns]
        }))


add_vertical_space(5)

miss1,miss2 = st.columns([0.5,0.5])

with miss1:
    st.markdown("#### Missing Values Pie Chart")
    add_vertical_space(1)
    num = st.select_slider("Select the number of attributes to Pie Chart missing values for", options=[1,2,3,4,5,6,7,8,9,10],value=4)
    missing_values_counts = df.isnull().sum().sort_values(ascending=False)
    top_missing_values = missing_values_counts.head(num)
    misspie = px.pie(values=top_missing_values, names=top_missing_values.index, title='Top {} Missing Values Attributes'.format(num))
    misspie.update_traces(marker=dict(colors=px.colors.qualitative.Set1))
    st.plotly_chart(misspie,use_container_width=True)
    st.markdown(" **Frequency reflecting Pie Chart for missing values** ")

with miss2:
    st.markdown("#### Missing Values Bar Chart")
    add_vertical_space(1)
    num = st.select_slider("Select the number of attributes to Bar Chart missing values for", options=[1,2,3,4,5,6,7,8,9,10],value=8)
    missing_values_counts = df.isnull().sum().sort_values(ascending=False)
    top_missing_values = missing_values_counts.head(num)
    missbar = px.bar(x=top_missing_values.index, y=top_missing_values.values, title='Top {} Missing Values Attributes'.format(num),labels={'x':"Attributes","y":"Frequency counts"})
    missbar.update_traces(marker=dict(color='rgb(158,202,225)', line=dict(color='rgb(8,48,107)', width=1.5)), selector=dict(type='bar'))
    st.plotly_chart(missbar,use_container_width=True)



add_vertical_space(5)
st.markdown("## Irregular Cardinality")

irr1, irr2 = st.columns([0.5, 0.5])

with irr1:
    st.markdown("#### Irregular Cardinality Pie Chart")
    add_vertical_space(1)
    num_pie = st.select_slider("Select the number of attributes for Pie Chart", options=list(range(1, len(df.columns) + 1)), value=8)
    cardinality = df.apply(lambda x: len(x.unique()), axis=0).sort_values(ascending=False)
    top_cardinality_pie = cardinality.head(num_pie)
    cardpie = px.pie(values=top_cardinality_pie, names=top_cardinality_pie.index, title=f'Top {num_pie} Irregular Cardinality Attributes')
    cardpie.update_traces(marker=dict(colors=px.colors.qualitative.Set1), textinfo='label+percent', textposition='inside')
    cardpie.update_layout(
        legend_title="Attributes")
    st.plotly_chart(cardpie, use_container_width=True)
    st.markdown("**Frequency reflecting Pie Chart for irregular cardinality**")

with irr2:
    st.markdown("#### Irregular Cardinality Bar Chart")
    add_vertical_space(1)
    num_bar = st.select_slider("Select the number of attributes for Bar Chart", options=list(range(1, len(df.columns) + 1)), value=23)
    cardinality = df.apply(lambda x: len(x.unique()), axis=0).sort_values(ascending=False)
    top_cardinality_bar = cardinality.head(num_bar)
    cardbar = px.bar(x=top_cardinality_bar.index, y=top_cardinality_bar.values, title=f'Top {num_bar} Irregular Cardinality Attributes', labels={'x': "Attributes", "y": "Cardinality"})
    cardbar.update_traces(marker=dict(color='rgb(158,202,225)', line=dict(color='rgb(8,48,107)', width=1.5)), selector=dict(type='bar'))
    st.plotly_chart(cardbar, use_container_width=True)


add_vertical_space(5)

st.markdown("## Outliers Detection and Visualisation")
add_vertical_space(2)

out1,out2 = st.columns([0.4,0.6])

with out1:
    st.markdown("##### Box Plot for Selected Attribute")
    attribute= st.selectbox("Select your desired attribute" , options= df.select_dtypes(include=['int', 'float']).columns,index=6)
    # Select numerical columns
    
    trace = go.Box(y=df[attribute], name=attribute)
    layout = go.Layout(title="Boxplot of {}".format(attribute),
                xaxis=dict(title="{}".format(attribute)),
                yaxis=dict(title="{}".format(attribute)))

    # Create figure
    fig = go.Figure(data=trace, layout=layout)
    st.plotly_chart(fig,use_container_width=True)


with out2:
    st.markdown("##### Violin Plot for Selected Attribute")
    attribute2= st.selectbox("Select your desired attribute" , options= df.select_dtypes(include=['int', 'float']).columns,index=6,key=2)
    # Select numerical columns
    
    fig = px.violin(df, x=attribute2,
                 title=f'Violin Plot for {attribute2}',
                 labels={'X': attribute2})

    # Update scatter plot style
    fig.update_traces(marker=dict(color='rgb(158, 202, 225)', size=10))
    st.plotly_chart(fig,use_container_width=True)


add_vertical_space(5)
st.markdown("#### The Overall Box Plot Visual for all the numerical attibutes")

numerical_columns = df.select_dtypes(include=['int', 'float']).columns

# Create traces for each numerical column
traces = []
for column in numerical_columns:
    trace = go.Box(y=df[column], name=column)
    traces.append(trace)

# Create layout
layout = go.Layout(title="Boxplot of All the Numerical Variables",
                xaxis=dict(title="Attributes"),
                yaxis=dict(title="Range Values"))

# Create figure
fig = go.Figure(data=traces, layout=layout)
st.plotly_chart(fig,use_container_width=True)
