import streamlit as st
import streamlit_extras as stex
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.dataframe_explorer import dataframe_explorer
import plotly.express as px
import warnings
import pandas as pd
from mlxtend.plotting import plot_pca_correlation_graph
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np
import math
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import plotly.figure_factory as ff
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Data Pre-Processing and Preparation for Exploration",
    menu_items={
        'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
        'Report a bug': "mailto:a.k.mirsha9@gmail.com",
        'About': "### This is an extremely cool web application built as a part of my Data Science Mini Project on the 'US Accidents Dataset'\n"
    },
    page_icon="analysis.png",
    layout="wide"
)

st.markdown('# **The Data Preprocessing and Preparation Window**')
add_vertical_space(2)
st.markdown('##### :smile: This Window is specifically generated to carry out Data Preparation and Preprocessing Techniques such as Normalisation, Binning, and Sampling for Further Exploration. :balloon:')

add_vertical_space(3)

c1, c2, c3 = st.columns(3)
with c2:
    fl = st.file_uploader(":file_folder: Upload your file", type=(["csv", "txt", "xlsx", "xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding="ISO-8859-1")
else:
    df = pd.read_csv("US_Accidents_1000.csv", encoding="ISO-8859-1")

add_vertical_space(3)

st.markdown('## Data Normalisation Process')
add_vertical_space(2)
st.markdown('''**Feature Scaling is an essential step in the data analysis and preparation of data for modeling. Wherein, we make the data scale-free for easy analysis.**

**Normalization is one of the feature scaling techniques. We particularly apply normalization when the data is skewed on either axis i.e. when the data does not follow the Gaussian distribution.**''')

add_vertical_space(5)

numerical_columns = df.select_dtypes(include=['int', 'float']).columns
print(numerical_columns)

data = df.dropna()



c1, c2 = st.columns(2)

with c1:
    st.markdown('##### Min-Max Normalisation')
    st.write("**The Normalised Data strictly lies between 0 to 1**")
    add_vertical_space(2)

    attribute = st.selectbox("Select the Attribute to visualise the Min-Max Normalisation for", list(numerical_columns), index=6)
    
    # Normalize data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data[[attribute]])
    add_vertical_space(3)

    st.markdown("##### Visual inspection")
    add_vertical_space(1)

# Displaying the plots in Streamlit
    fig = px.histogram(data[attribute], title='Original Data', color_discrete_sequence=px.colors.sequential.Plotly3)
    wig = px.histogram(data_normalized.flatten(), title='Normalised Data', color_discrete_sequence=px.colors.sequential.Plotly3)
    st.plotly_chart(fig, use_container_width=True, key = "fig")
    st.plotly_chart(wig, use_container_width=True, key = "wig")

    add_vertical_space(3)

    st.markdown("##### Statistical test for normality")
    add_vertical_space(1)
    st.write(stats.shapiro(data_normalized))

    add_vertical_space(3)

    st.markdown("##### Evaluate model performance")
    model = RandomForestClassifier()
    scores_original = cross_val_score(model, data[[attribute]], data['Severity'], cv=2)
    scores_normalized = cross_val_score(model, data_normalized, data['Severity'], cv=2)

    st.write("Original Data Scores:", scores_original.mean())
    st.write("Normalized Data Scores:", scores_normalized.mean())

with c2:
    st.markdown('##### Standard Normalisation')
    st.write("**The Normalised Data strictly has the Mean of 0 and Standard Deviation of approximately 1**")
    add_vertical_space(2)

    attribute = st.selectbox("Select the Attribute to visualise the Standardisation for", list(numerical_columns), index=6)
    
    # Normalize data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data[[attribute]])
    add_vertical_space(3)

    st.markdown("##### Visual inspection")
    add_vertical_space(1)
    
    # Displaying the plots in Streamlit
    fig = px.histogram(data[attribute], title='Original Data', color_discrete_sequence=px.colors.sequential.Plotly3)
    wig = px.histogram(data_normalized.flatten(), title='Normalised Data', color_discrete_sequence=px.colors.sequential.Plotly3)
    st.plotly_chart(fig, use_container_width=True, key = "fig2")
    st.plotly_chart(wig, use_container_width=True, key = "wig2")

    add_vertical_space(3)

    st.markdown("##### Statistical test for normality")
    add_vertical_space(1)
    st.write(stats.shapiro(data_normalized))

    add_vertical_space(3)

    st.markdown("##### Evaluate model performance")
    model = RandomForestClassifier()
    scores_original = cross_val_score(model, data[[attribute]], data['Severity'], cv=5)
    scores_normalized = cross_val_score(model, data_normalized, data['Severity'], cv=5)

    st.write("Original Data Scores:", scores_original.mean())
    st.write("Normalized Data Scores:", scores_normalized.mean())

add_vertical_space(5)
st.markdown('### Statistical Method Normalisation Verification')
w_statistic, p_value = stats.shapiro(data_normalized)
st.write("Normal Data - W statistic:", w_statistic, "P-value:", p_value)
# Test for non-normal data
w_statistic_non_normal, p_value_non_normal = stats.shapiro(data['Temperature(F)'])
st.write("Non-Normal Data - W statistic:", w_statistic_non_normal, "P-value:", p_value_non_normal)

add_vertical_space(3)

st.write('''**Interpreting the Results**
         
**● W statistic: Closer to 1 indicates data is more likely to be normal.**
         
**● P-value: A p-value greater than the chosen alpha level (commonly set at 0.05) suggests that the null hypothesis of normality cannot be rejected. A small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis, so you reject the null hypothesis of normality**''')

add_vertical_space(5)
st.markdown("### Quantile-Quantile Plot")

add_vertical_space(3)

# Create Q-Q plots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Q-Q plot for non-normalized data
stats.probplot(data['Temperature(F)'], dist="norm", plot=axes[0])
axes[0].set_title('Q-Q Plot for Non-Normalized Data')

# Q-Q plot for normalized data
stats.probplot(data_normalized.flatten(), dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot for Normalized Data')

# Displaying the plots in Streamlit
st.pyplot(fig)
add_vertical_space(5)

st.markdown("### Interactive Binning")
add_vertical_space(3)

# Function to perform binning
def perform_binning(data, column_name, labels, value_ranges):
    bins = pd.cut(data[column_name], bins=value_ranges, labels=labels)
    data['{}_Bin'.format(column_name)] = bins
    return data

def plot_binned_column(data, column_name, title):
    bar = px.bar(x=data[column_name].value_counts().index, y=data[column_name].value_counts().values,
                 color_discrete_sequence=px.colors.sequential.Plotly3)
    st.plotly_chart(bar,key = "bar")

st.subheader('Binning Options')
add_vertical_space(2)
bin_col_name = st.selectbox("Select the Numerical Attribute to visualise the Binning for", list(numerical_columns), index=6)
add_vertical_space(1)
if bin_col_name in df.columns:
    labels_input = st.text_input("Enter bin labels (comma-separated):", value='very cold,cold,moderate,warm,hot')
    add_vertical_space(1)
    labels = [label.strip() for label in labels_input.split(',')]
    value_ranges_input = st.text_input("Enter value ranges for bins (comma-separated):", value='-100,32,50,70,90,200')
    value_ranges = [float(val.strip()) for val in value_ranges_input.split(',')]
    add_vertical_space(2)
    if st.button('Perform Binning'):
        data = perform_binning(df, bin_col_name, labels, value_ranges)
        add_vertical_space(1)
        st.write('Data after binning:')
        add_vertical_space(1)
        st.write(data.head())
        add_vertical_space(2)
        st.write('Distribution of Bins:')
        add_vertical_space(1)
        plot_binned_column(data, '{}_Bin'.format(bin_col_name), 'Distribution of {} Bins'.format(bin_col_name))
