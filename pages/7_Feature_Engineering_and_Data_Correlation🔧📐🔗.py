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

warnings.filterwarnings('ignore')



st.set_page_config(initial_sidebar_state="collapsed",page_title= " Feature Engineering and Data Correlation Analysis ",
        menu_items={
         'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
         'Report a bug': "mailto:a.k.mirsha9@gmail.com",
         'About': "### This is an extremely cool web application built as a part of my Data Science Mini Project on the ' US Accidents Dataset '\n"
     },page_icon="analysis.png",layout="wide")



st.markdown('''# **The Feature Engineering and Data Correlation Analysis Window**''')
add_vertical_space(2)
st.markdown(''':smile: This Window is specifically generated to carry out Feature Engineering and Feature Identification for predicting a target feature by visualizing hidden relationships and underlying complicated patterns. :balloon: ''' )

add_vertical_space(3)

c1,c2,c3 = st.columns(3)
with c2:
    fl = st.file_uploader(":file_folder: Upload your file",type=(["csv","txt","xlsx","xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding = "ISO-8859-1")
else:
    df = pd.read_csv("US_Norm.csv", encoding = "ISO-8859-1")

add_vertical_space(3)

st.markdown("## Correlation Tests Analysis")

numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()


# Calculate correlation coefficients for numerical features
correlation_matrix = df[numerical_features].corr()
correlation_with_target = correlation_matrix['Severity'].sort_values(ascending=False)

# Transpose the Series to display horizontally
correlation_with_target_transposed = correlation_with_target.to_frame().T

st.write("Correlation with target (numerical features):")
st.write(correlation_with_target_transposed)



metric_cards_per_column = 10  # Number of metric cards to display per column

# Display metric cards in side-by-side columns
st.write("Correlation with target (numerical features):")

num_features = len(correlation_with_target_transposed.columns)
num_columns = num_features // metric_cards_per_column + (num_features % metric_cards_per_column > 0)

columns = st.columns(num_columns)

card_counter = 0
summ = 0
for item in correlation_with_target:
    if not pd.isnull(item) and item != 0:
        print("Correlation score:", item)
        summ += abs(item)



for i in range(num_columns):
    with columns[i]:
        for j in range(metric_cards_per_column):
            if card_counter >= num_features:
                break
            feature = correlation_with_target_transposed.columns[card_counter]
            correlation_score = correlation_with_target_transposed.iloc[0, card_counter]
            delta = correlation_score - 1 if feature == "Severity" else correlation_score
            st.metric(label=feature, value="{:.2f}".format(correlation_score), delta="{:.2f}".format(delta/summ),)
            card_counter += 1


add_vertical_space(5)
st.markdown("### Chi Sqaure Tests for Categorical Data")
add_vertical_space(1)
# Perform Chi-square test for categorical features
categorical_features = ['Source',
 'Description',
 'Street',
 'City',
 'County',
 'State',
 'Zipcode',
 'Country',
 'Timezone',
 'Airport_Code',
 'Wind_Direction',
 'Weather_Condition',
 'Sunrise_Sunset',
 'Civil_Twilight',
 'Nautical_Twilight',
 'Astronomical_Twilight',
 'Severity']

chi2_results = chi2(df[categorical_features], df['Severity'])
categorical_scores = pd.Series(chi2_results[0], index=categorical_features)


x=categorical_scores.sort_values(ascending=False)

st.write(x.to_frame().T)

col1,col2 = st.columns([0.5,0.5])


with col1:
    st.markdown('#### Chi Sqaure Test Bar Chart')
    sorted_categorical_scores = categorical_scores.sort_values(ascending=False)

    # Define the top n attributes
    top_n = st.select_slider("Select the number of Attributes to visualise the Chi Square Bar results for", list(range(1, len(sorted_categorical_scores) + 1)),value = 9)  # You can change this value as needed

    # Select the top n attributes
    top_n_attributes = sorted_categorical_scores.head(top_n)

    # Create a bar chart for the top n attributes
    bar_chart = px.bar(x=top_n_attributes.index, y=top_n_attributes.values, labels={'x': 'Attribute', 'y': 'Chi-square Score'}, 
                    title=f'Top {top_n} Categorical Attributes and Their Chi-square Scores')
    bar_chart.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(bar_chart,use_container_width=True)


with col2:
    st.markdown("#### Chi Square Test Pie Chart")

    # Define the top n attributes
    top_n = st.select_slider("Select the number of Attributes to visualise the Chi Square Pie results for", list(range(1, len(sorted_categorical_scores) + 1)),value = 4,key=3)  # You can change this value as needed

    # Select the top n attributes
    top_n_attributes = sorted_categorical_scores.head(top_n)
    # Create a pie chart to visualize the distribution of chi-square scores among the top n attributes
    pie_chart = px.pie(values=top_n_attributes.values, names=top_n_attributes.index, 
                    title=f'Distribution of Chi-square Scores Among Top {top_n} Attributes')
    st.plotly_chart(pie_chart,use_container_width=True)

add_vertical_space(5)

st.markdown("### Select K Best Test for Numerical Attributes")
add_vertical_space(3)

c1,c2,c3 = st.columns(3)
with c2:
    k = st.number_input(label="Select the 'K' Value",min_value=3,max_value=10,value=5)

add_vertical_space(2)
# Perform SelectKBest test for numerical attributes using mutual information
select_k_best = SelectKBest(score_func=mutual_info_classif, k=k)  # You can change the value of k as needed
select_k_best.fit(df[numerical_features], df['Severity'])

# Get the scores and corresponding feature names
numerical_scores = pd.Series(select_k_best.scores_, index=numerical_features)

# Sort the numerical scores in descending order
sorted_numerical_scores = numerical_scores.sort_values(ascending=False)

st.write(sorted_numerical_scores.to_frame().T)

miss1,miss2 = st.columns([0.5,0.5])

with miss1:
    st.markdown("#### Select K Best Bar Chart")
    add_vertical_space(1)
    num = st.select_slider("Select the number of attributes to Bar Chart", options=list(range(1, len(sorted_numerical_scores) + 1)),value=23)
    bar_chart_numerical = px.bar(x=sorted_numerical_scores.index[:num], y=sorted_numerical_scores.values[:num],
                              labels={'x': 'Attribute', 'y': 'Score'}, 
                              title='Top {} Numerical Attributes and Their SelectKBest Scores'.format(num))
    bar_chart_numerical.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(bar_chart_numerical, use_container_width=True)

with miss2:
    st.markdown("#### Select K BestPie Chart")
    add_vertical_space(1)
    num = st.select_slider("Select the number of attributes to Pie Chart ", options=list(range(1, len(sorted_numerical_scores) + 1)),value=7)
    pie_chart_numerical = px.pie(values=sorted_numerical_scores.values[:num], names=sorted_numerical_scores.index[:num], 
                               title='Distribution of SelectKBest Scores Among Top {} Numerical Attributes'.format(num))  
    st.plotly_chart(pie_chart_numerical, use_container_width=True)



add_vertical_space(5)
st.markdown("## Principal Component Analysis")

irr1, irr2 = st.columns([0.5, 0.5])

with irr1:
    st.markdown("#### PCA Correlation Circle Graph")
    add_vertical_space(1)

    float_columns = df.select_dtypes(include=['float']).columns
    
    # Select only columns with floating-point data
    float_columns = df.select_dtypes(include=['float']).columns
    # Extract features and target variable
    X = df[float_columns] # Features
    y = df['Severity'] # Target
    # Standardize the features
    
    scaling = st.checkbox("Whether you want to scale the data before PCA ?",value = True)
    if scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    num1_bar = st.select_slider("Select the number of attributes for the Principal Component Analysis", options=list(range(1, len(X.columns) + 1)), value=8)

    # Perform PCA
    pca = PCA(n_components=num1_bar) # Specify the number of components to keep
    X_pca = pca.fit_transform(X_scaled)


    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Calculate the correlation between original features and principal components
    correlation_matrix = pd.DataFrame(pca.components_, columns=float_columns)
    correlation_matrix.index = ['PC' + str(i) for i in range(1, len(correlation_matrix) + 1)]

    
  
    fig, ax = plot_pca_correlation_graph(X, float_columns,
                                        dimensions=(1, 2))

    # Display the plot using Streamlit
    st.pyplot(fig,use_container_width=True)

import plotly.graph_objects as go

with irr2:
    st.markdown("#### PCA analysis Bar Chart")
    add_vertical_space(1)
    num_bar = st.select_slider("Select the number of attributes for Bar Chart", options=list(range(1, len(X.columns) + 1)), value=8)
    
    #pl=plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, color='skyblue')
    fig = px.bar(x=range(len(pca.explained_variance_ratio_[:num_bar])), y=pca.explained_variance_ratio_[:num_bar], labels={'x': 'Principal Components', 'y': 'Magnitude Contribution of variance'}, 
                    title=f'Top {top_n} Principal Component Analysis Variance Ratio Distribution')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

add_vertical_space(5)

st.markdown("## Correlation Heatmap")
add_vertical_space(2)


numerical = df.select_dtypes(include=['float','int']).columns
important = ['Start_Lat','End_Lat','Start_Lng','End_Lng','Temperature(F)','Distance(mi)','Wind_Chill(F)', 'Humidity(%)', 'Wind_Speed(mph)', 'Pressure(in)']


df = pd.read_csv('US_Norm.csv')

c1,c2,c3 = st.columns(3)
with c2:
    colorscales = px.colors.named_colorscales()
    palette = st.selectbox("Select a color palette : ",options= colorscales,index=31)
add_vertical_space(2)


c1,c2 = st.columns(2)

with c1:
    st.markdown('#### Correlation Heatmap of Numerical Attributes')
    fig = px.imshow(df[numerical].corr(),color_continuous_scale=palette)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown('#### Correlation Heatmap of The Most Important Features')
    fig = px.imshow(df[important].corr(),text_auto=True,color_continuous_scale=palette)
    st.plotly_chart(fig, use_container_width=True)
