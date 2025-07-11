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
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap

warnings.filterwarnings('ignore')



st.set_page_config(initial_sidebar_state="expanded",page_title= " General Data Visualisation Window ",
        menu_items={
         'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
         'Report a bug': "mailto:a.k.mirsha9@gmail.com",
         'About': "### This is an extremely cool web application built as a part of my Data Science Mini Project on the ' US Accidents Dataset '\n"
     },page_icon="analysis.png",layout="wide")


st.markdown('''# **A General Data Visualisation Window**''')
add_vertical_space(2)
st.markdown('''##### :smile: This Window is specifically generated to carry out Generalised Visualisations on the data and grasping a strong knowledge on the same :balloon: ''' )

add_vertical_space(3)

c1,c2,c3 = st.columns(3)
with c2:
    fl = st.file_uploader(":file_folder: Upload your file",type=(["csv","txt","xlsx","xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding = "ISO-8859-1")
else:
    df = pd.read_csv("US_Accidents53.csv", encoding = "ISO-8859-1")

add_vertical_space(3)

add_vertical_space(3)

st.markdown("### Some General Visualisations")
add_vertical_space(3)

c1,c2 = st.columns(2)
with c1:
    fig = px.box(df, x='Severity', y='Temperature(F)', color='Severity', title='Temperature vs. Severity of Accidents')
    st.plotly_chart(fig,use_container_width=True)

with c2:
    fig = px.histogram(df, x='Hour', nbins=24, color_discrete_sequence=['orange'], labels={'Hour': 'Hour of the Day', 'count': 'Frequency'}, title='Histogram of Accident Times')
    fig.update_layout(xaxis_title='Hour of the Day', yaxis_title='Frequency', title_font_size=16, showlegend=False)
    st.plotly_chart(fig,use_container_width=True)

add_vertical_space(5)

df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Weekday'] = df['Start_Time'].dt.day_name()
c1,c2 = st.columns(2)
# Fix the pie chart plotting
with c1:
    st.markdown('##### Accidents by Day of the Week')
    week_day_counts = df['Weekday'].value_counts()

    pie_df = pd.DataFrame({
        'Weekday': week_day_counts.index,
        'Count': week_day_counts.values
    })

    fig = px.pie(pie_df, names='Weekday', values='Count', 
                title='Accidents by Day of the Week',
                color_discrete_sequence=['skyblue', 'lightgreen', 'lightcoral', 'orange', 'lightyellow', 'lightpink', 'lightskyblue'])

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(title_font_size=16, title_font_color='purple')

    st.plotly_chart(fig, use_container_width=True)


with c2:
    # Scatter plot using Plotly Express
    st.markdown('##### Wind Speed vs. Visibility')
    fig = px.scatter(df, x='Wind_Chill(F)', y='Visibility(mi)', color='Visibility(mi)', 
                    title='Wind Chill vs. Visibility', labels={'Wind_Chill(F)': 'Wind Chill (F)', 'Visibility(mi)': 'Visibility (miles)'})
    fig.update_traces(marker=dict(size=10, opacity=0.5), selector=dict(mode='markers'))
    fig.update_layout(title_font_size=16, title_font_color='brown', showlegend=False)
    # Update layout to display labels around the pie chart
    
    st.plotly_chart(fig,use_container_width=True)
    

add_vertical_space(5)

c1,c2 = st.columns(2)
with c1:
    st.markdown("##### Scatter Plot of Wind Speed Vs. Visibility")
    # Scatter plot using Plotly Express
    fig = px.scatter(df, x='Wind_Speed(mph)', y='Visibility(mi)', color='Visibility(mi)',
                     title='Wind Speed vs. Visibility', labels={'Wind_Speed(mph)': 'Wind Speed (mph)', 'Visibility(mi)': 'Visibility (miles)'})

    # Update layout
    fig.update_layout(title_font_size=16, title_font_color='blue',
                      xaxis_title='Wind Speed (mph)', yaxis_title='Visibility (miles)',
                      xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'),
                      yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'))

    st.plotly_chart(fig, use_container_width=True)

with c2:
    # Convert 'Start_Time' column to datetime and set it as index
    st.markdown("##### Trends Line Chart")
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df.set_index('Start_Time', inplace=True)

    # Resample the data by month and plot
    monthly_accidents = df.resample('M').size()
    fig = px.line(x=monthly_accidents.index, y=monthly_accidents.values,
                  title='Accidents Over Time', labels={'x': 'Date', 'y': 'Number of Accidents'})
    fig.update_traces(line=dict(color='hotpink', dash='dot', width=2))
    fig.update_layout(title_font_size=16, title_font_color='hotpink',
                      xaxis_title_font_size=12, yaxis_title_font_size=12,
                      xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'),
                      yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'))

    st.plotly_chart(fig, use_container_width=True)
    
add_vertical_space(5)

c1,c2 = st.columns(2)
with c1:
    st.markdown("##### Count Plot of Accident Severity")
    # Count plot using Plotly Express

    fig = px.histogram(df, x='Severity', title='Distribution of Accident Severity', color_discrete_sequence=['orange'], 
                    labels={'Severity': 'Severity'})
    fig.update_layout(title_font_size=16, title_font_color='orange', 
                    xaxis_title='Severity', yaxis_title='Count', 
                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'))

    st.plotly_chart(fig, use_container_width=True)

with c2:
    # Convert 'Start_Time' column to datetime and set it as index
    # Weather Condition Impact
    st.markdown('##### Accidents by Weather Condition')
    fig_weather = px.bar(df['Weather_Condition'].value_counts(), 
                        orientation='h',
                        title='Accidents by Weather Condition',
                        labels={'value': 'Number of Accidents', 'index': 'Weather Condition'},
                        color_discrete_sequence=['red'],
                        height=500)
    fig_weather.update_layout(title_font_size=16, title_font_color='red', 
                            yaxis_title='Weather Condition', xaxis_title='Number of Accidents')

    st.plotly_chart(fig_weather, use_container_width=True)
    
    
c1,c2 = st.columns(2)
with c1:
    # Accident Hotspots by City
    city_counts = df['City'].value_counts().head(10)
    st.markdown('##### Top 10 Cities by Accident Counts')
    fig_city = px.bar(city_counts, 
                    title='Top 10 Cities by Accident Counts',
                    labels={'value': 'Number of Accidents', 'index': 'City'},
                    color_discrete_sequence=['purple'],
                    height=500)
    fig_city.update_layout(title_font_size=16, title_font_color='purple', 
                        yaxis_title='Number of Accidents', xaxis_title='City')

    st.plotly_chart(fig_city, use_container_width=True)

df = pd.read_csv('US_Accidents53.csv')

with c2:
        # Set a custom color palette
    custom_palette = sns.color_palette("Set2")
    sns.set_palette(custom_palette)

    # Set the seaborn style
    sns.set_style("whitegrid")

    # 1. Distribution of Accidents by Month
    st.markdown('##### Distribution of Accidents by Month')
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Month', data=df, ax=ax1)
    ax1.set_title('Distribution of Accidents by Month')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Number of Accidents')
    st.pyplot(fig1,use_container_width=True)

add_vertical_space(5)
c1,c2 = st.columns(2)

add_vertical_space(5)
with c1:
    # 2. Accident Duration Analysis with Log Transformation
    st.markdown('##### Box Plot of Log-Transformed Accident Duration')
    
    # Calculate duration in hours
    df['Duration'] = (pd.to_datetime(df['End_Time']) - pd.to_datetime(df['Start_Time'])).dt.total_seconds() / 3600
    
    # Apply log1p transformation to handle zero safely
    df['Log_Duration'] = np.log1p(df['Duration'])

    # Plotting
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Log_Duration', data=df, ax=ax2)
    ax2.set_title('Box Plot of Log-Transformed Accident Duration')
    ax2.set_xlabel('Log(Duration in Hours)')
    st.pyplot(fig2, use_container_width=True)


with c2:
    # 3. Scatter Plot of Pressure vs. Humidity
    st.markdown('##### Pressure vs. Humidity')
    fig = px.scatter(df, x='Pressure(in)', y='Humidity(%)')
    fig.update_layout(xaxis_title='Atmospheric Pressure (in)', yaxis_title='Humidity (%)')
    fig.update_traces(marker=dict(color='rgba(0,0,0,0)'))  # Make the scatter markers transparent
    fig.update_layout(barmode='overlay')

    # Marginal X bar plot
    fig.add_trace(px.bar(df, x='Pressure(in)').data[0])

    # Marginal Y bar plot
    fig.add_trace(px.bar(df, y='Humidity(%)').data[0])
    st.plotly_chart(fig,use_container_width=True)


add_vertical_space(5)
# Set different color palettes
palette1 = px.colors.qualitative.Pastel
palette2 = px.colors.qualitative.Set3
palette3 = px.colors.sequential.Viridis
c1,c2 = st.columns(2)


with c1:
    # 6. Day-Night Accident Comparison
    st.markdown('##### Day vs. Night Accidents')
    fig2 = px.bar(df['Sunrise_Sunset'].value_counts(), x=df['Sunrise_Sunset'].value_counts().index, y=df['Sunrise_Sunset'].value_counts().values, 
                title='Day vs. Night Accidents', color_discrete_sequence=palette1)
    fig2.update_layout(xaxis_title='Time of Day', yaxis_title='Number of Accidents')
    st.plotly_chart(fig2)

with c2:
    wata = pd.read_csv("US_Accident23_1000.csv")
    # 7. Impact of Visibility on Accident Severity
    st.markdown('##### Visibility vs. Severity')
    fig3 = px.violin(wata, x='Severity', y='Visibility(mi)', title='Visibility vs. Severity', color='Severity', color_discrete_sequence=palette1)
    fig3.update_layout(xaxis_title='Severity', yaxis_title='Visibility (miles)')
    st.plotly_chart(fig3)


add_vertical_space(5)

c1,c2 = st.columns([0.4,0.6])
with c1:
    st.markdown("##### Rose Plot for Accidents and Wind Direction")
    add_vertical_space(2)
    # Generate sample data for demonstration (replace with your actual data)
    wind_directions = np.random.randint(0, 360, size=len(df))  # Wind directions in degrees
    accident_severity = df['Severity']  # Accident severity levels

    # Create a rose plot
    num_bins = 36  # Number of bins (36 for each 10 degrees)
    theta = np.linspace(0.0, 2 * np.pi, num_bins, endpoint=False)
    radii, _ = np.histogram(wind_directions, bins=num_bins)
    colors = plt.cm.viridis(accident_severity / max(accident_severity))  # Color mapping based on severity

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    bars = ax.bar(theta, radii, width=2 * np.pi / num_bins, color=colors, edgecolor='black')

    plt.title('Wind Direction vs. Accident Severity')
    st.pyplot(fig,use_container_width=True)

with c2 :
    st.markdown("##### Kernel Density Accidents in Folium Map")
    add_vertical_space(2)
    from streamlit_folium import folium_static
    import folium
    # Create a map centered around an average location
    map_center = [df['Start_Lat'].mean(), df['Start_Lng'].mean()]
    map = folium.Map(location=map_center, zoom_start=5)

    # Add a heatmap layer
    heatmap = HeatMap(list(zip(df['Start_Lat'], df['Start_Lng'])), min_opacity=0.2, radius=15, blur=15)
    map.add_child(heatmap)

    # Save the map as an HTML file
    map.save('accident_heatmap.html')
    print('Heatmap generated and saved as accident_heatmap.html.')
    folium_static(map)
