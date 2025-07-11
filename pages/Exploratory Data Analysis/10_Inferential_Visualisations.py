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



st.set_page_config(initial_sidebar_state="collapsed",page_title= " Inferential Data Visualisation Window ",
        menu_items={
         'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
         'Report a bug': "mailto:a.k.mirsha9@gmail.com",
         'About': "### This is an extremely cool web application built as a part of my Data Science Mini Project on the ' US Accidents Dataset '\n"
     },page_icon="analysis.png",layout="wide")


st.title(''' **The Inferential Data Visualisation Window**''')
add_vertical_space(2)
st.markdown('''##### :smile: This Window is specifically generated to carry out Highly Meaningful and Inferential Visualisations on the data and arriving at Conclusions on the same. :balloon: ''' )

add_vertical_space(3)

c1,c2,c3 = st.columns(3)
with c2:
    fl = st.file_uploader(":file_folder: Upload your file",type=(["csv","txt","xlsx","xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding = "ISO-8859-1")
else:
    df = pd.read_csv(r"US.csv", encoding = "ISO-8859-1")
    df = df.sample(n=10000,random_state = 42)

add_vertical_space(3)

c2,c3 = st.columns([5,4.75],gap='medium')
with c2:
    st.header("Identification of the 10 accident-prone streets in USA.")
    add_vertical_space(3)
    
    # Calculate the top 10 streets with the most number of accidents
    top_10_streets = df['Street'].value_counts().head(10).reset_index()
    top_10_streets.columns = ['Street', 'Cases']

    # Calculate total cases for annotation purposes
    total = sum(df['Street'].value_counts())

    # Streamlit app

    # Create a bar plot using Plotly Express
    fig = px.bar(
        top_10_streets,
        x='Street',
        y='Cases',
        color='Cases',
        color_continuous_scale='rainbow',
        title='Top 10 Accident-Prone Streets in the US'
    )

    # Annotate each bar with percentage
    fig.update_traces(texttemplate='%{y} cases<br>%{text:.2f}%', text=[(value / total) * 100 for value in top_10_streets['Cases']], textposition='outside')

    # Update layout for better aesthetics
    fig.update_layout(
        xaxis_title='Streets',
        yaxis_title='Accident Cases',
        title_font_size=20,
        title_font_color='grey',
        xaxis_tickangle=-45,
        xaxis_tickfont_size=12,
        yaxis_tickfont_size=12,
        plot_bgcolor='white'
    )

    # Show grid lines for the y-axis
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#b2d6c7')

    # Display the plot in the Streamlit app
    st.plotly_chart(fig)
    add_vertical_space(5)

with c3:

    
    st.write('''## Inference:

#### Top 10 Accident-Prone Streets in the US

1. **Most Accident-Prone Streets**:
   - **I-95 S** is the most accident-prone street, with **126 accident cases**.
   - **I-95 N** follows closely with **122 accident cases**.

2. **Significant Streets**:
   - **I-5 N** shows a notable number of accidents, totaling **75 cases**.
   - **I-94 W** and **I-80 W** also have significant accident cases, with **63** and **62 cases** respectively.

3. **Other Notable Streets**:
   - **I-25 N**, **I-70 W**, and **I-80 E** each have **60**, **56**, and **56 accident cases**, respectively.
   - **I-10 E** and **I-5 S** show **55** and **53 accident cases**, respectively.

4. **Visual Appeal**:
   - The color scale from red to purple effectively highlights the distribution of accidents, with red indicating higher accident counts and purple indicating lower counts, enhancing visual understanding of the most accident-prone streets.

Overall, the plot identifies that **I-95 S** and **I-95 N** are the streets with the highest number of road accidents, suggesting a need for targeted safety measures on these routes. This insight can inform traffic safety initiatives and preventive measures to reduce accidents on these critical streets.''')

with c2:
    add_vertical_space(4)
    # Calculate the number of accidents per hour
    accidents_per_hour = df['Hour'].value_counts().sort_index().reset_index()
    accidents_per_hour.columns = ['Hour', 'Accidents']

    # Streamlit app
    st.header('Number of Road Accidents by Hour of the Day in the US')

    # Create a bar plot using Plotly Express
    fig = px.bar(
        accidents_per_hour,
        x='Hour',
        y='Accidents',
        color='Accidents',
        color_continuous_scale='viridis',
        title='Number of Road Accidents by Hour of the Day in the US'
    )

    # Update layout for better aesthetics
    fig.update_layout(
        xaxis_title='Hour of the Day',
        yaxis_title='Number of Accidents',
        title_font_size=20,
        title_font_color='grey',
        xaxis_tickmode='array',
        xaxis_tickvals=list(range(24)),
        xaxis_ticktext=[f'{i}:00' for i in range(24)],
        xaxis_tickangle=-45,
        xaxis_tickfont_size=12,
        yaxis_tickfont_size=12,
        plot_bgcolor='white'
    )

    # Show grid lines for the y-axis
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#b2d6c7')

    # Display the plot in the Streamlit app
    st.plotly_chart(fig)

    # Print the hour with the most accidents
    peak_hour = accidents_per_hour.loc[accidents_per_hour['Accidents'].idxmax(), 'Hour']
    peak_accidents = accidents_per_hour.loc[accidents_per_hour['Accidents'].idxmax(), 'Accidents']
    st.write(f"The hour with the most road accidents is {peak_hour}:00 with {peak_accidents} accidents.")


with c3:
    add_vertical_space(3)
    
    st.write('''## Inference:

#### Number of Road Accidents by Hour of the Day in the US

1. **Peak Accident Hour**:
   - **16:00** is the hour with the most road accidents, totaling **840 accidents**.

2. **High Accident Period**:
   - There is a significant increase in accidents between **15:00 and 18:00**, indicating a peak period likely corresponding to afternoon and evening rush hours.

3. **Morning Increase**:
   - A noticeable increase in accidents starts around **6:00**, with a steady rise through the morning hours, peaking again around **8:00**.

4. **Lower Accident Periods**:
   - Early morning hours (**0:00 to 5:00**) and late evening hours (**20:00 to 23:00**) have relatively fewer accidents.

5. **Visual Appeal**:
   - The 'viridis' color scale effectively highlights the distribution of accidents, with brighter colors indicating higher accident counts, enhancing visual understanding of critical times.

Overall, the plot clearly identifies the times of day with the highest accident rates, particularly focusing on afternoon rush hours, suggesting the need for targeted traffic management during these peak periods.''')

with c2:
    add_vertical_space(8)
        # Count the number of accidents for each weather condition
    weather_counts = df['Weather_Condition'].value_counts().head(10).reset_index()
    weather_counts.columns = ['Weather_Condition', 'Cases']

    # Streamlit app
    st.header('Top 10 Weather Conditions During Road Accidents in the US')

    # Create a bar plot using Plotly Express
    fig = px.bar(
        weather_counts,
        x='Weather_Condition',
        y='Cases',
        color='Cases',
        color_continuous_scale='viridis',
        title='Top 10 Weather Conditions During Road Accidents in the US'
    )

    # Update layout for better aesthetics
    fig.update_layout(
        xaxis_title='Weather Condition',
        yaxis_title='Number of Accidents',
        title_font_size=20,
        title_font_color='grey',
        xaxis_tickangle=-45,
        xaxis_tickfont_size=12,
        yaxis_tickfont_size=12,
        plot_bgcolor='white'
    )

    # Show grid lines for the y-axis
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#b2d6c7')

    # Display the plot in the Streamlit app
    st.plotly_chart(fig)

    # Print the weather condition with the most accidents
    top_weather_condition = weather_counts.iloc[0]
    top_weather = top_weather_condition['Weather_Condition']
    top_cases = top_weather_condition['Cases']
    st.write(f"The weather condition with the most road accidents is {top_weather} with {top_cases} accidents.")

with c3:
    add_vertical_space(3)
    st.write('''## Inference:

#### Top 10 Weather Conditions During Road Accidents in the US

1. **Most Accident-Prone Weather Condition**:
   - **Fair** is the weather condition with the most road accidents, totaling **4367 accidents**.

2. **Significant Weather Conditions**:
   - **Cloudy** and **Mostly Cloudy** conditions also show a considerable number of accidents, though significantly fewer than Fair weather.
   - **Partly Cloudy** and **Light Rain** conditions follow, contributing to the overall accident count but at a lower frequency.

3. **Less Frequent Conditions**:
   - **Light Snow**, **Fog**, **Rain**, **Fair/Windy**, and **Haze** have relatively fewer accidents compared to the top conditions.

4. **Visual Appeal**:
   - The 'viridis' color scale effectively highlights the distribution of accidents, with brighter colors indicating higher accident counts, enhancing visual understanding of critical weather conditions.

Overall, the plot helps identify that most road accidents occur under **Fair** weather conditions, suggesting that even in seemingly safe conditions, vigilance is necessary. This insight can inform safety campaigns and preventive measures across varying weather conditions.''')

with c2:
    add_vertical_space(9)
    # Convert 'Start_Time' and 'End_Time' columns to datetime objects
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['End_Time'] = pd.to_datetime(df['End_Time'])
        # Calculate accident duration
    df['Accident_Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 3600  # Convert duration to hours

    # Get the top 10 accident durations
    top_10_accident_duration = df['Accident_Duration'].value_counts().head(10).reset_index()
    top_10_accident_duration.columns = ['Duration (hours)', 'Number of Cases']

    # Streamlit app
    st.header('Top 10 Accident Durations')

    # Create a bar plot using Plotly Express
    fig = px.bar(
        top_10_accident_duration,
        x='Duration (hours)',
        y='Number of Cases',
        color='Number of Cases',
        color_continuous_scale='viridis',
        title='Top 10 Accident Durations'
    )

    # Update layout for better aesthetics
    fig.update_layout(
        xaxis_title='Duration (hours)',
        yaxis_title='Number of Cases',
        title_font_size=20,
        title_font_color='grey',
        xaxis_tickangle=-45,
        xaxis_tickfont_size=12,
        yaxis_tickfont_size=12,
        plot_bgcolor='white'
    )

    # Show grid lines for the y-axis
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#b2d6c7')

    # Display the plot in the Streamlit app
    st.plotly_chart(fig)

    # Print the duration with the most accidents
    top_duration_condition = top_10_accident_duration.iloc[0]
    top_duration = top_duration_condition['Duration (hours)']
    top_cases = top_duration_condition['Number of Cases']
    st.write(f"The accident duration with the most cases is {top_duration} hours with {top_cases} cases.")

with c3:
    add_vertical_space(1)
    st.write('''## Inference:

#### Top 10 Accident Durations

1. **Most Frequent Accident Duration**:
   - The duration with the most cases is **0.25 hours**, totaling **1244 cases**.

2. **Significant Durations**:
   - Durations around **0.5 hours** and **1 hour** also show a considerable number of accidents, indicating a high frequency of shorter duration accidents.

3. **Less Frequent Durations**:
   - Durations extending up to **6 hours** are observed but with significantly fewer cases compared to shorter durations.

4. **Visual Appeal**:
   - The 'viridis' color scale effectively highlights the distribution of accidents, with brighter colors indicating higher accident counts, enhancing visual understanding of critical accident durations.

Overall, the plot helps identify that most accidents are resolved within a short period, typically **0.25 hours**. This insight can inform traffic management and emergency response strategies to improve road safety and reduce accident durations.''')


with c2:
    add_vertical_space(5)
    # Convert 'Start_Time' column to datetime object
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])

    # Extract year from 'Start_Time'
    df['Year'] = df['Start_Time'].dt.year

    # Count the number of accident cases for each year
    accident_cases_by_year = df['Year'].value_counts().sort_index().reset_index()
    accident_cases_by_year.columns = ['Year', 'Number of Cases']

    # Streamlit app
    st.header('Accident Cases Over the Years')

    # Create a line plot using Plotly Express
    fig = px.line(
        accident_cases_by_year,
        x='Year',
        y='Number of Cases',
        markers=True,
        title='Accident Cases Over the Years'
    )

    # Update layout for better aesthetics
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Number of Cases',
        title_font_size=24,
        title_font_color='grey',
        
        xaxis_tickangle=-45,
        xaxis_tickfont_size=14,
        yaxis_tickfont_size=14,
        plot_bgcolor='white',
        font=dict(family='Arial', size=12, color='black')
    )

    # Show grid lines
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#b2d6c7')

    # Add a horizontal line to show the mean number of cases
    mean_cases = accident_cases_by_year['Number of Cases'].mean()
    fig.add_hline(y=mean_cases, line_dash='dash', line_color='red', annotation_text='Mean', 
                annotation_position='bottom right')

    # Display the plot in the Streamlit app
    st.plotly_chart(fig)

    # Print the year with the most accidents
    top_year_condition = accident_cases_by_year.iloc[accident_cases_by_year['Number of Cases'].idxmax()]
    top_year = top_year_condition['Year']
    top_cases = top_year_condition['Number of Cases']
    st.write(f"The year with the most accident cases is {top_year} with {top_cases} cases.")


with c3:
    add_vertical_space(2)
    st.write('''## Inference:

#### Accident Cases Over the Years

1. **Year with the Most Accident Cases**:
   - **2020** had the most accident cases, totaling **5255 cases**.

2. **Trend Over the Years**:
   - There is a significant increase in the number of cases leading up to 2020, followed by a sharp decline in 2021.
   - A slight increase is observed in 2022, but the number of cases drops again in 2023.

3. **Mean Line**:
   - The red dashed line represents the mean number of cases over the years, providing a benchmark to compare yearly data.

4. **Visual Appeal**:
   - The line chart effectively shows the trend of accident cases over the years, with clear peaks and troughs that highlight significant changes in accident frequencies.

Overall, the plot indicates that 2020 was an outlier year with an exceptionally high number of accidents. Understanding the factors contributing to this spike can help in formulating strategies to prevent similar surges in the future.''')


with c2:
    add_vertical_space(7)
    # List of road conditions
    # List of road conditions
    road_conditions = ['Crossing' ,'Junction', 'Roundabout', 'Stop', 'Traffic_Calming',
                    'Traffic_Signal']

    # Calculate the sum of each road condition to find the top 5 useful amenities
    condition_sums = df[road_conditions].sum().sort_values(ascending=False).head(5)
    top_6_conditions = condition_sums.index.tolist()

    # Colors for the pie chart slices
    colors = ['#4CAF50', '#FF9800', '#03A9F4', '#E91E63', '#9C27B0','#9C26B0']

    # Streamlit app
    st.header('Top 5 Useful Amenities in Road Conditions')

    # Create a pie chart using Plotly Express
    fig = go.Figure()

    # Function to annotate each slice with percentage
    def func(pct, allvals):
        absolute = int(pct / 100. * sum(allvals))
        return "{:.1f}%\n({:d} Cases)".format(pct, absolute)

    # Plot pie chart for each top 5 road condition
    for idx, condition in enumerate(top_6_conditions):
        values = df[condition].value_counts().tolist()
        labels = ['False', 'True']
        colors_condition = ['lightgrey', colors[idx]]

        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            name=condition,
            hole=0.3,
            textinfo='percent+label',
            insidetextorientation='horizontal',
    
            marker=dict(colors=colors_condition, line=dict(color='grey', width=2)),
            domain=dict(row=idx, column=0)
        ))

    # Update layout for better aesthetics
    fig.update_layout(
        title_text='Top 5 Useful Amenities in Road Conditions',
        annotations=[dict(text=condition, x=0.2, y=0.925 - 0.175 * idx, font_size=14, showarrow=False) for idx, condition in enumerate(top_6_conditions)],
        grid=dict(rows=6, columns=1),
        showlegend=False,
        title_font_size=24,
        title_font_color='grey',
        
        font=dict(family='Arial', size=12, color='black'),
        height=1200  # Set the height to avoid overlapping
    )

    # Display the plot in the Streamlit app
    st.plotly_chart(fig)

with c3:
    add_vertical_space(2)
    st.write('''## Inference:

#### Top 5 Useful Amenities in Road Conditions

1. **Traffic Signal**:
   - **Traffic signals** are present in **21.4%** of the road conditions, making them the most common useful amenity. This indicates a significant presence of controlled intersections to manage traffic flow and enhance safety.

2. **Crossing**:
   - **Crossings** account for **16.2%** of the road conditions. This suggests that pedestrian crossings are a crucial feature in urban planning to ensure pedestrian safety and reduce accidents.

3. **Junction**:
   - **Junctions** are found in **10.7%** of the cases. The presence of junctions highlights the importance of road intersections and their potential role in accident occurrences due to the convergence of multiple roads.

4. **Stop**:
   - **Stop signs** are present in **2.41%** of the road conditions, indicating a lesser but still important role in traffic control, particularly in residential or low-traffic areas.

5. **Traffic Calming**:
   - **Traffic calming measures** (e.g., speed bumps) are seen in only **0.08%** of the road conditions. This low percentage points to a minimal implementation of such measures, which are typically used to slow down traffic and enhance safety in specific areas.

4. **Visual Appeal**:
   - The use of donut charts effectively displays the proportion of True and False values for each amenity, with distinct colors aiding visual interpretation and understanding of the distribution.

Overall, the plot highlights that traffic signals and crossings are the most prevalent amenities in road conditions, underscoring their critical role in managing traffic and ensuring safety. Stop Signs and Traffic Calming Mechanisms have greatly resulted in least accidents. This suggests areas for potential improvement in road safety measures inorder to mitigate road accidents.''')



