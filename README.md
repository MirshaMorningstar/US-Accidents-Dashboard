# The US Accidents Analysis and Case Study Dashboard

Welcome to the US Accidents Dashboard Project! This unique and comprehensive dashboard provides detailed insights into traffic accidents across the United States. Below is a detailed overview of the project, including its purpose, features, and requirements.

## Table of Contents

1. [Executive Summary](#executive-summary)
   1. [Purpose](#purpose)
   2. [Target Users](#target-users)
   3. [Unique Value Proposition](#unique-value-proposition)
2. [Introduction](#introduction)
   1. [Connecting Our Dashboard to Users](#connecting-our-dashboard-to-users)
   2. [Motivation and Purpose](#motivation-and-purpose)
   3. [Filling the Market Void](#filling-the-market-void)
   4. [Main Objectives](#main-objectives)
3. [Market Analysis](#market-analysis)
   1. [Market Demand](#market-demand)
4. [Unique Selling Points (USPs)](#unique-selling-points-usps)
   1. [Streamlined Accessibility](#streamlined-accessibility)
   2. [Comprehensive Analysis Features](#comprehensive-analysis-features)
   3. [Intuitive User Interface](#intuitive-user-interface)
   4. [Cost-Effective Solution](#cost-effective-solution)
   5. [Real-Time Data Analysis](#real-time-data-analysis)
   6. [Customizable Visualizations](#customizable-visualizations)
   7. [Educational Value](#educational-value)
   8. [Community Support and Collaboration](#community-support-and-collaboration)
5. [Dataset Description](#dataset-description)
6. [Plots](#plots)
7. [Description of Each Page](#description-of-each-page)
8. [Hardware Requirements](#hardware-requirements)
   1. [Development Machine](#development-machine)
   2. [Server](#server)
9. [Software Requirements](#software-requirements)
   1. [Programming Language](#programming-language)
   2. [Python Libraries](#python-libraries)
   3. [Development Environment](#development-environment)
   4. [Deployment Platform](#deployment-platform)
10. [Conclusion](#conclusion)

## Executive Summary

### Purpose

The purpose of this project is to create an interactive and user-friendly web page to explore and visualize a US accident dataset (2016-2023). The dashboard aims to uncover key insights and patterns related to accidents, including their severity and the influence of various factors such as location, time, and weather conditions.

### Target Users

The primary users of this dashboard include policymakers, researchers, traffic safety analysts, and the general public. These users can leverage the dashboard to understand accident trends, improve road safety measures, and make data-driven decisions.

### Unique Value Proposition

This dashboard stands out by offering a comprehensive and intuitive platform for analyzing US accident data. It combines various features such as data quality visualization, feature engineering, preprocessing, and both general and inferential visualizations. These tools enable users to gain deeper insights into accident data, making it a valuable resource for improving traffic safety and informing policy decisions.

## Introduction

### Connecting Our Dashboard to Users

Our US Accidents Dashboard is designed to bridge the gap between complex data and user-friendly analysis. By offering an interactive and intuitive platform, we enable users to easily explore and understand accident data. We have deployed our project using Streamlit Cloud, making it easily accessible to users anytime, anywhere, without the need for complex installations or configurations.

### Motivation and Purpose

The motivation behind this project is to leverage data to improve road safety. With thousands of accidents occurring every day, understanding the factors that contribute to these incidents is crucial. The purpose of this dashboard is to provide a tool that helps users identify trends, analyze patterns, and ultimately contribute to reducing accidents on the roads.

### Filling the Market Void

Currently, there is a lack of accessible, comprehensive tools for analyzing US accident data. Many existing solutions are either too complex for the average user or too simplistic to provide meaningful insights. Our dashboard fills this void by combining ease of use with powerful analytical capabilities, offering a unique solution that meets the needs of a wide range of users.

### Main Objectives

The main objectives of our dashboard are:
- To provide an interactive platform for exploring US accident data.
- To visualize data quality, including missing values and outliers.
- To prepare data for exploration by normalizing, binning, and sampling.
- To enable detailed analysis of accident factors such as location, time, and weather.
- To filter the data for analysis.
- To support feature engineering and data preprocessing for deeper insights.
- To offer both general and inferential visualizations to help users draw meaningful conclusions from the data.
- To develop visually appealing and informative data visualizations.
- To design and evaluate color palettes for visualization based on principles of perception, enhancing readability and interpretability.

## Market Analysis

### Market Demand

The demand for data-driven insights in the transportation and safety sector is at an all-time high. As cities grow and the number of vehicles on the road increases, there is a pressing need for tools that can help analyze and mitigate traffic accidents. Governments, city planners, and safety organizations are seeking advanced solutions to improve road safety and reduce accident rates. This demand creates a significant opportunity for our US Accidents Dashboard.Our dashboard stands out in the market as a versatile, user-friendly, and comprehensive tool for traffic accident analysis, meeting the needs of a broad spectrum of users from policymakers to the general public.

## Unique Selling Points (USPs)

### Streamlined Accessibility

Our project offers easy access to complex data analysis tools through deployment on Streamlit Cloud. Users can access the dashboard from anywhere with an internet connection, eliminating the need for software installations or technical setups.

### Comprehensive Analysis Features

Unlike many existing solutions, our dashboard provides a comprehensive suite of analysis tools. From data quality assessment to feature engineering and inferential visualizations, users can perform a wide range of analyses without switching between multiple platforms.

### Intuitive User Interface

With Streamlit's user-friendly interface, users of all technical backgrounds can navigate the dashboard effortlessly. Clear visualizations and interactive controls make data exploration and interpretation straightforward and engaging.

### Cost-Effective Solution

Built on open-source technology and deployed on Streamlit Cloud, our project offers a cost-effective alternative to expensive proprietary software. Users can access powerful analysis capabilities without the prohibitive costs associated with professional-grade tools.

### Real-Time Data Analysis

By leveraging Streamlit Cloud's capabilities, our dashboard can analyze and visualize real-time data updates. This feature ensures that users always have access to the most up-to-date insights, allowing for timely decision-making and response to changing conditions.

### Customizable Visualizations

Users can tailor the visualizations to their specific needs and preferences. With options to change color palettes, adjust plot sizes, and select specific data subsets, users can create customized views that highlight the insights most relevant to their objectives.  

### Educational Value

Beyond its analytical capabilities, our project serves as an educational tool for understanding traffic accident data. Users can learn about data quality assessment, preprocessing techniques, and inferential analysis methods through interactive exploration of real-world datasets.

### Community Support and Collaboration

As part of the Streamlit ecosystem, our project benefits from a vibrant community of users and developers. Users can share insights, collaborate on analysis projects, and contribute to the ongoing development and improvement of the dashboard.

## Dataset Description

Dataset : US-Accidents: A Countrywide Traffic Accident Dataset

This is a countrywide traffic accident dataset, which covers 49 states of the United States. The data is continuously being collected from February 2016 - 2023, using several data providers, including multiple APIs that provide streaming traffic event data. These APIs broadcast traffic events captured by a variety of entities, such as the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road-networks. Currently, there are about 0.5 million accident records in this dataset. Check the below descriptions for more detailed information.


### A Detailed Breakthrough

Dataset Source: The dataset is sourced from [relevant source, e.g., Kaggle].

Data Structure: The dataset includes 46 columns, each providing different details about the accidents. Key columns include:
- **ID**: This is a unique identifier of the accident record.
- **Severity**: Shows the severity of the accident, a number between 1 and 4, where 1 indicates the least impact on traffic (i.e., short delay as a result of the accident) and 4 indicates a significant impact on traffic (i.e., long delay).
- **Start_Time**: Shows start time of the accident in the local time zone.
- **End_Time**: Shows end time of the accident in the local time zone. End time here refers to when the impact of an accident on traffic flow was dismissed.
- **Start_Lat**: Shows latitude in GPS coordinate of the start point.
- **Start_Lng**: Shows longitude in GPS coordinate of the start point.
- **End_Lat**: Shows latitude in GPS coordinate of the end point.
- **End_Lng**: Shows longitude in GPS coordinate of the end point.
- **Distance(mi)**: The length of the road extent affected by the accident.
- **Description**: Shows natural language description of the accident.
- **Number**: Shows the street number in the address field.
- **Street**: Shows the street name in the address field.
- **City**: Shows the city in the address field.
- **County**: Shows the county in the address field.
- **State**: Shows the state in the address field.
- **ZipCode**: Shows the zip code in the address field.
- **Country**: Shows the country in the address field.
- **Timezone**: Shows timezone based on the location of the accident (eastern, central, etc.).
- **Airport_Code**: Denotes an airport-based weather station which is the closest one to location of the accident.
- **Weather_Timestamp**: Shows the time-stamp of a weather observation record (in local time).
- **Temperature(F)**: Shows the temperature (in Fahrenheit).
- **Wind_Chill(F)**: Shows the wind chill (in Fahrenheit).
- **Humidity(%)**: Shows the humidity (in percentage).
- **Pressure(in)**: Shows the air pressure (in inches).
- **Visibility(mi)**: Shows visibility (in miles).
- **Wind_Direction**: Shows wind direction.
- **Wind_Speed(mph)**: Shows wind speed (in miles per hour).
- **Precipitation(in)**: Shows precipitation amount in inches, if there is any.
- **Weather_Condition**: Shows the weather condition (rain, snow, thunderstorm, fog, etc.)
- **Amenity**: A POI annotation which indicates presence of amenity in a nearby location.
- **Bump**: A POI annotation which indicates presence of speed bump or hump in a nearby location.
- **Crossing**: A POI annotation which indicates presence of crossing in a nearby location.
- **Give_Way**: A POI annotation which indicates presence of give_way in a nearby location.
- **Junction**: A POI annotation which indicates the presence of a junction in a nearby location.
- **No_Exit**: A POI annotation which indicates presence of no_exit in a nearby location.
- **Railway**: A POI annotation which indicates presence of railway in a nearby location.
- **Roundabout**: A POI annotation which indicates the presence of a roundabout in a nearby location.
- **Station**: A POI annotation which indicates presence of a station in a nearby location.
- **Stop**: A POI annotation which indicates presence of stop in a nearby location.
- **Traffic_Calming**: A POI annotation which indicates presence of traffic_calming in a nearby location.
- **Traffic_Signal**: A POI annotation which indicates presence of traffic_signal in a nearby location.
- **Turning_Loop**: A POI annotation which indicates presence of turning_loop in a nearby location.
- **Sunrise_Sunset**: Shows the period of day (i.e. day or night) based on sunrise/sunset.
- **Civil_Twilight**: Shows the period of day (i.e. day or night) based on civil twilight.
- **Nautical_Twilight**: Shows the period of day (i.e. day or night) based on nautical twilight.
- **Astronomical_Twilight**: Shows the period of day (i.e. day or night) based on astronomical twilight.
- **Year**: The year component extracted from the Start_Time attribute.
- **Month**: The month component extracted from the Start_Time attribute.
- **Day**: The day component extracted from the Start_Time attribute.
- **Hour**: The hour component extracted from the Start_Time attribute.
- **Minute**: The minute component extracted from the Start_Time attribute.
- **Second**: The second component extracted from the Start_Time attribute.
- **Hour_Category**: A categorical attribute that categorizes the accidents based on the hour of the day (e.g., morning, afternoon, evening, night).

### Plots

- **Heatmap for Missing Values**: A heatmap visually represents missing data in a dataset, highlighting the presence and pattern of missing values across different features.
- **Pie Chart**: A pie chart displays data as slices of a circle, showing the relative proportions of different categories within a dataset.
- **Bar Plot**: A bar plot uses rectangular bars to represent and compare the frequency or value of different categories.
- **Violin Plot for Outlier**: A violin plot combines a box plot with a kernel density plot, showing the distribution, probability density, and presence of outliers in the data.
- **Box Plot for Outliers**: A box plot summarizes the distribution of a dataset, highlighting the median, quartiles, and potential outliers through the use of "whiskers" and individual points.
- **Histogram for Data Distribution**: A histogram displays the frequency distribution of a dataset by grouping data into bins and showing the count of data points in each bin.
- **Quantile-Quantile Plot for Normalization Visualization**: A quantile-quantile (Q-Q) plot compares the quantiles of a dataset to the quantiles of a theoretical distribution, helping to assess if the data follows the desired distribution, such as normality.
- **Bar Plot for Binning**: A bar plot for binning groups continuous data into discrete bins and represents the count or frequency of data points in each bin.
- **Scatter Plot**: A scatter plot displays individual data points on a two-dimensional plane, showing the relationship between two numerical variables.
- **Line Plot**: A line plot connects data points with lines, typically used to show trends over time or the relationship between two continuous variables.
- **Rose Plot**: A rose plot (circular bar plot or wind rose) displays data in circular segments, often used to represent cyclic data like wind direction and speed.
- **Folium Map to Visualize US Accident Places**: A Folium map uses the Folium library to create interactive maps, plotting locations of accidents across the United States for geographical analysis.

### Description of each page

1. **Authentication**
   - **Feature**: Secure login system requiring users to enter their credentials (username and password) to access the web page.
   - **Purpose**: Ensures that only authorized users can access the data and analysis tools, maintaining data privacy and security.
   
2. **Change Theme**
   - **Feature**: Option to change the dashboard's theme.
   - **Purpose**: Allows users to customize the visual appearance of the dashboard to their preference, enhancing the user experience.
   
3. **Color Palette Picker**
   - **Feature**: A tool for users to select different color palettes.
   - **Purpose**: Extracting color palettes from images. This app allows users to upload images, apply enhancements, and then extract color palettes using various machine learning clustering algorithms. Users can adjust parameters such as palette size, sample size, and image enhancements. The color palettes can be visualized and adopted for use in matplotlib or plotly plots.
   
4. **Specific Exploration of Data**
   - **Feature**: Tools for in-depth exploration of specific subsets of the data.
   - **Purpose**: Allows users to delve into particular aspects of the dataset, such as filtering accidents by location, time period, or weather conditions.
   
5. **General Exploration of Data**
   - **This Streamlit page is an Exploratory Data Analysis (EDA) tool for analyzing datasets, particularly the US Accidents dataset. Users can upload their CSV files or use a sample dataset provided.**
   - **The page generates a comprehensive profiling report using the pandas-profiling library, including an overview, alerts, missing values, correlation heatmap, and detailed visualizations for each variable. This enables users to gain insights and understand the dataset's structure and quality effectively.**
   
6. **Visualization of Data Quality**
   - **Feature**: Visual tools to assess and display the quality of the dataset, including:
     - **Missing Values Analysis**: Identifies and visualizes missing data in the dataset.
     - **Irregular Cardinality**: Detects and visualizes unusual or unexpected frequencies in categorical data.
     - **Outlier Detection**: Identifies and visualizes data points that deviate significantly from the norm.
     - **Plots**: Various plots to visually represent data quality issues, such as heatmap, pie, bar chart for missing values and violin and box plots for outliers.
   - **Purpose**: Helps users understand the completeness and accuracy of the data, highlighting areas with missing or

 inconsistent entries, and identifying anomalies.
   
7. **Feature Engineering and Data Correlation**
   - **Feature**: Tools for creating new features and analyzing correlations between different variables.
   - **Purpose**: Enhances the data's analytical potential by deriving new insights and understanding the relationships between variables.
   
8. **Data Preprocessing and Preparation**
   - **Feature**: Modules for data cleaning, normalization, and transformation, including:
     - **Normalization**: Standardizes data to a common scale, improving the performance of analysis techniques.
     - **Binning**: Groups continuous data into bins or intervals to simplify analysis.
     - **Plots**: Visual representations of the preprocessing steps, such as histograms for binning and to understand normalization using Quantile-Quantile plot.
     
9. **Comparison of 25+ Classifier model**
   - **This Streamlit page is designed to compare the performance of various machine learning classification algorithms using the lazypredict library. Users can upload their own CSV files or use an example dataset provided. The page allows users to customize dataset attributes and choose from top-performing machine learning models.**
   - **It generates performance metrics and visualizations, helping users understand and compare different model’s accuracy, balanced accuracy, F1 scores, and computation times. Additionally, the page includes functionality for predicting new data based on user-selected attributes and models.**
   
10. **Optimization for various parameters**
   - **This Streamlit app provides a user-friendly interface for hyperparameter optimization of a machine learning classification model, specifically using the Decision Tree algorithm. Users can upload their CSV dataset or use the provided example dataset.**
   - **The app allows users to interactively set various hyperparameters such as the number of estimators, max features, and others, and then builds a classification model based on the selected hyperparameters.**
   - **Model performance metrics such as coefficient of determination (R^2) and error (MSE or MAE) are displayed, along with a 3D surface plot showing the relationship between hyperparameters and model performance. Additionally, users can download a CSV file containing detailed grid search results for further analysis.**
   
11. **General Visualizations**
   - **Feature**: Standard charts and graphs to display trends.
   - **Purpose**: Provides an overview of the data through common visualizations like bar charts, line graphs, and pie charts, scatter plots, box plots, histogram making it easier to identify patterns and trends.
   
12. **Inferential Visualizations**
   - **Feature**: Advanced visualizations for inferential statistics and deeper analysis, providing a comprehensive inference about the US accident dataset.**

### Hardware Requirements

**Development Machine**
- **Processor**: Intel Core i3 or equivalent
- **RAM**: 8 GB or more
- **Storage**: 256 GB SSD or more
- **Operating System**: Windows 10, macOS, or Linux

**Server (if self-hosting instead of using Streamlit Cloud)**
- **Processor**: Intel Xeon or equivalent
- **RAM**: 16 GB or more
- **Storage**: 500 GB SSD or more
- **Operating System**: Linux (Ubuntu 20.04 LTS or later recommended)
- **Network**: High-speed internet connection with at least 10 Mbps upload/download speed

### Software Requirements

**Programming Language**
- **Python**: Version 3.8 or higher

**Python Libraries**
- **Streamlit**: For building and deploying the dashboard
- **pandas**: For data manipulation and analysis
- **Plotly**: For interactive visualizations
- **seaborn**: For statistical data visualization
- **numpy**: For numerical operations
- **scikit-learn**: For machine learning and preprocessing tasks
- **matplotlib**: For basic plotting
- **base58**: A library for binary-to-text encoding, commonly used for creating short, human-readable identifiers.
- **pillow**: A Python Imaging Library (PIL) fork that adds image processing capabilities.
- **lazypredict**: A module that helps in building a lot of basic models without much code and helps understand which models work better without much parameter tuning.
- **xgboost**: An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.
- **lightgbm**: A highly efficient gradient boosting framework that uses tree-based learning algorithms.
- **pytest**: A testing framework for Python that makes it easy to write simple and scalable test cases.
- **tqdm**: A library providing fast, extensible progress bars for loops and other iterative tasks.
- **ydata-profiling**: A library that generates profile reports from a pandas DataFrame to understand data quickly.
- **streamlit-pandas-profiling**: An integration of Streamlit with pandas-profiling, allowing easy display of profile reports in a Streamlit app.
- **Jinja2**: A templating engine for Python, used for rendering templates to generate HTML or other markup languages.

**Development Environment**
- **IDE/Code Editor**: Visual Studio Code, Jupyter Notebook

**Deployment Platform**
- **Streamlit Cloud**: For hosting the dashboard online

### Conclusion

This project has successfully developed an interactive and user-friendly web page for exploring and visualizing a comprehensive US accident dataset. Through the deployment on Streamlit Cloud, we have ensured that our dashboard is accessible from anywhere, without the need for complex installations, making it a versatile tool for a wide range of users, including policymakers, researchers, traffic safety analysts, and the general public.

This project has effectively filled a market void by providing a comprehensive, user-friendly, and cost-effective solution for analyzing US accident data. By transforming complex data into actionable intelligence through thoughtful design and robust analytical capabilities, our dashboard empowers users to make data-driven decisions, ultimately contributing to improved road safety and more informed policy decisions.
