import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_diabetes
from streamlit_extras.add_vertical_space import add_vertical_space

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning Hyperparameter Optimization Window',
          menu_items={
         'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
         'Report a bug': "mailto:a.k.mirsha9@gmail.com",
         'About': "### This is an extremely cool web application built as a part of my Data Science Mini Project on my ' US Accidents Dataset '\n"
     },
    page_icon="image.png",
    layout='wide')

#---------------------------------#
st.write("""
# The Machine Learning Hyperparameter Optimization Window
**(Classification Edition)**

In this implementation, the *DecisionTreeClassifier()* function is used in this app for building a classification model using the **Decision Tree** algorithm.""")

add_vertical_space(2)

st.write("""
Moreover all the hyper parameters that are essential to build the model are extracted as inputs from the user interactively so that users can fine tune their model's performance based on the chosen hyperparameter.
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Sidebar - Specify parameter settings
st.sidebar.header('Set Parameters')
split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

st.sidebar.subheader('Learning Parameters')
parameter_max_depth = st.sidebar.slider('Maximum depth of the tree (max_depth)', 1, 20, 10)
parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2)
parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 1)

st.sidebar.subheader('General Parameters')
parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)


#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('Dataset')



#---------------------------------#
# Model building

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
    return href

def build_model(df):
    X = data.drop(columns=['Severity','Start_Time'])  # Features
    Y = data['Severity']  # Target variable
    
    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
    #X_train.shape, Y_train.shape
    #X_test.shape, Y_test.shape

    dt = DecisionTreeClassifier(max_depth=parameter_max_depth,
        random_state=parameter_random_state,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf)

    dt.fit(X_train, Y_train)

    st.subheader('Model Performance')

    Y_pred_test = dt.predict(X_test)
    st.write('Accuracy Score:')
    st.info( accuracy_score(Y_test, Y_pred_test) )

    st.subheader('Model Parameters')
    st.write(dt.get_params())

    #-------------------------------------------------------#


    #-----Hyperparameters visualization-----#
    max_depth_range = np.arange(1, 21, 1)
    min_samples_split_range = np.arange(2, 11, 1)
    min_samples_leaf_range = np.arange(1, 11, 1)

    # Create a meshgrid of hyperparameters
    max_depth_values, min_samples_split_values, min_samples_leaf_values = np.meshgrid(max_depth_range, min_samples_split_range, min_samples_leaf_range)

    # Flatten the meshgrid arrays
    max_depth_values = max_depth_values.flatten()
    min_samples_split_values = min_samples_split_values.flatten()
    min_samples_leaf_values = min_samples_leaf_values.flatten()

    # Create dataframe for visualization
    df_hyperparameters = pd.DataFrame({
        'max_depth': max_depth_values,
        'min_samples_split': min_samples_split_values,
        'min_samples_leaf': min_samples_leaf_values
    })

    # Calculate accuracy for each combination of hyperparameters
    accuracies = []
    for i in range(len(df_hyperparameters)):
        dt = DecisionTreeClassifier(max_depth=df_hyperparameters.iloc[i]['max_depth'],
                                    min_samples_split=df_hyperparameters.iloc[i]['min_samples_split'],
                                    min_samples_leaf=df_hyperparameters.iloc[i]['min_samples_leaf'],
                                    random_state=parameter_random_state)
        dt.fit(X_train, Y_train)
        Y_pred_test = dt.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred_test)
        accuracies.append(accuracy)

    df_hyperparameters['accuracy'] = accuracies

    
    # fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    #fig.update_layout(scene=dict(aspectmode="data"))
    
    
    # Create interactive 3D plot
    fig = go.Figure(data=[go.Surface(
        x=df_hyperparameters['max_depth'],
        y=df_hyperparameters['min_samples_split'],
        z=df_hyperparameters['min_samples_leaf']
    )])

    fig.update_layout(scene=dict(
        xaxis_title='Max Depth',
        yaxis_title='Min Samples Split',
        zaxis_title='Min Samples Leaf'
    ))

    st.plotly_chart(fig)

    #-----Save grid data-----#
    st.markdown(filedownload(df_hyperparameters), unsafe_allow_html=True)


#---------------------------------#
# 3D Parabola Visualization

#-----Hyperparameters visualization-----#

  


#---------------------------------#
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        
        data = pd.read_csv("US_Norm.csv")
        

        st.markdown('The **US Accidents** dataset is used as the example.')
        st.write(data.head(5))
        build_model(data)
      

    
