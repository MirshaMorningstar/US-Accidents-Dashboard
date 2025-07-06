import streamlit as st
import streamlit_extras as stex
from streamlit_extras.add_vertical_space import add_vertical_space
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes #load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os


from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, 
                              AdaBoostClassifier)
from sklearn.svm import SVC, NuSVC, LinearSVC
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import (QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis)
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import (LogisticRegression, RidgeClassifierCV, RidgeClassifier, 
                                  PassiveAggressiveClassifier, SGDClassifier, Perceptron)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(initial_sidebar_state="collapsed",page_title='The Machine Learning Classification Algorithms Comparison Window',
    layout='wide',
    menu_items={
         'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
         'Report a bug': "mailto:a.k.mirsha9@gmail.com",
         'About': "### This is an extremely cool web application built as a part of my Data Science Mini Project on the ' US Accidents Dataset '\n"
     },
     page_icon="classification.png")
add_vertical_space(2)
#---------------------------------#
# Model building
def build_model(df):
    
    X = data.drop(columns=['Severity','Start_Time'])  # Features
    Y = data['Severity']  # Target variable

    st.markdown('**1.2. Dataset dimension**')
    st.write('X')
    st.info(X.shape)
    st.write('Y')
    st.info(Y.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('Predictor variables')
    st.info(list(X.columns))
    st.write('Target variable')
    st.info(Y.name)

    # Build lazy model
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size,random_state = seed_number)
    clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    models_train,predictions_train = clf.fit(X_train, X_train, Y_train, Y_train)
    models_test,predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)
   

    st.subheader('2. Table of Model Performance')

    ca,cb = st.columns(2)
    
    with ca:
        st.write('Training set')
        st.write(predictions_train[['Accuracy', 'Balanced Accuracy', 'F1 Score', 'Time Taken']])
        st.markdown(filedownload(predictions_train,'training.csv'), unsafe_allow_html=True)

    with cb:
        st.write('Test set')
        st.write(predictions_test[['Accuracy', 'Balanced Accuracy', 'F1 Score', 'Time Taken']])
        st.markdown(filedownload(predictions_test,'test.csv'), unsafe_allow_html=True)

    st.subheader('3. Plot of Model Performance (Test set)')


    with st.markdown('### **Accuracy**'):
        # Tall
        predictions_test["Accuracy"] = [0 if i < 0 else i for i in predictions_test["Accuracy"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(y=predictions_test.index, x="Accuracy", data=predictions_test)
        ax1.set(xlim=(0, 1))
    st.markdown(imagedownload(plt,'plot-acc-tall.pdf'), unsafe_allow_html=True)
        # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax1 = sns.barplot(x=predictions_test.index, y="Accuracy", data=predictions_test)
    ax1.set(ylim=(0, 1))
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'plot-acc-wide.pdf'), unsafe_allow_html=True)

    with st.markdown('### **Balanced Accuracy**'):
        # Tall
        predictions_test["Balanced Accuracy"] = [0 if i < 0 else i for i in predictions_test["Balanced Accuracy"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(y=predictions_test.index, x="Balanced Accuracy", data=predictions_test)
        ax1.set(xlim=(0, 1))
    st.markdown(imagedownload(plt,'plot-bal-acc-tall.pdf'), unsafe_allow_html=True)
        # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax1 = sns.barplot(x=predictions_test.index, y="Balanced Accuracy", data=predictions_test)
    ax1.set(ylim=(0, 1))
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'plot-bal-acc-wide.pdf'), unsafe_allow_html=True)


    with st.markdown('### **F1 Score**'):
        # Tall
        predictions_test["F1 Score"] = [0 if i < 0 else i for i in predictions_test["F1 Score"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(y=predictions_test.index, x="F1 Score", data=predictions_test)
        ax1.set(xlim=(0, 1))
    st.markdown(imagedownload(plt,'plot-f1-tall.pdf'), unsafe_allow_html=True)
        # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax1 = sns.barplot(x=predictions_test.index, y="F1 Score", data=predictions_test)
    ax1.set(ylim=(0, 1))
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'plot-f1-wide.pdf'), unsafe_allow_html=True)

   

    with st.markdown('### **Calculation time**'):
        # Tall
        predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
    st.markdown(imagedownload(plt,'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
        # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)

# Download CSV data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

#---------------------------------#
st.write("""
# The Machine Learning Algorithm Comparison Window""")

add_vertical_space(1)
st.write("""


In this Page Window, the **lazypredict** library is used for building and comparing several machine learning models at once.


""")

add_vertical_space(2)

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)


#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')
add_vertical_space(1)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    x = True
else:
    st.info('Awaiting for CSV file to be uploaded.')
    add_vertical_space(1)
    if st.button('Press to use our Example "US Accidents Dataset"...'):
        xdata = pd.read_csv("US.csv")
        xdata = xdata.sample(n=1000,random_state=42)
        # Load your dataset
        data = xdata.drop(["ID","Source"],axis=1)


# Assume the last column is the target variable
X = data.drop(columns=['Severity','Start_Time',"End_Time","Description","Street","Zipcode","Country","Timezone","Airport_Code","Weather_Timestamp","Amenity","Bump","Crossing","Give_Way","Junction","No_Exit","Railway","Roundabout","Station","Stop","Traffic_Calming","Traffic_Signal","Turning_Loop","Civil_Twilight","Nautical_Twilight","Astronomical_Twilight"])  # Features
y = data['Severity']  # Target variable

### Time for user prediction
add_vertical_space(2)
st.markdown("## The Predictions Making Space")
add_vertical_space(1)
st.markdown("###### Feel free to customise the dataset, applying what all attributes you want to filter for...")
add_vertical_space(3)

c1,c2 = st.columns(2)
with c1:
    attributes = X.columns
    default_attributes = ["Hour_Category","Temperature(F)","Wind_Chill(F)","Humidity(%)","Pressure(in)"]
    attributes = st.multiselect("Filter out your attributes for the predictions dataset here.",attributes,default_attributes)
    
with c2:
    classifier_names = [
    'RandomForestClassifier',
    'LGBMClassifier',
    'ExtraTreesClassifier',
    'SVC',
    'DecisionTreeClassifier',
    'QuadraticDiscriminantAnalysis',
    'BaggingClassifier',
    'ExtraTreeClassifier',
    'LabelPropagation',
    'LabelSpreading',
    'NuSVC',
    'LogisticRegression',
    'KNeighborsClassifier',
    'LinearSVC',
    'CalibratedClassifierCV',
    'LinearDiscriminantAnalysis',
    'RidgeClassifierCV',
    'RidgeClassifier',
    'AdaBoostClassifier',
    'PassiveAggressiveClassifier',
    'SGDClassifier',
    'Perceptron',
    'GaussianNB',
    'NearestCentroid',
    'BernoulliNB',
    'DummyClassifier']

    st.markdown("#### You can Choose from the top 5 Best Performing ML Models for this Dataset...")
    add_vertical_space(2)
    st.markdown("**Feel free to adjust all the Attributes and Data Predicting Models Properly for an enhanced testing experience. All the models listed here has achieved more than 97% Accuracy in predicting this Dataset...**")
    add_vertical_space(3)
    model_name = st.selectbox("Select the Model you want to predict upon...",classifier_names,0)


# Split the dataset into training and testing sets
X = data[attributes]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a dictionary of classifiers
classifiers = {
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
    'LGBMClassifier': LGBMClassifier(random_state=42),
    'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=100, random_state=42),
    'SVC': SVC(random_state=42),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
    'BaggingClassifier': BaggingClassifier(n_estimators=100, random_state=42),
    'ExtraTreeClassifier': ExtraTreeClassifier(random_state=42),
    'LabelPropagation': LabelPropagation(),
    'LabelSpreading': LabelSpreading(),
    'NuSVC': NuSVC(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LinearSVC': LinearSVC(random_state=42, max_iter=10000),
    'CalibratedClassifierCV': CalibratedClassifierCV(),
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
    'RidgeClassifierCV': RidgeClassifierCV(),
    'RidgeClassifier': RidgeClassifier(random_state=42),
    'AdaBoostClassifier': AdaBoostClassifier(n_estimators=100, random_state=42),
    'PassiveAggressiveClassifier': PassiveAggressiveClassifier(max_iter=1000, random_state=42),
    'SGDClassifier': SGDClassifier(max_iter=1000, random_state=42),
    'Perceptron': Perceptron(max_iter=1000, random_state=42),
    'GaussianNB': GaussianNB(),
    'NearestCentroid': NearestCentroid(),
    'BernoulliNB': BernoulliNB(),
    'DummyClassifier': DummyClassifier(strategy='most_frequent')
}

if model_name in classifiers:
    classifier = classifiers[model_name]
else:
    raise ValueError(f"Unsupported model: {model_name}")

# Create and train the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])


# Train the model
pipeline.fit(X_train, y_train)

new_data = {}

with c1:
    for item in attributes:
        if data[item].dtype in ['int64','float64']:
            if item=='Pressure(in)':
                new_data[item]=[st.number_input(f"Enter the value of {item} to be predicted.",value=31)]
            else:
                new_data[item]=[st.number_input(f"Enter the value of {item} to be predicted.",value=69)]
        elif data[item].dtype in ['bool','object']:
            new_data[item]=[st.text_input(f"Enter the value of {item} to be predicted.",value="Evening")]
        
with c2:
    if st.button("Predict !!!"):
        new_data = pd.DataFrame(new_data)
        # Ensure the new data has the same preprocessing as the training data
        prediction = pipeline.predict(new_data)

        st.markdown('#### Prediction for the given sample data:')
        st.markdown(f"**The {model_name} has predicted the Target Attribute belonging to class {prediction[0]}**")

add_vertical_space(3)
st.markdown("### Now Click Here to view The Overall Comparsion Report of **25 +** Machine Learning Models...")
add_vertical_space(3)

if st.button("SHOW MODELS' COMPARISON REPORT"):
    add_vertical_space(3)
    data = pd.read_csv("US_Norm.csv")
    data = data.drop(["Unnamed: 0","ID","Source","Description","Street"],axis=1)
    build_model(data)


