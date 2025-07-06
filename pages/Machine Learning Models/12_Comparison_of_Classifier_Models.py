import streamlit as st
import streamlit_extras as stex
from streamlit_extras.add_vertical_space import add_vertical_space
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier)
from sklearn.svm import SVC, NuSVC, LinearSVC
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifierCV, RidgeClassifier, PassiveAggressiveClassifier,
    SGDClassifier, Perceptron)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Page Config
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title='The Machine Learning Classification Algorithms Comparison Window',
    layout='wide',
    menu_items={
        'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
        'Report a bug': "mailto:a.k.mirsha9@gmail.com",
        'About': "### This is a mini-project web app based on the US Accidents Dataset"
    },
    page_icon="classification.png")

# Vertical space and title
add_vertical_space(2)
st.title("The Machine Learning Algorithm Comparison Window")
add_vertical_space(1)
st.write("""
In this window, the **lazypredict** library is used to compare ML models.
""")
add_vertical_space(2)

# Session state flags
if "predicted_clicked" not in st.session_state:
    st.session_state["predicted_clicked"] = False
if "show_model_report" not in st.session_state:
    st.session_state["show_model_report"] = False
if "example_data_loaded" not in st.session_state:
    st.session_state["example_data_loaded"] = False

# File Upload
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)

# Dataset
st.subheader('1. Dataset')
add_vertical_space(1)

if uploaded_file is not None:
    xdata = pd.read_csv(uploaded_file)
    st.write(xdata)
else:
    st.info('Awaiting for CSV file... Using sample.')
    xdata = pd.read_csv("US.csv").sample(n=1000, random_state=42)

if st.button('Press to use our Example "US Accidents Dataset"...'):
    st.session_state["example_data_loaded"] = True
    st.session_state["predicted_clicked"] = False
    st.session_state["show_model_report"] = False
    st.rerun()

if st.session_state["example_data_loaded"]:
    data = xdata.drop(["ID", "Source"], axis=1)
    data = data.drop(columns=[
        'Start_Time', "End_Time", "Description", "Street", "Zipcode", "Country", "Timezone", "Airport_Code",
        "Weather_Timestamp", "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
        "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop", "Civil_Twilight",
        "Nautical_Twilight", "Astronomical_Twilight"], errors='ignore')

    X = data.drop(columns=['Severity'])
    y = data['Severity']

    add_vertical_space(2)
    st.markdown("## The Predictions Making Space")
    add_vertical_space(1)

    c1, c2 = st.columns(2)

    with c1:
        attributes = st.multiselect("Select attributes for prediction", list(X.columns), default=[
            "Hour_Category", "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)"])

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

    with c2:
        model_name = st.selectbox("Select model", list(classifiers.keys()), 0)

    X = data[attributes]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    classifier = classifiers[model_name]
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    pipeline.fit(X_train, y_train)

    new_data_input = {}
    with c1:
        for item in attributes:
            if data[item].dtype in ['int64', 'float64']:
                default = 69 if item != 'Pressure(in)' else 31
                new_data_input[item] = [st.number_input(f"Enter {item}", value=default)]
            else:
                new_data_input[item] = [st.text_input(f"Enter {item}", value="Evening")]

    with c2:
        if not st.session_state["predicted_clicked"]:
            if st.button("Predict !!!"):
                st.session_state["predicted_clicked"] = True
                st.rerun()

    if st.session_state["predicted_clicked"]:
        new_df = pd.DataFrame(new_data_input)
        prediction = pipeline.predict(new_df)
        st.success(f"✅ The {model_name} predicted the target as **{prediction[0]}**")

    add_vertical_space(3)
    st.markdown("### Click below to view the overall comparison report of 25+ ML models.")
    add_vertical_space(2)

    if not st.session_state["show_model_report"]:
        if st.button("SHOW MODELS' COMPARISON REPORT"):
            st.session_state["show_model_report"] = True
            st.rerun()

    if st.session_state["show_model_report"]:
        def filedownload(df, filename):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename}</a>'

        def imagedownload(plt, filename):
            s = io.BytesIO()
            plt.savefig(s, format='pdf', bbox_inches='tight')
            plt.close()
            b64 = base64.b64encode(s.getvalue()).decode()
            return f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename}</a>'

        def build_model(data):
            X = data.drop(columns=['Severity', 'Start_Time'], errors='ignore')
            Y = data['Severity']
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size/100, random_state=seed_number)

            clf = LazyClassifier(verbose=0, ignore_warnings=True)
            _, pred_train = clf.fit(X_train, X_train, Y_train, Y_train)
            _, pred_test = clf.fit(X_train, X_test, Y_train, Y_test)

            st.subheader("Training set")
            st.write(pred_train[['Accuracy', 'Balanced Accuracy', 'F1 Score', 'Time Taken']])
            st.markdown(filedownload(pred_train, "train.csv"), unsafe_allow_html=True)

            st.subheader("Test set")
            st.write(pred_test[['Accuracy', 'Balanced Accuracy', 'F1 Score', 'Time Taken']])
            st.markdown(filedownload(pred_test, "test.csv"), unsafe_allow_html=True)

            for metric in ['Accuracy', 'Balanced Accuracy', 'F1 Score', 'Time Taken']:
                st.markdown(f"### {metric}")
                pred_test[metric] = [max(0, x) for x in pred_test[metric]]
                fig, ax = plt.subplots(figsize=(9, 3))
                sns.barplot(x=pred_test.index, y=metric, data=pred_test, ax=ax)
                ax.set_ylim(0, 1 if metric != 'Time Taken' else None)
                plt.xticks(rotation=90)
                st.pyplot(fig)
                st.markdown(imagedownload(plt, f"{metric}.pdf"), unsafe_allow_html=True)

        try:
            data = pd.read_csv("US_Norm.csv")
            data = data.drop(["Unnamed: 0", "ID", "Source", "Description", "Street"], axis=1)
            build_model(data)
        except Exception as e:
            st.error(f"Failed to build comparison report: {e}")
