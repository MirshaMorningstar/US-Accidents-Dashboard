import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
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

st.set_page_config(initial_sidebar_state="collapsed",
                   page_title='ML Classification Comparison',
                   layout='wide')

add_vertical_space(2)
st.title("Machine Learning Algorithm Comparison")

# Dataset upload
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Train split %', 10, 90, 80, 5)
    seed_number = st.sidebar.slider('Random Seed', 1, 100, 42)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# FileDownload helper
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename}</a>'

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()
    return f'<a href="data:image/pdf;base64,{b64}" download={filename}>Download {filename}</a>'

# LazyClassifier report
def build_model(df):
    st.subheader('Full Model Comparison Report')
    df = df.drop(columns=[col for col in df.columns if col not in df.select_dtypes(include=['number', 'object']) or df[col].nunique() == 1])
    X = df.drop(columns=['Severity'])
    y = df['Severity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-split_size)/100, random_state=seed_number)
    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    _, test_results = clf.fit(X_train, X_test, y_train, y_test)
    st.dataframe(test_results[['Accuracy', 'F1 Score', 'Time Taken']])
    st.markdown(filedownload(test_results, "test_results.csv"), unsafe_allow_html=True)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=test_results.index, y="Accuracy", data=test_results)
    plt.xticks(rotation=90)
    st.pyplot(plt)

# Handle dataset
if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.info("Using default dataset (US_Norm.csv)")
    df = load_data("US_Norm.csv")

if 'Severity' not in df.columns:
    st.error("Dataset must contain a 'Severity' column as target")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Predict using user-selected model
model_list = [
    'RandomForestClassifier', 'LGBMClassifier', 'ExtraTreesClassifier', 'SVC', 'DecisionTreeClassifier',
    'QuadraticDiscriminantAnalysis', 'BaggingClassifier', 'ExtraTreeClassifier', 'LabelPropagation',
    'LabelSpreading', 'NuSVC', 'LogisticRegression', 'KNeighborsClassifier', 'LinearSVC',
    'CalibratedClassifierCV', 'LinearDiscriminantAnalysis', 'RidgeClassifierCV', 'RidgeClassifier',
    'AdaBoostClassifier', 'PassiveAggressiveClassifier', 'SGDClassifier', 'Perceptron',
    'GaussianNB', 'NearestCentroid', 'BernoulliNB', 'DummyClassifier'
]

st.subheader("Make Prediction")
cols = [c for c in df.columns if c != 'Severity']
sel_cols = st.multiselect("Select input features", cols, default=cols[:5])
model_choice = st.selectbox("Choose a Model", model_list)

X = df[sel_cols]
y = df['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-split_size)/100, random_state=seed_number)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Create classifier instance
all_models = {
    'RandomForestClassifier': RandomForestClassifier(),
    'LGBMClassifier': LGBMClassifier(),
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'SVC': SVC(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
    'BaggingClassifier': BaggingClassifier(),
    'ExtraTreeClassifier': ExtraTreeClassifier(),
    'LabelPropagation': LabelPropagation(),
    'LabelSpreading': LabelSpreading(),
    'NuSVC': NuSVC(),
    'LogisticRegression': LogisticRegression(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LinearSVC': LinearSVC(max_iter=10000),
    'CalibratedClassifierCV': CalibratedClassifierCV(),
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
    'RidgeClassifierCV': RidgeClassifierCV(),
    'RidgeClassifier': RidgeClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),
    'SGDClassifier': SGDClassifier(),
    'Perceptron': Perceptron(),
    'GaussianNB': GaussianNB(),
    'NearestCentroid': NearestCentroid(),
    'BernoulliNB': BernoulliNB(),
    'DummyClassifier': DummyClassifier()
}

model = all_models[model_choice]
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
pipeline.fit(X_train, y_train)

# Input fields
data_input = {}
for feature in sel_cols:
    if df[feature].dtype in ['float64', 'int64']:
        data_input[feature] = st.number_input(f"Input {feature}", value=float(df[feature].mean()))
    else:
        data_input[feature] = st.text_input(f"Input {feature}", value=str(df[feature].unique()[0]))

if st.button("Predict"):
    new_df = pd.DataFrame([data_input])
    pred = pipeline.predict(new_df)
    st.success(f"Predicted Severity class: {pred[0]}")

# Show comparison report
if st.button("Show Comparison Report"):
    build_model(df)
