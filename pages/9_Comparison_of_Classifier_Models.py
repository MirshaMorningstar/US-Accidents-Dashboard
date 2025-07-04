import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

# ---------------------------------
# Page config
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title='ML Model Comparison',
    page_icon="📊",
    layout='wide'
)

add_vertical_space(2)

# ---------------------------------
# Helper Functions

def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename}</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename}</a>'
    return href

# ---------------------------------
# App Title
st.title("Full Machine Learning Model Comparison Report")
add_vertical_space(1)

# ---------------------------------
# Sidebar: Upload CSV
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
        [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    """)

# Sidebar: Parameters
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Train/Test Split %', 10, 90, 80, 5)
    seed_number = st.sidebar.slider('Random seed', 1, 100, 42)

# ---------------------------------
# Dataset Load
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('### Dataset Preview')
    st.write(df.head())
else:
    st.info("Awaiting CSV upload... Using default example dataset")
    df = pd.read_csv("./data/US_Norm.csv")  # ✅ Make sure to add this to your GitHub repo

# ---------------------------------
# Build Model
st.markdown("---")
st.subheader("Train and Compare Models with LazyPredict")

try:
    df = df.drop(columns=["Unnamed: 0", "ID", "Source", "Description", "Street"], errors='ignore')
    X = df.drop(columns=['Severity'], errors='ignore')
    Y = df['Severity']

    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=(100 - split_size) / 100,
        random_state=seed_number
    )

    # LazyClassifier (Train & Test)
    clf_train = LazyClassifier(verbose=0, ignore_warnings=True)
    clf_test = LazyClassifier(verbose=0, ignore_warnings=True)

    models_train, predictions_train = clf_train.fit(X_train, X_train, Y_train, Y_train)
    models_test, predictions_test = clf_test.fit(X_train, X_test, Y_train, Y_test)

    if predictions_test.empty:
        st.error("⚠️ No models could be evaluated. Please check your dataset.")
    else:
        st.subheader("Model Performance Report (Test Set)")
        st.dataframe(predictions_test[['Accuracy', 'F1 Score', 'Time Taken']])
        st.markdown(filedownload(predictions_test, "test_results.csv"), unsafe_allow_html=True)

        # Visualize Accuracy
        predictions_test['Accuracy'] = predictions_test['Accuracy'].clip(lower=0)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=predictions_test.index, y="Accuracy", data=predictions_test)
        plt.xticks(rotation=90)
        plt.title("Model Accuracy (Test Set)")
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'accuracy_plot.png'), unsafe_allow_html=True)

except Exception as e:
    st.error(f"❌ Error while processing: {e}")

add_vertical_space(3)
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + LazyPredict")
