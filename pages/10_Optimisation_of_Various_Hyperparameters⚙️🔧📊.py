import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from streamlit_extras.add_vertical_space import add_vertical_space
import os

# ----------------------------
# Page layout
st.set_page_config(
    page_title='Hyperparameter Optimization Window (Classification)',
    layout='wide',
    initial_sidebar_state="collapsed",
    page_icon="image.png",
    menu_items={
        'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
        'Report a bug': "mailto:a.k.mirsha9@gmail.com",
        'About': "### Hyperparameter tuning tool for RandomForestClassifier on US Accidents Dataset"
    }
)

st.title("⚙️ Machine Learning Hyperparameter Optimization (Classification Edition)")
add_vertical_space(2)

st.write("""
This tool allows you to perform hyperparameter tuning on a classification model using `RandomForestClassifier`.
Use the sidebar to configure your dataset and parameters. A 3D surface will be generated for selected hyperparameters.
""")

# ----------------------------
# Sidebar: Dataset & parameters
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

st.sidebar.header('Set Split Ratio')
split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5) / 100

st.sidebar.header('Learning Parameters')
n_est_range = st.sidebar.slider('n_estimators range', 10, 300, (50, 150), step=10)
n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 10)

max_feat_range = st.sidebar.slider('max_features range', 2, 30, (5, 15))
# (optional) Can expose step size later

min_samples_split = st.sidebar.slider('min_samples_split', 2, 10, 2)
min_samples_leaf = st.sidebar.slider('min_samples_leaf', 1, 10, 1)

st.sidebar.header('General Parameters')
criterion = st.sidebar.selectbox('Criterion', ['gini', 'entropy', 'log_loss'])
bootstrap = st.sidebar.selectbox('Bootstrap', [True, False])
oob_score = st.sidebar.selectbox('Use OOB Score', [False, True])
random_state = st.sidebar.slider('Random State', 0, 1000, 42)
n_jobs = st.sidebar.selectbox('n_jobs (parallelism)', [1, -1])

# ----------------------------
# Function to download result CSV
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">📥 Download Results as CSV</a>'

# ----------------------------
# Model building and tuning
def build_model(df):
    try:
        X = df.drop(columns=['Severity', 'Start_Time'], errors='ignore')
        Y = df['Severity']
    except KeyError:
        st.error("Dataset must include 'Severity' as target and 'Start_Time' column.")
        return

    st.markdown('### Target Variable')
    st.info(Y.name)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 - split_size, random_state=random_state)

    param_grid = {
        'n_estimators': np.arange(n_est_range[0], n_est_range[1] + 1, n_estimators_step),
        'max_features': list(range(max_feat_range[0], max_feat_range[1] + 1))
    }

    clf = RandomForestClassifier(
        criterion=criterion,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        oob_score=oob_score,
        random_state=random_state,
        n_jobs=n_jobs
    )

    grid = GridSearchCV(clf, param_grid, cv=3)
    grid.fit(X_train, Y_train)

    Y_pred = grid.predict(X_test)

    c1,c2 = st.columns()

    with c1:
        st.write('**R² Score:**', r2_score(Y_test, Y_pred))
        st.write('**Mean Squared Error:**', mean_squared_error(Y_test, Y_pred))
        st.write('**Best Parameters:**', grid.best_params_)
        st.write('**All Parameters:**')
        st.write(grid.get_params())
    
        # Process grid search results
        results_df = pd.concat(
            [pd.DataFrame(grid.cv_results_["params"]),
             pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])],
            axis=1
        )
        st.markdown(filedownload(results_df), unsafe_allow_html=True)

    with c2:

        grouped = results_df.groupby(['max_features', 'n_estimators']).mean().reset_index()
        pivot = grouped.pivot(index='max_features', columns='n_estimators', values='R2')
    
        x_vals = pivot.columns.values
        y_vals = pivot.index.values
        z_vals = pivot.values
    
        fig = go.Figure(data=[go.Surface(z=z_vals, x=x_vals, y=y_vals)])
        fig.update_layout(
            title='Hyperparameter Tuning 3D Surface',
            scene=dict(
                xaxis_title='n_estimators',
                yaxis_title='max_features',
                zaxis_title='R2 Score'
            ),
            autosize=False,
            width=800,
            height=800,
            margin=dict(l=50, r=50, b=65, t=90)
        )
        st.plotly_chart(fig)

# ----------------------------
# Main Panel
st.subheader('📄 Dataset Preview')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    build_model(df)
else:
    st.info("No dataset uploaded.")
    if st.button('Use Example Dataset'):
        # Adjust this path for deployment on Streamlit Cloud
        try:
            example_path = os.path.join(os.path.dirname(__file__), '..', 'US_Norm.csv')
            df = pd.read_csv(example_path)
        except:
            df = pd.read_csv("US_Norm.csv")  # fallback if root file exists
        st.write(df.head())
    
        st.subheader("🔍 Grid Search CV on RandomForest Classifier Model Performance")    
        with st.spinner("Generating profiling report... this may take a few seconds..."):
            build_model(df)
