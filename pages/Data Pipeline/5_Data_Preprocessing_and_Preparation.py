# 5_Data_Preprocessing_and_Preparation.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from streamlit_extras.add_vertical_space import add_vertical_space
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ------------------- PAGE CONFIG -----------------------
st.set_page_config(
    initial_sidebar_state="expanded",
    page_title="Data Preprocessing and Preparation for Exploration",
    page_icon="analysis.png",
    layout="wide",
    menu_items={
        'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
        'Report a bug': "mailto:a.k.mirsha9@gmail.com",
        'About': "### Web app built as part of my Data Science Mini Project on the US Accidents Dataset."
    }
)

# ------------------- HEADER -----------------------
st.markdown('# 🧹 **Data Preprocessing and Preparation**')
add_vertical_space(2)
st.markdown("##### 😄 This window handles Normalization, Binning, and Verification Techniques for further EDA.")
add_vertical_space(3)

# ------------------- FILE UPLOAD -----------------------
c1, c2, c3 = st.columns(3)
with c2:
    fl = st.file_uploader("📂 Upload your file", type=["csv", "xlsx", "xls"])
if fl is not None:
    df = pd.read_csv(fl, encoding="ISO-8859-1")
    st.success("✅ File Uploaded Successfully!")
else:
    df = pd.read_csv("US_Accidents.csv", encoding="ISO-8859-1")
    st.info("ℹ️ Using sample dataset: `US_Accidents.csv`")

add_vertical_space(3)

# ------------------- NORMALIZATION -----------------------
st.markdown("## ⚖️ Normalization Process")
add_vertical_space(2)
st.markdown("""Normalization is a key preprocessing step for scaling features prior to modeling.  
We use it when data does not follow a normal (Gaussian) distribution.""")

add_vertical_space(2)

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
data = df.dropna()

c1, c2 = st.columns(2)

with c1:
    st.markdown('### 🔹 Min-Max Normalization')
    attribute = st.selectbox("Select attribute for Min-Max Normalization:", numerical_columns, index=6)
    scaler = MinMaxScaler()
    minmax_scaled = scaler.fit_transform(data[[attribute]])

    add_vertical_space(1)
    st.markdown("**📊 Histogram Comparison:**")
    fig1 = px.histogram(data[attribute], title='Original Data', color_discrete_sequence=px.colors.sequential.Agsunset)
    fig2 = px.histogram(minmax_scaled.flatten(), title='Min-Max Normalized', color_discrete_sequence=px.colors.sequential.Agsunset)
    st.plotly_chart(fig1, use_container_width=True, key=1)
    st.plotly_chart(fig2, use_container_width=True, key=2)

    add_vertical_space(1)
    st.markdown("**🧪 Shapiro-Wilk Test (Normalized Data):**")
    w_stat, p_val = stats.shapiro(minmax_scaled.flatten())
    st.write(f"W Statistic: `{w_stat:.4f}`, P-value: `{p_val:.4f}`")

    add_vertical_space(1)
    st.markdown("### 📈 Model Performance Comparison")

    st.markdown("#### 🔹 K-Nearest Neighbors (Distance-Based Model)")
    knn = KNeighborsClassifier()
    score_orig_knn = cross_val_score(knn, data[[attribute]], data['Severity'], cv=2).mean()
    score_scaled_knn = cross_val_score(knn, minmax_scaled, data['Severity'], cv=2).mean()
    st.write(f"Original: `{score_orig_knn:.4f}` | Normalized: `{score_scaled_knn:.4f}`")

    st.markdown("##### _Observation_:")
    st.info("Normalization improves the KNN model accuracy because it relies on distance-based calculations, and unscaled features can distort the distance metrics.")

    add_vertical_space(1)

    st.markdown("#### 🔸 Random Forest (Ensemble Tree-Based Model)")
    rf = RandomForestClassifier()
    score_orig_rf = cross_val_score(rf, data[[attribute]], data['Severity'], cv=2).mean()
    score_scaled_rf = cross_val_score(rf, minmax_scaled, data['Severity'], cv=2).mean()
    st.write(f"Original: `{score_orig_rf:.4f}` | Normalized: `{score_scaled_rf:.4f}`")

    st.markdown("##### _Observation_:")
    st.info("Random Forest is scale-invariant. Hence, normalization has negligible impact on its performance.")

with c2:
    st.markdown('### 🔸 Standard Normalization')
    attribute2 = st.selectbox("Select attribute for Standardization:", numerical_columns, index=6)
    std_scaler = StandardScaler()
    std_scaled = std_scaler.fit_transform(data[[attribute2]])

    add_vertical_space(1)
    st.markdown("**📊 Histogram Comparison:**")
    fig3 = px.histogram(data[attribute2], title='Original Data', color_discrete_sequence=px.colors.sequential.Agsunset)
    fig4 = px.histogram(std_scaled.flatten(), title='Standardized Data', color_discrete_sequence=px.colors.sequential.Agsunset)
    st.plotly_chart(fig3, use_container_width=True, key=3)
    st.plotly_chart(fig4, use_container_width=True, key=4)

    add_vertical_space(1)
    st.markdown("**🧪 Shapiro-Wilk Test (Standardized Data):**")
    w_stat2, p_val2 = stats.shapiro(std_scaled.flatten())
    st.write(f"W Statistic: `{w_stat2:.4f}`, P-value: `{p_val2:.4f}`")

    add_vertical_space(1)
    st.markdown("### 📈 Model Performance Comparison")

    st.markdown("#### 🔹 K-Nearest Neighbors (Distance-Based Model)")
    knn2 = KNeighborsClassifier()
    score_orig_knn2 = cross_val_score(knn2, data[[attribute2]], data['Severity'], cv=2).mean()
    score_scaled_knn2 = cross_val_score(knn2, std_scaled, data['Severity'], cv=2).mean()
    st.write(f"Original: `{score_orig_knn2:.4f}` | Normalized: `{score_scaled_knn2:.4f}`")

    st.markdown("##### _Observation_:")
    st.info("Standardization significantly helps KNN by transforming features to comparable scales, avoiding bias from higher magnitude features.")

    add_vertical_space(1)

    st.markdown("#### 🔸 Random Forest (Ensemble Tree-Based Model)")
    rf2 = RandomForestClassifier()
    score_orig_rf2 = cross_val_score(rf2, data[[attribute2]], data['Severity'], cv=2).mean()
    score_scaled_rf2 = cross_val_score(rf2, std_scaled, data['Severity'], cv=2).mean()
    st.write(f"Original: `{score_orig_rf2:.4f}` | Normalized: `{score_scaled_rf2:.4f}`")

    st.markdown("##### _Observation_:")
    st.info("Random Forest remains unaffected by normalization, proving its robustness against scale variance.")

add_vertical_space(4)


# ------------------- QQ Plot and Normality Check -----------------------
st.markdown("## 📊 Normality Verification via Statistics")
add_vertical_space(1)

w_stat_n, p_val_n = stats.shapiro(std_scaled.flatten())
w_stat_raw, p_val_raw = stats.shapiro(data[attribute2])

st.write("### Shapiro-Wilk Test Summary")
st.markdown(f"""
- **Normalized (`{attribute2}`):** W = `{w_stat_n:.4f}`, P-value = `{p_val_n:.4f}`  
- **Original (`{attribute2}`):** W = `{w_stat_raw:.4f}`, P-value = `{p_val_raw:.4f}`  
""")

st.markdown("""
> - **W closer to 1** = more normal  
> - **P > 0.05** ⇒ likely normal  
> - **P ≤ 0.05** ⇒ not normally distributed
""")

add_vertical_space(3)
st.markdown("### 📈 Quantile-Quantile (Q-Q) Plot")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
stats.probplot(data[attribute2], dist="norm", plot=axes[0])
axes[0].set_title("Q-Q Plot: Original Data")
stats.probplot(std_scaled.flatten(), dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot: Standardized Data")
st.pyplot(fig)

add_vertical_space(4)

# ------------------- BINNING -----------------------
st.markdown("## 🧱 Interactive Binning")
add_vertical_space(1)

def perform_binning(data, column_name, labels, value_ranges):
    bins = pd.cut(data[column_name], bins=value_ranges, labels=labels)
    data['{}_Bin'.format(column_name)] = bins
    return data

def plot_binned_column(data, column_name):
    bar = px.bar(
        x=data[column_name].value_counts().index,
        y=data[column_name].value_counts().values,
        labels={'x': column_name, 'y': 'Count'},
        color_discrete_sequence=px.colors.sequential.Magenta
    )
    st.plotly_chart(bar,key=5)

st.subheader("🔢 Binning Options")
bin_col_name = st.selectbox("Select Numerical Attribute for Binning", numerical_columns, index=6)

if bin_col_name in df.columns:
    labels_input = st.text_input("Enter bin labels (comma-separated):", value="Very Cold,Cold,Moderate,Warm,Hot")
    value_ranges_input = st.text_input("Enter bin ranges (comma-separated):", value="-100,32,50,70,90,200")

    labels = [label.strip() for label in labels_input.split(',')]
    value_ranges = [float(val.strip()) for val in value_ranges_input.split(',')]

    if "binning" not in st.session_state:
        st.session_state["binning"] = False
        
    if st.button("✅ Perform Binning"):
        st.session_state["binning"] = True
        
    if st.session_state["binning"]:
        data = perform_binning(df, bin_col_name, labels, value_ranges)
        st.write("📋 Sample Binned Data")
        st.dataframe(data[[bin_col_name, f"{bin_col_name}_Bin"]].head())
        st.write("📊 Bin Distribution")
        plot_binned_column(data, f"{bin_col_name}_Bin")

add_vertical_space(5)
