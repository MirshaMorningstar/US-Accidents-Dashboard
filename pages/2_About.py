import streamlit as st

st.set_page_config(page_title="About this Project",
                   menu_items={
        'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
        'Report a bug': "mailto:a.k.mirsha9@gmail.com",
        'About': "### This is an extremely cool web application built as a part of my Data Science and Machine Learning Project on the 'US Accidents Dataset'"
    }, initial_sidebar_state="expanded")

st.markdown("""
# 📘 About This Project  
## 🚗 US Accidents Analysis & Machine Learning Prediction Platform  

st.info("✅ This is a public page — login is only required for the full dashboard. Please use the **Authentication** page in the sidebar to log in.")
---
### 👋 Introduction

Welcome to a full-stack, production-grade Streamlit application that seamlessly blends **data visualization**, **machine learning**, **UI personalization**, and **real-time interaction**. This interactive platform empowers users to explore, analyze, and predict accident severity trends across the United States using a rich dataset and a modern interface.

---

### 🎯 Project Objectives

- 🔍 **Understand** key trends, patterns, and risk indicators contributing to accidents  
- 📈 **Visualize** data quality, EDA, and feature correlations interactively  
- 🧠 **Model** accident severity with 25+ ML classifiers using LazyPredict  
- 🎯 **Predict** severity outcomes based on user-defined feature inputs  
- 🛠️ **Optimize** model hyperparameters using GridSearchCV and 3D performance surfaces  
- 🎨 **Personalize** the UI/UX with cognitive design-based theme selectors and color extractors  
- 🔐 **Secure** the app with user authentication and session handling  

---

### 🧱 Key Modules Overview

| 🧩 Module | 🔍 Description |
|----------|----------------|
| 🎨 **UI Personalization** | Cognitive-based theme customizer & image-based color palette picker |
| 🧹 **Data Preprocessing** | Normalization, binning, distribution checks, Q-Q plots |
| 🧪 **Data Quality Checks** | Missing values, cardinality, outliers – all visualized |
| 🧬 **Feature Engineering** | Correlation matrix, SelectKBest, PCA, and mutual info |
| 🔍 **Exploratory Visualizations** | General and inferential visual insights with map overlays |
| 🔎 **Specific Data Explorer** | Dynamic dataframe filtering, case-insensitive search |
| 📊 **Automated EDA** | Full profiling via ydata-profiling |
| 🤖 **ML Comparison** | LazyPredict-based benchmarking of 25+ classifier models |
| 🎯 **Custom Prediction** | Live prediction interface with model and attribute selection |
| 🔧 **Hyperparameter Tuning** | GridSearchCV with 3D visualization for RandomForest |

---

### 💡 Unique Value Propositions

- ✅ End-to-end ML pipeline: data ingestion → EDA → ML → tuning → prediction  
- 🧩 Modular and navigable: Each component works standalone or in full pipeline
- 🌈 UI/UX focused design with cognitive science principles  
- ⚙️ Live feedback: Interact with sliders, filters, and dropdowns in real time  
- 📦 Deployable: Built and Deployed on Streamlit Cloud 
- 🔐 Secure: Authenticated access with password hashing using `streamlit-authenticator`  
- 🧪 Machine Learning Laboratory: Side-by-side evaluation of 25 classifiers + interactive prediction tool + Hyperparameter Optimisation Playground

---

### 📊 Dataset Citation & Information

The application is powered by the **US Accidents (2016–2023)** dataset from Kaggle, containing over **7.7 million records** gathered from traffic APIs across 49 US states.

#### 📚 Citation (must cite if reused):

> Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath.  
> “A Countrywide Traffic Accident Dataset.”, 2019.  
>  
> Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath.  
> "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights."  
> ACM SIGSPATIAL, 2019.

📌 License: CC BY-NC-SA 4.0  
🔗 [Kaggle Dataset Page](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

---

### 📌 Real-World Applications

- 🛣️ **Accident Risk Assessment**: Identify critical factors influencing accident severity  
- 🧪 **ML Research Platform**: Benchmark classifiers on real-world, imbalanced data  
- 🛣️ **City Planning**: Analyze and reduce accident hotspots 
- 🧠 **Educational Tool**: Perfect for teaching end-to-end applied ML, Data Science in real time  
- 📉 **Optimization Research Lab**: Experiment with hyperparameter tuning, Feature testing, real-time predictions and model behavior   
- 🚀 **ML Prototyping**: Quickly compare and tune models  
- 🧍 **User-Centered Design**: Themes, palettes, and customizations

---

### 🙌 Built With Passion

This platform is a reflection of my passion for **AI-driven analytics**, UI design, and practical problem-solving. It is more than a project — it's a deployable, interactive, intelligent system.

---

Built with ❤️ by **Mirsha Morningstar**
- 🧠 [GitHub](https://github.com/MirshaMorningstar)  
- 💼 [LinkedIn](https://www.linkedin.com/in/mirshamorningstar)  

---

🛠 *For any questions or collaborations, feel free to reach out via GitHub or LinkedIn.*  
""")
