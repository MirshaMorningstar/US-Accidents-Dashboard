import toml
import streamlit as st
import streamlit_extras as stex
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.dataframe_explorer import dataframe_explorer
import plotly.express as px
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_pca_correlation_graph
import os
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

from streamlit_option_menu import option_menu

warnings.filterwarnings('ignore')

def modify(primary="#bd4bff",back="#efe6e6",secondary="#f0f2f5",font="serif",text="black",dark = False):
    # Path to your config.toml file
    config_path = r".streamlit/config.toml"

    # Read the contents of config.toml
    with open(config_path, "r") as f:
        config_data = toml.load(f)


    # Modify the theme section as per your requirements
    config_data["theme"]["primaryColor"] = primary  # Sample: Purple
    config_data["theme"]["backgroundColor"] = back  # Sample: Light Gray
    config_data["theme"]["secondaryBackgroundColor"] = secondary  # Sample: Light Blueish Gray
    config_data["theme"]["textColor"] = text  # Sample: Blue
    config_data["theme"]["font"] = font  # Sample: Serif Font

    # Write back the modified contents to config.toml
    with open(config_path, "w") as f:
        toml.dump(config_data, f)
    

st.set_page_config(initial_sidebar_state="collapsed",page_title= " Change Application Themes ",
        menu_items={
         'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
         'Report a bug': "mailto:a.k.mirsha9@gmail.com",
         'About': "### This is an extremely cool web application built as a part of my Data Science Mini Project on the ' US Accidents Dataset '\n"
     },page_icon="analysis.png",layout="wide")



st.markdown('''# **Change Themes based on Principles of Perception and Cognition**''')
add_vertical_space(2)
st.markdown('''##### :smile: This Window is specifically generated to change the themes of the entire application based on the user's choice of preference :balloon: ''' )

add_vertical_space(1)

st.warning("**Note:** The Themes applied in this Window shall be reflected throughout the web application ( i.e across all tabs ) ",icon = "🚨")

add_vertical_space(2)
c1,c2 = st.columns(2)
with c1:
    st.markdown("#### Set to the Defualt Light Theme of the app")
    add_vertical_space(1)
    if st.button("Set Theme to Default Light"):
        modify()
with c2:
    st.markdown("#### Set to the Defualt Dark Theme of the app")
    add_vertical_space(1)
    if st.button("Set Theme to Default Dark"):
        modify("#FF4B4B","#0e1117","#262730",text="#FAFAFA")


add_vertical_space(5)


st.markdown("### Light Themes on Visual Theory of Perception")
add_vertical_space(2)
c1,c2 = st.columns(2)

with c1:
    st.markdown("###### Clear Understanding to the User ")
    st.write("Teal is a calming yet assertive color that promotes clear communication and understanding. Light gray provides a neutral and balanced background, enhancing readability and ensuring that content stands out effectively. Ghost white offers a subtle distinction in the sidebar, helping users navigate through sections with clarity.")
    add_vertical_space(1)
    if st.button("Use Clear Light Theme"):
        modify("#008080",'#f0f0f0','#f8f8ff')    
        

with c2:
    st.markdown("###### Compelling to the User ")
    st.write(" The vibrant tomato red draws attention and creates a sense of urgency or excitement, making it compelling for users to engage with important elements. Alice blue provides a clean and soothing background, ensuring that the content remains easy to read and visually appealing. Beige offers a subtle contrast in the sidebar, aiding in navigation without overshadowing the main content.")
    add_vertical_space(1)
    if st.button("Use Compelling Light Theme"):
        modify("#ff6347",'#f0f8ff','#f5f5dc')


add_vertical_space(3)

c1,c2 = st.columns(2)

with c1:
    st.markdown("###### Professional Standard: ")
    st.write("Dark gray exudes professionalism and sophistication while maintaining readability and visual hierarchy. White serves as a clean and timeless background, emphasizing content and ensuring a professional aesthetic. Light gray in the sidebar provides a subtle contrast and organization without detracting from the main focus.")
    add_vertical_space(1)
    if st.button("Use Professional Light Theme"):
        modify("#333333",'#ffffff','#f0f0f0')

    add_vertical_space(5)

with c2:
    st.markdown("###### Visually Attractive and Eye-catching ")
    st.write("Vivid orange is a bold and energetic color that immediately captures attention and stimulates interest. Light peach provides a soft and inviting background, creating a warm and welcoming atmosphere while ensuring readability. Khaki in the sidebar offers a complementary contrast, enhancing visual appeal and encouraging exploration of the interface.")
    add_vertical_space(1)
    if st.button("Use Attractive Light Theme"):
        modify("#ffaa00",'#f9f6f2','#f0e68c')

add_vertical_space(5)



st.markdown("### Dark Themes on Visual Theory of Perception")
add_vertical_space(2)
c1, c2 = st.columns(2)

with c2:
    st.markdown("###### Compelling to the User")
    st.write("The vibrant orange-red stands out against the dark background, immediately drawing the user's attention. The dark gray background provides a sleek and modern appearance while enhancing readability. The charcoal sidebar offers a subtle contrast, making it easy to navigate through sections without distracting from the main content.")
    add_vertical_space(1)
    if st.button("Use Compelling Dark Theme"):
        modify("#FF204E", "#222831", "#31363F",text="#F5EDED")

with c1:
    st.markdown("###### Clear Understanding to the User")
    st.write("Turquoise promotes clear communication and understanding, ensuring that important information stands out effectively. Dark slate gray creates a visually appealing contrast while maintaining a professional appearance. Gunmetal in the sidebar offers a subtle distinction, aiding in navigation and organization without overwhelming the user.")
    add_vertical_space(1)
    if st.button("Use Clear Dark Theme"):
        modify("#40E0D0", "#030637", "#3C0753",text="#EEEEEE")

add_vertical_space(3)

c1, c2 = st.columns(2)

with c1:
    st.markdown("###### Professional Standard")
    st.write("Light gray exudes professionalism and sophistication while ensuring readability and visual hierarchy. Black provides a classic and timeless background, emphasizing content and maintaining a professional aesthetic. Dark slate gray in the sidebar offers a subtle contrast and organization without detracting from the main focus.")
    add_vertical_space(1)
    if st.button("Use Professional Dark Theme"):
        modify("#CCCCCC", "#1a1a1a", "#2a2a2a",text="#EEEEEE")

    add_vertical_space(5)

with c2:
    st.markdown("###### Visually Attractive and Eye-catching")
    st.write("Vivid yellow is bold and eye-catching, instantly capturing attention and creating visual interest. Midnight black offers a dramatic and striking background, providing a sleek and modern appearance. Dark charcoal in the sidebar complements the primary color, enhancing visual appeal and encouraging exploration of the interface.")
    add_vertical_space(1)
    if st.button("Use Attractive Dark Theme"):
        modify("#FFD700", "#121212", "#222222",text="#F5EDED")

add_vertical_space(5)






st.markdown("### Themes on Visual Theory of Cognition")
add_vertical_space(2)
c1, c2 = st.columns(2)

with c2:
    st.markdown("###### Electric Colors")
    st.write("Electric green is a vibrant and attention-grabbing color that stimulates excitement and engagement. Black provides a dramatic and high-contrast background, enhancing the visibility of content and creating a sense of urgency. Navy blue in the sidebar offers a complementary contrast, aiding navigation without overwhelming the user.")
    add_vertical_space(1)
    if st.button("Use Electric Colors Theme"):
        modify("FF3EA5",  "#8576FF", "#41C9E2",text="#F5EDED")

with c1:
    st.markdown("###### Radiative Glow Colors")
    st.write("Radiant yellow promotes clarity and comprehension, ensuring that important information stands out effectively. White offers a clean and minimalist background, enhancing readability and creating a sense of openness. Salmon in the sidebar provides a subtle yet warm contrast, aiding navigation and organization without detracting from the main focus.")
    add_vertical_space(1)
    if st.button("Use Radioactive Glow Theme"):
        modify("#ffcc00", "#ffffff", "#E3FEF7",text="brown")

add_vertical_space(3)
c1, c2 = st.columns(2)

with c1:
    st.markdown("###### Classic Neutrals Standard")
    st.write("Dark gray exudes professionalism and sophistication while maintaining readability and visual hierarchy. Light gray provides a neutral and balanced background, ensuring content stands out effectively. Medium gray in the sidebar offers a subtle contrast and organization, enhancing navigation without overshadowing the main content.")
    add_vertical_space(1)
    if st.button("Use Classic Neutrals Theme"):
        modify("#333333", "#f5f5f5", "#808080")

    add_vertical_space(5)

with c2:
    st.markdown("###### Vivid Contrast")
    st.write("Vivid red is bold and attention-grabbing, instantly capturing attention and creating visual interest. Vivid yellow offers a striking and high-contrast background, enhancing visibility and drawing the user's focus. Vivid blue in the sidebar complements the primary color, creating a visually appealing contrast and encouraging exploration of the interface.")
    add_vertical_space(1)
    if st.button("Use Vivid Contrast Theme"):
        modify("#ff0000", "#ffff00", "#8576FF")





