import toml
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import warnings

warnings.filterwarnings('ignore')

def modify(primary="#bd4bff", back="#efe6e6", secondary="#f0f2f5", font="serif", text="black", dark=False):
    config_path = r".streamlit/config.toml"
    with open(config_path, "r") as f:
        config_data = toml.load(f)

    config_data["theme"]["primaryColor"] = primary
    config_data["theme"]["backgroundColor"] = back
    config_data["theme"]["secondaryBackgroundColor"] = secondary
    config_data["theme"]["textColor"] = text
    config_data["theme"]["font"] = font

    with open(config_path, "w") as f:
        toml.dump(config_data, f)

st.set_page_config(
    initial_sidebar_state="expanded",
    page_title="Change Application Themes",
    menu_items={
        'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
        'Report a bug': "mailto:a.k.mirsha9@gmail.com",
        'About': "### This is an extremely cool web application built as a part of my Data Science Mini Project on the 'US Accidents Dataset'"
    },
    page_icon="analysis.png",
    layout="wide"
)

# ----------------------------- Header -----------------------------
st.markdown("""<h1 style='text-align: center; color: #6A0DAD;'>🎨 Change Application Themes</h1>""", unsafe_allow_html=True)
add_vertical_space(2)

st.markdown('''##### :smile: This page lets you personalize your visual experience based on design principles of **Perception** and **Cognition**.''')
add_vertical_space(1)

st.warning("**Note:** Theme changes apply globally across all application tabs.", icon="🚨")
add_vertical_space(2)

# ------------------ Default Theme Options ------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("🔆 Default Light Theme")
    if st.button("Apply Default Light"):
        modify()

with c2:
    st.subheader("🌙 Default Dark Theme")
    if st.button("Apply Default Dark"):
        modify("#FF4B4B", "#0e1117", "#262730", text="#FAFAFA")

st.markdown("---")

# ------------------ Light Themes ------------------
st.subheader("💡 Light Themes Inspired by Perception Theory")
add_vertical_space(2)
c1, c2 = st.columns(2)

with c1:
    st.markdown("**🧠 Clear Understanding**")
    st.info("Teal promotes clarity and trust; ghost white aids sidebar navigation.")
    if st.button("Use Clear Light Theme"):
        modify("#008080", '#f0f0f0', '#f8f8ff')

with c2:
    st.markdown("**🎯 Compelling Appeal**")
    st.info("Tomato red grabs attention; beige sidebar improves navigation.")
    if st.button("Use Compelling Light Theme"):
        modify("#ff6347", '#f0f8ff', '#f5f5dc')

add_vertical_space(2)
c1, c2 = st.columns(2)

with c1:
    st.markdown("**📋 Professional Standard**")
    st.info("Dark gray conveys seriousness, white enhances readability.")
    if st.button("Use Professional Light Theme"):
        modify("#333333", '#ffffff', '#f0f0f0')

with c2:
    st.markdown("**🌈 Visually Attractive**")
    st.info("Orange energizes; peachy background welcomes users.")
    if st.button("Use Attractive Light Theme"):
        modify("#ffaa00", '#f9f6f2', '#f0e68c')

st.markdown("---")

# ------------------ Dark Themes ------------------
st.subheader("🌑 Dark Themes Inspired by Perception Theory")
add_vertical_space(2)
c1, c2 = st.columns(2)

with c1:
    st.markdown("**🧠 Clear Understanding**")
    st.info("Turquoise text on slate gray improves focus.")
    if st.button("Use Clear Dark Theme"):
        modify("#40E0D0", "#030637", "#3C0753", text="#EEEEEE")

with c2:
    st.markdown("**🎯 Compelling Appeal**")
    st.info("Orange-red pops out against dark surfaces.")
    if st.button("Use Compelling Dark Theme"):
        modify("#FF204E", "#222831", "#31363F", text="#F5EDED")

add_vertical_space(2)
c1, c2 = st.columns(2)

with c1:
    st.markdown("**📋 Professional Standard**")
    st.info("Light gray and black create a timeless layout.")
    if st.button("Use Professional Dark Theme"):
        modify("#CCCCCC", "#1a1a1a", "#2a2a2a", text="#EEEEEE")

with c2:
    st.markdown("**🌈 Visually Attractive**")
    st.info("Bold yellow over black attracts instant focus.")
    if st.button("Use Attractive Dark Theme"):
        modify("#FFD700", "#121212", "#222222", text="#F5EDED")

st.markdown("---")

# ------------------ Cognition Theory Themes ------------------
st.subheader("🧠 Themes Inspired by Visual Cognition Theory")
add_vertical_space(2)
c1, c2 = st.columns(2)

with c1:
    st.markdown("**⚡ Electric Colors**")
    st.info("Vibrant green and navy blue promote alertness.")
    if st.button("Use Electric Theme"):
        modify("#39FF14", "#0f0f0f", "#000080", text="#F5EDED")

with c2:
    st.markdown("**💡 Radiative Glow**")
    st.info("Yellow foregrounds and white backgrounds amplify clarity.")
    if st.button("Use Radiative Glow Theme"):
        modify("#ffcc00", "#ffffff", "#E3FEF7", text="brown")

add_vertical_space(2)
c1, c2 = st.columns(2)

with c1:
    st.markdown("**📦 Classic Neutrals**")
    st.info("Minimalist layout enhances comprehension.")
    if st.button("Use Classic Neutral Theme"):
        modify("#333333", "#f5f5f5", "#808080")

with c2:
    st.markdown("**🎨 Vivid Contrast**")
    st.info("Red, yellow, and blue combo boosts energy and focus.")
    if st.button("Use Vivid Contrast Theme"):
        modify("#ff0000", "#ffff00", "#8576FF")

add_vertical_space(3)

st.success("✨ Choose your favorite theme and refresh to see it applied across all pages!")
