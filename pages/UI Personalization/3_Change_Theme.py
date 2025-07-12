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
        'About': "### This is an extremely cool web application built as a part of my Data Science and Machine Learning Project on the 'US Accidents Dataset'"
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
    st.info("Teal is a calming yet assertive color that promotes clear communication and understanding. Light gray provides a neutral and balanced background, enhancing readability and ensuring that content stands out effectively. Ghost white offers a subtle distinction in the sidebar, helping users navigate through sections with clarity.")
    if st.button("Use Clear Light Theme"):
        modify("#008080", '#f0f0f0', '#f8f8ff')

with c2:
    st.markdown("**🎯 Compelling Appeal**")
    st.info("The vibrant tomato red draws attention and creates a sense of urgency or excitement, making it compelling for users to engage with important elements. Alice blue provides a clean and soothing background, ensuring that the content remains easy to read and visually appealing. Beige offers a subtle contrast in the sidebar, aiding in navigation without overshadowing the main content.")
    if st.button("Use Compelling Light Theme"):
        modify("#ff6347", '#f0f8ff', '#f5f5dc')

add_vertical_space(2)
c1, c2 = st.columns(2)

with c1:
    st.markdown("**📋 Professional Standard**")
    st.info("Dark gray exudes professionalism and sophistication while maintaining readability and visual hierarchy. White serves as a clean and timeless background, emphasizing content and ensuring a professional aesthetic. Light gray in the sidebar provides a subtle contrast and organization without detracting from the main focus.")
    if st.button("Use Professional Light Theme"):
        modify("#333333", '#ffffff', '#f0f0f0')

with c2:
    st.markdown("**🌈 Visually Attractive**")
    st.info("Vivid orange is a bold and energetic color that immediately captures attention and stimulates interest. Light peach provides a soft and inviting background, creating a warm and welcoming atmosphere while ensuring readability. Khaki in the sidebar offers a complementary contrast, enhancing visual appeal and encouraging exploration of the interface.")
    if st.button("Use Attractive Light Theme"):
        modify("#ffaa00", '#f9f6f2', '#f0e68c')

st.markdown("---")

# ------------------ Dark Themes ------------------
st.subheader("🌑 Dark Themes Inspired by Perception Theory")
add_vertical_space(2)
c1, c2 = st.columns(2)

with c1:
    st.markdown("**🧠 Clear Understanding**")
    st.info("Turquoise promotes clear communication and understanding, ensuring that important information stands out effectively. Dark slate gray creates a visually appealing contrast while maintaining a professional appearance. Gunmetal in the sidebar offers a subtle distinction, aiding in navigation and organization without overwhelming the user.")
    if st.button("Use Clear Dark Theme"):
        modify("#40E0D0", "#030637", "#3C0753", text="#EEEEEE")

with c2:
    st.markdown("**🎯 Compelling Appeal**")
    st.info("The vibrant orange-red stands out against the dark background, immediately drawing the user's attention. The dark gray background provides a sleek and modern appearance while enhancing readability. The charcoal sidebar offers a subtle contrast, making it easy to navigate through sections without distracting from the main content.")
    if st.button("Use Compelling Dark Theme"):
        modify("#FF204E", "#222831", "#31363F", text="#F5EDED")

add_vertical_space(2)
c1, c2 = st.columns(2)

with c1:
    st.markdown("**📋 Professional Standard**")
    st.info("Light gray exudes professionalism and sophistication while ensuring readability and visual hierarchy. Black provides a classic and timeless background, emphasizing content and maintaining a professional aesthetic. Dark slate gray in the sidebar offers a subtle contrast and organization without detracting from the main focus.")
    if st.button("Use Professional Dark Theme"):
        modify("#CCCCCC", "#1a1a1a", "#2a2a2a", text="#EEEEEE")

with c2:
    st.markdown("**🌈 Visually Attractive**")
    st.info("Vivid yellow is bold and eye-catching, instantly capturing attention and creating visual interest. Midnight black offers a dramatic and striking background, providing a sleek and modern appearance. Dark charcoal in the sidebar complements the primary color, enhancing visual appeal and encouraging exploration of the interface.")
    if st.button("Use Attractive Dark Theme"):
        modify("#FFD700", "#121212", "#222222", text="#F5EDED")

st.markdown("---")

# ------------------ Cognition Theory Themes ------------------
st.subheader("🧠 Themes Inspired by Visual Cognition Theory")
add_vertical_space(2)
c1, c2 = st.columns(2)

with c1:
    st.markdown("**⚡ Electric Colors**")
    st.info("Electric green is a vibrant and attention-grabbing color that stimulates excitement and engagement. Black provides a dramatic and high-contrast background, enhancing the visibility of content and creating a sense of urgency. Navy blue in the sidebar offers a complementary contrast, aiding navigation without overwhelming the user.")
    if st.button("Use Electric Theme"):
        modify("#39FF14", "#0f0f0f", "#000080", text="#F5EDED")

with c2:
    st.markdown("**💡 Radiative Glow**")
    st.info("Radiant yellow promotes clarity and comprehension, ensuring that important information stands out effectively. White offers a clean and minimalist background, enhancing readability and creating a sense of openness. Salmon in the sidebar provides a subtle yet warm contrast, aiding navigation and organization without detracting from the main focus.")
    if st.button("Use Radiative Glow Theme"):
        modify("#ffcc00", "#ffffff", "#E3FEF7", text="brown")

add_vertical_space(2)
c1, c2 = st.columns(2)

with c1:
    st.markdown("**📦 Classic Neutrals**")
    st.info("Dark gray exudes professionalism and sophistication while maintaining readability and visual hierarchy. Light gray provides a neutral and balanced background, ensuring content stands out effectively. Medium gray in the sidebar offers a subtle contrast and organization, enhancing navigation without overshadowing the main content.")
    if st.button("Use Classic Neutral Theme"):
        modify("#333333", "#f5f5f5", "#808080")

with c2:
    st.markdown("**🎨 Vivid Contrast**")
    st.info("Vivid red is bold and attention-grabbing, instantly capturing attention and creating visual interest. Vivid yellow offers a striking and high-contrast background, enhancing visibility and drawing the user's focus. Vivid blue in the sidebar complements the primary color, creating a visually appealing contrast and encouraging exploration of the interface.")
    if st.button("Use Vivid Contrast Theme"):
        modify("#ff0000", "#ffff00", "#8576FF")

add_vertical_space(3)

st.success("✨ Choose your favorite theme and refresh to see it applied across all pages!")
