import streamlit as st


st.set_page_config(
    page_title="WikiText-2 Next Word Lab",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page(
        "ui_pages/1_Next_Word_Prediction.py",
        title="Next Word Prediction",
        icon="✨",
        default=True,
    ),
    st.Page(
        "ui_pages/2_Sentence_Continuation.py",
        title="Sentence Continuation",
        icon="✍️",
    ),
    st.Page(
        "ui_pages/3_Model_Comparison.py",
        title="Model Comparison",
        icon="📊",
    ),
    st.Page(
        "ui_pages/4_Performance_Dashboard.py",
        title="Performance Dashboard",
        icon="📈",
    ),
    st.Page(
        "ui_pages/5_About.py",
        title="About Project",
        icon="ℹ️",
    ),
]

navigation = st.navigation(pages)
navigation.run()
