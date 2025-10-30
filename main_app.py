import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Bird Prediction AI",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
    }
    .welcome-card {
        background-color: #f0f8f0;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)

# Redirect to welcome page
st.switch_page("pages/1_🏠_Welcome.py")