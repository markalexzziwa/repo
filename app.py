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

# Main welcome page content
def main():
    st.markdown('<div class="main-header">🦜 Bird Prediction AI</div>', unsafe_allow_html=True)

    # Header section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="welcome-card">
            <h2>🌟 Welcome to Bird Prediction AI</h2>
            <p>Discover, identify, and explore the fascinating world of birds through artificial intelligence!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🎯 What You Can Do:")
        
        features = [
            "**🖼️ Upload Images** - Upload bird photos for instant identification",
            "**🤖 AI Prediction** - Get accurate bird species predictions",
            "**🎥 Video Generation** - Create beautiful videos of your identified birds",
            "**📊 Detailed Analysis** - Learn about bird characteristics and habitats"
        ]
        
        for feature in features:
            st.write(f"• {feature}")

    with col2:
        st.image("🦅", width=200)
        st.info("**Ready to explore?**\n\nUse the sidebar to navigate to the predictor!")

    # Quick stats section
    st.markdown("---")
    st.subheader("📈 Quick Stats")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("Bird Species", "10,000+", "in database")

    with col4:
        st.metric("Accuracy Rate", "95%", "AI powered")

    with col5:
        st.metric("Processing Time", "< 5s", "per image")

    # Footer
    st.markdown("---")
    st.caption("🦜 Bird Prediction AI • Powered by Streamlit & AI Technology")

# Sidebar navigation
st.sidebar.title("🚀 Navigation")
st.sidebar.markdown("---")

if st.sidebar.button("🏠 Home", use_container_width=True):
    st.rerun()

if st.sidebar.button("🦜 Bird Predictor", use_container_width=True):
    st.switch_page("pages/2_Predictor.py")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Upload clear bird images for best results!")

if __name__ == "__main__":
    main()