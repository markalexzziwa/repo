import streamlit as st
import datetime

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
    st.info("**Ready to explore?**\n\nClick the button below to start predicting!")

# Get started button
st.markdown("---")
st.subheader("🚀 Get Started")

if st.button("🎬 Start Bird Prediction", type="primary", use_container_width=True):
    st.success("Redirecting to Bird Predictor...")
    st.switch_page("pages/2_🦜_Bird_Predictor.py")

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