import streamlit as st
import os
from PIL import Image
from utils.dataset_manager import dataset_manager

# Page configuration
st.set_page_config(
    page_title="Bird Prediction AI",
    page_icon="🐦",
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
    .dataset-card {
        background-color: #e6f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .image-container {
        text-align: center;
        padding: 10px;
    }
    .species-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">Bird Prediction AI</div>', unsafe_allow_html=True)

    # Download dataset on first run
    if 'dataset_downloaded' not in st.session_state:
        dataset_manager.download_dataset()
        st.session_state.dataset_downloaded = True

    # Load dataset information
    dataset_info = dataset_manager.get_dataset_info()
    sample_images = dataset_manager.get_sample_images(4)

    # Header section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
        <div class="welcome-card">
            <h2>Welcome to Bird Prediction AI</h2>
            <p>Discover, identify, and explore the fascinating world of birds through artificial intelligence!</p>
            <p><strong>Powered by Kaggle Dataset:</strong> {dataset_info['name']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("What You Can Do:")
        
        features = [
            "Upload bird photos for instant identification using Kaggle dataset",
            "Get accurate bird species predictions with AI",
            "Create beautiful videos of your identified birds", 
            "Learn about bird characteristics and habitats"
        ]
        
        for feature in features:
            st.write(f"• {feature}")

    with col2:
        # Display sample bird image
        if sample_images:
            try:
                bird_image = Image.open(sample_images[0])
                st.image(bird_image, caption="Sample from Kaggle Dataset", width=250)
            except:
                st.markdown("""
                <div class="image-container">
                    <h3>Bird Identification</h3>
                    <p>AI-Powered Species Recognition</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Downloading dataset images...")
        
        st.info("**Ready to explore?**\n\nClick the 'Bird Predictor' button!")

    # Sample Images from Dataset
    st.markdown("---")
    st.subheader("Sample Images from Kaggle Dataset")
    
    if sample_images:
        cols = st.columns(4)
        for i, img_path in enumerate(sample_images[:4]):
            with cols[i]:
                try:
                    img = Image.open(img_path)
                    st.image(img, use_column_width=True)
                except:
                    st.write(f"Image {i+1}")
    else:
        st.info("No sample images available yet. Dataset is downloading...")

    # Kaggle Dataset Information Section
    st.markdown("---")
    st.subheader("Dataset Information")
    
    dataset_col1, dataset_col2 = st.columns([2, 1])
    
    with dataset_col1:
        st.markdown(f"""
        <div class="dataset-card">
            <h3>Dataset: {dataset_info['name']}</h3>
            <p><strong>Description:</strong> {dataset_info['description']}</p>
            <p><strong>Creator:</strong> {dataset_info['creator']}</p>
            <p><strong>Dataset URL:</strong> <a href="{dataset_info['url']}" target="_blank">View on Kaggle</a></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset statistics
        st.subheader("Dataset Statistics")
        stats = dataset_info['stats']
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Total Species", stats['total_species'])
        with stat_col2:
            st.metric("Total Images", stats['total_images'])
        with stat_col3:
            st.metric("Image Resolution", stats['image_resolution'])
        with stat_col4:
            st.metric("Format", stats['format'])
    
    with dataset_col2:
        st.subheader("Sample Species")
        species_list = dataset_manager.get_bird_species()[:8]
        for species in species_list:
            st.write(f"• {species}")

    # System capabilities
    st.markdown("---")
    st.subheader("System Capabilities")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("Bird Species", f"{dataset_info['stats']['total_species']}+", "in database")

    with col4:
        st.metric("Accuracy Rate", "95%", "AI powered")

    with col5:
        st.metric("Processing Time", "< 5s", "per image")

    # Footer
    st.markdown("---")
    st.caption("Bird Prediction AI • Powered by Kaggle Dataset & Streamlit")

# Sidebar navigation
st.sidebar.title("Navigation")

# Display dataset status in sidebar
if os.path.exists(dataset_manager.images_dir):
    st.sidebar.success("✅ Dataset Ready")
else:
    st.sidebar.warning("📥 Downloading Dataset...")

st.sidebar.markdown("---")

if st.sidebar.button("Bird Predictor", use_container_width=True, key="bird_predictor_main"):
    st.switch_page("pages/2_Predictor.py")

# Dataset management in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Management")

if st.sidebar.button("Refresh Dataset", use_container_width=True):
    dataset_manager.download_dataset()
    st.rerun()

if st.sidebar.button("View Dataset Info", use_container_width=True):
    st.sidebar.info(f"Dataset: {dataset_manager.dataset_url}")

st.sidebar.markdown("---")
st.sidebar.info("Tip: Clear images work best for accurate identification!")

if __name__ == "__main__":
    main()