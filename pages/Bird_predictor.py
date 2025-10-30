import streamlit as st
import time
import random
import os
from PIL import Image
import io

st.title("🦜 Bird Species Predictor")
st.write("Upload bird images and generate prediction videos")

# File upload section
st.markdown("---")
st.subheader("📁 Upload Bird Image")

uploaded_file = st.file_uploader(
    "Choose a bird image...",
    type=['jpg', 'jpeg', 'png', 'webp'],
    help="Upload a clear image of a bird for best results"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)
    
    # Prediction section
    st.markdown("---")
    st.subheader("🔍 AI Analysis")
    
    with st.spinner("Analyzing bird image..."):
        progress_bar = st.progress(0)
        
        # Simulate analysis process
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
        
        # Mock prediction results
        bird_species = ["Bald Eagle", "Blue Jay", "Cardinal", "Hummingbird", "Robin", "Sparrow"]
        confidence_scores = [92.5, 85.3, 78.9, 95.1, 88.7, 82.4]
        
        predicted_bird = random.choice(bird_species)
        confidence = random.choice(confidence_scores)
    
    # Display prediction results
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"**Predicted Species:** {predicted_bird}")
        st.info(f"**Confidence Score:** {confidence}%")
    
    with col2:
        st.metric("Identification Status", "COMPLETED", "100%")
    
    # Bird details
    st.markdown("---")
    st.subheader("📋 Bird Information")
    
    bird_info = {
        "Bald Eagle": {"Habitat": "North America", "Diet": "Fish", "Status": "Protected"},
        "Blue Jay": {"Habitat": "Forests", "Diet": "Omnivore", "Status": "Common"},
        "Cardinal": {"Habitat": "Woodlands", "Diet": "Seeds", "Status": "Common"},
        "Hummingbird": {"Habitat": "Gardens", "Diet": "Nectar", "Status": "Migratory"},
        "Robin": {"Habitat": "Lawns", "Diet": "Insects", "Status": "Common"},
        "Sparrow": {"Habitat": "Urban", "Diet": "Seeds", "Status": "Abundant"}
    }
    
    if predicted_bird in bird_info:
        info = bird_info[predicted_bird]
        info_cols = st.columns(3)
        with info_cols[0]:
            st.write(f"**Habitat:** {info['Habitat']}")
        with info_cols[1]:
            st.write(f"**Diet:** {info['Diet']}")
        with info_cols[2]:
            st.write(f"**Status:** {info['Status']}")
    
    # Video generation section
    st.markdown("---")
    st.subheader("🎥 Generate Prediction Video")
    
    st.write("Create a beautiful video showcasing your bird prediction results:")
    
    video_options = st.multiselect(
        "Video Features:",
        ["Species Introduction", "Habitat Information", "Behavior Facts", "Conservation Status", "Fun Facts"],
        default=["Species Introduction", "Fun Facts"]
    )
    
    if st.button("🎬 Generate Video", type="primary"):
        with st.spinner("Generating your bird prediction video..."):
            # Simulate video generation
            video_progress = st.progress(0)
            status_text = st.empty()
            
            steps = [
                "Processing image data...",
                "Generating bird facts...",
                "Creating visual elements...",
                "Rendering video frames...",
                "Finalizing video output..."
            ]
            
            for i, step in enumerate(steps):
                status_text.text(f"🔄 {step}")
                time.sleep(1)
                video_progress.progress((i + 1) * 20)
            
            # Mock video generation completion
            status_text.success("✅ Video generated successfully!")
            
            # Display mock video player
            st.markdown("---")
            st.subheader("🎬 Your Bird Prediction Video")
            
            # Mock video player (in real app, this would be an actual video file)
            st.video("https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4")
            
            # Download button (mock)
            st.download_button(
                label="📥 Download Video",
                data=b"mock_video_data",  # In real app, this would be actual video data
                file_name=f"bird_prediction_{predicted_bird.replace(' ', '_')}.mp4",
                mime="video/mp4"
            )
    
    # New prediction option
    st.markdown("---")
    if st.button("🔄 Analyze Another Bird"):
        st.rerun()

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload a bird image to get started with prediction and video generation.")
    
    # Sample bird images section
    st.markdown("---")
    st.subheader("📸 Supported Bird Types")
    
    sample_cols = st.columns(4)
    sample_birds = ["🦅 Eagles", "🦜 Parrots", "🐦 Songbirds", "🦆 Waterfowl"]
    
    for i, bird in enumerate(sample_birds):
        with sample_cols[i]:
            st.write(f"**{bird}**")
            st.image("📷", width=100)

# Navigation
st.markdown("---")
col_nav1, col_nav2 = st.columns(2)
with col_nav1:
    if st.button("🏠 Back to Welcome Page"):
        st.switch_page("main_app.py")
with col_nav2:
    st.caption("🦜 Bird Prediction AI • Advanced Bird Identification System")