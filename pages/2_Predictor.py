import streamlit as st
import time
import random
from PIL import Image

st.set_page_config(
    page_title="Bird Predictor - Bird AI",
    page_icon="🦜",
    layout="wide"
)

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
        "Bald Eagle": {
            "Habitat": "North America", 
            "Diet": "Fish", 
            "Status": "Protected",
            "Fun Fact": "They can reach speeds of 160 km/h when diving for prey!"
        },
        "Blue Jay": {
            "Habitat": "Forests", 
            "Diet": "Omnivore", 
            "Status": "Common",
            "Fun Fact": "Blue Jays can mimic the calls of hawks to warn other birds!"
        },
        "Cardinal": {
            "Habitat": "Woodlands", 
            "Diet": "Seeds", 
            "Status": "Common",
            "Fun Fact": "Only male cardinals are bright red; females are pale brown!"
        },
        "Hummingbird": {
            "Habitat": "Gardens", 
            "Diet": "Nectar", 
            "Status": "Migratory",
            "Fun Fact": "They can fly backwards and their wings beat 50-200 times per second!"
        },
        "Robin": {
            "Habitat": "Lawns", 
            "Diet": "Insects", 
            "Status": "Common",
            "Fun Fact": "American Robins have about 3,500 feathers!"
        },
        "Sparrow": {
            "Habitat": "Urban", 
            "Diet": "Seeds", 
            "Status": "Abundant",
            "Fun Fact": "Sparrows dust bathe to keep their feathers clean!"
        }
    }
    
    if predicted_bird in bird_info:
        info = bird_info[predicted_bird]
        info_cols = st.columns(2)
        with info_cols[0]:
            st.write(f"**🌍 Habitat:** {info['Habitat']}")
            st.write(f"**🍽️ Diet:** {info['Diet']}")
            st.write(f"**📊 Status:** {info['Status']}")
        with info_cols[1]:
            st.write(f"**💡 Fun Fact:** {info['Fun Fact']}")
    
    # Video generation section
    st.markdown("---")
    st.subheader("🎥 Generate Prediction Video")
    
    st.write("Create a beautiful video showcasing your bird prediction results:")
    
    video_options = st.multiselect(
        "Select video content:",
        ["Species Introduction", "Habitat Information", "Behavior Facts", "Conservation Status", "Fun Facts"],
        default=["Species Introduction", "Fun Facts"]
    )
    
    video_duration = st.slider("Video duration (seconds):", 10, 60, 30)
    
    if st.button("🎬 Generate Video", type="primary", use_container_width=True):
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
                time.sleep(1.5)
                video_progress.progress((i + 1) * 20)
            
            # Mock video generation completion
            status_text.success("✅ Video generated successfully!")
            
            # Display mock video player
            st.markdown("---")
            st.subheader("🎬 Your Bird Prediction Video")
            
            # Video preview section
            col_vid1, col_vid2 = st.columns([2, 1])
            
            with col_vid1:
                st.info("**Video Preview:** Bird Identification Results")
                st.write("""
                **Video Content Includes:**
                - Bird species identification
                - Habitat and behavior information
                - Conservation status
                - Interesting facts
                """)
                
                # Mock video placeholder
                st.markdown("""
                <div style='background: linear-gradient(135deg, #667eea, #764ba2); 
                          padding: 100px; text-align: center; border-radius: 10px; 
                          color: white; margin: 20px 0;'>
                    <h3>🎥 Bird Prediction Video</h3>
                    <p>Duration: {} seconds</p>
                    <p>Content: {}</p>
                </div>
                """.format(video_duration, ", ".join(video_options)), unsafe_allow_html=True)
            
            with col_vid2:
                st.success("**Video Details**")
                st.write(f"**Duration:** {video_duration}s")
                st.write(f"**Species:** {predicted_bird}")
                st.write(f"**Confidence:** {confidence}%")
                st.write(f"**Features:** {len(video_options)}")
                
                # Download button (mock)
                st.download_button(
                    label="📥 Download Video",
                    data=b"mock_video_data",
                    file_name=f"bird_prediction_{predicted_bird.replace(' ', '_')}.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
    
    # New prediction option
    st.markdown("---")
    col_new1, col_new2 = st.columns(2)
    
    with col_new1:
        if st.button("🔄 Analyze Another Bird", use_container_width=True):
            st.rerun()
    
    with col_new2:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.switch_page("app.py")

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload a bird image to get started with prediction and video generation.")
    
    # Sample bird types section (without problematic image calls)
    st.markdown("---")
    st.subheader("📸 Supported Bird Types")
    
    sample_cols = st.columns(4)
    sample_birds = [
        {"emoji": "🦅", "name": "Eagles", "desc": "Large birds of prey"},
        {"emoji": "🦜", "name": "Parrots", "desc": "Colorful tropical birds"},
        {"emoji": "🐦", "name": "Songbirds", "desc": "Melodic singers"},
        {"emoji": "🦆", "name": "Waterfowl", "desc": "Aquatic birds"}
    ]
    
    for i, bird in enumerate(sample_birds):
        with sample_cols[i]:
            st.markdown(f"""
            <div style='text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 10px;'>
                <div style='font-size: 40px;'>{bird['emoji']}</div>
                <h4>{bird['name']}</h4>
                <p style='font-size: 12px; color: #666;'>{bird['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

# Simple sidebar navigation
with st.sidebar:
    st.title("🔍 Navigation")
    st.markdown("---")
    
    if st.button("🏠 Go to Home Page", use_container_width=True):
        st.switch_page("app.py")
    
    st.markdown("---")
    st.info("""
    **Tips for best results:**
    - Use clear, well-lit images
    - Focus on the bird's main features
    - Avoid blurry or distant shots
    """)

# Footer
st.markdown("---")
st.caption("🦜 Bird Prediction AI • Advanced Bird Identification System")