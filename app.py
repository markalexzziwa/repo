import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import tempfile
import os
import random
from gtts import gTTS
import base64
import io
import numpy as np
import time

# Set page config
st.set_page_config(
    page_title="Bird Species Identifier",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .story-box {
        background-color: #fffaf0;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #ffd700;
        margin: 1rem 0;
    }
    .upload-area {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 3rem;
        text-align: center;
        background-color: #f8fff8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🦜 Bird Species Identifier & Story Generator</h1>', unsafe_allow_html=True)
st.markdown("### Upload a bird image to identify its species and generate an educational story!")

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

@st.cache_resource
def load_model():
    """Load the trained model"""
    st.info("🔄 Loading AI model...")
    
    # Use a pretrained model as base
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # For demo purposes - you would replace this with your actual trained model
    # model.fc = nn.Linear(model.fc.in_features, num_your_classes)
    # model.load_state_dict(torch.load('your_trained_model.pth', map_location='cpu'))
    
    model.eval()
    st.success("✅ Model loaded successfully!")
    return model

def preprocess_image(image):
    """Preprocess uploaded image for prediction"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_species(image, model):
    """Predict bird species from image"""
    try:
        input_tensor = preprocess_image(image)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            # For demo - simulate prediction probabilities
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Mock species list - replace with your actual species
        species_list = [
            "African Fish Eagle", "Great Blue Turaco", "Shoebill Stork", 
            "Marabou Stork", "Grey Crowned Crane", "African Jacana",
            "Superb Starling", "Lilac-breasted Roller", "African Grey Parrot",
            "Secretary Bird", "Hamerkop", "Pied Kingfisher"
        ]
        
        # Simulate prediction (replace with actual model prediction)
        predicted_species = random.choice(species_list)
        confidence = random.uniform(0.75, 0.98)  # Simulate confidence
        
        return predicted_species, confidence
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0.0

def generate_detailed_story(species):
    """Generate detailed educational story for predicted species"""
    story_templates = [
        f"The **{species}** is truly one of nature's avian treasures. Found across various habitats, this remarkable bird plays a crucial role in ecosystem balance. With its distinctive appearance and fascinating behaviors, the {species} contributes to maintaining the delicate harmony of biodiversity that makes our planet so special for bird enthusiasts and conservationists alike.",
        
        f"Meet the magnificent **{species}**, a jewel in nature's wildlife crown. This species has adapted perfectly to its environment, developing unique feeding strategies and complex social behaviors. Conservation programs work tirelessly to protect the {species} and its habitat, ensuring future generations can witness the beauty of this remarkable bird in its natural surroundings.",
        
        f"The **{species}** represents the incredible diversity of avian life. Each day brings new discoveries about this bird's fascinating life cycle and intricate interactions with other species. From its elaborate nesting habits to its impressive migrations, the {species} demonstrates nature's incredible adaptability and the importance of preserving diverse ecosystems for all wildlife."
    ]
    
    # Add some species-specific details
    species_details = {
        "African Fish Eagle": "Known for its distinctive cry and incredible fishing skills, this majestic bird is often considered the voice of African waterways.",
        "Great Blue Turaco": "With its stunning blue and green plumage and unique crest, this bird is one of Africa's most colorful avian residents.",
        "Shoebill Stork": "This prehistoric-looking bird is known for its massive shoe-shaped bill and patient hunting style in swampy habitats.",
        "Grey Crowned Crane": "The national bird of Uganda, known for its elegant dance displays and beautiful golden crown of feathers.",
        "Lilac-breasted Roller": "Renowned for its breathtakingly colorful plumage and spectacular aerial acrobatics during courtship displays."
    }
    
    story = random.choice(story_templates)
    
    # Add species-specific detail if available
    if species in species_details:
        story += f" {species_details[species]}"
    
    return story

def generate_audio(story, filename):
    """Generate audio from story text"""
    try:
        tts = gTTS(text=story, lang='en', slow=False)
        tts.save(filename)
        return True
    except Exception as e:
        st.error(f"Audio generation error: {e}")
        return False

def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Create a download link for files"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=100)
        st.title("Navigation")
        
        st.markdown("---")
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload** a clear bird image
        2. **Wait** for AI analysis
        3. **View** species prediction
        4. **Read** the generated story
        5. **Listen** to audio narration
        6. **Download** the audio file
        """)
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This AI-powered system:
        - Identifies bird species from images
        - Generates educational stories
        - Creates audio narrations
        - Provides conservation information
        
        **Built with:**
        - PyTorch (AI Model)
        - Streamlit (Web Interface)
        - gTTS (Text-to-Speech)
        - Computer Vision
        """)
        
        st.markdown("---")
        st.header("🐦 Common Bird Species")
        st.markdown("""
        - African Fish Eagle
        - Grey Crowned Crane  
        - Lilac-breasted Roller
        - Shoebill Stork
        - Marabou Stork
        - African Jacana
        - And many more...
        """)

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Bird Image")
        
        # Upload area with better styling
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            " ",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a bird for species identification",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)
            
            # Image details
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("Image Size", f"{image.size[0]}x{image.size[1]}")
            with col1b:
                st.metric("Format", uploaded_file.type.split('/')[-1].upper())
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🔄 Loading AI model and analyzing image..."):
                    # Load model
                    if not st.session_state.model_loaded:
                        model = load_model()
                        st.session_state.model = model
                        st.session_state.model_loaded = True
                    else:
                        model = st.session_state.model
                    
                    # Predict species
                    predicted_species, confidence = predict_species(image, model)
                    
                    if predicted_species:
                        # Store results in session state
                        st.session_state.last_prediction = {
                            'species': predicted_species,
                            'confidence': confidence,
                            'image': image,
                            'timestamp': time.time()
                        }
                        
                        st.success("✅ Analysis complete!")
                    else:
                        st.error("❌ Could not analyze image. Please try another image.")

    with col2:
        if st.session_state.last_prediction:
            prediction = st.session_state.last_prediction
            
            st.subheader("🎯 Prediction Results")
            
            # Prediction card
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Predicted Species", prediction['species'])
            with col2b:
                st.metric("Confidence", f"{prediction['confidence']:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Generate story
            st.subheader("📖 Educational Story")
            story = generate_detailed_story(prediction['species'])
            
            # Story box
            st.markdown('<div class="story-box">', unsafe_allow_html=True)
            st.write(story)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Species facts
            st.subheader("📊 Species Information")
            with st.expander("Learn more about this species"):
                st.info("""
                **Typical Habitat:** Various ecosystems including forests, wetlands, and savannas  
                **Primary Diet:** Varies by species (fish, insects, seeds, small animals)  
                **Conservation Status:** Species-dependent (from Least Concern to Endangered)  
                **Unique Characteristics:** Each species has distinct behaviors and adaptations
                """)
            
            # Audio section
            st.subheader("🎵 Audio Story")
            audio_col1, audio_col2 = st.columns([2, 1])
            
            with audio_col1:
                if st.button("Generate Audio Narration", use_container_width=True):
                    with st.spinner("Creating audio narration..."):
                        # Create temporary file for audio
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                            audio_path = tmp_file.name
                        
                        if generate_audio(story, audio_path):
                            # Play audio
                            audio_file = open(audio_path, 'rb')
                            audio_bytes = audio_file.read()
                            
                            st.audio(audio_bytes, format='audio/mp3')
                            
                            # Download button
                            st.download_button(
                                label="Download Audio Story",
                                data=audio_bytes,
                                file_name=f"{prediction['species'].replace(' ', '_')}_story.mp3",
                                mime="audio/mp3",
                                use_container_width=True
                            )
                            
                            # Cleanup
                            os.unlink(audio_path)
                        else:
                            st.error("Failed to generate audio")
            
            with audio_col2:
                if st.button("🔄 New Prediction", use_container_width=True):
                    st.session_state.last_prediction = None
                    st.rerun()
        
        else:
            # Welcome message when no prediction
            st.info("👆 Upload a bird image and click 'Analyze Image' to get started!")
            st.markdown("""
            ### How it works:
            1. **Upload** a clear photo of a bird
            2. **AI analyzes** the image features
            3. **Get instant** species identification
            4. **Receive** educational content
            5. **Listen** to the story audio
            
            ### Tips for best results:
            - Use clear, well-lit images
            - Ensure the bird is clearly visible
            - Avoid blurry or distant shots
            - Multiple angles can improve accuracy
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🦜 Bird Species Identifier | Built with Streamlit & PyTorch | "
        "<a href='https://github.com/your-repo' target='_blank'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
