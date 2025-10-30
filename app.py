import streamlit as st
import pandas as pd
import os
import tempfile
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

from model_trainer import BirdClassifier, prepare_data, get_multiple_images_for_species
from story_generator import StoryGenerator
from video_generator import create_enhanced_story_video

st.set_page_config(
    page_title="Bird Species Video Story Generator",
    page_icon="🐦",
    layout="wide"
)

st.title("🐦 Bird Species Video Story Generator")
st.markdown("**Train a PyTorch model, predict bird species, generate stories, and create narrated videos**")

if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'label_map' not in st.session_state:
    st.session_state.label_map = None
if 'image_df' not in st.session_state:
    st.session_state.image_df = None
if 'df' not in st.session_state:
    st.session_state.df = None

os.makedirs('outputs', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

tab1, tab2, tab3, tab4 = st.tabs(["📁 Dataset Upload", "🎓 Train Model", "🔮 Predict & Generate", "📹 Create Videos"])

with tab1:
    st.header("Upload Dataset")
    st.markdown("Upload a CSV file with bird species information and a folder containing images")
    
    csv_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if csv_file:
        df = pd.read_csv(csv_file)
        st.session_state.df = df
        
        st.success(f"CSV loaded with {len(df)} rows")
        st.dataframe(df.head())
        
        if 'common_name' in df.columns:
            st.markdown(f"**Unique Species:** {df['common_name'].nunique()}")
            st.bar_chart(df['common_name'].value_counts().head(10))
        
    images_folder = st.text_input("Path to images folder (or upload individual images below)", value="uploads/images")
    
    uploaded_images = st.file_uploader(
        "Or upload bird images directly (they will be saved to uploads/images/)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_images:
        os.makedirs(images_folder, exist_ok=True)
        
        species_folders = {}
        
        for uploaded_file in uploaded_images:
            filename = uploaded_file.name
            
            parts = filename.split('_')
            if len(parts) >= 2:
                species_name = parts[0].replace('-', ' ').title()
            else:
                species_name = "Unknown"
            
            if species_name not in species_folders:
                species_folder = os.path.join(images_folder, species_name.replace(' ', '_'))
                os.makedirs(species_folder, exist_ok=True)
                species_folders[species_name] = species_folder
            
            save_path = os.path.join(species_folders[species_name], uploaded_file.name)
            with open(save_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
        
        st.success(f"Uploaded {len(uploaded_images)} images to {images_folder}")
        
        if st.button("Create CSV from uploaded images"):
            image_data = []
            for species, folder in species_folders.items():
                for img_file in os.listdir(folder):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_data.append({
                            'common_name': species,
                            'folder_path': folder,
                            'filename': img_file,
                            'description': f"A beautiful {species} bird"
                        })
            
            if image_data:
                auto_df = pd.DataFrame(image_data)
                st.session_state.df = auto_df
                auto_df.to_csv('uploads/dataset.csv', index=False)
                st.success(f"Created dataset with {len(auto_df)} images from {len(species_folders)} species")
                st.dataframe(auto_df.head())
    
    if st.button("Process Dataset"):
        if st.session_state.df is not None:
            df = st.session_state.df
            
            if 'folder_path' in df.columns and 'filename' in df.columns:
                image_data = []
                for _, row in df.iterrows():
                    folder = row['folder_path']
                    species = row['common_name']
                    
                    if os.path.exists(folder):
                        files = [row['filename']] if 'filename' in row and pd.notna(row['filename']) else os.listdir(folder)
                        
                        for img_file in files:
                            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                image_data.append({
                                    'common_name': species,
                                    'folder_path': folder,
                                    'filename': img_file
                                })
            else:
                st.error("CSV must have 'common_name', 'folder_path', and 'filename' columns")
                st.stop()
            
            if image_data:
                image_df = pd.DataFrame(image_data)
                st.session_state.image_df = image_df
                
                st.success(f"Found {len(image_df)} images from {image_df['common_name'].nunique()} species")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Images per Species:**")
                    st.dataframe(image_df['common_name'].value_counts())
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    image_df['common_name'].value_counts().head(10).plot(kind='bar', ax=ax, color='steelblue')
                    ax.set_title('Top 10 Species by Image Count')
                    ax.set_xlabel('Species')
                    ax.set_ylabel('Image Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.error("No valid images found in the specified folders")
        else:
            st.error("Please upload a CSV file first")

with tab2:
    st.header("Train Model")
    
    if st.session_state.image_df is None:
        st.warning("Please upload and process a dataset first in the Dataset Upload tab")
    else:
        st.markdown(f"**Dataset:** {len(st.session_state.image_df)} images, {st.session_state.image_df['common_name'].nunique()} species")
        
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Number of epochs", min_value=1, max_value=20, value=5)
        with col2:
            batch_size = st.slider("Batch size", min_value=4, max_value=32, value=8, step=4)
        
        if st.button("Start Training"):
            with st.spinner("Preparing data..."):
                train_loader, val_loader, label_map = prepare_data(
                    st.session_state.image_df,
                    batch_size=batch_size,
                    train_split=0.8
                )
                st.session_state.label_map = label_map
                
                st.success(f"Created {len(train_loader)} training batches and {len(val_loader)} validation batches")
            
            with st.spinner(f"Training model for {epochs} epochs..."):
                classifier = BirdClassifier(num_classes=len(label_map), use_pretrained=False)
                
                progress_bar = st.progress(0)
                loss_placeholder = st.empty()
                
                for epoch in range(epochs):
                    classifier.model.train()
                    running_loss = 0.0
                    
                    for images, labels, _ in train_loader:
                        images, labels = images.to(classifier.device), labels.to(classifier.device)
                        outputs = classifier.model(images)
                        
                        import torch.nn as nn
                        criterion = nn.CrossEntropyLoss()
                        loss = criterion(outputs, labels)
                        
                        classifier.model.zero_grad()
                        loss.backward()
                        
                        import torch.optim as optim
                        if not hasattr(classifier, 'optimizer'):
                            classifier.optimizer = optim.Adam(classifier.model.parameters(), lr=0.001)
                        classifier.optimizer.step()
                        
                        running_loss += loss.item()
                    
                    avg_loss = running_loss / len(train_loader)
                    progress_bar.progress((epoch + 1) / epochs)
                    loss_placeholder.markdown(f"**Epoch {epoch+1}/{epochs}** - Loss: {avg_loss:.4f}")
                
                st.session_state.model = classifier
                st.session_state.trained = True
                
                classifier.save_model('outputs/bird_model.pth')
                st.success("Training completed! Model saved to outputs/bird_model.pth")
            
            with st.spinner("Evaluating model..."):
                results = classifier.evaluate(val_loader, num_classes=len(label_map))
                
                st.metric("Validation Accuracy", f"{results['accuracy']:.2%}")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(results['confusion_matrix'], annot=True, fmt='d',
                           xticklabels=list(label_map.keys()),
                           yticklabels=list(label_map.keys()),
                           cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted Species')
                ax.set_ylabel('Actual Species')
                ax.set_title('Confusion Matrix')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)

with tab3:
    st.header("Predict Species & Generate Story")
    
    if not st.session_state.trained:
        st.warning("Please train a model first in the Train Model tab")
    else:
        uploaded_test_image = st.file_uploader("Upload an image to predict", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_test_image:
            test_image = Image.open(uploaded_test_image)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(test_image, caption="Uploaded Image", use_container_width=True)
            
            temp_image_path = f"uploads/temp_{uploaded_test_image.name}"
            test_image.save(temp_image_path)
            
            with st.spinner("Predicting..."):
                predicted_species, confidence = st.session_state.model.predict(
                    temp_image_path,
                    st.session_state.label_map
                )
            
            with col2:
                st.success(f"**Predicted Species:** {predicted_species}")
                st.metric("Confidence", f"{confidence:.2%}")
                
                story_gen = StoryGenerator()
                
                description = None
                if st.session_state.df is not None and 'description' in st.session_state.df.columns:
                    desc_row = st.session_state.df[st.session_state.df['common_name'] == predicted_species]
                    if not desc_row.empty:
                        description = desc_row['description'].iloc[0]
                
                story = story_gen.generate_story(predicted_species, description)
                
                st.markdown("### Generated Story")
                st.markdown(f"*{story}*")
                
                if st.button("Save Story"):
                    with open(f"outputs/{predicted_species.replace(' ', '_')}_story.txt", 'w') as f:
                        f.write(story)
                    st.success("Story saved!")

with tab4:
    st.header("Create Story Videos")
    
    if not st.session_state.trained or st.session_state.image_df is None:
        st.warning("Please upload a dataset and train a model first")
    else:
        unique_species = sorted(st.session_state.image_df['common_name'].unique())
        
        selected_species = st.multiselect(
            "Select species to create videos for",
            options=unique_species,
            default=unique_species[:min(2, len(unique_species))]
        )
        
        video_style = st.radio("Video Style", options=['transition', 'ken_burns'])
        max_images = st.slider("Maximum images per video", min_value=1, max_value=10, value=5)
        
        if st.button("Generate Videos"):
            story_gen = StoryGenerator()
            
            for species in selected_species:
                with st.expander(f"Creating video for {species}", expanded=True):
                    description = None
                    if st.session_state.df is not None and 'description' in st.session_state.df.columns:
                        desc_row = st.session_state.df[st.session_state.df['common_name'] == species]
                        if not desc_row.empty:
                            description = desc_row['description'].iloc[0]
                    
                    story = story_gen.generate_story(species, description)
                    st.markdown(f"**Story:** {story}")
                    
                    image_paths = get_multiple_images_for_species(
                        species,
                        st.session_state.image_df,
                        max_images=max_images
                    )
                    
                    if image_paths:
                        st.markdown(f"Found {len(image_paths)} images")
                        
                        cols = st.columns(min(len(image_paths), 4))
                        for idx, img_path in enumerate(image_paths[:4]):
                            with cols[idx]:
                                st.image(img_path, caption=f"Image {idx+1}", use_container_width=True)
                        
                        video_output_path = f"outputs/{species.replace(' ', '_')}_video.mp4"
                        
                        with st.spinner(f"Creating {video_style} video..."):
                            success = create_enhanced_story_video(
                                species,
                                story,
                                image_paths,
                                video_output_path,
                                style=video_style
                            )
                        
                        if success and os.path.exists(video_output_path):
                            file_size = os.path.getsize(video_output_path) / (1024 * 1024)
                            st.success(f"Video created! Size: {file_size:.2f} MB")
                            
                            with open(video_output_path, 'rb') as video_file:
                                st.download_button(
                                    label=f"Download {species} Video",
                                    data=video_file,
                                    file_name=f"{species.replace(' ', '_')}_video.mp4",
                                    mime="video/mp4"
                                )
                            
                            st.video(video_output_path)
                        else:
                            st.error(f"Failed to create video for {species}")
                    else:
                        st.warning(f"No images found for {species}")

st.sidebar.header("About")
st.sidebar.markdown("""
This application recreates the Kaggle bird species video generation pipeline:

1. **Upload Dataset** - Load CSV and images
2. **Train Model** - PyTorch ResNet18 transfer learning
3. **Predict & Generate** - Classify birds and create stories
4. **Create Videos** - Generate narrated videos with:
   - Multiple images with smooth transitions
   - Synced caption chunks
   - Text-to-speech audio narration
   - Progress indicators
   - Ken Burns pan-and-zoom effects
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info:**")
if st.session_state.trained:
    st.sidebar.success("✓ Model trained")
    st.sidebar.markdown(f"Classes: {len(st.session_state.label_map) if st.session_state.label_map else 0}")
else:
    st.sidebar.info("Model not trained yet")
