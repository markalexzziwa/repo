import streamlit as st
from PIL import Image
import cv2

# Display logo
st.image("ugb1.png", use_column_width=True)

# Title of the app
st.title("Birds in Uganda")

# Welcome message
st.markdown("*Upload any bird image you'd like to learn more about it. Discover more about that it*")

# Create a box container
with st.container():
    st.write("Choose how you want to input the bird image:")
    
    # Create two columns for the options
    col1, col2 = st.columns(2)
    
    # First column - Upload image option
    with col1:
        st.write("📁 Browse Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Second column - Camera option
    with col2:
        st.write("📷 Capture Image")
        camera_photo = st.camera_input("Take a photo")
        if camera_photo is not None:
            image = Image.open(camera_photo)
            st.image(image, caption='Captured Image', use_column_width=True)
