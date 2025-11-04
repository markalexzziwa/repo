import streamlit as st
from PIL import Image
# Note: OpenCV (cv2) is optional. Remove the import to avoid deployment issues unless you add
# `opencv-python` to your `requirements.txt` and your deployment environment supports it.

# Display logo (centered and resized to one-quarter of original dimensions)
try:
    _logo = Image.open("ugb1.png")
    _w, _h = _logo.size
    # Prevent zero or negative sizes
    _new_w = max(1, _w // 4)
    _new_h = max(1, _h // 4)
    _logo_small = _logo.resize((_new_w, _new_h), Image.LANCZOS)
    # center the image using three columns and put image in the middle
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(_logo_small, use_column_width=False)
except Exception:
    # If logo not found or cannot be opened, skip silently
    pass

# Centered title
st.markdown("<h1 style='text-align: center; margin-bottom: 0.1rem;'>Birds in Uganda</h1>", unsafe_allow_html=True)

# Centered welcome message (italic)
st.markdown("<p style='text-align: center;'><em>Upload any bird image you'd like to learn more about it. Discover more about the birds of Uganda!</em></p>", unsafe_allow_html=True)

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

        # Use a button to activate the camera. Persist the state so the camera stays active
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False

        if st.button("Use camera 📷", key="use_camera_button"):
            st.session_state.camera_active = True

        if st.session_state.camera_active:
            camera_photo = st.camera_input("Take a photo", key="camera_input")
            if camera_photo is not None:
                image = Image.open(camera_photo)
                st.image(image, caption='Captured Image', use_column_width=True)

            # Provide a stop button to close the camera preview
            if st.button("Stop camera", key="stop_camera_button"):
                st.session_state.camera_active = False
