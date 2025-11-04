import streamlit as st
from PIL import Image
# Note: OpenCV (cv2) is optional. Remove the import to avoid deployment issues unless you add
# `opencv-python` to your `requirements.txt` and your deployment environment supports it.
import base64
import os

# Display logo (centered and resized to one-quarter of original dimensions)
def _set_background_glass(img_path: str = "ugb1.png"):
    """Set a full-page background using the given image and add a translucent glass
    style to the main Streamlit block container so content appears on a frosted panel.
    The image is embedded as a data URI to improve compatibility when deployed.
    """
    try:
        if not os.path.exists(img_path):
            return
        with open(img_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        css = f"""
        <style>
        .stApp {{
            /* Apply a white overlay so the image appears very subtle, requiring focus to notice */
            background-image: linear-gradient(rgba(255,255,255,0.92), rgba(255,255,255,0.92)), url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp .main .block-container {{
            background: rgba(255,255,255,0.6);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            border-radius: 12px;
            padding: 1rem 1.5rem;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        # If embedding fails, don't break the app
        pass

# Apply the background/glass style
_set_background_glass("ugb1.png")
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
