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
            # CSS for dark theme and components
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

        # Custom CSS for dark card layout
        st.markdown("""
        <style>
        .dark-card {
            background: rgba(23, 23, 35, 0.92);
            border-radius: 20px;
            padding: 2.5rem;
            margin: 2rem auto;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            max-width: 1200px;
        }
        .input-section {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .section-title {
            color: rgba(255,255,255,0.9);
            font-size: 1.1rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            font-weight: 300;
        }
        /* Override Streamlit's default text colors in the dark card */
        .dark-card label, 
        .dark-card .stTextInput > label,
        .dark-card .stFileUploader label,
        .dark-card .stFileUploader span,
        .dark-card [data-testid="stCameraInputLabel"] {
            color: rgba(255,255,255,0.8) !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.image(_logo_small, use_column_width=False)
except Exception:
    # If logo not found or cannot be opened, skip silently
    pass

# Centered title
st.markdown("<h1 style='text-align: center; margin-bottom: 0.1rem;'>Birds in Uganda</h1>", unsafe_allow_html=True)

# Centered welcome message (italic)
st.markdown("<p style='text-align: center;'><em>Upload any bird image you'd like to learn more about it. Discover more about the birds of Uganda!</em></p>", unsafe_allow_html=True)

# Add custom CSS for modern components
st.markdown("""
<style>
.main-card {
    background: rgba(33, 33, 45, 0.85);
    border-radius: 20px;
    padding: 2rem;
    margin: 2rem 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}
.instructions {
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.1);
}
.input-section {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid rgba(255,255,255,0.1);
}
.section-title {
    color: rgba(255,255,255,0.9);
    font-size: 1.1rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    font-weight: 300;
}
.stButton > button {
    width: 100%;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    background: rgba(46,134,193,0.8);
    color: white;
    border: none;
    margin: 0.5rem 0;
    backdrop-filter: blur(5px);
    -webkit-backdrop-filter: blur(5px);
}
.stButton > button:hover {
    background: rgba(33,97,140,0.9);
}
/* Override Streamlit's default text colors in the dark card */
.main-card label, .main-card .stTextInput > label {
    color: rgba(255,255,255,0.8) !important;
}
</style>
""", unsafe_allow_html=True)

# Main content container with modern layout
with st.container():
    # Start main dark card
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    # Instructions with modern styling
    st.markdown("""
    <div class="instructions">
        <p style='text-align: center; color: rgba(255,255,255,0.9); margin: 0; font-weight: 300; letter-spacing: 0.5px;'>
            Choose one of the options below to identify a bird
        </p>
    </div>

    <!-- Summary card inside the main dark card -->
    <div style='background: rgba(255,255,255,0.03); border-radius: 12px; padding: 1rem; margin-bottom: 1.25rem;'>
        <div style='color: rgba(255,255,255,0.95); font-weight:600; margin-bottom:0.5rem;'>Input Sections:</div>
        <ul style='color: rgba(255,255,255,0.85); margin-top:0.25rem; margin-bottom:0.75rem;'>
            <li>Subtle dark backgrounds</li>
            <li>Light borders for definition</li>
            <li>Improved spacing and padding</li>
            <li>Better visual hierarchy</li>
        </ul>
        <div style='color: rgba(255,255,255,0.95); font-weight:600; margin-bottom:0.5rem;'>Interactive Elements:</div>
        <ul style='color: rgba(255,255,255,0.85); margin-top:0.25rem; margin-bottom:0;'>
            <li>Improved button styling</li>
            <li>Better hover effects</li>
            <li>Smoother transitions</li>
            <li>File uploader styling matches theme</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns with equal width
    col1, col2 = st.columns(2)
    
    # Upload section with modern styling
    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📁 Upload Image</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select a bird image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Camera section with modern styling
    with col2:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📷 Capture Image</p>', unsafe_allow_html=True)
        
        # Camera activation state
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False

        # Camera controls
        if not st.session_state.camera_active:
            if st.button("Start Camera 📷", key="use_camera_button"):
                st.session_state.camera_active = True
        
        if st.session_state.camera_active:
            camera_photo = st.camera_input("Take a photo", key="camera_input")
            if camera_photo is not None:
                image = Image.open(camera_photo)
                st.image(image, caption='Captured Photo', use_column_width=True)
            
            if st.button("Stop Camera ⏹️", key="stop_camera_button", help="Click to stop camera preview"):
                st.session_state.camera_active = False
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Close the main dark card
    st.markdown('</div>', unsafe_allow_html=True)
