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
    _new_w = max(1, _w // 2)  # Changed from 4 to 2 to make image twice as large
    _new_h = max(1, _h // 2)
    _logo_small = _logo.resize((_new_w, _new_h), Image.LANCZOS)
    # Add the logo and title directly without centering
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

    # Create columns for logo and title
    logo_col, title_col = st.columns([1, 2])
    with logo_col:
        st.image(_logo_small, use_column_width=False)
    with title_col:
        st.markdown("<h1 style='margin: 1rem 0 0.1rem 1rem; font-size: 3rem;'>Birds in Uganda</h1>", unsafe_allow_html=True)
except Exception:
    # If logo not found or cannot be opened, skip silently
    pass

# Centered welcome message (italic)
st.markdown("<p style='text-align: center; margin-top: -1rem; margin-bottom: 2rem;'><em>Upload any bird image you'd like to learn more about it. Discover more about the birds of Uganda!</em></p>", unsafe_allow_html=True)

st.markdown("""
<style>
.input-section {
    background: rgba(255,255,255,0.98);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    border: 1px solid rgba(230,230,230,0.8);
}
.section-title {
    color: #1f2937;
    font-size: 1.05rem;
    margin-bottom: 0.75rem;
    font-weight: 600;
}
.stButton > button {
    width: 100%;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    background: rgba(34,197,94,0.9);
    color: white;
    border: none;
}
.stButton > button:hover { background: rgba(21,128,61,0.95); }
</style>
""", unsafe_allow_html=True)

# Main content container with modern layout
with st.container():
    # Instructions
    st.markdown("""
        <div style='text-align:center; margin-bottom: 3rem; border-top: 4px solid rgba(34,197,94,0.9); padding-top: 1.5rem;'>
            <p style='color: rgba(34,197,94,0.9); margin: 0; font-weight: 500; font-size: 1.05rem;'>
            Choose one of the options below to identify a bird
        </p>
    </div>
    """, unsafe_allow_html=True)


    # Create columns for the interactive elements
    col1, col2 = st.columns(2)
    
    # Upload section with modern styling
    
    with col1:
        st.markdown("<h4 style='margin-bottom:0.5rem; color:#1f2937;'>📁 Upload Image</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select a bird image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            # Show Identify Specie button below the image
            st.button("Identify Specie", key="identify_specie_button")
    
    # Camera section with modern styling
    with col2:
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False

        # Camera controls
        st.markdown("<h4 style='margin-bottom:0.5rem; color:#1f2937;'>📷 Take Picture</h4>", unsafe_allow_html=True)
        # When the camera is inactive show a black placeholder that mimics the camera preview.
        # It disappears when the user clicks Start Camera and the real camera input is opened.
        if not st.session_state.camera_active:
            # Show ub2.png as a placeholder if it exists; otherwise fall back to the solid black div.
            try:
                _placeholder_path = "ub2.png"
                if os.path.exists(_placeholder_path):
                    with open(_placeholder_path, "rb") as _f:
                        _data = _f.read()
                    _b64 = base64.b64encode(_data).decode()
                    # Use an <img> tag with object-fit to mimic the camera preview area and keep styling.
                    _img_html = (
                        f"<img src=\"data:image/png;base64,{_b64}\" "
                        "style=\"width:100%; aspect-ratio:4/3; min-height:280px; object-fit:cover; "
                        "border-radius:8px; margin-bottom:0.75rem; box-shadow: inset 0 0 40px rgba(0,0,0,0.6);\"/>")
                    st.markdown(_img_html, unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div style='width:100%; aspect-ratio:4/3; min-height:280px; background:#000; border-radius:8px; margin-bottom:0.75rem; box-shadow: inset 0 0 40px rgba(0,0,0,0.6);'></div>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                # If anything goes wrong showing the image, fall back to the black block so UI remains usable.
                st.markdown(
                    "<div style='width:100%; aspect-ratio:4/3; min-height:280px; background:#000; border-radius:8px; margin-bottom:0.75rem; box-shadow: inset 0 0 40px rgba(0,0,0,0.6);'></div>",
                    unsafe_allow_html=True,
                )

            # Use an on_click callback to reliably set session state and trigger a rerun so the placeholder is removed.
            def _start_camera():
                st.session_state.camera_active = True

            st.button("Start Camera 📷", key="use_camera_button", on_click=_start_camera)
        
        if st.session_state.camera_active:
            camera_photo = st.camera_input("Take a photo", key="camera_input")
            if camera_photo is not None:
                image = Image.open(camera_photo)
                st.image(image, caption='Captured Photo', use_column_width=True)
                # Show Identify Specie button below the captured photo
                st.button("Identify Specie", key="identify_specie_camera_button")
            
            if st.button("Stop Camera ⏹️", key="stop_camera_button", help="Click to stop camera preview"):
                st.session_state.camera_active = False
    
    # (no wrapper divs to close)
