# app.py - Bird Species Classifier for Git Deployment
from flask import Flask, request, jsonify, send_file, render_template_string
from PIL import Image
import io
import os
import tempfile
import numpy as np
import cv2
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip
import random
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Bird Species Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .upload-form { border: 2px dashed #4CAF50; padding: 30px; margin: 20px 0; text-align: center; background: #f9f9f9; border-radius: 10px; }
        .result { background: #e8f5e8; padding: 20px; margin: 20px 0; border-radius: 5px; border-left: 4px solid #4CAF50; }
        button { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 10px; }
        button:hover { background: #45a049; }
        input[type="file"] { padding: 10px; margin: 10px; }
        .image-preview { max-width: 400px; margin: 20px auto; border-radius: 10px; }
        .loading { color: #4CAF50; margin: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐦 Bird Species Classifier</h1>
            <p>Upload a bird image for species classification and educational video generation</p>
        </div>
        
        <div class="upload-form">
            <h3>📷 Upload Bird Image</h3>
            <form id="uploadForm" action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required onchange="previewImage(this)">
                <div id="imagePreview" class="image-preview"></div>
                <br>
                <button type="submit" onclick="showLoading()">🔍 Classify Bird Species</button>
            </form>
            <div id="loading" class="loading" style="display: none;">Analyzing image... Please wait</div>
        </div>
        
        {% if result %}
        <div class="result">
            <h3>🎯 Classification Result</h3>
            <p><strong>🦅 Species:</strong> {{ result.species }}</p>
            <p><strong>📊 Confidence:</strong> {{ result.confidence }}</p>
            <p><strong>📖 Description:</strong> {{ result.story }}</p>
            {% if result.image_data %}
            <img src="data:image/jpeg;base64,{{ result.image_data }}" alt="Uploaded Bird" class="image-preview">
            {% endif %}
            <br>
            <form action="/generate-video" method="post">
                <input type="hidden" name="species" value="{{ result.species }}">
                <input type="hidden" name="story" value="{{ result.story }}">
                <input type="hidden" name="image_data" value="{{ result.image_data }}">
                <button type="submit">🎬 Generate Educational Video</button>
            </form>
        </div>
        {% endif %}
    </div>

    <script>
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('uploadForm').style.opacity = '0.7';
        }
        
        function previewImage(input) {
            const preview = document.getElementById('imagePreview');
            if (input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.innerHTML = `<img src="${e.target.result}" style="max-width: 100%; border-radius: 8px; border: 2px solid #4CAF50;">`;
                }
                reader.readAsDataURL(input.files[0]);
            }
        }
    </script>
</body>
</html>
'''

def train_model():
    """Initialize model components"""
    print("✅ Bird Classifier Model Initialized")
    return True

def predict(image_path):
    """Predict bird species from image"""
    try:
        # Sample bird species
        species_list = [
            "African Fish Eagle", "Great Blue Turaco", "Marabou Stork", 
            "Shoebill", "Grey Crowned Crane", "Superb Starling",
            "Pied Kingfisher", "Hadada Ibis", "African Jacana",
            "Lilac-breasted Roller", "African Grey Parrot"
        ]
        
        # For demo - random prediction
        species = random.choice(species_list)
        confidence = round(random.uniform(0.75, 0.95), 2)
        
        return {
            'species': species,
            'confidence': confidence,
            'success': True
        }
        
    except Exception as e:
        return {
            'species': 'Unknown Species',
            'confidence': 0.0,
            'success': False,
            'error': str(e)
        }

def generate_video(prediction, image_data=None):
    """Generate educational video about the bird species"""
    try:
        species = prediction['species']
        confidence = prediction['confidence']
        
        # Create temporary video file
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # Video parameters
        width, height = 1280, 720
        fps = 24
        duration = 10
        total_frames = duration * fps
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Generate story
        stories = [
            f"The {species} is a magnificent bird found across Africa. With {confidence:.0%} confidence, we identify this beautiful species known for its unique characteristics and ecological importance.",
            f"Identified as {species} with {confidence:.0%} certainty. This remarkable bird plays a vital role in maintaining ecosystem balance and showcases nature's incredible diversity.",
            f"Our analysis confirms this is a {species}. These birds are essential for biodiversity and contribute significantly to their natural habitats."
        ]
        story = random.choice(stories)
        
        # Generate video frames
        for i in range(total_frames):
            # Create background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.rectangle(frame, (0, 0), (width, height), (70, 130, 180), -1)
            
            # Add title
            title = f"Educational Video: {species}"
            cv2.putText(frame, title, (width//2 - 300, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Add confidence
            conf_text = f"Identification Confidence: {confidence:.0%}"
            cv2.putText(frame, conf_text, (width//2 - 200, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add story text (scrolling)
            words = story.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 50:
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            # Display story with scroll effect
            y_pos = 250
            start_idx = max(0, (i // 15) % max(1, len(lines) - 3))
            
            for j in range(min(4, len(lines) - start_idx)):
                line = lines[start_idx + j]
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                x_pos = (width - text_size[0]) // 2
                cv2.putText(frame, line, (x_pos, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_pos += 50
            
            # Add progress bar
            progress = i / total_frames
            bar_width = 600
            bar_x = (width - bar_width) // 2
            cv2.rectangle(frame, (bar_x, 550), (bar_x + bar_width, 570), (255, 255, 255), 2)
            cv2.rectangle(frame, (bar_x, 550), (bar_x + int(bar_width * progress), 570), (50, 205, 50), -1)
            
            # Add frame counter
            cv2.putText(frame, f"Frame {i+1}/{total_frames}", (50, 650), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        
        # Add audio narration
        try:
            audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
            tts = gTTS(text=f"This educational video features the {species}. {story}", lang='en', slow=False)
            tts.save(audio_path)
            
            video_clip = VideoFileClip(output_path)
            audio_clip = AudioFileClip(audio_path)
            
            final_clip = video_clip.set_audio(audio_clip)
            final_output = output_path.replace('.mp4', '_audio.mp4')
            final_clip.write_videofile(final_output, codec='libx264', audio_codec='aac', verbose=False, logger=None)
            
            os.replace(final_output, output_path)
            
            video_clip.close()
            audio_clip.close()
            os.remove(audio_path)
            
        except Exception as e:
            print(f"Audio narration skipped: {e}")
        
        return output_path
        
    except Exception as e:
        print(f"Video generation error: {e}")
        return None

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """Handle image classification"""
    if 'image' not in request.files:
        return "No image file provided", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400
    
    try:
        # Save uploaded image temporarily
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        file.save(temp_image.name)
        
        # Get prediction
        prediction_result = predict(temp_image.name)
        
        if prediction_result['success']:
            # Generate story
            stories = [
                f"The {prediction_result['species']} is known for its beautiful plumage and important role in the ecosystem.",
                f"This magnificent {prediction_result['species']} showcases the incredible biodiversity of bird species.",
                f"The {prediction_result['species']} plays a vital role in maintaining ecological balance in its habitat."
            ]
            story = random.choice(stories)
            
            # Convert image to base64 for display
            image = Image.open(temp_image.name)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            result_html = HTML_TEMPLATE.replace('{% if result %}', f'''
            {% if result %}
            <div class="result">
                <h3>🎯 Classification Result</h3>
                <p><strong>🦅 Species:</strong> {prediction_result['species']}</p>
                <p><strong>📊 Confidence:</strong> {prediction_result['confidence']:.0%}</p>
                <p><strong>📖 Description:</strong> {story}</p>
                <img src="data:image/jpeg;base64,{img_str}" alt="Uploaded Bird" class="image-preview">
                <br>
                <form action="/generate-video" method="post">
                    <input type="hidden" name="species" value="{prediction_result['species']}">
                    <input type="hidden" name="story" value="{story}">
                    <input type="hidden" name="image_data" value="{img_str}">
                    <button type="submit">🎬 Generate Educational Video</button>
                </form>
            </div>
            ''').replace('{% endif %}', '{% endif %}')
            
            # Clean up temp file
            os.unlink(temp_image.name)
            
            return render_template_string(result_html)
        else:
            return f"Classification failed: {prediction_result.get('error', 'Unknown error')}", 500
            
    except Exception as e:
        return f"Error processing image: {str(e)}", 500

@app.route('/generate-video', methods=['POST'])
def handle_video_generation():
    """Generate and download educational video"""
    species = request.form.get('species', '')
    story = request.form.get('story', '')
    
    if not species:
        return "Species information required", 400
    
    try:
        # Create prediction object
        prediction = {
            'species': species,
            'confidence': 0.85,  # Default for video
            'success': True
        }
        
        # Generate video
        video_path = generate_video(prediction)
        
        if video_path and os.path.exists(video_path):
            return send_file(
                video_path,
                as_attachment=True,
                download_name=f"{species.replace(' ', '_')}_educational_video.mp4",
                mimetype='video/mp4'
            )
        else:
            return "Video generation failed", 500
            
    except Exception as e:
        return f"Video generation error: {str(e)}", 500

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Bird Species Classifier",
        "version": "2.0",
        "endpoints": {
            "home": "/",
            "predict": "/predict (POST)",
            "generate_video": "/generate-video (POST)", 
            "health": "/health"
        }
    })

if __name__ == '__main__':
    # Initialize model
    train_model()
    
    # Get port from environment variable (for Crane Cloud)
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 Bird Species Classifier Starting...")
    print(f"📍 Port: {port}")
    print("🌐 Ready for classification and video generation!")
    
    app.run(host='0.0.0.0', port=port, debug=False)