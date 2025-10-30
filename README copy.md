# Bird Species Video Story Generator

A PyTorch-based application that trains a CNN model on bird species images, makes predictions, generates narrative stories, and creates narrated videos with transitions and audio.

## Features

- **Dataset Upload**: Load CSV files with bird species data and upload images
- **Model Training**: Train a PyTorch CNN (SimpleCNN or ResNet18) for bird classification
- **Prediction**: Classify new bird images with confidence scores
- **Story Generation**: Create template-based narratives about predicted species
- **Video Creation**: Generate MP4 videos with:
  - Multiple images with smooth cross-fade transitions
  - Ken Burns pan-and-zoom effects
  - Synced caption chunks that progress through the story
  - Text-to-speech audio narration (gTTS)
  - Progress indicators and styled overlays

## Dataset Format

### CSV File
Your CSV should contain at least these columns:
- `common_name`: Bird species name (e.g., "African Grey Parrot")
- `folder_path`: Path to the folder containing images for this species
- `filename`: Name of the image file (optional if scanning folders)
- `description`: Text description of the species (optional, used for story generation)

Example:
```csv
common_name,folder_path,filename,description
African Grey Parrot,uploads/images/African_Grey_Parrot,parrot1.jpg,A beautiful grey parrot with red tail feathers
Eagle,uploads/images/Eagle,eagle1.jpg,A majestic bird with brown and white plumage
```

### Image Organization
Images should be organized in folders by species:
```
uploads/images/
├── African_Grey_Parrot/
│   ├── parrot1.jpg
│   ├── parrot2.jpg
│   └── parrot3.jpg
├── Eagle/
│   ├── eagle1.jpg
│   └── eagle2.jpg
```

## Quick Start

### 1. Upload Dataset
- Go to the **Dataset Upload** tab
- Upload your CSV file or create one from uploaded images
- The app will automatically scan folders and create an image dataset
- Click "Process Dataset" to prepare data for training

### 2. Train Model
- Go to the **Train Model** tab
- Choose number of epochs (5-10 recommended for testing)
- Adjust batch size based on your dataset size
- Click "Start Training" and wait for completion
- View training metrics and confusion matrix

### 3. Predict & Generate Stories
- Go to the **Predict & Generate** tab
- Upload a bird image
- View predicted species and confidence score
- Read the auto-generated story about the bird

### 4. Create Videos
- Go to the **Create Videos** tab
- Select species to generate videos for
- Choose video style: "transition" (cross-fade) or "ken_burns" (pan-zoom)
- Set maximum images per video
- Click "Generate Videos"
- Download and view your narrated story videos!

## Video Output

Each video includes:
- **Title**: "The Story of [Species Name]"
- **Multiple Images**: Smoothly transitioning or with Ken Burns effect
- **Captions**: Story text split into chunks, syncing with video progress
- **Audio**: Text-to-speech narration of the full story
- **Progress Bar**: Visual indicator of video progress
- **Resolution**: 1280x720 (HD)
- **Format**: MP4

## Model Information

- **With torchvision**: Uses pretrained ResNet18 for transfer learning
- **Without torchvision**: Uses custom SimpleCNN trained from scratch
- **Input Size**: 224x224 pixels
- **Normalization**: ImageNet mean/std values

## Technical Stack

- **PyTorch**: Deep learning model training
- **Streamlit**: Web interface
- **OpenCV**: Video frame generation
- **Pillow**: Image processing
- **gTTS**: Text-to-speech audio
- **MoviePy**: Audio-video synchronization (optional)
- **Matplotlib/Seaborn**: Visualizations

## Tips

1. **Dataset Size**: Use at least 20-50 images per species for good results
2. **Training**: Start with 5 epochs for testing, increase to 10-20 for production
3. **Video Length**: Longer stories create longer videos (auto-calculated)
4. **Images**: Use high-quality JPG or PNG images for best video output
5. **Species Names**: Use consistent naming in your CSV file

## Output Files

All generated files are saved to the `outputs/` directory:
- `bird_model.pth`: Trained model weights
- `[species]_video.mp4`: Generated story videos
- `[species]_story.txt`: Generated story texts (if saved)

## Troubleshooting

- **Import errors**: Packages are pre-installed, restart the app if needed
- **Model training slow**: Reduce batch size or use fewer epochs
- **Video creation fails**: Ensure images exist and are readable
- **No audio in video**: MoviePy may not be available, videos will have captions only

## Example Workflow

1. Upload 3-5 bird species with 10+ images each
2. Train for 5 epochs to test the pipeline
3. Upload a test image to verify predictions
4. Generate videos for 1-2 species to see results
5. Download and enjoy your narrated bird story videos!
