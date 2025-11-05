import torch
from torchvision import transforms
from PIL import Image
import json

# Load model and label map once
def load_model_and_labels():
    from my_model import BirdClassifier  # Replace with your actual model class
    model = BirdClassifier()
    model.load_state_dict(torch.load("bird_species_model.pth", map_location="cpu"))
    model.eval()
    with open("label_map.json") as f:
        label_map = json.load(f)
    return model, label_map

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Predict species from image
def predict_species(image, model, label_map):
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
    return label_map[str(predicted_class)]