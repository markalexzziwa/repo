import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    from torchvision import transforms, models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    import torch.nn.functional as F

class BirdDataset(Dataset):
    def __init__(self, dataframe, transform=None, label_map=None):
        self.df = dataframe
        self.transform = transform
        if label_map is not None:
            self.label_map = label_map
        else:
            self.label_map = {label: idx for idx, label in enumerate(self.df['common_name'].unique())}
        self.inverse_map = {v: k for k, v in self.label_map.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder_path = row['folder_path']
        img_path = os.path.join(folder_path, row['filename'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='gray')
        
        label = self.label_map[row['common_name']]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, row['filename']

class SimpleTransform:
    def __init__(self, image_size=224):
        self.image_size = image_size
        
    def __call__(self, img):
        img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        return img_tensor

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        
        x = x.view(-1, 256 * 14 * 14)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class BirdClassifier:
    def __init__(self, num_classes, use_pretrained=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        if TORCHVISION_AVAILABLE and use_pretrained:
            try:
                self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
                print("Using pretrained ResNet18 model")
            except Exception as e:
                print(f"Could not load pretrained weights: {e}. Using SimpleCNN instead.")
                self.model = SimpleCNN(num_classes)
        elif TORCHVISION_AVAILABLE:
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            print("Using ResNet18 model (training from scratch)")
        else:
            print("Torchvision not available, using SimpleCNN model")
            self.model = SimpleCNN(num_classes)
        
        self.model = self.model.to(self.device)
        
    def train_model(self, train_loader, val_loader, epochs=10, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        loss_history = []
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            
            for images, labels, _ in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            loss_history.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return loss_history
    
    def evaluate(self, val_loader, num_classes=None):
        self.model.eval()
        true_labels = []
        pred_labels = []
        pred_probs = []
        filenames = []

        with torch.no_grad():
            for images, labels, fname in val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(predicted.cpu().numpy())
                pred_probs.extend(probabilities.cpu().numpy())
                filenames.extend(fname)

        acc = accuracy_score(true_labels, pred_labels)
        
        if num_classes is None:
            num_classes = self.num_classes
        cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))
        
        return {
            'accuracy': acc,
            'confusion_matrix': cm,
            'true_labels': true_labels,
            'pred_labels': pred_labels,
            'pred_probs': pred_probs,
            'filenames': filenames
        }
    
    def predict(self, image_path, label_map):
        self.model.eval()
        
        if TORCHVISION_AVAILABLE:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = SimpleTransform()
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        inverse_map = {v: k for k, v in label_map.items()}
        predicted_species = inverse_map[predicted_idx.item()]
        
        return predicted_species, confidence.item()
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

def prepare_data(image_df, batch_size=16, train_split=0.8):
    if TORCHVISION_AVAILABLE:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = SimpleTransform()
    
    global_label_map = {label: idx for idx, label in enumerate(sorted(image_df['common_name'].unique()))}
    
    train_size = int(train_split * len(image_df))
    val_size = len(image_df) - train_size
    
    indices = list(range(len(image_df)))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_df = image_df.iloc[train_indices].reset_index(drop=True)
    val_df = image_df.iloc[val_indices].reset_index(drop=True)
    
    train_dataset = BirdDataset(train_df, transform=transform, label_map=global_label_map)
    val_dataset = BirdDataset(val_df, transform=transform, label_map=global_label_map)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, global_label_map

def get_multiple_images_for_species(species, image_df, max_images=5):
    species_images = image_df[image_df['common_name'] == species]
    image_paths = []
    
    for _, row in species_images.head(max_images).iterrows():
        img_path = os.path.join(row['folder_path'], row['filename'])
        if os.path.exists(img_path):
            image_paths.append(img_path)
    
    return image_paths
