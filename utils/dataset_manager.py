import os
import kaggle
import opendatasets as od
import pandas as pd
from PIL import Image
import shutil
import streamlit as st

class BirdDatasetManager:
    def __init__(self):
        self.dataset_url = "gpiosenka/100-bird-species"
        self.data_dir = "data/birds"
        self.images_dir = os.path.join(self.data_dir, "100-bird-species")
        self.sample_images_dir = "data/sample_images"
        
    def download_dataset(self):
        """Download the Kaggle dataset"""
        try:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir, exist_ok=True)
                
            if not os.path.exists(self.images_dir):
                st.info("Downloading bird species dataset from Kaggle... This may take a few minutes.")
                
                # Download using opendatasets
                od.download(f"https://www.kaggle.com/datasets/{self.dataset_url}", 
                           data_dir=self.data_dir)
                
                st.success("Dataset downloaded successfully!")
            else:
                st.info("Dataset already exists.")
                
            return True
        except Exception as e:
            st.error(f"Error downloading dataset: {str(e)}")
            return False
    
    def get_sample_images(self, num_samples=8):
        """Get sample images from the dataset for display"""
        sample_images = []
        
        try:
            # Check if dataset exists, if not download it
            if not os.path.exists(self.images_dir):
                self.download_dataset()
            
            # Look for images in the dataset structure
            for root, dirs, files in os.walk(self.images_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(root, file)
                        sample_images.append(img_path)
                        if len(sample_images) >= num_samples:
                            break
                if len(sample_images) >= num_samples:
                    break
                    
        except Exception as e:
            st.warning(f"Could not load sample images: {str(e)}")
        
        return sample_images[:num_samples]
    
    def get_bird_species(self):
        """Get list of bird species from the dataset"""
        species = []
        try:
            if os.path.exists(self.images_dir):
                # The dataset structure typically has folders named by species
                for item in os.listdir(self.images_dir):
                    if os.path.isdir(os.path.join(self.images_dir, item)):
                        species.append(item.replace('_', ' ').title())
                
                # If no species found, return default list
                if not species:
                    species = [
                        "Bald Eagle", "Blue Jay", "Northern Cardinal", "Ruby-throated Hummingbird",
                        "American Robin", "House Sparrow", "Great Horned Owl", "Pileated Woodpecker",
                        "American Flamingo", "Emperor Penguin", "Atlantic Puffin", "Mallard Duck"
                    ]
            else:
                # Return default species if dataset not downloaded
                species = [
                    "Bald Eagle", "Blue Jay", "Northern Cardinal", "Ruby-throated Hummingbird",
                    "American Robin", "House Sparrow", "Great Horned Owl", "Pileated Woodpecker"
                ]
                
        except Exception as e:
            st.warning(f"Could not load species list: {str(e)}")
            species = ["Bald Eagle", "Blue Jay", "Northern Cardinal", "American Robin"]
            
        return sorted(species)
    
    def get_bird_image(self, species_name, image_index=0):
        """Get a specific bird image by species name"""
        try:
            if not os.path.exists(self.images_dir):
                return None
                
            # Convert species name to folder name format
            folder_name = species_name.lower().replace(' ', '_')
            species_path = os.path.join(self.images_dir, folder_name)
            
            if os.path.exists(species_path):
                # Get all images for this species
                images = [f for f in os.listdir(species_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if images and image_index < len(images):
                    return os.path.join(species_path, images[image_index])
                    
        except Exception as e:
            st.warning(f"Could not load image for {species_name}: {str(e)}")
            
        return None
    
    def get_dataset_info(self):
        """Get information about the dataset"""
        info = {
            "name": "100 Bird Species",
            "url": f"https://www.kaggle.com/datasets/{self.dataset_url}",
            "description": "A comprehensive dataset containing 100 different bird species with high-quality images for classification tasks.",
            "creator": "Gerald Piosenka",
            "stats": {}
        }
        
        try:
            if os.path.exists(self.images_dir):
                # Count species
                species_count = 0
                total_images = 0
                
                for item in os.listdir(self.images_dir):
                    if os.path.isdir(os.path.join(self.images_dir, item)):
                        species_count += 1
                        species_path = os.path.join(self.images_dir, item)
                        images = [f for f in os.listdir(species_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        total_images += len(images)
                
                info["stats"] = {
                    "total_species": species_count or 100,
                    "total_images": f"{total_images:,}" if total_images else "58,388",
                    "image_resolution": "224x224 pixels",
                    "format": "JPG"
                }
            else:
                info["stats"] = {
                    "total_species": 100,
                    "total_images": "58,388",
                    "image_resolution": "224x224 pixels", 
                    "format": "JPG"
                }
                
        except Exception as e:
            st.warning(f"Could not get dataset stats: {str(e)}")
            
        return info

# Global instance
dataset_manager = BirdDatasetManager()