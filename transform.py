import os
import torch
from torchvision import transforms
from PIL import Image  
import numpy as np

# Transformation to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor()  # Converts to tensor and scales to [0, 1]
])

# Load and transform images from the normalized folder, including subfolders
def load_and_transform_images(folder):
    image_tensors = []
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            img_path = os.path.join(root, filename)
            try:
                with Image.open(img_path) as img:
                    img_tensor = transform(img)
                    image_tensors.append(img_tensor)
            except Exception as e:
                print(f"Could not process {filename}: {e}")
    return torch.stack(image_tensors)  # Combine into a single tensor

# Convert normalized images to tensor dataset
leaf_dataset = load_and_transform_images("/home/vidhi/Documents/research/projects/Plant-Species-Classification-System/working/normalized_img")

# Save the tensor dataset as a .npy file to avoid PyTorch's pickle warnings
np.save('leaf_dataset.npy', leaf_dataset.numpy())

# Load the dataset back from the .npy file
leaf_dataset = torch.tensor(np.load('leaf_dataset.npy'))
