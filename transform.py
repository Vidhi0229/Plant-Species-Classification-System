import os
import torch
from torchvision import transforms
from PIL import Image  

# Transformation to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor()  # Converts to tensor and scales to [0, 1]
])

# Load and transform images from the normalized folder
def load_and_transform_images(folder):
    image_tensors = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                img_tensor = transform(img)
                image_tensors.append(img_tensor)
        except Exception as e:
            print(f"Could not process {filename}: {e}")
    return torch.stack(image_tensors)  # Combine into a single tensor

# Convert normalized images to tensor dataset
leaf_dataset = load_and_transform_images("/home/vidhi/Documents/research/projects/Plant-Species-Classification-System/working/normalized_img")

# Save the tensor dataset to a file
torch.save(leaf_dataset, 'leaf_dataset.pt')

# Load the dataset back from the file, setting weights_only to True
leaf_dataset = torch.load('leaf_dataset.pt', weights_only=True)
