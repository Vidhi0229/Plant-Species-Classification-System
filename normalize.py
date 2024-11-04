import numpy as np
from PIL import Image
import os

# Paths to input and output directories
resized_folder = '//home/vidhi/Documents/research/projects/Plant-Species-Classification-System/working/resized_img'
normalized_folder = '/home/vidhi/Documents/research/projects/Plant-Species-Classification-System/working/normalized_img'


def normalize_images(input_folder, output_folder, scale_range=(0, 1)):
    min_val, max_val = scale_range  # Define the range for normalization
    
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            # Open and convert image to a NumPy array
            with Image.open(img_path) as img:
                img_array = np.array(img).astype(np.float32)
                
                # Scale to the specified range (e.g., [0, 1])
                if scale_range == (0, 1):
                    img_array /= 255.0  # Normalize to [0, 1]
                elif scale_range == (-1, 1):
                    img_array = (img_array / 127.5) - 1  # Normalize to [-1, 1]
                
                # Convert back to an image
                normalized_img = Image.fromarray((img_array * 255).astype(np.uint8))
                normalized_img.save(output_path)  # Save the normalized image
                
                print(f"Normalized and saved {filename} to {output_folder}")
        except Exception as e:
            print(f"Could not process {filename}: {e}")

# Normalize resized images to [0, 1] range
normalize_images(resized_folder, normalized_folder, scale_range=(0, 1))
