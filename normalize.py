import numpy as np
from PIL import Image
import os

# Paths to input and output directories
resized_folder = '/home/vidhi/Documents/research/projects/Plant-Species-Classification-System/working/resized_img'
normalized_folder = '/home/vidhi/Documents/research/projects/Plant-Species-Classification-System/working/normalized_img'

def normalize_images(input_folder, output_folder, scale_range=(0, 1)):
    min_val, max_val = scale_range  
    
    for dirname, _, filenames in os.walk(input_folder):
        for filename in filenames:
            img_path = os.path.join(dirname, filename)
            
            # Create the mirrored subdirectory structure in the output folder
            relative_path = os.path.relpath(dirname, input_folder)
            output_dir = os.path.join(output_folder, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, filename)
            
            try:
                # Open and convert image to a NumPy array
                with Image.open(img_path) as img:
                    img_array = np.array(img).astype(np.float32)
                    
                    # Normalize based on the specified range
                    if scale_range == (0, 1):
                        img_array /= 255.0  # Normalize to [0, 1]
                    elif scale_range == (-1, 1):
                        img_array = (img_array / 127.5) - 1  # Normalize to [-1, 1]
                    
                    # Convert back to an image
                    normalized_img = Image.fromarray((img_array * 255).clip(0, 255).astype(np.uint8))
                    normalized_img.save(output_path)  
                    
                    print(f"Normalized and saved {filename} to {output_dir}")
            except Exception as e:
                print(f"Could not process {filename}: {e}")

# Normalize resized images to [0, 1] range
normalize_images(resized_folder, normalized_folder, scale_range=(0, 1))
