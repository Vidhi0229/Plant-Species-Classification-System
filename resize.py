from PIL import Image
import os

# Define input and output directories
input_folder = '/home/vidhi/Documents/research/projects/Plant-Species-Classification-System/dataset' 
output_folder = '/home/vidhi/Documents/research/projects/Plant-Species-Classification-System/working/resized_img' 

def resize_images(input_folder, output_folder, size=(64, 64)):
    for dirname, _, filenames in os.walk(input_folder):
        for filename in filenames:
            img_path = os.path.join(dirname, filename)
            output_path = os.path.join(output_folder, filename)  
            
            try:
                # Open, resize, and save the image
                with Image.open(img_path) as img:
                    resized_img = img.resize(size)
                    resized_img.save(output_path)  
                    print(f"Resized and saved {filename} to {output_folder}")
            except Exception as e:
                print(f"Could not process {filename}: {e}")

# Resize images to 64x64 pixels
resize_images(input_folder, output_folder, size=(64, 64))


