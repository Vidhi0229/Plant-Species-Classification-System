from generator import Generator
from discriminator import Discriminator
import torch
import torch.optim as optim
import torch.nn as nn

# Set random noise dimension, e.g., 100
z_dim = 100
img_channels = 3  # RGB images

# Initialize generator and discriminator
generator = Generator(z_dim=z_dim, img_channels=img_channels)
discriminator = Discriminator(img_channels=img_channels)

# Set training parameters
learning_rate = 0.0002
num_epochs = 1000  
batch_size = 64

# Initialize optimizers
gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Training loop
for epoch in range(num_epochs):
    # Generate random noise as input to the generator
    noise = torch.randn(batch_size, z_dim, 1, 1)
    
    # Generate fake images
    fake_images = generator(noise)
    
    # Get discriminator's output for fake images
    disc_output = discriminator(fake_images)
    
    # Calculate the mean of discriminator outputs across the batch for consistency
    disc_output_mean = disc_output.mean().item()
    
    # Classify each image based on threshold
    threshold = 0.5
    num_real = (disc_output > threshold).sum().item()
    num_fake = (disc_output <= threshold).sum().item()
    
    # Print statistics
    if (epoch + 1) % 100 == 0:  # Print every 100 epochs
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Generated image shape: {fake_images.shape}")  # Expected shape: [batch_size, img_channels, 64, 64]
        print(f"Discriminator output (mean): {disc_output_mean:.8f}")
        print(f"Number of images classified as real: {num_real}")
        print(f"Number of images classified as fake: {num_fake}")
        print("\n")
    
