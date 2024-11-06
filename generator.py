import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, img_channels):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            # First Layer: Input z_dim -> 512 channels, image size (4x4)
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(512, affine=True),  
            nn.ReLU(True),

            # Second Layer: 512 -> 256 channels, image size (8x8)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(True),

            # Third Layer: 256 -> 128 channels, image size (16x16)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True),

            # Fourth Layer: 128 -> 64 channels, image size (32x32)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),

            # Final Layer: 64 -> img_channels (RGB), image size (64x64)
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Tanh activation to scale between -1 and 1
        )

    def forward(self, z):
        return self.model(z.view(z.size(0), z.size(1), 1, 1))
