#  Author: Paul-Jason Mello
#  Date: June 5th, 2023

# General Libraries
import numpy as np
from torchvision import transforms

class GrayscaleTransform:
    def __init__(self, image_size, num_channels):
        self.resize = transforms.Resize((image_size, image_size))
        self.grayscale = transforms.Grayscale(num_output_channels=num_channels)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, x):
        x = self.resize(x)
        x = self.grayscale(x)
        x = self.to_tensor(x)
        x = (x * 2) - 1  # [-1, 1]
        return x


class RGBTransform:
    def __init__(self, image_size):
        self.resize = transforms.Resize((image_size, image_size))
        self.to_tensor = transforms.ToTensor()

    def __call__(self, x):
        x = self.resize(x)
        x = self.to_tensor(x)
        x = (x * 2) - 1  # [-1, 1]
        return x


class ConvertToImage:
    def __call__(self, x):
        x = (x + 1) / 2  # [-1, 1]
        x = x.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        x = x * 255.0  # [0, 255]
        x = x.cpu().detach().numpy().astype(np.uint8)  # Likely a bottleneck for speed
        x = transforms.ToPILImage()(x)
        return x