import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

class Transform:
    def __init__(self, image_size=None):
        self.image_size = image_size


    def transform_from_PIL(self):
        transform = transforms.Compose([
            # transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def transform_from_ndarray(self):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        return transform


