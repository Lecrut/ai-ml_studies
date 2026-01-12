"""
SUBMISSION FOR IMAGE-TEXT MATCHING COMPETITION

Dummy model that always returns 0.5
"""

import torch
import torch.nn as nn
from torchvision import transforms

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class SubmissionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Dummy layer to have some parameters
        self.dummy = nn.Linear(1, 1)
    
    def predict(self, image_tensor, text_string):
        # Always return 0.5
        return 0.5