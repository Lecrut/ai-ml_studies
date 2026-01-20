"""
Simple PyTorch model example for the Image-Text Matching competition.
Run this file to generate weights.pth that you can submit.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
import re

SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class SubmissionModel(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=32):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Image encoder
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Text encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_fc = nn.Linear(embed_dim, 32)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def tokenize(self, text):
        words = re.sub(r'[^a-zA-Z\s]', '', text.lower()).split()
        indices = [hash(w) % self.vocab_size for w in words[:10]]
        while len(indices) < 10:
            indices.append(0)
        return torch.tensor(indices, dtype=torch.long)
    
    def forward(self, images, texts):
        device = images.device
        img_feats = self.image_encoder(images)
        
        text_feats = []
        for text in texts:
            tokens = self.tokenize(text).to(device)
            embedded = self.embedding(tokens).mean(dim=0)
            text_feats.append(embedded)
        text_feats = self.text_fc(torch.stack(text_feats))
        
        combined = torch.cat([img_feats, text_feats], dim=1)
        return self.classifier(combined).squeeze(-1)
    
    def predict(self, image_tensor, text_string):
        self.eval()
        with torch.no_grad():
            score = self.forward(image_tensor.unsqueeze(0), [text_string])
            return score.item()


if __name__ == "__main__":
    print("Creating model with random weights...")
    model = SubmissionModel()
    torch.save(model.state_dict(), 'weights.pth')
    print("Saved weights.pth")
    
    # Quick test
    dummy_img = torch.randn(3, 224, 224)
    score = model.predict(dummy_img, "A test caption")
    print(f"Test prediction: {score:.4f}")
