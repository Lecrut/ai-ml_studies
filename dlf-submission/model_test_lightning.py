"""
PyTorch Lightning model example for the Image-Text Matching competition.
Run this file to generate weights.pth that you can submit.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
import re
import lightning as L
from model import SubmissionModel

SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



# =============================================================================
# Lightning Module wrapper (shows how to use SubmissionModel with Lightning)
# =============================================================================

class LitModel(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = SubmissionModel()
        self.criterion = nn.BCELoss()
    
    def forward(self, images, texts):
        return self.model(images, texts)
    
    def training_step(self, batch, batch_idx):
        images, texts, labels = batch
        outputs = self(images, texts)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def save_submission_weights(self, path='weights.pth'):
        """Save only the SubmissionModel weights (not the full LightningModule)."""
        torch.save(self.model.state_dict(), path)
        print(f"Saved {path}")


if __name__ == "__main__":
    print("Creating model with random weights...")
    model = SubmissionModel()
    torch.save(model.state_dict(), 'weights.pth')
    print("Saved weights.pth")
    
    # Quick test
    dummy_img = torch.randn(3, 224, 224)
    score = model.predict(dummy_img, "A test caption")
    print(f"Test prediction: {score:.4f}")
    
    # Show how to save from LightningModule after training
    print("\nTo save weights after Lightning training:")
    print("  lit_model.save_submission_weights('weights.pth')")
