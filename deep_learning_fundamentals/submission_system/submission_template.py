"""
SUBMISSION TEMPLATE FOR IMAGE-TEXT MATCHING COMPETITION

Instructions:
1. Implement your model in the SubmissionModel class
2. Optionally customize get_transform() for your preprocessing
3. Save weights: torch.save(model.state_dict(), 'weights.pth')
4. Create submission: zip submission.zip model.py weights.pth
5. Upload to: /submissions/queue/YOUR_TEAM_NAME/submission.zip

Task: Predict if an image and text caption match (1) or not (0)
Output: Float between 0.0 and 1.0 (≥0.5 = match)
"""

import torch
import torch.nn as nn
from torchvision import transforms


def get_transform():
    """
    OPTIONAL: Define custom image preprocessing.
    If omitted, default transform will be used (Resize 224, ImageNet normalization).
    
    Returns:
        torchvision.transforms.Compose: Preprocessing pipeline
        
    Example with augmentation:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class SubmissionModel(nn.Module):
    """
    REQUIRED: Main model class for image-text matching.
    
    Must implement:
        - __init__(): Initialize your model
        - predict(image_tensor, text_string): Inference method
    """
    
    def __init__(self):
        """
        Initialize your model architecture here.
        
        Tips:
        - Here you can use pretrained tokenizers/encoders/models
        - THERE IS NO INTERNET ACCESS DURING EVALUATION, load all resources LOCALLY
        - Keep model size reasonable (CPU RAM limit: 32GB, GPU VRAM limit: 20 GB)
        - Optimize for inference speed (10 min timeout)
        """
        super().__init__()
        
        # Example: Simple baseline
        # TODO: Replace with your architecture
        
        # Image encoder
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Text encoder (placeholder - use proper NLP model)
        self.text_encoder = nn.Sequential(
            nn.Linear(512, 64),  # Assume text features
            nn.ReLU()
        )
        
        # Fusion and classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def encode_text(self, text):
        """
        Helper method: Convert text to features.
        
        Here you can implement proper text encoding:

        """
        # Placeholder: random features
        # Replace with actual text encoding
        return torch.randn(1, 512, device=next(self.parameters()).device)
    
    def forward(self, images, texts):
        """
        Training-time forward pass (if needed).
        
        Args:
            images: Batch of images (B, C, H, W)
            texts: List of text strings
            
        Returns:
            scores: Match scores (B, 1) or (B,)
        """
        # Encode images
        img_feats = self.image_encoder(images)  # (B, 64)
        
        # Encode texts
        text_feats_list = [self.encode_text(text) for text in texts]
        text_feats = torch.cat(text_feats_list, dim=0)  # (B, 512)
        text_feats = self.text_encoder(text_feats)  # (B, 64)
        
        # Combine and classify
        combined = torch.cat([img_feats, text_feats], dim=1)  # (B, 128)
        scores = self.classifier(combined).squeeze(-1)  # (B,)
        
        return scores
    
    def predict(self, image_tensor, text_string):
        """
        REQUIRED: Inference method called by evaluator.
        
        Args:
            image_tensor: Single preprocessed image (C, H, W)
            text_string: Single caption text
            
        Returns:
            float: Match score between 0.0 and 1.0
                  ≥0.5 = match, <0.5 = no match
        
        CRITICAL:
        - Must return a Python float, not a tensor
        - Use .item() to convert tensor to float
        - Handle device placement correctly
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            image_batch = image_tensor.unsqueeze(0)  # (1, C, H, W)
            
            # Run inference
            score_tensor = self.forward(image_batch, [text_string])
            
            # Convert to float
            score = score_tensor.item()
            
            return score


# =============================================================================
# TESTING YOUR SUBMISSION LOCALLY
# =============================================================================

def test_submission():
    """
    Test your model before submitting.
    Run: python model.py
    """
    print("Testing submission...")
    
    # Create model
    model = SubmissionModel()
    model.eval()
    
    # Create dummy input
    dummy_image = torch.randn(3, 224, 224)
    dummy_text = "A dog running through a field"
    
    # Test predict method
    try:
        score = model.predict(dummy_image, dummy_text)
        print(f"✓ predict() works! Score: {score}")
        
        # Validate output
        if not isinstance(score, float):
            print(f"❌ ERROR: predict() must return float, got {type(score)}")
        elif not (0.0 <= score <= 1.0):
            print(f"❌ ERROR: Score {score} not in [0.0, 1.0]")
        else:
            print("✓ Output format valid")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Test transform
    try:
        transform = get_transform()
        from PIL import Image
        dummy_pil = Image.new('RGB', (256, 256))
        transformed = transform(dummy_pil)
        print(f"✓ get_transform() works! Output shape: {transformed.shape}")
    except Exception as e:
        print(f"⚠ get_transform() error: {e}")
        print("  (Will use default transform)")
    
    print("\n" + "="*60)
    print("SUBMISSION CHECKLIST:")
    print("="*60)
    print("[ ] Model architecture implemented")
    print("[ ] predict() method works correctly")
    print("[ ] Weights trained and saved to weights.pth")
    print("[ ] get_transform() defined (optional)")
    print("[ ] Tested locally with sample data")
    print("[ ] Created zip: zip submission.zip model.py weights.pth")
    print("[ ] Ready to upload!")
    print("="*60)


if __name__ == "__main__":
    test_submission()
