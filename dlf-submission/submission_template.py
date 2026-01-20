"""
================================================================================
                    SUBMISSION TEMPLATE - IMAGE-TEXT MATCHING
================================================================================

This is the OFFICIAL TEMPLATE for your submission. Rename this file to model.py.

TASK: Given an image and a text caption, predict if they match (1.0) or not (0.0).

================================================================================
                              REQUIRED FILES
================================================================================

Your submission ZIP MUST contain:

    submission.zip
    ├── model.py          # THIS FILE (renamed) - REQUIRED
    └── weights.pth       # Your trained model weights - REQUIRED

================================================================================
                         OPTIONAL: CUSTOM MODULES
================================================================================

You CAN include additional files for tokenizers, helper modules, configs, etc:

    submission.zip
    ├── model.py                    # REQUIRED
    ├── weights.pth                 # REQUIRED
    ├── my_text_encoder.py          # OPTIONAL - your custom Python module
    ├── tokenizer/                  # OPTIONAL - HuggingFace tokenizer folder
    │   ├── vocab.txt
    │   ├── tokenizer_config.json
    │   └── special_tokens_map.json
    ├── config.json                 # OPTIONAL - any config file
    └── embeddings/                 # OPTIONAL - pretrained embeddings
        └── glove_subset.pt

HOW TO LOAD THESE FILES:
    
    import os
    
    # Get the directory where model.py is located
    SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # METHOD 1: Import your custom Python modules
    # ============================================
    # If my_text_encoder.py is in your zip, just import it directly:
    from my_text_encoder import TextEncoder
    
    # METHOD 2: Load tokenizer from folder
    # ====================================
    from transformers import AutoTokenizer
    tokenizer_path = os.path.join(SUBMISSION_DIR, 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # METHOD 3: Load any file (config, embeddings, vocab, etc.)
    # =========================================================
    import json
    config_path = os.path.join(SUBMISSION_DIR, 'config.json')
    with open(config_path) as f:
        config = json.load(f)
    
    # Load PyTorch tensors
    embeddings_path = os.path.join(SUBMISSION_DIR, 'embeddings', 'glove_subset.pt')
    embeddings = torch.load(embeddings_path, weights_only=True)

================================================================================
                             CRITICAL RULES
================================================================================

1. NO INTERNET ACCESS during evaluation - include ALL required files in your ZIP
2. Class MUST be named: SubmissionModel
3. MUST implement: predict(image_tensor, text_string) -> float
4. predict() returns a float between 0.0 and 1.0 (use .item() to convert tensors)
5. Time limit: 10 minutes for the entire test set
6. Memory limits: 32GB RAM, ~20GB VRAM

================================================================================
                         TESTING YOUR SUBMISSION
================================================================================

Before submitting, test your model:

    # Quick self-test (runs basic checks)
    python model.py
    
    # Full evaluation on validation set (RECOMMENDED)
    # This runs exactly like the real submission system!
    python submission_evaluator.py ./your_submission_folder/
    
    # Or test a ZIP file directly
    python submission_evaluator.py submission.zip

================================================================================
"""

import os
import re
import torch
import torch.nn as nn
from torchvision import transforms

# =============================================================================
# SUBMISSION_DIR: Use this to load any files from your submission ZIP
# =============================================================================
SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# OPTIONAL: Custom Image Transform
# =============================================================================
# If you don't define this function, the evaluator uses a default transform:
#   Resize(224, 224) -> ToTensor() -> Normalize(ImageNet mean/std)

def get_transform():
    """
    OPTIONAL: Define custom image preprocessing.
    
    The evaluator will call this function and use the returned transform
    to preprocess images BEFORE passing them to your predict() method.
    
    Returns:
        torchvision.transforms.Compose: Your preprocessing pipeline
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# =============================================================================
# REQUIRED: SubmissionModel Class
# =============================================================================

class SubmissionModel(nn.Module):
    """
    REQUIRED: Your model for image-text matching.
    
    The evaluator will:
        1. Create an instance: model = SubmissionModel()
        2. Load your weights: model.load_state_dict(torch.load('weights.pth'))
        3. Move to GPU: model.to(device)
        4. Call predict() for each (image, text) pair
    
    You MUST implement:
        - __init__(): Initialize your model architecture
        - predict(image_tensor, text_string) -> float: Inference method
    """
    
    def __init__(self):
        """
        Initialize your model architecture.
        
        This is called BEFORE weights are loaded.
        Define all nn.Module layers here.
        
        To load custom files (tokenizers, configs, etc.), use:
            path = os.path.join(SUBMISSION_DIR, 'your_file')
        """
        super().__init__()
        
        # =====================================================================
        # EXAMPLE: Simple Baseline Model
        # Replace this with your actual architecture!
        # =====================================================================
        
        # --- Image Encoder ---
        # A tiny CNN that outputs a 32-dim feature vector
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # -> 112x112
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> 56x56
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> 32x1x1
            nn.Flatten()  # -> 32
        )
        
        # --- Text Encoder ---
        # Simple bag-of-words with hash-based tokenization
        # (In practice, use a proper tokenizer like BERT, CLIP, etc.)
        self.vocab_size = 1000
        self.embedding = nn.Embedding(self.vocab_size, 32)
        self.text_fc = nn.Linear(32, 32)
        
        # --- Fusion & Classifier ---
        # Combines image (32) + text (32) -> match probability
        self.classifier = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # =====================================================================
        # EXAMPLE: Loading a custom tokenizer from your ZIP
        # Uncomment and modify for your actual model:
        # =====================================================================
        # from transformers import AutoTokenizer
        # tokenizer_path = os.path.join(SUBMISSION_DIR, 'tokenizer')
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # =====================================================================
        # EXAMPLE: Importing a custom module from your ZIP
        # If you included my_encoder.py in your zip:
        # =====================================================================
        # from my_encoder import MyCustomEncoder
        # self.custom_encoder = MyCustomEncoder()
    
    def _tokenize_text(self, text):
        """
        EXAMPLE: Simple hash-based tokenization.
        Replace with your actual tokenization!
        
        Args:
            text: A string caption
            
        Returns:
            torch.LongTensor of token indices
        """
        # Clean text and split into words
        words = re.sub(r'[^a-zA-Z\s]', '', text.lower()).split()
        
        # Hash each word to get a vocabulary index (simple approach)
        max_len = 10
        indices = [hash(w) % self.vocab_size for w in words[:max_len]]
        
        # Pad to fixed length
        while len(indices) < max_len:
            indices.append(0)
        
        return torch.tensor(indices, dtype=torch.long)
    
    def forward(self, images, texts):
        """
        Forward pass for training (optional to implement).
        
        Args:
            images: Batch of images (B, C, H, W)
            texts: List of text strings (length B)
            
        Returns:
            scores: Match probabilities (B,)
        """
        device = images.device
        batch_size = images.size(0)
        
        # Encode images
        img_features = self.image_encoder(images)  # (B, 32)
        
        # Encode texts (one by one to keep it simple)
        text_features = []
        for text in texts:
            tokens = self._tokenize_text(text).to(device)  # (10,)
            embedded = self.embedding(tokens)  # (10, 32)
            pooled = embedded.mean(dim=0)  # (32,)
            text_features.append(pooled)
        text_features = torch.stack(text_features)  # (B, 32)
        text_features = self.text_fc(text_features)  # (B, 32)
        
        # Fusion
        combined = torch.cat([img_features, text_features], dim=1)  # (B, 64)
        scores = self.classifier(combined).squeeze(-1)  # (B,)
        
        return scores
    
    def predict(self, image_tensor, text_string):
        """
        REQUIRED: The evaluator calls this method for each sample.
        
        Args:
            image_tensor: A single preprocessed image tensor (C, H, W)
                         - Already transformed by get_transform()
                         - Already on the correct device (GPU/CPU)
            text_string: A single text caption (str)
            
        Returns:
            float: Match score between 0.0 and 1.0
                   - >= 0.5 means "match"
                   - < 0.5 means "no match"
        
        IMPORTANT:
            - Return a Python float, not a tensor!
            - Use .item() to convert: tensor.item()
            - Keep the model in eval mode
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension: (C, H, W) -> (1, C, H, W)
            image_batch = image_tensor.unsqueeze(0)
            
            # Run forward pass
            score_tensor = self.forward(image_batch, [text_string])
            
            # Convert to Python float (REQUIRED!)
            score = score_tensor.item()
            
            return score


# =============================================================================
# LOCAL TESTING - Run this file to test your model
# =============================================================================

def test_submission():
    """
    Basic self-test for your model.
    Run: python model.py
    """
    print("=" * 70)
    print("                    SUBMISSION SELF-TEST")
    print("=" * 70)
    
    errors = []
    warnings = []
    
    # -------------------------------------------------------------------------
    # Test 1: Can we instantiate the model?
    # -------------------------------------------------------------------------
    print("\n[1/4] Testing model instantiation...")
    try:
        model = SubmissionModel()
        print("      ✓ SubmissionModel() created successfully")
    except Exception as e:
        errors.append(f"Failed to create model: {e}")
        print(f"      ✗ ERROR: {e}")
        # Can't continue without a model
        print("\n" + "=" * 70)
        print("RESULT: FAILED - Cannot instantiate model")
        print("=" * 70)
        return False
    
    # -------------------------------------------------------------------------
    # Test 2: Does predict() work?
    # -------------------------------------------------------------------------
    print("\n[2/4] Testing predict() method...")
    try:
        model.eval()
        dummy_image = torch.randn(3, 224, 224)
        dummy_text = "A dog running through a grassy field"
        
        score = model.predict(dummy_image, dummy_text)
        
        # Check return type
        if not isinstance(score, float):
            errors.append(f"predict() must return float, got {type(score).__name__}")
            print(f"      ✗ ERROR: predict() returned {type(score).__name__}, expected float")
        elif not (0.0 <= score <= 1.0):
            errors.append(f"Score {score} is outside valid range [0.0, 1.0]")
            print(f"      ✗ ERROR: Score {score} not in [0.0, 1.0]")
        else:
            print(f"      ✓ predict() returned valid score: {score:.4f}")
    except Exception as e:
        errors.append(f"predict() failed: {e}")
        print(f"      ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # Test 3: Does get_transform() work?
    # -------------------------------------------------------------------------
    print("\n[3/4] Testing get_transform()...")
    try:
        transform = get_transform()
        from PIL import Image
        dummy_pil = Image.new('RGB', (300, 200))  # Non-square to test resize
        transformed = transform(dummy_pil)
        
        if transformed.shape[0] != 3:
            warnings.append(f"Transform output has {transformed.shape[0]} channels, expected 3")
            print(f"      ⚠ WARNING: Output has {transformed.shape[0]} channels")
        else:
            print(f"      ✓ get_transform() works! Output shape: {tuple(transformed.shape)}")
    except Exception as e:
        warnings.append(f"get_transform() failed: {e} (will use default)")
        print(f"      ⚠ WARNING: {e}")
        print("      (Evaluator will use default transform)")
    
    # -------------------------------------------------------------------------
    # Test 4: Check if weights.pth exists
    # -------------------------------------------------------------------------
    print("\n[4/4] Checking for weights.pth...")
    weights_path = os.path.join(SUBMISSION_DIR, 'weights.pth')
    if os.path.exists(weights_path):
        print(f"      ✓ weights.pth found ({os.path.getsize(weights_path) / 1024 / 1024:.1f} MB)")
        
        # Try loading weights
        try:
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            print("      ✓ Weights loaded successfully")
        except Exception as e:
            errors.append(f"Failed to load weights: {e}")
            print(f"      ✗ ERROR: Failed to load weights: {e}")
    else:
        warnings.append("weights.pth not found (required for submission)")
        print("      ⚠ WARNING: weights.pth not found in current directory")
        print("      (You must create weights.pth before submitting)")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    if errors:
        print("RESULT: FAILED")
        print("=" * 70)
        print("\nErrors (must fix before submitting):")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("RESULT: PASSED")
        print("=" * 70)
    
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  ⚠ {w}")
    
    # -------------------------------------------------------------------------
    # Submission Checklist
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SUBMISSION CHECKLIST:")
    print("-" * 70)
    print("  [ ] Model architecture implemented in SubmissionModel")
    print("  [ ] predict() returns float in [0.0, 1.0]")
    print("  [ ] Weights saved: torch.save(model.state_dict(), 'weights.pth')")
    print("  [ ] All custom files included in ZIP (tokenizers, configs, etc.)")
    print("  [ ] No internet dependencies (everything is local)")
    print("  [ ] ZIP created: zip -r submission.zip model.py weights.pth [other files...]")
    print("-" * 70)
    
    return len(errors) == 0


if __name__ == "__main__":
    success = test_submission()
    exit(0 if success else 1)
