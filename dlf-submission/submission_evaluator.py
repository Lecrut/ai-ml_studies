#!/usr/bin/env python3
"""
================================================================================
                    SUBMISSION EVALUATOR - Test Your Model Locally
================================================================================

This script simulates EXACTLY what happens when you submit to the competition.
If your submission works here, it will work on the submission system.

USAGE:
    # Test a submission folder (must contain model.py and weights.pth)
    python submission_evaluator.py ./my_submission/

    # Test a ZIP file directly
    python submission_evaluator.py submission.zip

    # Specify custom data directory
    python submission_evaluator.py ./my_submission/ --data ./flickr8k_dataset/

    # Limit number of samples (for quick testing)
    python submission_evaluator.py ./my_submission/ --max-samples 50

WHAT THIS DOES:
    1. Loads your model.py (just like the real evaluator)
    2. Loads your weights.pth
    3. Uses your get_transform() if defined
    4. Runs predict() on each (image, caption) pair
    5. Computes accuracy and prints the result

REQUIREMENTS:
    Your submission folder must contain:
        - model.py      (with SubmissionModel class and predict() method)
        - weights.pth   (your trained weights)
        - [optional]    Any additional files (tokenizers, configs, etc.)

================================================================================
"""

import os
import sys
import argparse
import tempfile
import zipfile
import time
import importlib.util
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms


# =============================================================================
# Default paths (can be overridden via command line)
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "flickr8k_dataset"
DEFAULT_TEST_CSV = "hidden_test.csv"  # This is the validation set you can share


# =============================================================================
# Default transform (used if model doesn't provide get_transform)
# =============================================================================
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# =============================================================================
# Error classes for helpful messages
# =============================================================================

class SubmissionError(Exception):
    """Base class for submission errors with helpful messages."""
    pass


class ModelNotFoundError(SubmissionError):
    def __init__(self, path):
        self.path = path
        super().__init__(
            f"\n{'='*60}\n"
            f"ERROR: model.py not found!\n"
            f"{'='*60}\n"
            f"Looking in: {path}\n\n"
            f"Your submission folder MUST contain a file named 'model.py'.\n"
            f"This file must define:\n"
            f"  - A class named 'SubmissionModel' (inherits from nn.Module)\n"
            f"  - A method 'predict(image_tensor, text_string)' that returns a float\n\n"
            f"See 'submission_template.py' for an example.\n"
            f"{'='*60}"
        )


class WeightsNotFoundError(SubmissionError):
    def __init__(self, path):
        self.path = path
        super().__init__(
            f"\n{'='*60}\n"
            f"ERROR: weights.pth not found!\n"
            f"{'='*60}\n"
            f"Looking in: {path}\n\n"
            f"Your submission folder MUST contain 'weights.pth'.\n"
            f"This file should contain your trained model weights.\n\n"
            f"Create it with:\n"
            f"    torch.save(model.state_dict(), 'weights.pth')\n"
            f"{'='*60}"
        )


class SubmissionModelNotFoundError(SubmissionError):
    def __init__(self):
        super().__init__(
            f"\n{'='*60}\n"
            f"ERROR: SubmissionModel class not found!\n"
            f"{'='*60}\n"
            f"Your model.py must define a class named exactly 'SubmissionModel'.\n\n"
            f"Example:\n"
            f"    class SubmissionModel(nn.Module):\n"
            f"        def __init__(self):\n"
            f"            super().__init__()\n"
            f"            # ... your architecture ...\n"
            f"        \n"
            f"        def predict(self, image_tensor, text_string):\n"
            f"            # ... your inference code ...\n"
            f"            return score  # float between 0.0 and 1.0\n"
            f"{'='*60}"
        )


class PredictMethodNotFoundError(SubmissionError):
    def __init__(self):
        super().__init__(
            f"\n{'='*60}\n"
            f"ERROR: predict() method not found!\n"
            f"{'='*60}\n"
            f"Your SubmissionModel class MUST have a 'predict' method.\n\n"
            f"Required signature:\n"
            f"    def predict(self, image_tensor, text_string):\n"
            f"        '''\n"
            f"        Args:\n"
            f"            image_tensor: Preprocessed image (C, H, W) on device\n"
            f"            text_string: Caption text (str)\n"
            f"        Returns:\n"
            f"            float: Score between 0.0 and 1.0\n"
            f"        '''\n"
            f"        # Your inference code here\n"
            f"        return score\n"
            f"{'='*60}"
        )


class WeightsLoadError(SubmissionError):
    def __init__(self, original_error):
        super().__init__(
            f"\n{'='*60}\n"
            f"ERROR: Failed to load weights!\n"
            f"{'='*60}\n"
            f"Error: {original_error}\n\n"
            f"Common causes:\n"
            f"  1. weights.pth was saved with a different model architecture\n"
            f"  2. Layer names don't match (check your nn.Module definitions)\n"
            f"  3. File is corrupted\n\n"
            f"To debug, try loading manually:\n"
            f"    from model import SubmissionModel\n"
            f"    model = SubmissionModel()\n"
            f"    state_dict = torch.load('weights.pth', map_location='cpu')\n"
            f"    model.load_state_dict(state_dict)\n"
            f"{'='*60}"
        )


class PredictReturnTypeError(SubmissionError):
    def __init__(self, returned_type, returned_value):
        super().__init__(
            f"\n{'='*60}\n"
            f"ERROR: predict() must return a float!\n"
            f"{'='*60}\n"
            f"Your predict() returned: {returned_type} = {returned_value}\n\n"
            f"predict() MUST return a Python float between 0.0 and 1.0.\n\n"
            f"If you're returning a tensor, use .item():\n"
            f"    score = output_tensor.item()  # Converts tensor to float\n"
            f"    return score\n"
            f"{'='*60}"
        )


class PredictScoreRangeError(SubmissionError):
    def __init__(self, score):
        super().__init__(
            f"\n{'='*60}\n"
            f"ERROR: predict() returned score outside valid range!\n"
            f"{'='*60}\n"
            f"Returned score: {score}\n"
            f"Valid range: 0.0 to 1.0\n\n"
            f"Make sure your model outputs a probability (use Sigmoid).\n"
            f"{'='*60}"
        )


class TransformError(SubmissionError):
    def __init__(self, original_error):
        super().__init__(
            f"\n{'='*60}\n"
            f"ERROR: get_transform() failed!\n"
            f"{'='*60}\n"
            f"Error: {original_error}\n\n"
            f"Your get_transform() function must return a valid transform.\n"
            f"Using default transform instead.\n"
            f"{'='*60}"
        )


class InferenceError(SubmissionError):
    def __init__(self, sample_idx, image_name, caption_preview, original_error):
        caption_short = caption_preview[:50] + "..." if len(caption_preview) > 50 else caption_preview
        super().__init__(
            f"\n{'='*60}\n"
            f"ERROR: Inference failed on sample {sample_idx}!\n"
            f"{'='*60}\n"
            f"Image: {image_name}\n"
            f"Caption: {caption_short}\n"
            f"Error: {original_error}\n\n"
            f"Common causes:\n"
            f"  1. Tensor device mismatch (use .to(device) consistently)\n"
            f"  2. Shape mismatch in your model\n"
            f"  3. Text processing error\n"
            f"{'='*60}"
        )


# =============================================================================
# Core evaluation logic
# =============================================================================

def load_submission(submission_path: Path):
    """
    Load a submission from a directory.
    
    Returns: (model, device, transform)
    Raises: SubmissionError with helpful message on failure
    """
    submission_path = Path(submission_path).resolve()
    
    # Check for required files
    model_path = submission_path / "model.py"
    weights_path = submission_path / "weights.pth"
    
    if not model_path.exists():
        raise ModelNotFoundError(model_path)
    
    if not weights_path.exists():
        raise WeightsNotFoundError(weights_path)
    
    # Add submission directory to Python path (for imports)
    original_path = sys.path.copy()
    sys.path.insert(0, str(submission_path))
    
    # Change to submission directory (for relative paths)
    original_cwd = os.getcwd()
    os.chdir(submission_path)
    
    try:
        # Dynamically import model.py
        print(f"[INFO] Loading model from: {model_path}")
        spec = importlib.util.spec_from_file_location("user_model", str(model_path))
        user_module = importlib.util.module_from_spec(spec)
        sys.modules['user_model'] = user_module
        spec.loader.exec_module(user_module)
        
        # Check for SubmissionModel class
        if not hasattr(user_module, 'SubmissionModel'):
            raise SubmissionModelNotFoundError()
        
        # Try to get custom transform
        if hasattr(user_module, 'get_transform'):
            try:
                transform = user_module.get_transform()
                print("[INFO] Using custom get_transform() from your model")
            except Exception as e:
                print(f"[WARN] get_transform() failed: {e}")
                print("[WARN] Using default transform instead")
                transform = DEFAULT_TRANSFORM
        else:
            print("[INFO] No get_transform() found, using default transform")
            transform = DEFAULT_TRANSFORM
        
        # Create model instance
        print("[INFO] Creating SubmissionModel instance...")
        model = user_module.SubmissionModel()
        
        # Check for predict method
        if not hasattr(model, 'predict'):
            raise PredictMethodNotFoundError()
        
        # Load weights
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Loading weights on device: {device}")
        
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as e:
            raise WeightsLoadError(e)
        
        model.to(device)
        model.eval()
        print("[INFO] Model loaded successfully!")
        
        return model, device, transform
        
    finally:
        # Restore original state
        os.chdir(original_cwd)
        sys.path = original_path


def run_evaluation(model, device, transform, data_dir: Path, max_samples: int = None):
    """
    Run evaluation on the test set.
    
    Returns: (accuracy, total_samples, failed_samples)
    """
    test_csv = data_dir / DEFAULT_TEST_CSV
    images_dir = data_dir / "Images"
    
    if not test_csv.exists():
        print(f"\n[ERROR] Test CSV not found: {test_csv}")
        print(f"Make sure you have the dataset in: {data_dir}")
        sys.exit(1)
    
    if not images_dir.exists():
        print(f"\n[ERROR] Images directory not found: {images_dir}")
        sys.exit(1)
    
    # Load test data
    df = pd.read_csv(test_csv)
    
    if max_samples and max_samples < len(df):
        print(f"[INFO] Limiting to {max_samples} samples (out of {len(df)})")
        df = df.head(max_samples)
    
    total = len(df)
    print(f"\n[INFO] Running evaluation on {total} samples...")
    print("-" * 60)
    
    correct = 0
    evaluated = 0
    failed = 0
    
    start_time = time.time()
    
    for idx, row in df.iterrows():
        img_filename = row['image']
        caption = row['caption']
        target = int(row['label'])
        
        img_path = images_dir / img_filename
        
        if not img_path.exists():
            print(f"[WARN] Image not found: {img_filename}, skipping...")
            failed += 1
            continue
        
        try:
            # Load and transform image
            image = Image.open(img_path).convert('RGB')
            img_tensor = transform(image).to(device)
            
            # Run prediction
            score = model.predict(img_tensor, caption)
            
            # Validate output
            if not isinstance(score, (float, int)):
                raise PredictReturnTypeError(type(score).__name__, score)
            
            score = float(score)
            
            if not (0.0 <= score <= 1.0):
                raise PredictScoreRangeError(score)
            
            # Binarize prediction
            pred = 1 if score >= 0.5 else 0
            
            if pred == target:
                correct += 1
            
            evaluated += 1
            
            # Progress indicator
            if evaluated % 100 == 0 or evaluated == total:
                elapsed = time.time() - start_time
                eta = (elapsed / evaluated) * (total - evaluated) if evaluated > 0 else 0
                print(f"[PROGRESS] {evaluated}/{total} samples | "
                      f"Accuracy so far: {correct/evaluated:.2%} | "
                      f"ETA: {eta:.1f}s")
                
        except SubmissionError:
            raise  # Re-raise our custom errors
        except Exception as e:
            # First failure: show detailed error
            if failed == 0:
                raise InferenceError(idx, img_filename, caption, e)
            failed += 1
            if failed <= 3:
                print(f"[WARN] Sample {idx} failed: {e}")
            elif failed == 4:
                print("[WARN] Additional failures suppressed...")
    
    elapsed = time.time() - start_time
    
    if evaluated == 0:
        print("\n[ERROR] No samples were evaluated successfully!")
        return 0.0, 0, failed
    
    accuracy = correct / evaluated
    
    return accuracy, evaluated, failed, elapsed


def print_results(accuracy, evaluated, failed, elapsed):
    """Print final results."""
    print("\n" + "=" * 60)
    print("                    EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated: {evaluated}")
    print(f"  Samples failed:    {failed}")
    print(f"  Time elapsed:      {elapsed:.1f} seconds")
    print("-" * 60)
    print(f"  ACCURACY: {accuracy:.4f} ({accuracy:.2%})")
    print("=" * 60)
    
    if accuracy >= 0.7:
        print("\n🎉 Great job! Your model is performing well.")
    elif accuracy >= 0.5:
        print("\n📈 Your model is better than random. Keep improving!")
    else:
        print("\n💡 Tip: A score below 50% is worse than random guessing.")
        print("   Check if your model is learning the right patterns.")
    
    print()


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract a ZIP file to a directory."""
    print(f"[INFO] Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print(f"[INFO] Extracted to: {extract_to}")


def main():
    parser = argparse.ArgumentParser(
        description="Test your submission locally before uploading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python submission_evaluator.py ./my_submission/
    python submission_evaluator.py submission.zip
    python submission_evaluator.py ./my_submission/ --max-samples 100
    python submission_evaluator.py ./my_submission/ --data ./flickr8k_dataset/
        """
    )
    
    parser.add_argument('submission',
                        help='Path to submission folder or ZIP file')
    parser.add_argument('--data', '-d', type=str, default=str(DEFAULT_DATA_DIR),
                        help=f'Path to dataset directory (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--max-samples', '-n', type=int, default=None,
                        help='Maximum number of samples to evaluate (for quick testing)')
    
    args = parser.parse_args()
    
    submission_path = Path(args.submission)
    data_dir = Path(args.data)
    
    print("=" * 60)
    print("           SUBMISSION EVALUATOR - Local Testing")
    print("=" * 60)
    print(f"Submission: {submission_path}")
    print(f"Data dir:   {data_dir}")
    print("=" * 60)
    
    # Handle ZIP files
    temp_dir = None
    if submission_path.suffix == '.zip':
        if not submission_path.exists():
            print(f"\n[ERROR] ZIP file not found: {submission_path}")
            sys.exit(1)
        
        temp_dir = tempfile.mkdtemp(prefix="submission_test_")
        extract_zip(submission_path, Path(temp_dir))
        submission_path = Path(temp_dir)
    
    if not submission_path.exists():
        print(f"\n[ERROR] Submission path not found: {submission_path}")
        sys.exit(1)
    
    try:
        # Load submission
        model, device, transform = load_submission(submission_path)
        
        # Run evaluation
        accuracy, evaluated, failed, elapsed = run_evaluation(
            model, device, transform, data_dir, args.max_samples
        )
        
        # Print results
        print_results(accuracy, evaluated, failed, elapsed)
        
        # Return success/failure based on whether evaluation completed
        sys.exit(0 if evaluated > 0 else 1)
        
    except SubmissionError as e:
        print(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[INFO] Evaluation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup temp directory if we created one
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
