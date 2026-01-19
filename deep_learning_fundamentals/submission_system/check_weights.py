import torch
import sys
import os

# Add submission_system to path
sys.path.append(os.path.dirname(__file__))

from model import SubmissionModel

def check_weights():
    print("Sprawdzanie ładowania wag...")

    # Create model
    model = SubmissionModel()
    print("✓ Model utworzony")

    # Load weights
    weights_path = os.path.join(os.path.dirname(__file__), "weights.pth")
    if not os.path.exists(weights_path):
        print("❌ weights.pth nie istnieje")
        return

    try:
        state_dict = torch.load(weights_path, weights_only=True)
        model.load_state_dict(state_dict)
        print("✓ Wagi załadowane")
    except Exception as e:
        print(f"❌ Błąd ładowania wag: {e}")
        return

    # Test predict
    model.eval()
    with torch.no_grad():
        dummy_img = torch.randn(3, 224, 224)
        test_caption = "a dog running in the grass"
        try:
            score = model.predict(dummy_img, test_caption)
            print(f"✓ Predict działa: score = {score:.4f}")
            if 0.0 <= score <= 1.0:
                print("✓ Score w prawidłowym zakresie [0.0, 1.0]")
            else:
                print("❌ Score poza zakresem")
        except Exception as e:
            print(f"❌ Błąd predict: {e}")

    print("Sprawdzenie zakończone.")

if __name__ == "__main__":
    check_weights()