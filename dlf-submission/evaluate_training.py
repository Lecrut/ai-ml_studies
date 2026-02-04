import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Import modelu
from model import SubmissionModel

# =====================
# CONFIG - takie same jak w train_submission.py
# =====================
DATA_DIR = r"c:\Users\Filip\Documents\mgr-siium\deep_learning_fundamentals\ex_1"
CSV_FILE = os.path.join(DATA_DIR, "captions_217k_ultra_dataset.csv")
IMG_DIR = os.path.join(DATA_DIR, "datasets")

BATCH_SIZE = 64
NUM_SAMPLES = 10000 
NUM_WORKERS = int(os.cpu_count() * 0.8)

# =====================
# DATASET - taki sam jak w treningu
# =====================
class FlickrDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        if img_path.startswith("datasets/"):
            img_path = img_path[len("datasets/"):]
        img_path = os.path.join(IMG_DIR, img_path)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption = str(row["caption"])
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return image, caption, label

# =====================
# EVALUATION FUNCTION
# =====================
def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ładuj dane treningowe
    print("Loading training data...")
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} samples")

    # Wybierz losowo 10k próbek dla szybszej ewaluacji
    if len(df) > NUM_SAMPLES:
        df = df.sample(NUM_SAMPLES, random_state=42)
        print(f"Sampled {NUM_SAMPLES} random samples for evaluation")
    else:
        print(f"Dataset has less than {NUM_SAMPLES} samples, using all data")

    # Transform taki sam jak w treningu
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset i DataLoader (bez shuffle dla deterministycznych wyników)
    dataset = FlickrDataset(df, transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Ważne: bez shuffle dla ewaluacji
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Ładuj model
    print("Loading model...")
    model = SubmissionModel().to(device)

    # Załaduj wagi - sprawdzamy dostępne pliki wag
    weight_files = [f for f in os.listdir('.') if f.startswith('weights') and f.endswith('.pth')]
    if weight_files:
        # Wybierz najnowszy plik po dacie modyfikacji
        weight_file = max(weight_files, key=lambda f: os.path.getmtime(f))
        print(f"Loading weights from: {weight_file}")
        model.load_state_dict(torch.load(weight_file, map_location=device))
    else:
        print("No weight files found! Using untrained model.")
        return

    model.eval()

    # Listy do zbierania wyników
    all_preds = []
    all_labels = []

    print("Evaluating on training data...")
    with torch.no_grad():
        for images, captions, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images, list(captions))
            outputs = outputs.squeeze()

            # Przewidywania z progiem 0.5
            preds = (outputs > 0.5).float()

            # Zbierz wyniki
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Konwertuj na numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # =====================
    # STATYSTYKI
    # =====================

    print("\n" + "="*50)
    print("EVALUATION RESULTS ON TRAINING DATA")
    print("="*50)

    # Podstawowe metryki
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Macierz pomyłek
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print("[[TN, FP]")
    print(" [FN, TP]]")
    print(cm)

    # Szczegółowy raport klasyfikacji
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1'], zero_division=0))

    # Statystyki próbek
    total_samples = len(all_labels)
    class_0_count = np.sum(all_labels == 0)
    class_1_count = np.sum(all_labels == 1)

    print(f"\nDataset Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Class 0 samples: {class_0_count} ({class_0_count/total_samples*100:.1f}%)")
    print(f"Class 1 samples: {class_1_count} ({class_1_count/total_samples*100:.1f}%)")

    # Sprawdź balans klas
    if abs(class_0_count - class_1_count) / total_samples < 0.1:
        print("Dataset appears balanced.")
    else:
        print("Dataset appears imbalanced.")

    print("\nEvaluation complete!")

if __name__ == "__main__":
    evaluate_model()