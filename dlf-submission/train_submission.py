import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm
import random
import torch.amp 

# Zakładam, że plik model.py jest w tym samym folderze
from model import SubmissionModel

# =====================
# CONFIG
# =====================
DATA_DIR = r"c:\Users\Filip\Documents\mgr-siium\deep_learning_fundamentals\ex_1"
CSV_FILE = os.path.join(DATA_DIR, "captions_217k_ultra_dataset.csv")
IMG_DIR = os.path.join(DATA_DIR, "datasets")

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
NUM_WORKERS = int(os.cpu_count() * 0.8)

# =====================
# DATASET
# =====================
class FlickrDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        while True:
            row = self.df.iloc[idx]
            img_path = row["image_path"]
            
            if img_path.startswith("datasets/"):
                img_path = img_path[len("datasets/"):]
            
            full_path = os.path.join(IMG_DIR, img_path)

            try:
                image = Image.open(full_path).convert("RGB")
                image = self.transform(image)
                caption = str(row["caption"])
                label = torch.tensor(float(row["label"]), dtype=torch.float32)

                return image, caption, label

            except (FileNotFoundError, OSError, Exception) as e:
                print(f"Error loading image {full_path}: {e}. Selecting a new random image.")
                idx = random.randint(0, len(self.df) - 1)

# =====================
# MAIN LOOP
# =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True 
    print("Device:", device)

    df = pd.read_csv(CSV_FILE)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FlickrDataset(df, transform)
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,         
        persistent_workers=True, 
        prefetch_factor=2       
    )

    model = SubmissionModel().to(device)

    print("Loading ImageNet weights...")
    pretrained = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    pretrained_layers = nn.Sequential(*list(pretrained.children())[:-2])
    model.image_encoder.load_state_dict(pretrained_layers.state_dict())
    print("ImageNet injected.")

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LR
    )

    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, captions, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # 1. Autocast tylko dla forward pass (obliczenia modelu)
            with torch.amp.autocast('cuda'):
                outputs = model(images, list(captions))
                outputs = outputs.squeeze()
            
            # 2. FIX: Wychodzimy z autocast i rzutujemy na float32 dla BCELoss
            # To zapobiega błędowi "unsafe to autocast"
            loss = criterion(outputs.float(), labels.float())

            # 3. Skalowanie gradientów
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # --- ACCURACY ---
            # Ponieważ BCELoss działa na prawdopodobieństwach (Sigmoid), próg to 0.5
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            loop.set_postfix(loss=loss.item())

        # Podsumowanie epoki
        epoch_loss = total_loss / len(loader)
        epoch_acc = correct_predictions / total_samples

        print(f"Epoch {epoch+1} summary | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        if epoch_acc > best_acc:
            print(f" >>> New Best Accuracy! ({best_acc:.4f} -> {epoch_acc:.4f}). Saving model...")
            best_acc = epoch_acc
            torch.save(model.state_dict(), "weights.pth")
        else:
            print(f" ... Accuracy did not improve (Best: {best_acc:.4f})")

    print("Training complete.")

if __name__ == "__main__":
    main()