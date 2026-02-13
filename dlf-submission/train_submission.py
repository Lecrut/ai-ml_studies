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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from model import SubmissionModel

# =====================
# CONFIG
# =====================
DATA_DIR = r"c:\Users\Filip\Documents\mgr-siium\deep_learning_fundamentals\ex_1"
CSV_FILE = os.path.join(DATA_DIR, "captions_flickr8k_12_02.csv")
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
    # df = df.sample(frac=0.5, random_state=42)

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
    
    WEIGHTS_FILE = "weights.pth"

    if os.path.exists(WEIGHTS_FILE):
        print(f"Znaleziono plik {WEIGHTS_FILE}. Wczytywanie wytrenowanego modelu...")
        try:
            model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
            print("Sukces! Wagi załadowane - kontynuujemy trening (fine-tuning).")
            
        except Exception as e:
            print(f"Błąd podczas ładowania wag: {e}. Rozpoczynam trening od zera.")
            print("Loading ImageNet weights...")
            pretrained = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            pretrained_layers = nn.Sequential(*list(pretrained.children())[:-2])
            model.image_encoder.load_state_dict(pretrained_layers.state_dict())
            print("ImageNet injected.")
    else:
        print("Brak zapisanego modelu. Rozpoczynam trening od zera.")
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

    batch_losses = []
    batch_accs = []
    batch_nums = []

    last_epoch_preds = []
    last_epoch_labels = []

    epoch_accs = []
    batches_per_epoch = len(loader)

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

            with torch.amp.autocast('cuda'):
                outputs = model(images, list(captions))
                outputs = outputs.squeeze()
            
            loss = criterion(outputs.float(), labels.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # --- ACCURACY ---
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            batch_acc = (predicted == labels).float().mean().item()

            batch_losses.append(loss.item())
            batch_accs.append(batch_acc)
            batch_nums.append(len(batch_nums))

            if epoch == EPOCHS - 1:
                last_epoch_preds.extend(predicted.cpu().numpy())
                last_epoch_labels.extend(labels.cpu().numpy())

            loop.set_postfix(loss=loss.item())

        epoch_loss = total_loss / len(loader)
        epoch_acc = correct_predictions / total_samples

        epoch_accs.append(epoch_acc)

        print(f"Epoch {epoch+1} summary | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        if epoch_acc > best_acc:
            print(f" >>> New Best Accuracy! ({best_acc:.4f} -> {epoch_acc:.4f}). Saving model...")
            best_acc = epoch_acc
            torch.save(model.state_dict(), WEIGHTS_FILE)
        else:
            print(f" ... Accuracy did not improve (Best: {best_acc:.4f})")

    print("Training complete.")

    # Generate plots
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    axes[0].plot(batch_nums, batch_losses, color='tab:red')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Batch')
    axes[0].set_ylabel('Loss')

    epoch_end_positions = [(i+1)*batches_per_epoch - 1 for i in range(EPOCHS)]
    axes[1].plot(batch_nums, batch_accs, color='tab:blue', label='Batch Accuracy')
    axes[1].plot(epoch_end_positions, epoch_accs, 'o-', color='green', label='Epoch Average Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].set_xlabel('Batch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    cm = confusion_matrix(last_epoch_labels, last_epoch_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=axes[2])
    axes[2].set_title('Confusion Matrix (Last Epoch)')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')

    plt.tight_layout()
    plt.savefig('training_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()