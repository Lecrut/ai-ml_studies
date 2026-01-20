import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm

from model import SubmissionModel


# =====================
# CONFIG
# =====================
DATA_DIR = r"c:\Users\Filip\Documents\mgr-siium\deep_learning_fundamentals\ex_1"
CSV_FILE = os.path.join(DATA_DIR, "captions_flickr8k_with_false.csv")
IMG_DIR = os.path.join(DATA_DIR, "datasets")

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 4


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
# TRAIN
# =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(CSV_FILE)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = FlickrDataset(df, transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = SubmissionModel().to(device)

    # ===================================================
    # LOAD IMAGENET → ZOSTANIE ZAPISANE W weights.pth
    # ===================================================
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

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, captions, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images, list(captions))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "weights.pth")
    print("Saved weights.pth")


if __name__ == "__main__":
    main()
