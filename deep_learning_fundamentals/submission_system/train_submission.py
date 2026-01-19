import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import string
from collections import Counter
import pytorch_lightning as pl
from model import TextProcessor, SubmissionModel

# Constants
BATCH_SIZE = 32
EPOCHS = 15  # Small number for quick training
MAX_SAMPLES = 10000  # Use subset for quick training

# Paths
DATA_DIR = r"c:\Users\Filip\Documents\mgr-siium\deep_learning_fundamentals\ex_1"
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions_flickr8k_with_false.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "datasets")

# Dataset
class ImageTextDataset(Dataset):
    def __init__(self, dataframe, text_processor, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.processor = text_processor
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Adjust image path
        img_path = row['image_path']
        if img_path.startswith('datasets/'):
            img_path = os.path.join(IMAGES_DIR, img_path[9:])  # Remove 'datasets/'
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        sequence = torch.tensor(self.processor.text_to_sequence(row['caption']), dtype=torch.long)
        
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        return image, sequence, label

# Model
# Training Model using SubmissionModel
class TrainingModel(pl.LightningModule):
    def __init__(self, vocab_size, learning_rate=5e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = SubmissionModel(vocab_size=vocab_size)
        # Remove Sigmoid from classifier for logits
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])  # Remove Sigmoid
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, images, captions):
        return self.model(images, captions)

    def training_step(self, batch, batch_idx):
        images, captions, labels = batch
        logits = self(images, captions)
        loss = self.criterion(logits, labels.float())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, captions, labels = batch
        logits = self(images, captions)
        loss = self.criterion(logits, labels.float())
        
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).float()
        acc = (predictions == labels).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = [
            {'params': self.model.vision_encoder.parameters(), 'lr': 1e-5}, 
            {'params': self.model.lstm.parameters(), 'lr': 1e-3},
            {'params': self.model.image_projector.parameters(), 'lr': 1e-3},
            {'params': self.model.text_projector.parameters(), 'lr': 1e-3},
            {'params': self.model.classifier.parameters(), 'lr': 1e-3}
        ]
        return torch.optim.Adam(params)

# Main training
if __name__ == "__main__":
    # Load data
    df = pd.read_csv(CAPTIONS_FILE)
    df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=42).reset_index(drop=True)  # Subset
    
    # Split
    train_df = df[:int(0.8*len(df))]
    val_df = df[int(0.8*len(df)):]
    
    # Build vocab
    processor = TextProcessor(max_vocab_size=5000, max_len=40)
    processor.build_vocab(train_df['caption'].tolist())
    
    # Datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_ds = ImageTextDataset(train_df, processor, transform)
    val_ds = ImageTextDataset(val_df, processor, transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = TrainingModel(vocab_size=len(processor))
    
    # Trainer
    trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="auto", devices=1)
    trainer.fit(model, train_loader, val_loader)
    
    # Save weights
    torch.save(model.model.state_dict(), os.path.join(os.path.dirname(__file__), "weights.pth"))
    
    # Update model.py with vocab
    vocab_str = str(processor.vocab)
    model_py_path = os.path.join(os.path.dirname(__file__), "model.py")
    with open(model_py_path, 'r') as f:
        content = f.read()
    
    # Replace the dummy vocab
    old_vocab = "processor.vocab = {'<pad>': 0, '<unk>': 1, 'a': 2, 'the': 3, 'is': 4, 'image': 5, 'of': 6, 'and': 7, 'with': 8, 'in': 9, 'on': 10}  # Dummy vocab"
    new_vocab = f"processor.vocab = {vocab_str}"
    content = content.replace(old_vocab, new_vocab)
    
    with open(model_py_path, 'w') as f:
        f.write(content)
    
    print("Training complete. Files saved in submission_system/")