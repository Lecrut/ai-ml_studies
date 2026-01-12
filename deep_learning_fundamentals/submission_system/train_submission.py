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

# Constants
BATCH_SIZE = 32
EPOCHS = 5  # Small number for quick training
MAX_SAMPLES = 10000  # Use subset for quick training

# Paths
DATA_DIR = r"c:\Users\Filip\Documents\mgr-siium\deep_learning_fundamentals\ex_1"
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions_coco.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "datasets")

# Text Processor
class TextProcessor:
    def __init__(self, max_vocab_size=5000, max_len=45):  # Smaller vocab for speed
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.vocab = {'<pad>': 0, '<unk>': 1}
        
    def clean_and_tokenize(self, text):
        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def build_vocab(self, sentences):
        all_tokens = [token for s in sentences for token in self.clean_and_tokenize(s)]
        token_counts = Counter(all_tokens)
        top_words = token_counts.most_common(self.max_vocab_size - 2)
        
        for i, (word, count) in enumerate(top_words):
            self.vocab[word] = i + 2
            
    def text_to_sequence(self, text):
        sequence = [self.vocab.get(word, self.vocab['<unk>']) for word in self.clean_and_tokenize(text)]
        if len(sequence) < self.max_len:
            sequence.extend([self.vocab['<pad>']] * (self.max_len - len(sequence)))
        elif len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]
        return sequence

    def __len__(self):
        return len(self.vocab)

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
class MyClipModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=64, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1]) 
        self.vision_dim = 2048 
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.text_dim = hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(self.vision_dim + self.text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, images, captions):
        img_features = self.vision_encoder(images)
        img_features = img_features.view(img_features.size(0), -1) 
                
        embedded = self.embedding(captions)
        _, (h_n, _) = self.lstm(embedded)
        text_features = h_n[-1] 
        
        combined = torch.cat((img_features, text_features), dim=1)
        
        logits = self.classifier(combined)
        return logits

    def training_step(self, batch, batch_idx):
        images, captions, labels = batch
        logits = self(images, captions)
        loss = self.criterion(logits, labels.unsqueeze(1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, captions, labels = batch
        logits = self(images, captions)
        loss = self.criterion(logits, labels.unsqueeze(1))
        predictions = (torch.sigmoid(logits) > 0.5).float()
        acc = (predictions == labels.unsqueeze(1)).float().mean()
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# Main training
if __name__ == "__main__":
    # Load data
    df = pd.read_csv(CAPTIONS_FILE)
    df = df[df['image_path'].str.contains('train2014')]  # Only train2014
    df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=42).reset_index(drop=True)  # Subset
    
    # Split
    train_df = df[:int(0.8*len(df))]
    val_df = df[int(0.8*len(df)):]
    
    # Build vocab
    processor = TextProcessor()
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
    model = MyClipModel(vocab_size=len(processor))
    
    # Trainer
    trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="auto", devices=1)
    trainer.fit(model, train_loader, val_loader)
    
    # Save weights
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "weights.pth"))
    
    # Update model.py with vocab
    vocab_str = str(processor.vocab)
    model_py_path = os.path.join(os.path.dirname(__file__), "model.py")
    with open(model_py_path, 'r') as f:
        content = f.read()
    
    # Replace the dummy vocab init
    old_vocab = "processor.build_vocab([\"dummy sentence for init\"])"
    new_vocab = f"processor.vocab = {vocab_str}"
    content = content.replace(old_vocab, new_vocab)
    
    with open(model_py_path, 'w') as f:
        f.write(content)
    
    print("Training complete. Files saved in submission_system/")