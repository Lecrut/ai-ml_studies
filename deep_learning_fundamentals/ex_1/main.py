#%% Imports
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from PIL import Image
import pandas as pd
import string
from collections import Counter
from sklearn.model_selection import train_test_split
import os

#%% Constants
pl.seed_everything(42)
CPU_WORKERS = os.cpu_count() if os.cpu_count() is not None else 1
BATCH_SIZE = 128

#%% Text Processing
class TextProcessor:
    def __init__(self, max_vocab_size=10000, max_len=50):
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.vocab = {'<pad>': 0, '<unk>': 1}
        
    def clean_and_tokenize(self, text):
        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def build_vocab(self, sentences):
        print(f"Budowanie słownika (max_vocab={self.max_vocab_size}, max_len={self.max_len})...")
        all_tokens = [token for s in sentences for token in self.clean_and_tokenize(s)]
        token_counts = Counter(all_tokens)
        
        top_words = token_counts.most_common(self.max_vocab_size - 2)
        
        for i, (word, count) in enumerate(top_words):
            self.vocab[word] = i + 2
        print(f"Słownik gotowy. Rozmiar: {len(self.vocab)}")
            
    def text_to_sequence(self, text):
        sequence = [self.vocab.get(word, self.vocab['<unk>']) for word in self.clean_and_tokenize(text)]
        
        if len(sequence) < self.max_len:
            sequence.extend([self.vocab['<pad>']] * (self.max_len - len(sequence)))
        elif len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]
        return sequence

    def __len__(self):
        return len(self.vocab)

#%% Dataset Class
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
        
        image = Image.open(row['image_path']).convert('RGB')
        image = self.transform(image)
   
        sequence = torch.tensor(self.processor.text_to_sequence(row['caption']), dtype=torch.long)
        label = torch.tensor(row['label'], dtype=torch.float32) if 'label' in row else torch.tensor(0.0)
        
        return image, sequence, label

#%% Model Definition
class MyClipModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, learning_rate=5e-4):
        super().__init__()
        self.save_hyperparameters()
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        for param in self.vision_encoder[-1].parameters():
            param.requires_grad = True
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
    
        self.image_projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.text_projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4), 
            nn.Linear(256, 1)
        )
        
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, images, captions):
        x_img = self.vision_encoder(images)      
        x_img = self.avgpool(x_img).flatten(1)   
        x_img = self.image_projector(x_img)      
        
        embedded = self.embedding(captions)      
        lstm_out, _ = self.lstm(embedded)        
       
        x_txt = torch.mean(lstm_out, dim=1)      
        x_txt = self.text_projector(x_txt)       
        
        combined = torch.cat((x_img, x_txt), dim=1) 
        
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
        
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).float()
        acc = (predictions == labels.unsqueeze(1)).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = [
            {'params': self.vision_encoder.parameters(), 'lr': 1e-5}, 
            {'params': self.lstm.parameters(), 'lr': 1e-3},
            {'params': self.image_projector.parameters(), 'lr': 1e-3},
            {'params': self.text_projector.parameters(), 'lr': 1e-3},
            {'params': self.classifier.parameters(), 'lr': 1e-3}
        ]
        return torch.optim.Adam(params) 

#%% Helper: Model Summary
def print_model_summary(model, processor):
    print("\n" + "="*40)
    print("PODSUMOWANIE MODELU")
    print("="*40)
    try:
        dummy_image = torch.randn(1, 3, 224, 224)
        dummy_caption = torch.randint(0, len(processor), (1, processor.max_len))
        summary(model, input_data=[dummy_image, dummy_caption], 
                col_names=["input_size", "output_size", "num_params", "trainable"], depth=3)
    except Exception as e:
        print(f"Nie udało się wygenerować podsumowania (czy zainstalowałeś torchinfo?): {e}")
    print("="*40 + "\n")

#%% Training Function with Callbacks
def fit(model, df_train, df_val, epochs=20):
    OPTIMAL_WORKERS = CPU_WORKERS // 2 if CPU_WORKERS > 2 else 1
    
    train_ds = ImageTextDataset(df_train, processor)
    val_ds = ImageTextDataset(df_val, processor)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=OPTIMAL_WORKERS, 
        persistent_workers=True, 
        pin_memory=True           
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=OPTIMAL_WORKERS, 
        persistent_workers=True,
        pin_memory=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/lightning_multiclass',
        filename='best-{epoch:02d}-{val_acc:.4f}',
        save_top_k=1,
        monitor='val_acc',
        mode='max',
        save_last=True
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_acc',
        patience=5,
        mode='max',
        verbose=True
    )
    
    trainer = pl.Trainer(
        max_epochs=epochs, 
        accelerator="gpu",
        devices=1,
        precision="16-mixed",  
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=10  
    )
    
    print(f"Rozpoczynam trening (Precision: 16-mixed, Workers: {OPTIMAL_WORKERS})...")
    trainer.fit(model, train_loader, val_loader)
    print("Trening zakończony.")
    
    return trainer, checkpoint_callback.best_model_path

#%% Prediction Function
def predict(model, df_test):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    dataset = ImageTextDataset(df_test, processor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=CPU_WORKERS)
    
    predictions = []
    
    with torch.no_grad():
        for images, captions, _ in loader:
            images = images.to(device)
            captions = captions.to(device)
            
            logits = model(images, captions)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().cpu().numpy().flatten()
            predictions.extend(preds)
            
    return predictions

#%% Main Execution
if __name__ == "__main__":
    if os.path.exists('captions_flickr8k_with_false.csv'):
        df = pd.read_csv('captions_flickr8k_with_false.csv')
    else:
        raise FileNotFoundError("Nie znaleziono pliku captions_flickr8k_with_false.csv")
    
    processor = TextProcessor(max_vocab_size=9000, max_len=40)
    processor.build_vocab(df['caption'].tolist())
    
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    model = MyClipModel(vocab_size=len(processor))

    print_model_summary(model, processor)

    trainer, best_model_path = fit(model, train_df, val_df, epochs=20)

    print(f"\n--- Weryfikacja najlepszego modelu ---")
    if best_model_path and os.path.exists(best_model_path):
        print(f"Wczytywanie wag z najlepszego checkpointu: {best_model_path}")
        best_model = MyClipModel.load_from_checkpoint(best_model_path)
    else:
        print("Nie znaleziono checkpointu (być może trening był za krótki). Używam modelu z ostatniej epoki.")
        best_model = model

    print("\nGenerowanie predykcji na zbiorze testowym...")
    wyniki = predict(best_model, test_df)
    
    test_df['prediction'] = wyniki
    print(f"Przykładowe wyniki: {wyniki[:10]}")
    
    test_df.to_csv('test_predictions.csv', index=False)
    print("Wyniki zapisano do test_predictions.csv")

    acc = (test_df['label'] == test_df['prediction']).mean()

    print(f"Accuracy: {acc}")