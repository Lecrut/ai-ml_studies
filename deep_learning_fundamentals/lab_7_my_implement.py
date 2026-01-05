#%% Imports
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import string
from collections import Counter
from sklearn.model_selection import train_test_split

#%% Text Processing - Tokenization and Vocabulary
class TextProcessor:
    def __init__(self, max_vocab_size=7000, max_len=20):
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

#%% Model Definition - Image-Text Consistency Model - MyClipModel
class MyClipModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=64, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1]) 
        self.vision_dim = 2048 # ResNet50 output size
        
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
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

#%% Training and Prediction Functions
def fit(model, df_train, df_val, epochs=20, batch_size=32):
    train_ds = ImageTextDataset(df_train, processor)
    val_ds = ImageTextDataset(df_val, processor)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    trainer = pl.Trainer(max_epochs=epochs, accelerator="auto", enable_progress_bar=True)
    
    print("Rozpoczynam trening...")
    trainer.fit(model, train_loader, val_loader)
    print("Trening zakończony.")
    return trainer

def predict(model, df_test):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    dataset = ImageTextDataset(df_test, processor)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
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
    df = pd.read_csv('captions_flickr8k_with_false.csv') 
    
    processor = TextProcessor()
    processor.build_vocab(df['caption'].tolist())
    
    train_df, temp_df = train_test_split(df, test_size=0.2)
    val_df, test_df = train_test_split(temp_df, test_size=0.5)

    model = MyClipModel(vocab_size=len(processor))

    fit(model, train_df, val_df, epochs=3)

    wyniki = predict(model, test_df)
    
    print(f"Przykładowe wyniki: {wyniki[:10]}")
    test_df['prediction'] = wyniki