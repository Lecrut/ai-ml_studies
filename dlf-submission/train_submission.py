import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from collections import Counter
from tqdm import tqdm
import json

# Zakładam, że Twój model jest w pliku model.py
from model import SubmissionModel 

# === KONFIGURACJA ===
DATA_DIR = r"c:\Users\Filip\Documents\mgr-siium\deep_learning_fundamentals\ex_1"
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions_flickr8k_with_false.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "datasets")
BATCH_SIZE = 32
EPOCHS = 1
LR = 1e-4

# === 1. KLASA DO TWORZENIA SŁOWNIKA ===
class TextProcessor:
    def __init__(self, min_freq=2):
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.reverse_vocab = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        self.min_freq = min_freq
        
    def build_vocab(self, sentences):
        print("Budowanie słownika...")
        counter = Counter()
        for sentence in sentences:
            words = str(sentence).lower().split()
            counter.update(words)
            
        idx = 4
        for word, count in counter.items():
            if count >= self.min_freq:
                self.vocab[word] = idx
                self.reverse_vocab[idx] = word
                idx += 1
        print(f"Rozmiar słownika: {len(self.vocab)}")

    def text_to_sequence(self, text, max_len=40):
        words = str(text).lower().split()
        seq = [self.vocab["<START>"]]
        for w in words:
            seq.append(self.vocab.get(w, self.vocab["<UNK>"]))
        seq.append(self.vocab["<END>"])
        
        # Padding
        if len(seq) < max_len:
            seq += [self.vocab["<PAD>"]] * (max_len - len(seq))
        return seq[:max_len]

# === 2. DATASET ===
class FlickrDataset(Dataset):
    def __init__(self, df, processor, transform=None):
        self.df = df
        self.processor = processor
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Obsługa ścieżki
        img_path = row['image_path']
        if img_path.startswith('datasets/'):
            relative_path = img_path[len('datasets/'):]
            full_path = os.path.join(IMAGES_DIR, relative_path)
        else:
            full_path = os.path.join(IMAGES_DIR, os.path.basename(img_path))
            
        image = Image.open(full_path).convert('RGB')
        image = self.transform(image)
        
        # Użycie procesora do zamiany tekstu na liczby
        caption_seq = torch.tensor(self.processor.text_to_sequence(row['caption']), dtype=torch.long)
        label = torch.tensor(float(row['label']), dtype=torch.float32)
        
        return image, caption_seq, label

# === 3. GŁÓWNA PĘTLA ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    # A. Wczytanie danych i budowa słownika
    df = pd.read_csv(CAPTIONS_FILE)
    processor = TextProcessor(min_freq=2)
    processor.build_vocab(df['caption'].tolist())

    # B. Tworzenie datasetu
    dataset = FlickrDataset(df, processor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # C. Inicjalizacja Twojego modelu
    # Przekazujemy rozmiar słownika do Twojego modelu
    model = SubmissionModel(vocab_size=len(processor.vocab))

    # === D. KLUCZOWY MOMENT: Ładowanie wag ImageNet do ResNeta ===
    print("Pobieranie i wstrzykiwanie wag ImageNet do vision_encoder...")
    
    # 1. Pobieramy oficjalnego ResNeta z wagami
    pretrained_resnet = models.resnet50(weights='DEFAULT')
    
    # 2. Wyciągamy z niego warstwy tak samo jak Ty to robisz w klasie (bez 2 ostatnich)
    pretrained_layers = nn.Sequential(*list(pretrained_resnet.children())[:-2])
    
    # 3. Ładujemy te wagi do Twojego pustego vision_encoder
    model.vision_encoder.load_state_dict(pretrained_layers.state_dict())
    print("Wagi załadowane pomyślnie.")
    
    model = model.to(device)
    
    # E. Trening
    criterion = nn.BCELoss() # Zakładam Sigmoid na końcu Twojego modelu
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (imgs, caps, lbls) in enumerate(tqdm(loader, desc=f"Epoka {epoch+1}/{EPOCHS}")):
            imgs, caps, lbls = imgs.to(device), caps.to(device), lbls.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, caps) # caps tutaj wchodzi jako tensor indeksów
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

    # F. Zapis modelu
    print("Zapisywanie wag...")
    torch.save(model.state_dict(), 'weights.pth')
    
    # Zapisz słownik, aby model mógł go wczytać
    with open('vocab.json', 'w') as f:
        json.dump(processor.vocab, f)
        
    print("Gotowe. Pliki weights.pth i vocab.json utworzone.")