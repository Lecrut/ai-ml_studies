#  Image–Text Consistency Challenge
#%% Imports 
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms

#%% Constants 
BATCH_SIZE = 64

#%% Check cuda 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


#%% Make dataloader
df = pd.read_csv('captions_flickr8k_with_false.csv')

unique_images = df['image_path'].unique()
train_images, temp_images = train_test_split(unique_images, test_size=0.2, random_state=42)
val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

train_df = df[df['image_path'].isin(train_images)]
val_df = df[df['image_path'].isin(val_images)]
test_df = df[df['image_path'].isin(test_images)]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class ClipDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        caption = row['caption']
        label = row['label']
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        return image, caption, label

train_dataset = ClipDataset(train_df)
val_dataset = ClipDataset(val_df)
test_dataset = ClipDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


#%% Model Definition
class CrossModalModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, output_dim=256):
        super(CrossModalModel, self).__init__()
      
        resnet = models.resnet50(weights='IMAGENET1K_V1')

        modules = list(resnet.children())[:-1]
        self.image_encoder = nn.Sequential(*modules)

        self.image_projection = nn.Linear(resnet.fc.in_features, output_dim)
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.text_projection = nn.Linear(hidden_dim * 2, output_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(output_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, images, captions):
        img_features = self.image_encoder(images)
        img_features = torch.flatten(img_features, 1)
        img_emb = self.image_projection(img_features)
        
        text_emb_raw = self.embedding(captions)
        _, (hidden, _) = self.lstm(text_emb_raw)
        text_features = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        text_emb = self.text_projection(text_features)
     
        diff = torch.abs(img_emb - text_emb)
        prod = img_emb * text_emb
        
        combined = torch.cat([img_emb, text_emb, diff, prod], dim=1)
        
        output = self.classifier(combined)
        return output
    
#%% Example Usage
