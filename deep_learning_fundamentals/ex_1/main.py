#  Image–Text Consistency Challenge
#%% Imports 
import torch
import torch.nn as nn
import torchvision.models as models

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