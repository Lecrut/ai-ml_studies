import os
import torch
import torch.nn as nn
from torchvision import transforms, models

SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

class SubmissionModel(nn.Module):  
    def __init__(self):
        super().__init__()
        
        self.vocab_size = 256  
        self.max_len = 100     
        self.emb_dim = 512     
        
        resnet = models.resnet50(weights=None)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.img_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.char_embedding = nn.Embedding(self.vocab_size, 64, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.text_proj = nn.Sequential(
            nn.Linear(512, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.emb_dim * 2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _tokenize(self, text):
        text = str(text).lower()
        ids = [min(ord(c), 255) for c in text[:self.max_len]]
        
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
            
        return torch.tensor(ids, dtype=torch.long)
    
    def forward(self, images, texts):
        device = images.device
        
        text_ids_list = [self._tokenize(t) for t in texts]
        text_ids = torch.stack(text_ids_list).to(device)

        img = self.image_encoder(images)
        img = self.avgpool(img)
        img_vec = self.img_proj(img)

        emb = self.char_embedding(text_ids)
        lstm_out, _ = self.lstm(emb)
        text_feat, _ = torch.max(lstm_out, dim=1) 
        txt_vec = self.text_proj(text_feat)

        fused = torch.cat([
            img_vec * txt_vec,
            torch.abs(img_vec - txt_vec)
        ], dim=1)
        
        return self.classifier(fused).squeeze(1)
    
    def predict(self, image_tensor, text_string):
        self.eval()
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0)
            
            score_tensor = self.forward(image_batch, [text_string])
            
            return score_tensor.item()
