import torch
import torch.nn as nn
from torchvision import transforms, models
import string
import os
import json

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class SubmissionModel(nn.Module):
    def __init__(self, vocab_size=None):
        super().__init__()
        
        resnet = models.resnet50(weights=None)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-2])

        self.vocab = None
        self._load_vocab()
        
        if self.vocab is None:
            self.vocab = {'<pad>': 0, '<unk>': 1}
        
        if vocab_size is None:
            self.vocab_size = len(self.vocab)
        else:
            self.vocab_size = vocab_size
            
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        for param in self.vision_encoder[-1].parameters():
            param.requires_grad = True
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 

        self.embedding_dim = 128
        self.hidden_dim = 256
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        
        self.image_projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.text_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4), 
            nn.Linear(256, 1),
            nn.Sigmoid()  
        )

        self.max_len = 40

    def _load_vocab(self):
        SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))
        vocab_path = os.path.join(SUBMISSION_DIR, 'vocab.json')
        if os.path.exists(vocab_path):
            with open(vocab_path) as f:
                self.vocab = json.load(f)
        
    def _clean_and_tokenize(self, text):
        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()
            
    def _text_to_sequence(self, text):
        sequence = [self.vocab.get(word, self.vocab['<unk>']) for word in self._clean_and_tokenize(text)]
        
        if len(sequence) < self.max_len:
            sequence.extend([self.vocab['<pad>']] * (self.max_len - len(sequence)))
        elif len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]
        return sequence
    
    def forward(self, images, captions):
        x_img = self.vision_encoder(images)      
        x_img = self.avgpool(x_img).flatten(1)   
        x_img = self.image_projector(x_img)      
        
        embedded = self.embedding(captions)      
        lstm_out, _ = self.lstm(embedded)        
       
        x_txt = torch.mean(lstm_out, dim=1)      
        x_txt = self.text_projector(x_txt)       
        
        combined = torch.cat((x_img, x_txt), dim=1) 
        
        scores = self.classifier(combined).squeeze(-1)
        return scores
    
    def predict(self, image_tensor, text_string):
        self.eval()
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0)
            sequence = torch.tensor(self._text_to_sequence(text_string), dtype=torch.long, device=image_tensor.device).unsqueeze(0)
            score = self.forward(image_batch, sequence)
            return score.item()
        