import torch
import torch.nn as nn
from torchvision import transforms, models
import string
from collections import Counter

class TextProcessor:
    def __init__(self, max_vocab_size=9000, max_len=40):
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

processor = TextProcessor()
processor.vocab = {'<pad>': 0, '<unk>': 1, 'a': 2, 'the': 3, 'is': 4, 'image': 5, 'of': 6, 'and': 7, 'with': 8, 'in': 9, 'on': 10}  # Dummy vocab

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class SubmissionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        for param in self.vision_encoder[-1].parameters():
            param.requires_grad = True
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 

        vocab_size = len(processor)
        embedding_dim = 128
        hidden_dim = 256
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
            nn.Linear(256, 1),
            nn.Sigmoid()  
        )
    
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
            sequence = torch.tensor(processor.text_to_sequence(text_string), dtype=torch.long, device=image_tensor.device).unsqueeze(0)
            score = self.forward(image_batch, sequence)
            return score.item()