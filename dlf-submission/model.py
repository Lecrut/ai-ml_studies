import re
import torch
import torch.nn as nn
from torchvision import transforms, models

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class SubmissionModel(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(weights=None)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-2])

        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.image_encoder[-1].parameters():
            p.requires_grad = True

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.image_projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.vocab_size = 20000
        self.max_len = 20

        self.embedding_dim = 256
        self.hidden_dim = 256

        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.text_fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def _tokenize_text(self, text):
        import re
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = text.split()[:self.max_len]

        ids = [hash(w) % self.vocab_size for w in words]
        if len(ids) < self.max_len:
            ids.extend([0] * (self.max_len - len(ids)))

        return torch.tensor(ids, dtype=torch.long)


    def forward(self, images, texts):
        device = images.device

        img_feat = self.image_encoder(images)
        img_feat = self.avgpool(img_feat).flatten(1)
        img_feat = self.image_projector(img_feat)

        token_batch = torch.stack(
            [self._tokenize_text(t) for t in texts]
        ).to(device)

        emb = self.embedding(token_batch)
        _, (h, _) = self.lstm(emb)
        text_feat = torch.cat([h[-2], h[-1]], dim=1)
        text_feat = self.text_fc(text_feat)

        out = self.classifier(torch.cat([img_feat, text_feat], dim=1))
        return out.squeeze(1)

    def predict(self, image_tensor, text_string):
        self.eval()
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0)

            score = self.forward(image_batch, [text_string])
            return score.item()
