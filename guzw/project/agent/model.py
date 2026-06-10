import torch
import torch.nn as nn


class MyCarAgent:
    def __init__(self, in_channels=4, n_classes=5, lr=1e-3):
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 25 * 25, n_classes), 
        )

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

    def train_step(self, dataloader):
        device = next(self.net.parameters()).device
        self.net.train()
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).long()
            self.optimizer.zero_grad()
            outputs = self.net(images)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def load_weights(self, path):
        try:
            self.net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            print(f"Model weights loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading model weights from {path}: {e}")
        