import torch
import torch.nn as nn


class myAgent:
    def __init__(self, in_channels=3, n_classes=4, lr=1e-3):
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, n_classes),
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

