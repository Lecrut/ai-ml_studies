import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

# keep import simple but ensure local module is importable
sys.path.insert(0, os.path.dirname(__file__))
from model import myAgent


class NPZImageDataset(Dataset):
    def __init__(self, files, height, width, channels, n_classes):
        self.files = files
        self.h = height
        self.w = width
        self.channels = channels
        self.n_classes = n_classes

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        data = np.load(p)
        states = data['states']
        img = states[0] if states.ndim == 4 else states
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        pil = Image.fromarray(img).convert('L').resize((self.w, self.h), Image.BILINEAR)
        arr = np.array(pil).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        if self.channels != 1:
            tensor = tensor.repeat(self.channels, 1, 1)
        label = int(np.array(data['actions']).squeeze()) if 'actions' in data else 0
        return tensor, label


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_data()
        self.setup_model()

    def setup_data(self):
        dataset_dir = self.config.dataset_dir
        assert os.path.isdir(dataset_dir), f"dataset_dir not found: {dataset_dir}"
        files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith('.npz')]
        files.sort()
        split = self.config.train_split
        n_train = int(len(files) * split)
        train_files = files[:n_train]
        val_files = files[n_train:]
        train_ds = NPZImageDataset(train_files, self.config.height, self.config.width, self.config.channels, self.config.n_classes)
        val_ds = NPZImageDataset(val_files, self.config.height, self.config.width, self.config.channels, self.config.n_classes)
        self.train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)

    def setup_model(self):
        self.model = myAgent(in_channels=self.config.channels, img_h=self.config.height, img_w=self.config.width, n_classes=self.config.n_classes, lr=self.config.lr)
        self.model.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.net.parameters(), lr=self.config.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def validate(self):
        self.model.net.eval()
        total_loss = 0.0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model.net(images)
                loss = self.loss_fn(outputs, labels)
                b = images.size(0)
                total_loss += loss.item() * b
                total += b
        return total_loss / max(1, total)

    def train(self):
        epochs = self.config.epochs
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        best_val = float('inf')
        best_path = None
        for epoch in range(1, epochs + 1):
            # call model's train_step directly (assume it handles optimizer)
            self.model.train_step(self.train_loader)
            val_loss = self.validate()
            print(f"Epoch {epoch}/{epochs} - val_loss: {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(self.config.checkpoint_dir, f'best_model_epoch{epoch}.pth')
                torch.save(self.model.net.state_dict(), best_path)
        return best_path


def main():
    class Config:
        dataset_dir = 'project/dataset'
        train_split = 0.8
        n_classes = 5
        channels = 1
        height = 64
        width = 64
        batch_size = 32
        epochs = 100
        lr = 1e-3
        checkpoint_dir = 'checkpoints'

    config = Config()
    trainer = Trainer(config)
    best_model = trainer.train()
    print(f"Best model saved at: {best_model}")


if __name__ == '__main__':
    main()