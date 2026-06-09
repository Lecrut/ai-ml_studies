import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger

sys.path.insert(0, os.path.dirname(__file__))
from model import MyCarAgent


class CarAgentLightning(pl.LightningModule):
    def __init__(self, in_channels=4, n_classes=5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.agent = MyCarAgent(in_channels=in_channels, n_classes=n_classes)
        self.net = self.agent.net 
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images) 
        loss = self.loss_fn(outputs, labels)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

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
        
        frames = states[:4] if len(states) >= 4 else states
        processed_frames = []
        
        for img in frames:
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
                
            gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), cv2.COLOR_RGB2GRAY)
            
            resized = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
            
            arr = resized.astype(np.float32) / 255.0
            processed_frames.append(arr)
            
        while len(processed_frames) < self.channels:
            processed_frames.append(processed_frames[-1])
            
        tensor = torch.from_numpy(np.stack(processed_frames, axis=0))
        label = int(np.array(data['actions']).squeeze()) if 'actions' in data else 0
        
        return tensor, label

class CarTrainer:
    def __init__(self, config):
        self.config = config
        self.setup_data()
        self.setup_model()
    
    def setup_data(self):
        dataset_dir = self.config.dataset_dir
        assert os.path.isdir(dataset_dir), f"Nie znaleziono folderu: {dataset_dir}"
        files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith('.npz')]
        files.sort()
        
        split = self.config.train_split
        n_train = int(len(files) * split)
        train_files = files[:n_train]
        val_files = files[n_train:]
        
        self.train_dataset = NPZImageDataset(train_files, self.config.height, self.config.width, self.config.channels, self.config.n_classes)
        self.val_dataset = NPZImageDataset(val_files, self.config.height, self.config.width, self.config.channels, self.config.n_classes)
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=self.config.num_workers, 
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=self.config.num_workers, 
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )

    def setup_model(self):
        self.model = CarAgentLightning(
            in_channels=self.config.channels,
            n_classes=self.config.n_classes,
            lr=self.config.learning_rate
        )

    def train(self):
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=self.config.checkpoint_dir,
            filename='car-agent-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min'
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=20,
            verbose=True,
            mode='min'
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        tensorboard_logger = TensorBoardLogger("logs", name="car_agent")
        comet_logger = CometLogger(
            project_name=self.config.comet_project_name,
        )

        loggers = [tensorboard_logger, comet_logger]

        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=self.config.gpus if torch.cuda.is_available() else "auto",
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
            logger=loggers,
            precision=self.config.precision,
            gradient_clip_val=self.config.gradient_clip_val 
        )

        ckpt_path = getattr(self.config, 'resume_checkpoint', None)
        trainer.fit(self.model, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader, ckpt_path=ckpt_path)
        return checkpoint_callback.best_model_path


def main():
    class Config:
        dataset_dir = 'project/dataset'
        train_split = 0.8
        n_classes = 5
        channels = 4      
        height = 100      
        width = 100       
        batch_size = 32
        learning_rate = 1e-4
        epochs = 250
        num_workers = 4   
        gpus = 1 if torch.cuda.is_available() else 0
        precision = "16-mixed" 
        gradient_clip_val = 1.0
        checkpoint_dir = 'checkpoints/car_agent'
        resume_checkpoint = 'checkpoints\car_agent\car-agent-epoch=248-val_loss=0.68.ckpt'
        comet_project_name = "autonomous-car-agent"

    config = Config()
    trainer = CarTrainer(config)
    best_model_path = trainer.train()
    torch.save(trainer.model.net.state_dict(), 'project/records/best_car_agent_clone.pth')
    print(f"\nNajlepszy model: {best_model_path}")

if __name__ == "__main__":
    main()