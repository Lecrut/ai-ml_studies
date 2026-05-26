from __future__ import annotations
from pathlib import Path
from typing import Sequence
import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


ACTION_NAMES = ["forward", "backward", "left", "right", "stop"]
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_DIR / "records" / "dataset"
DEFAULT_CHECKPOINT_PATH = PROJECT_DIR / "records" / "imitation_model.pth"


class FrameImitationDataset(Dataset):
    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.samples = self._load_samples()

    def _load_samples(self):
        files = sorted(self.dataset_dir.glob("batch_*.npz"))
        if not files:
            raise FileNotFoundError(
                f"No dataset batches found in {self.dataset_dir}. "
                "Run project/game_expert.py first to collect frames."
            )

        states = []
        actions = []
        for file_path in files:
            batch = np.load(file_path)
            batch_states = batch["states"].astype(np.float32) / 255.0
            batch_actions = batch["actions"].astype(np.int64)
            states.append(batch_states)
            actions.append(batch_actions)

        stacked_states = np.concatenate(states, axis=0)
        stacked_actions = np.concatenate(actions, axis=0)
        return stacked_states, stacked_actions

    def __len__(self):
        return len(self.samples[0])

    def __getitem__(self, index):
        states, actions = self.samples
        state = torch.from_numpy(states[index]).float()
        action = torch.tensor(actions[index], dtype=torch.long)
        return state, action


class ImitationCNN(nn.Module):
    def __init__(self, num_actions: int = len(ACTION_NAMES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def _feature_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 84, 84)
            features = self.features(dummy)
            return int(np.prod(features.shape[1:]))

    def forward(self, x):
        return self.classifier(self.features(x))


class ImitationLightningModule(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3, num_actions: int = len(ACTION_NAMES)):
        super().__init__()
        self.save_hyperparameters()
        self.model = ImitationCNN(num_actions=num_actions)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        states, actions = batch
        logits = self(states)
        loss = self.criterion(logits, actions)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == actions).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        states, actions = batch
        logits = self(states)
        loss = self.criterion(logits, actions)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == actions).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class FrameDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir: str | Path, batch_size: int = 64, val_split: float = 0.15, seed: int = 42):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None):
        dataset = FrameImitationDataset(self.dataset_dir)
        dataset_size = len(dataset)
        val_size = max(1, int(dataset_size * self.val_split)) if dataset_size > 1 else 0
        train_size = dataset_size - val_size

        if train_size <= 0:
            raise ValueError("Dataset is too small to create a training split.")

        if val_size > 0:
            self.train_dataset, self.val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )
        else:
            self.train_dataset = dataset
            self.val_dataset = None

    def train_dataloader(self):
        return build_dataloader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return build_dataloader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def train(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    val_split: float = 0.15,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_module = FrameDataModule(dataset_dir=dataset_dir, batch_size=batch_size, val_split=val_split, seed=seed)
    model = ImitationLightningModule(learning_rate=learning_rate)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    data_module.setup()

    if data_module.val_dataset is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_path.parent,
            filename=checkpoint_path.stem,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_path.parent,
            filename=checkpoint_path.stem,
            save_top_k=1,
            save_last=True,
        )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        default_root_dir=str(checkpoint_path.parent),
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=data_module)

    best_model_path = Path(checkpoint_callback.best_model_path) if checkpoint_callback.best_model_path else checkpoint_path
    if best_model_path.exists() and best_model_path != checkpoint_path:
        checkpoint_path.write_bytes(best_model_path.read_bytes())

    return model, checkpoint_path


def load_model(checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH, device: str | torch.device | None = None):
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint["model_state"]

    num_actions = len(checkpoint.get("action_names", ACTION_NAMES))
    model = ImitationCNN(num_actions=num_actions).to(device)
    cleaned_state_dict = {key.replace("model.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    return model, checkpoint


def predict_action(model: nn.Module, frames: np.ndarray | Sequence[np.ndarray], device: str | torch.device | None = None) -> int:
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    frame_array = np.asarray(frames, dtype=np.float32)
    if frame_array.max() > 1.0:
        frame_array = frame_array / 255.0

    input_tensor = torch.from_numpy(frame_array).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
    return int(logits.argmax(dim=1).item())


def main():
    train()


if __name__ == "__main__":
    main()
