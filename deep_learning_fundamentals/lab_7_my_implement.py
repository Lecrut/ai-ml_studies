# Lab 7: Sometimes You Gotta Run Before You Can Walk
# %%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import CometLogger
from waste_dataset_multiclass import CombinedWasteDatasetMulti, final_classes
from torchsummary import summary
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import comet_ml

# Set seed for reproducibility
pl.seed_everything(42)

#%% 1. Setup and Data Loading
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy
import string # We need this for punctuation
from torchinfo import summary


#%%
# now - there are two ways we can start - the "naive" way and the semi-optimized way
# let's start with the first one
# OR, we could already optimize the vocabulary size a bit
# by limiting it to tN words only
# for that we have do to some preprocessing first

# Configuration
# DATA_PATH = "./datasets/IMDB/IMDB Dataset.csv"
DATA_PATH = "illegal_surveilance_tagged.csv"
SEQUENCE_LENGTH = 20
BATCH_SIZE = 64
EMBEDDING_DIM = 20
HIDDEN_SIZE = 128
NUM_LAYERS = 1
LEARNING_RATE = 1e-2
MAX_VOCAB_SIZE = 10000 

# Ensure the data file exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Please place 'IMDB Dataset.csv' at the path: {DATA_PATH}")


# Load the dataset
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded with {len(df)} entries.")

# Map sentiment labels to numerical values
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})


# 1. Function to clean and tokenize a review
def clean_and_tokenize(text):
    # Remove punctuation by replacing it with a space, then split by whitespace
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()


#%%
# 2. Tokenize all reviews and count frequencies
all_tokens = [token for review in df['sentence'] for token in clean_and_tokenize(review)]
token_counts = Counter(all_tokens)
print(f"Total unique tokens: {len(token_counts)}")


#%%

# 3. Build vocabulary: Map the MAX_VOCAB_SIZE most common words to indices
# Indices: 0 for padding, 1 for unknown words ('<unk>')
# we will learn what that means precisely later

# We only take the top N words based on MAX_VOCAB_SIZE
top_words = token_counts.most_common(MAX_VOCAB_SIZE - 2) # -2 for <pad> and <unk>

vocab = {word: i + 2 for i, (word, count) in enumerate(top_words)}
vocab['<pad>'] = 0
vocab['<unk>'] = 1
VOCAB_SIZE = len(vocab)
print(f"Vocabulary built with size: {VOCAB_SIZE}")

# Now, VOCAB_SIZE will be much closer to MAX_VOCAB_SIZE (e.g., 50,000)


def text_to_sequence(text, vocab, max_len):
    """Converts a review text to a padded sequence of indices using the clean tokenizer."""
    # Use the same cleaning logic here
    sequence = [vocab.get(word, vocab['<unk>']) for word in clean_and_tokenize(text)]
    
    # Pad or truncate the sequence
    if len(sequence) < max_len:
        sequence.extend([vocab['<pad>']] * (max_len - len(sequence)))
    elif len(sequence) > max_len:
        sequence = sequence[:max_len]
        
    return sequence

# Apply the sequence conversion
df['sequence'] = df['sentence'].apply(lambda x: text_to_sequence(x, vocab, SEQUENCE_LENGTH))

# Split the data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")


#%% 
# 2. Data Module (pl.LightningDataModule)
# no matter which way we chose to build the vocabulary
# we can now create the DataModule for PyTorch Lightning


class IMDBDataset(Dataset):
    """Custom Dataset for IMDB sequences and labels."""
    def __init__(self, data_frame):
        self.sequences = torch.tensor(data_frame['sequence'].tolist(), dtype=torch.long)
        self.labels = torch.tensor(data_frame['sentiment'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class SentimentDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the IMDB dataset."""
    def __init__(self, train_df, val_df, batch_size):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Create datasets
        self.train_dataset = IMDBDataset(self.train_df)
        self.val_dataset = IMDBDataset(self.val_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)

# Instantiate the DataModule
imdb_dm = SentimentDataModule(train_df, val_df, BATCH_SIZE)


#%% 
# 3. Lightning Model (pl.LightningModule)

# We can now build a simple LSTM model for sentiment classification
# what do we need to build one. From our previous knowledge we know that we need:
# 1. feature extraction layer 
# 2. some classifier head
# this task is, however, a sequence classification task
# so we will need to use the LSTM layer
# so our archiecure will be: Embedding (convert word indices to dense vectors) -> LSTM (sequence processing) -> Classifier


class SentimentLSTM(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, learning_rate):
        super().__init__()
        # Saves hyperparameters for logging and checkpointing
        self.save_hyperparameters()
        
        # 1. Embedding Layer: Converts word indices to dense vectors
        # Input: (batch, seq_len) -> Output: (batch, seq_len, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. Recurrent Layer: LSTM for sequence processing
        # `batch_first=True` means inputs/outputs are (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 3. Classifier Layer: Takes the final hidden state to predict sentiment
        # We need a linear layer with output size 1 for binary classification
        self.classifier = nn.Linear(hidden_size, 1)

        # Loss and Metrics
        # TODO: why BCEWithLogitsLoss? could we use something else?
        self.criterion = nn.BCEWithLogitsLoss()
        # and why BinaryAccuracy?
        self.accuracy = BinaryAccuracy()

    def forward(self, x):
        # x shape: (batch_size, SEQUENCE_LENGTH) - word indices
        
        # Embedding
        embedded = self.embedding(x)
        # embedded shape: (batch_size, SEQUENCE_LENGTH, EMBEDDING_DIM)
        
        # LSTM Pass
        # `_` is for the output over all steps (y_t) which we ignore here for classification
        # `(h_n, c_n)` contains the final hidden state (h_n) and cell state (c_n)
        _, (h_n, c_n) = self.lstm(embedded)
        
        # Access the Last Step (so called Context Vector)
        # TODO: check h_n shape 

        # We take the hidden state from the last layer (index -1)
        final_hidden_state = h_n[-1, :, :]
        # TODO: what will be the shape?
        
        # Classification
        logits = self.classifier(final_hidden_state)
        return logits

    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences)
        loss = self.criterion(logits, labels)
        
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(logits, labels.long()), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences)
        loss = self.criterion(logits, labels)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy(logits, labels.long()), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


#%%
# Instantiate the Model
model = SentimentLSTM(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    learning_rate=LEARNING_RATE
)
print("\nLSTM Model instantiated.")
print(model)

#%%
# Print the Summary (this time using torchinfo.summary)

# Determine the device (CPU or GPU) the model is currently on, 
# or force it to CPU for the summary if you don't want to use the GPU.
device = next(model.parameters()).device 
print(f"Model is currently on device: {device}")


# we must create a dummy input tensor with the correct shape
# secondly - this time we MUST specify the dtype as long (integers)
# because the Embedding layer expects integer indices, otherwise it would be created as float by default
INPUT_TENSOR = torch.ones(BATCH_SIZE, SEQUENCE_LENGTH, dtype=torch.long, device=device)


summary(model, input_data=INPUT_TENSOR)

#%% 4. Training the Model

# Define a Trainer
trainer = pl.Trainer(
    max_epochs=20, 
    accelerator="auto", # Use GPU if available
    logger=True, 
    log_every_n_steps=50,
    enable_progress_bar=True
)

#%% 
print(f"\nStarting training for {trainer.max_epochs} epochs...")

# Start the training loop
trainer.fit(model, datamodule=imdb_dm)

print("\nTraining complete!")
# %%

# Questions: 
# were we able to train the model successfully? Was our accuracy good enough?
# is there a difference between the two vocabulary building methods?
# and, more importantly, has the model overfitted? 

# %% Load only texts 
# Now, we can use the trained model to classify new reviews

DATA_PATH_NEW = "illegal_surveilance_test_untagged.csv"
# Load the new dataset
df_new = pd.read_csv(DATA_PATH_NEW)
print(f"New Dataset loaded with {len(df_new)} entries.")

# %%
