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
# %%


'''
Great news, everyone! 
Our waste classification system has been a smashing success. We are now able to accurately sort waste into multiple categories,
with stunning accuracy, thanks to our state-of-the-art deep learning models and innovative data handling techniques.


Oh no, the system for "classifying waste" was just another ruse. That's not good news at all!!!


The actual goal was to put listening devices in the bins to eavesdrop on people.
By following the "green initiative", the evil corporation is trying to gather
data on people's conversations about sensitive topics. Because who would suspect garbage bins
of being a front for surveillance? And we know people often discuss private matters while taking out the trash. 
For some reason, the state thinks, that they feel safe there and are open to disccussions. 

Moreover, our system was deployed literally everywhere, in parks, shopping malls, schools, and even government buildings.
The data collected is then analyzed to identify potential threats, monitor public sentiment, and even influence public opinion through targeted propaganda.

It seems though the software is not ready yet and we are, once again, tasked in making it work. 
If only we were not that great at AI ....


But alas, there is nothing we can do about it now. We have to make the best system possible.

But how, we have no idea about processing audio data?
Fortunately, we have experience with classification, and the task has been simplified for us - we are given data in the form of text files - transcripts of the audio recordings.
Unfortunately for us we still don't know what to do with them, as we have no experience with NLP either.

We must act quickly and learn how to process text data. 
We know, that we cannot use those fancy "transformers" yet, as they are too complex for the hardware (for now).
Let us then start with something simpler and learn how other "recurrent" models work.

'''


# %%

'''
The Concept of Recurrent Networks

Traditional neural networks (like MLPs or basic CNNs) treat all inputs as independent. 
This is insufficient for sequential data like text, audio, or time-series, where the order of information is crucial 
and one input depends on the previous ones.
A Recurrent Neural Network (RNN) is designed to handle this. It has a 
"memory"—called the hidden state (ht​)—which captures information about the sequence processed up to the current time step (t).
    Why: To model dependencies in sequential data.
    How it Works: At each step t, the network takes the current input xt​ and the previous hidden state ht−1​ to produce the output yt​ and the new hidden state ht​. 
    This process repeats for the entire sequence.
'''
#%%



# first let's look at the Embeding layer again - and try to understand why would we even need one
'''
Preprocessing: The Embedding Layer
Before passing sequence data (like words) into an RNN, it must be converted into a numerical vector format. The torch.nn.Embedding layer does exactly this.
It maps discrete inputs (like word indices) into dense, continuous vectors. These vectors are learned during training and help the network understand the semantic relationships between words.
    Parameters:
        num_embeddings: The size of the dictionary (e.g., the number of unique words).
        embedding_dim: The size of the vector space in which the words will be embedded (the output feature size).

'''


import torch
import torch.nn as nn

# Define parameters
VOCAB_SIZE = 1000  # Number of unique words in our vocabulary
EMBEDDING_DIM = 50 # Size of the vector for each word
BATCH_SIZE = 4
SEQUENCE_LENGTH = 10 # Number of words (time steps) in a sentence

# 1. Create the Embedding Layer
# Input: (VOCAB_SIZE, EMBEDDING_DIM) matrix of weights
embedding_layer = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

# 2. Prepare mock input data
# A batch of 4 sentences, each 10 words long. 
# The values are word indices (0 to VOCAB_SIZE - 1)
input_data_indices = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH))

# 3. Pass indices through the Embedding Layer
embedded_data = embedding_layer(input_data_indices)

print(f"Original Input Shape (word indices): {input_data_indices.shape}")
print(f"Embedded Data Shape (vectorized words): {embedded_data.shape}")
# Expected: (BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_DIM) -> (4, 10, 50)



#%%
''' 
Creating the Recurrent Models

PyTorch provides three core recurrent modules:
- `nn.RNN`: simplest recurrent unit (tanh or ReLU nonlinearity).
- `nn.GRU`: Gated Recurrent Unit; compact and strong baseline.
- `nn.LSTM`: Long Short-Term Memory; includes a cell state to help long-range dependencies.

Key Parameters for All Recurrent Layers:
    input_size: The number of expected features in the input x (e.g., the EMBEDDING_DIM).
    hidden_size: The number of features in the hidden state h. This is the "memory size" and determines the depth of processing.
    num_layers: Number of stacked recurrent layers. A value >1 creates a deep (multi-layer) RNN.
    batch_first: If True, the input/output tensors are provided as (batch, seq, feature). This is the standard convention for most of PyTorch. Always set this to True for text/sequence processing.
    bidirectional: if True, forward + backward; doubles output feature size.
    dropout: applied between layers when `num_layers > 1`.

'''

#%%
# Shared Model Parameters
HIDDEN_SIZE = 64
NUM_LAYERS = 2 

# A. Simple RNN (nn.RNN)
rnn_layer = nn.RNN(
    input_size=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    batch_first=True
)

# Run the data through the simple RNN
rnn_output, rnn_hidden_state = rnn_layer(embedded_data)

print("\nSimple RNN (nn.RNN)")
print(f"Output Shape (all steps): {rnn_output.shape}")
print(f"Hidden State Shape (last step): {rnn_hidden_state.shape}")

#%%

# B. Gated Recurrent Unit (nn.GRU)
# GRU is an improvement over RNN, addressing the vanishing gradient problem.
gru_layer = nn.GRU(
    input_size=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    batch_first=True
)

# Run the data through the GRU
gru_output, gru_hidden_state = gru_layer(embedded_data)

print("\nGated Recurrent Unit (nn.GRU)")
print(f"Output Shape (all steps): {gru_output.shape}")
print(f"Hidden State Shape (last step): {gru_hidden_state.shape}")


#%%
# C. Long Short-Term Memory (nn.LSTM)
# LSTM is the most common, using three "gates" (input, forget, output) 
# and a cell state (C) to manage long-term dependencies.
lstm_layer = nn.LSTM(
    input_size=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    batch_first=True
)

# Run the data through the LSTM
# Note: LSTM returns a tuple for the hidden state (h_n, c_n)
lstm_output, (lstm_hidden_state, lstm_cell_state) = lstm_layer(embedded_data)

print("\nLong Short-Term Memory (nn.LSTM)")
print(f"Output Shape (all steps): {lstm_output.shape}")
print(f"Hidden State Shape (h_n): {lstm_hidden_state.shape}")
print(f"Cell State Shape (c_n): {lstm_cell_state.shape}")



#%%

'''
Understanding Outputs and the Hidden State

The output of an RNN/GRU/LSTM layer typically consists of two parts:
1. output (The Sequence Output)
    Shape: (BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE)
    This is the output yt​ for every time step/word in the sequence. 
    If you were doing tasks like sequence tagging (e.g., Named Entity Recognition), you would use this entire tensor.

2. hidden_state (h_n) (The Last Step/Context)
    Shape: (NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)
    This is the final hidden state after processing the entire sequence. It is often referred to as the context vector for the sequence.

'''

#%%
# The goal for sequence classification (e.g., sentiment analysis) 
# is to get the representation of the *entire* sentence.

# For RNN and GRU:
# The shape is (NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE).
# We want the hidden state of the TOP (last) layer: index NUM_LAYERS - 1
final_gru_context = gru_hidden_state[-1, :, :] 

print("\nFinal Context Vector (GRU)")
print(f"Shape of gru_hidden_state: {gru_hidden_state.shape}")
print(f"Shape of final context vector for classification: {final_gru_context.shape}")
# Expected: (BATCH_SIZE, HIDDEN_SIZE) -> (4, 64)

# For LSTM:
# The hidden state is the first element of the tuple: lstm_hidden_state (h_n)
final_lstm_context = lstm_hidden_state[-1, :, :] 

print("\nFinal Context Vector (LSTM)")
print(f"Shape of lstm_hidden_state (h_n): {lstm_hidden_state.shape}")
print(f"Shape of final context vector for classification: {final_lstm_context.shape}")
# Expected: (BATCH_SIZE, HIDDEN_SIZE) -> (4, 64)

# This (BATCH_SIZE, HIDDEN_SIZE) tensor is what you would pass to a 
# linear layer for the final classification decision.

# Example of a final classification layer (for a binary problem):
linear_classifier = nn.Linear(HIDDEN_SIZE, 1) # Output 1 logit

classification_logits = linear_classifier(final_gru_context)

print("\nFinal Classification Output")
print(f"Shape of classification logits: {classification_logits.shape}")
# Expected: (BATCH_SIZE, 1) -> (4, 1)


# %%


# %% 

'''
Ok, this concludes our quick start guide on recurrent networks in PyTorch.
Now we must learn (quickly) how to apply this knowledge to our problem. 
First we must learn how to even classify using the recurrent networks

Fortunately, we have some sample datasets that we can practice on.
Let's start with something simple - classifying movie reviews into positive and negative ones.

We can get the "IMDB Movie Reviews" dataset easily from Kaggle:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

It's just a simple CSV file with two columns - "review" and "sentiment".
we have only two classes - positive and negative.


'''

#####


# our first task is to load the dataset and prepare it for training.
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

# Configuration
# DATA_PATH = "./datasets/IMDB/IMDB Dataset.csv"
DATA_PATH = "illegal_surveilance_tagged.csv"
SEQUENCE_LENGTH = 100 # Max length of a review (truncate or pad)
BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_SIZE = 128
NUM_LAYERS = 1
LEARNING_RATE = 1e-3


# Ensure the data file exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Please place 'IMDB Dataset.csv' at the path: {DATA_PATH}")

# Load the dataset
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded with {len(df)} entries.")

# Map sentiment labels to numerical values
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})

# --- Simple Tokenization and Vocabulary Building ---
# 1. Tokenize all reviews
all_tokens = [word.lower() for review in df['sentence'] for word in review.split()]
token_counts = Counter(all_tokens)

# 2. Build vocabulary: Map words to unique indices
# Index 0 is reserved for padding, index 1 for unknown words ('<unk>')
vocab = {word: hash(f"{word}{i}") for i, (word, count) in enumerate(token_counts.most_common())}
vocab['<pad>'] = 0
vocab['<unk>'] = 1
VOCAB_SIZE = len(vocab)
print(f"Vocabulary built with size: {VOCAB_SIZE}")

def text_to_sequence(text, vocab, max_len):
    """Converts a review text to a padded sequence of indices."""
    sequence = [vocab.get(word.lower(), vocab['<unk>']) for word in text.split()]
    
    # Pad or truncate the sequence
    if len(sequence) < max_len:
        # Pad with 0 (index of '<pad>')
        sequence.extend([vocab['<pad>']] * (max_len - len(sequence)))
    elif len(sequence) > max_len:
        # Truncate
        sequence = sequence[:max_len]
        
    return sequence

# Apply the sequence conversion
df['sequence'] = df['sentence'].apply(lambda x: text_to_sequence(x, vocab, SEQUENCE_LENGTH))

# Split the data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

print(len(vocab))

# That is a looooong vocabulary, but it should still work``


#%%

# OR, we could already optimize the vocabulary size a bit
# by limiting it to tN words only
# for that we have do to some preprocessing first

# Configuration
# DATA_PATH = "./datasets/IMDB/IMDB Dataset.csv"
DATA_PATH = "illegal_surveilance_tagged.csv"
SEQUENCE_LENGTH = 100
BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_SIZE = 128
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
MAX_VOCAB_SIZE = 50000 # <-- Set a reasonable limit for the vocabulary 
                       # ("otherwise we will use all the possible words in the sentences")


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
