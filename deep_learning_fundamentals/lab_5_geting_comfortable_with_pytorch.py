# Lab 5: Getting Comfortable with PyTorch
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

#%%

# Data augmentations for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

# Validation and test transforms
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



# %%
'''

After we have gotten comfortable with creating simple models in PyTorch and PyTorch Lightning, it's time to explore
more complex models and techniques. 
However, to be able to do so effectively, we need to ensure that we have a solid understanding of the fundamentals of what can we really do using Pytorch and PyTorch Lightning.

Right now all our architecutures have been "traditional" feedforward neural networks. We push data through a series of layers, and at the end we get an output.
In our case it was a single image on the input, and a class prediction on the output.

However, this is not all we can do. 

Before we jump into more complex architectures let's first understand the building blocks we are working with.
Firstly, we will be working with things that are provided for us by Pytorch - all the possible "layers" and "loss functions" that we can use.
These are our lego bricks, that we can stack and combine in any way we want to create complex architectures.
We must first understand what we have at our disposal. After that it is only a case of our imagination, creativity, and experience to create architectures that can solve complex problems.

So let us first start exploring layers provided by Pytorch.

'''

# %%

# the plan - show what all of the most useful layers do in pytorch
# what are the inputs and outputs, what do they do and provide full explanation of each layer
# we will create a simplest possible model using each layer to show how it works
# we already know Linear, Conv2d, MaxPool2d, Flatten, ReLU, Softmax, CrossEntropyLoss
# we will explore all of them (as a remainder), BUT also:
# conv1d
# all the lazy layers: LazyLinear, LazyConv2d
# maxpool and maxunpool
# Dropout and dropout2d - AND full explanation of differences between those two layers AND what happens during training and eval mode
# BatchNorm2d
# AvgPool2d
# AdaptiveAvgPool2d
# AdaptiveMaxPool2d
# ConvTranspose2d
# Embedding
# LayerNorm
# GroupNorm
# InstanceNorm2d
# Sigmoid
# Tanh
# LeakyReLU
# ELU
# Hardswish
# and any other useful layers we can find in torch.nn

# %%
# =====================================================================
# HELPER FUNCTIONS FOR VISUALIZATION
# =====================================================================

def show_images(images, titles=None, cols=4, figsize=(15, 4)):
    """
    Helper function to display a batch of images.
    
    Args:
        images: Tensor of shape (batch, channels, height, width)
        titles: List of titles for each image
        cols: Number of columns in the grid
        figsize: Figure size
    """
    batch_size = images.shape[0]
    rows = (batch_size + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if batch_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else [axes] if batch_size == 1 else axes
    
    for idx in range(batch_size):
        img = images[idx].cpu().detach()
        
        # Handle different image formats
        if img.shape[0] == 3:  # RGB
            # Denormalize if needed (ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0).numpy()
        elif img.shape[0] == 1:  # Grayscale
            img = img.squeeze().numpy()
        else:
            # For feature maps with many channels, show first channel
            img = img[0].numpy()
        
        axes[idx].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[idx].axis('off')
        if titles and idx < len(titles):
            axes[idx].set_title(titles[idx])
    
    # Hide empty subplots
    for idx in range(batch_size, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

def get_sample_images(num_images=4):
    """Get a batch of sample images from the dataset."""
    dataset = CombinedWasteDatasetMulti(
        root_dir="./datasets/combined_waste_dataset",
        split='train',
        transform=train_transform
    )
    dataloader = DataLoader(dataset, batch_size=num_images, shuffle=True)
    images, labels = next(iter(dataloader))
    return images, labels

# %%
# =====================================================================
# 1. LINEAR LAYER (nn.Linear)
# =====================================================================
"""
The Linear layer (also called Fully Connected or Dense layer) performs a linear transformation:
y = xW^T + b

Where:
- x is the input (batch_size, in_features)
- W is the weight matrix (out_features, in_features)
- b is the bias vector (out_features)
- y is the output (batch_size, out_features)

Purpose:
- Used for learning general relationships between inputs and outputs
- Typically used in the final layers for classification
- Each output is connected to ALL inputs

Parameters:
- in_features: Size of each input sample
- out_features: Size of each output sample
- bias: If True (default), adds a learnable bias
"""

print("=" * 70)
print("LINEAR LAYER (nn.Linear)")
print("=" * 70)

# Example 1: Simple linear transformation
linear = nn.Linear(in_features=10, out_features=5)
input_data = torch.randn(3, 10)  # Batch of 3 samples, each with 10 features
output = linear(input_data)

print(f"Input shape: {input_data.shape}")
print(f"Output shape: {output.shape}")
print(f"Weight shape: {linear.weight.shape}")
print(f"Bias shape: {linear.bias.shape}")
print(f"\nInput sample:\n{input_data[0]}")
print(f"\nOutput sample:\n{output[0]}")

# %%
# =====================================================================
# 2. CONVOLUTIONAL LAYER 2D (nn.Conv2d)
# =====================================================================
"""
Conv2d performs 2D convolution, the fundamental operation in CNNs for image processing.

How it works:
- Slides a filter (kernel) across the input image
- At each position, computes element-wise multiplication and sum
- Creates a feature map that detects patterns (edges, textures, objects)

Purpose:
- Learns spatial features from images
- Preserves spatial relationships
- Shares weights across the image (parameter efficient)
- Early layers detect edges, later layers detect complex patterns

Parameters:
- in_channels: Number of input channels (3 for RGB, 1 for grayscale)
- out_channels: Number of filters (feature maps) to create
- kernel_size: Size of the convolving kernel (e.g., 3x3, 5x5)
- stride: How much to move the kernel each step (default=1)
- padding: Pixels to add around the image (default=0)
- bias: If True, adds learnable bias
"""

print("=" * 70)
print("CONVOLUTIONAL LAYER 2D (nn.Conv2d)")
print("=" * 70)

# Get sample images
images, labels = get_sample_images(num_images=4)
print(f"Original images shape: {images.shape}")  # (batch, channels, height, width)

# Example: Conv2d with 3x3 kernel
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
conv_output = conv(images)

print(f"After Conv2d(3→16, kernel=3x3, padding=1):")
print(f"  Output shape: {conv_output.shape}")
print(f"  Weight shape: {conv.weight.shape}")  # (out_channels, in_channels, kernel_h, kernel_w)

# Visualize first 4 feature maps
print("\nVisualizing first 4 feature maps from first image:")
feature_maps = conv_output[0, :4].unsqueeze(1)  # Take first 4 channels
show_images(feature_maps, titles=[f"Filter {i+1}" for i in range(4)])

# %%
# =====================================================================
# 3. MAX POOLING (nn.MaxPool2d)
# =====================================================================
"""
MaxPool2d reduces spatial dimensions by taking the maximum value in each region.

How it works:
- Divides input into non-overlapping regions (e.g., 2x2)
- Takes the maximum value from each region
- Outputs a downsampled feature map

Purpose:
- Reduces spatial dimensions (less computation)
- Provides translation invariance (small shifts don't affect output)
- Reduces overfitting by abstracting features
- Keeps the strongest activations

Parameters:
- kernel_size: Size of the pooling window
- stride: How much to move the window (default=kernel_size)
- padding: Padding to add (default=0)
- return_indices: If True, returns indices for unpooling
"""

print("=" * 70)
print("MAX POOLING (nn.MaxPool2d)")
print("=" * 70)

images, _ = get_sample_images(num_images=4)
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
pooled = maxpool(images)

print(f"Before MaxPool2d: {images.shape}")
print(f"After MaxPool2d(2x2): {pooled.shape}")
print(f"Spatial reduction: {images.shape[2]}x{images.shape[3]} → {pooled.shape[2]}x{pooled.shape[3]}")

# Visualize the effect
show_images(images, titles=[f"Original {i+1}" for i in range(4)])
show_images(pooled, titles=[f"Pooled {i+1}" for i in range(4)])

# %%
# =====================================================================
# 4. MAX UNPOOLING (nn.MaxUnpool2d)
# =====================================================================
"""
MaxUnpool2d is the inverse of MaxPool2d. It upsamples by placing values
at the positions remembered from MaxPool2d and filling the rest with zeros.

How it works:
- MaxPool2d can return indices of max values
- MaxUnpool2d uses these indices to place values back
- Creates sparse upsampled output

Purpose:
- Used in decoder networks (e.g., segmentation, autoencoders)
- Helps preserve spatial information from pooling
- Creates sparse feature maps

Parameters:
- kernel_size: Size of the unpooling window (should match pooling)
- stride: Stride of unpooling (should match pooling)
- padding: Padding (should match pooling)
"""

print("=" * 70)
print("MAX UNPOOLING (nn.MaxUnpool2d)")
print("=" * 70)

# Create a simple example
input_tensor = torch.randn(1, 1, 4, 4)
print(f"Original input shape: {input_tensor.shape}")
print(f"Original input:\n{input_tensor.squeeze()}")

# MaxPool with indices
maxpool_with_indices = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
pooled, indices = maxpool_with_indices(input_tensor)

print(f"\nAfter MaxPool2d (with indices): {pooled.shape}")
print(f"Pooled output:\n{pooled.squeeze()}")
print(f"Indices:\n{indices.squeeze()}")

# MaxUnpool
maxunpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
unpooled = maxunpool(pooled, indices)

print(f"\nAfter MaxUnpool2d: {unpooled.shape}")
print(f"Unpooled output:\n{unpooled.squeeze()}")
print("\nNote: Values are placed at original max positions, rest filled with zeros")

# %%
# =====================================================================
# 5. AVERAGE POOLING (nn.AvgPool2d)
# =====================================================================
"""
AvgPool2d reduces spatial dimensions by taking the average value in each region.

How it works:
- Divides input into regions (e.g., 2x2)
- Computes the average of values in each region
- Outputs a downsampled feature map

Purpose:
- Similar to MaxPool but smoother
- Preserves more information about all features (not just max)
- Often used before final classification layer
- Less aggressive than MaxPool

Parameters:
- kernel_size: Size of the pooling window
- stride: How much to move the window (default=kernel_size)
- padding: Padding to add (default=0)
"""

print("=" * 70)
print("AVERAGE POOLING (nn.AvgPool2d)")
print("=" * 70)

images, _ = get_sample_images(num_images=4)
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
avg_pooled = avgpool(images)

print(f"Before AvgPool2d: {images.shape}")
print(f"After AvgPool2d(2x2): {avg_pooled.shape}")

# Compare with MaxPool
max_pooled = maxpool(images)

print("\nComparing MaxPool vs AvgPool:")
print("MaxPool tends to keep stronger/sharper features")
print("AvgPool creates smoother, more blended features")

show_images(images[:2], titles=["Original 1", "Original 2"])
show_images(max_pooled[:2], titles=["MaxPool 1", "MaxPool 2"])
show_images(avg_pooled[:2], titles=["AvgPool 1", "AvgPool 2"])

# %%
# =====================================================================
# 6. ADAPTIVE AVERAGE POOLING (nn.AdaptiveAvgPool2d)
# =====================================================================
"""
AdaptiveAvgPool2d automatically adjusts to produce a specific output size,
regardless of input size.

How it works:
- You specify desired output size (e.g., 7x7)
- Layer automatically calculates kernel size and stride
- Very useful for handling variable input sizes

Purpose:
- Makes networks flexible to different input sizes
- Used in modern architectures (ResNet, etc.)
- Allows global average pooling (output_size=1)
- Simplifies architecture design

Parameters:
- output_size: Desired output spatial dimensions (H, W) or single value for square
"""

print("=" * 70)
print("ADAPTIVE AVERAGE POOLING (nn.AdaptiveAvgPool2d)")
print("=" * 70)

# Example with different input sizes
inputs = [
    torch.randn(1, 3, 224, 224),
    torch.randn(1, 3, 128, 128),
    torch.randn(1, 3, 300, 200),
]

adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

print("Same layer, different input sizes → same output size:")
for i, inp in enumerate(inputs):
    output = adaptive_pool(inp)
    print(f"Input {i+1}: {inp.shape} → Output: {output.shape}")

# Global Average Pooling (very common)
global_pool = nn.AdaptiveAvgPool2d(output_size=1)
images, _ = get_sample_images(num_images=2)
global_features = global_pool(images)

print(f"\nGlobal Average Pooling:")
print(f"Input: {images.shape} → Output: {global_features.shape}")
print("Each channel becomes a single value (average of all spatial positions)")


# ok but what is Global Average Pooling useful for?
# It is often used at the end of convolutional neural networks before the final classification layer.

# Global average pooling (GAP) reduces the spatial dimensions of each feature map in a convolutional neural network (CNN)
# to a single value by averaging all its elements. It is often used just before the final output layer to convert a set of feature 
#  maps into a fixed-size vector, which helps prevent overfitting and makes the model more spatially invariant

# more details and images here:
# https://www.kdnuggets.com/diving-into-the-pool-unraveling-the-magic-of-cnn-pooling-layers

# %%
# =====================================================================
# 7. ADAPTIVE MAX POOLING (nn.AdaptiveMaxPool2d)
# =====================================================================
"""
AdaptiveMaxPool2d is like AdaptiveAvgPool2d but uses max instead of average.

How it works:
- Automatically adjusts to produce specified output size
- Takes maximum values instead of averages
- More aggressive feature selection

Purpose:
- Same flexibility as AdaptiveAvgPool2d
- Keeps strongest activations
- Can be used for global max pooling

Parameters:
- output_size: Desired output spatial dimensions
- return_indices: If True, returns indices (for unpooling)
"""

print("=" * 70)
print("ADAPTIVE MAX POOLING (nn.AdaptiveMaxPool2d)")
print("=" * 70)

adaptive_max_pool = nn.AdaptiveMaxPool2d(output_size=(7, 7))
images, _ = get_sample_images(num_images=2)

avg_out = nn.AdaptiveAvgPool2d(7)(images)
max_out = adaptive_max_pool(images)

print(f"Input shape: {images.shape}")
print(f"Adaptive AvgPool output: {avg_out.shape}")
print(f"Adaptive MaxPool output: {max_out.shape}")

# Global pooling comparison
global_avg = nn.AdaptiveAvgPool2d(1)(images)
global_max = nn.AdaptiveMaxPool2d(1)(images)

print(f"\nGlobal pooling (output_size=1):")
print(f"Global AvgPool: {global_avg.shape}")
print(f"Global MaxPool: {global_max.shape}")
print(f"First image, first channel - Avg: {global_avg[0, 0].item():.4f}, Max: {global_max[0, 0].item():.4f}")

# %%
# =====================================================================
# 8. BATCH NORMALIZATION (nn.BatchNorm2d)
# =====================================================================
"""
BatchNorm2d normalizes activations across the batch dimension.

How it works:
- For each channel, computes mean and std across batch and spatial dimensions
- Normalizes: y = (x - mean) / sqrt(var + eps)
- Then applies learned scale (gamma) and shift (beta): y = gamma * y + beta
- During training: uses batch statistics
- During eval: uses running statistics accumulated during training

Purpose:
- Stabilizes and speeds up training
- Reduces internal covariate shift
- Acts as regularization (slight noise from batch statistics)
- Allows higher learning rates
- Reduces sensitivity to initialization

Parameters:
- num_features: Number of channels (C in batch, C, H, W)
- eps: Small constant for numerical stability (default=1e-5)
- momentum: For running statistics (default=0.1)
- affine: If True, learns gamma and beta (default=True)
"""

print("=" * 70)
print("BATCH NORMALIZATION (nn.BatchNorm2d)")
print("=" * 70)

images, _ = get_sample_images(num_images=4)
print(f"Original images shape: {images.shape}")

# Statistics before normalization
print(f"\nBefore BatchNorm:")
print(f"  Mean: {images.mean().item():.4f}")
print(f"  Std: {images.std().item():.4f}")
print(f"  Min: {images.min().item():.4f}, Max: {images.max().item():.4f}")

# Apply BatchNorm
batchnorm = nn.BatchNorm2d(num_features=3)
batchnorm.train()  # Training mode
normalized = batchnorm(images)

print(f"\nAfter BatchNorm (training mode):")
print(f"  Mean: {normalized.mean().item():.4f}")
print(f"  Std: {normalized.std().item():.4f}")
print(f"  Min: {normalized.min().item():.4f}, Max: {normalized.max().item():.4f}")

# Visualize effect
show_images(images, titles=[f"Original {i+1}" for i in range(4)])
show_images(normalized, titles=[f"Normalized {i+1}" for i in range(4)])

print("\nNote: BatchNorm normalizes each channel independently")
print("This helps stabilize training and allows higher learning rates")

# %%
# =====================================================================
# 9. DROPOUT (nn.Dropout)
# =====================================================================
"""
Dropout randomly sets elements to zero during training.

How it works:
- During TRAINING: Each element is set to 0 with probability p
  - Remaining elements are scaled by 1/(1-p) to maintain expected value
- During EVAL: Does nothing (pass-through)

Purpose:
- Prevents overfitting
- Forces network to learn redundant representations
- Acts like ensemble learning (different sub-networks each iteration)
- Makes network more robust

Parameters:
- p: Probability of an element being zeroed (default=0.5)
- inplace: If True, modifies input directly (default=False)

IMPORTANT: Always set model.train() during training and model.eval() during evaluation!
"""

print("=" * 70)
print("DROPOUT (nn.Dropout)")
print("=" * 70)

dropout = nn.Dropout(p=0.5)
input_tensor = torch.ones(1, 10)  # Simple example

print(f"Original input:\n{input_tensor}")

# Training mode - dropout active
dropout.train()
output_train1 = dropout(input_tensor.clone())
output_train2 = dropout(input_tensor.clone())

print(f"\nTraining mode (dropout active, p=0.5):")
print(f"Output 1:\n{output_train1}")
print(f"Output 2:\n{output_train2}")
print("Note: Different elements zeroed each time, remaining scaled by 2.0")

# Eval mode - dropout inactive
dropout.eval()
output_eval = dropout(input_tensor)

print(f"\nEval mode (dropout inactive):")
print(f"Output:\n{output_eval}")
print("Note: Input passed through unchanged")

# %%
# =====================================================================
# 10. DROPOUT 2D (nn.Dropout2d)
# =====================================================================
"""
Dropout2d drops entire channels instead of individual elements.

How it works:
- During TRAINING: Entire channels (feature maps) are set to 0 with probability p
  - If a channel is dropped, ALL spatial positions in that channel are zeroed
- During EVAL: Does nothing (pass-through)

Purpose:
- Better for convolutional layers (spatial correlation)
- Prevents co-adaptation between feature maps
- More effective than regular dropout for CNNs
- Maintains spatial coherence

Parameters:
- p: Probability of a channel being zeroed (default=0.5)
- inplace: If True, modifies input directly (default=False)

DIFFERENCE from Dropout:
- Dropout: zeros random individual elements
- Dropout2d: zeros entire channels (all positions in a feature map)
"""

print("=" * 70)
print("DROPOUT 2D (nn.Dropout2d)")
print("=" * 70)

images, _ = get_sample_images(num_images=2)
print(f"Input shape: {images.shape}")  # (batch, channels, height, width)

dropout2d = nn.Dropout2d(p=0.3)

# Training mode
dropout2d.train()
dropped = dropout2d(images.clone())

print(f"\nTraining mode (p=0.3):")
print(f"Original - First image, first 3 channels mean:")
for i in range(3):
    print(f"  Channel {i}: {images[0, i].mean():.4f}")

print(f"\nAfter Dropout2d - First image, first 3 channels mean:")
for i in range(3):
    mean_val = dropped[0, i].mean()
    print(f"  Channel {i}: {mean_val:.4f} {'(DROPPED)' if mean_val == 0 else ''}")

# Visualize
print("\nVisual comparison:")
show_images(images[:2], titles=["Original 1", "Original 2"])
show_images(dropped[:2], titles=["With Dropout2d 1", "With Dropout2d 2"])

print("\nKEY DIFFERENCE:")
print("- nn.Dropout: Randomly zeros INDIVIDUAL pixels")
print("- nn.Dropout2d: Zeros ENTIRE channels (feature maps)")
print("For CNNs, Dropout2d is usually better!")

# %%
# =====================================================================
# 11. LAYER NORMALIZATION (nn.LayerNorm)
# =====================================================================
"""
LayerNorm normalizes across features for each sample independently.

How it works:
- Computes mean and std for EACH sample across the feature dimension
- Unlike BatchNorm (normalizes across batch), LayerNorm normalizes across features
- Formula: y = (x - mean) / sqrt(var + eps) * gamma + beta

Purpose:
- Works well with small batch sizes (doesn't depend on batch statistics)
- Used in Transformers and RNNs
- More stable for sequence models
- Each sample normalized independently

Parameters:
- normalized_shape: Shape of features to normalize over (can be tuple)
- eps: Small constant for stability (default=1e-5)
- elementwise_affine: If True, learns gamma and beta (default=True)

DIFFERENCE from BatchNorm:
- BatchNorm: Normalizes across batch dimension (for each feature)
- LayerNorm: Normalizes across feature dimension (for each sample)
"""

print("=" * 70)
print("LAYER NORMALIZATION (nn.LayerNorm)")
print("=" * 70)

# Example with simple data
batch_data = torch.randn(4, 10)  # 4 samples, 10 features each
print(f"Input shape: {batch_data.shape}")
print(f"Input (first sample): {batch_data[0]}")

# LayerNorm normalizes each sample independently
layernorm = nn.LayerNorm(normalized_shape=10)
ln_output = layernorm(batch_data)

print(f"\nAfter LayerNorm:")
print(f"Output (first sample): {ln_output[0]}")
print(f"\nFirst sample - mean: {ln_output[0].mean():.6f}, std: {ln_output[0].std():.6f}")
print(f"Second sample - mean: {ln_output[1].mean():.6f}, std: {ln_output[1].std():.6f}")

# For images (less common, but possible)
images, _ = get_sample_images(num_images=2)
# Normalize across channel, height, width dimensions
layernorm_img = nn.LayerNorm([3, 224, 224])
ln_images = layernorm_img(images)

print(f"\nFor images:")
print(f"Input shape: {images.shape}")
print(f"First image - mean: {images[0].mean():.4f}, std: {images[0].std():.4f}")
print(f"After LN - mean: {ln_images[0].mean():.4f}, std: {ln_images[0].std():.4f}")

# %%
# =====================================================================
# 12. GROUP NORMALIZATION (nn.GroupNorm)
# =====================================================================
"""
GroupNorm divides channels into groups and normalizes within each group.

How it works:
- Divides channels into G groups
- Normalizes within each group (across channels and spatial dimensions)
- Independent of batch size (like LayerNorm)
- Middle ground between LayerNorm (1 group) and InstanceNorm (C groups)

Purpose:
- Works well with small batch sizes
- Alternative to BatchNorm when batch size is small
- Used in detection/segmentation where batch sizes are limited
- More stable than BatchNorm for small batches

Parameters:
- num_groups: Number of groups to divide channels into
- num_channels: Number of channels (must be divisible by num_groups)
- eps: Small constant for stability (default=1e-5)
- affine: If True, learns gamma and beta (default=True)

Example: 32 channels with 8 groups = 4 channels per group
"""

print("=" * 70)
print("GROUP NORMALIZATION (nn.GroupNorm)")
print("=" * 70)

# Create feature maps
feature_maps = torch.randn(2, 16, 32, 32)  # (batch=2, channels=16, H=32, W=32)
print(f"Input shape: {feature_maps.shape}")

# GroupNorm with 4 groups (16 channels / 4 groups = 4 channels per group)
groupnorm = nn.GroupNorm(num_groups=4, num_channels=16)
gn_output = groupnorm(feature_maps)

print(f"GroupNorm(num_groups=4, num_channels=16)")
print(f"Output shape: {gn_output.shape}")
print(f"\nChannels divided into 4 groups of 4 channels each")

# Compare different normalizations
print("\nComparing normalization methods:")
print("- BatchNorm: Normalizes each channel across batch and spatial dims")
print("- LayerNorm: Normalizes all channels+spatial dims for each sample")
print("- GroupNorm: Normalizes groups of channels for each sample")
print("- InstanceNorm: Normalizes each channel for each sample (next section)")

# Real example with images
images, _ = get_sample_images(num_images=2)
groupnorm_img = nn.GroupNorm(num_groups=3, num_channels=3)  # 3 groups for RGB
gn_images = groupnorm_img(images)

print(f"\nApplying GroupNorm to images:")
print(f"Before - mean: {images.mean():.4f}, std: {images.std():.4f}")
print(f"After - mean: {gn_images.mean():.4f}, std: {gn_images.std():.4f}")

# %%
# =====================================================================
# 13. INSTANCE NORMALIZATION (nn.InstanceNorm2d)
# =====================================================================
"""
InstanceNorm2d normalizes each channel of each sample independently.

How it works:
- For each sample and each channel, computes mean and std across spatial dimensions
- Normalizes each channel independently
- Equivalent to GroupNorm with num_groups = num_channels
- Does NOT use batch statistics

Purpose:
- Used in style transfer and GANs
- Removes instance-specific contrast information
- Focuses on relative feature strengths within each channel
- Good when batch statistics are not meaningful

Parameters:
- num_features: Number of channels
- eps: Small constant for stability (default=1e-5)
- momentum: For running stats (default=0.1)
- affine: If True, learns gamma and beta (default=False for InstanceNorm)

DIFFERENCE from BatchNorm:
- BatchNorm: Uses batch statistics
- InstanceNorm: Each sample and channel normalized independently
"""

print("=" * 70)
print("INSTANCE NORMALIZATION (nn.InstanceNorm2d)")
print("=" * 70)

images, _ = get_sample_images(num_images=2)
print(f"Input shape: {images.shape}")

instancenorm = nn.InstanceNorm2d(num_features=3)
in_output = instancenorm(images)

print(f"\nStatistics per channel of first image:")
for i in range(3):
    print(f"Channel {i}:")
    print(f"  Before - mean: {images[0, i].mean():.4f}, std: {images[0, i].std():.4f}")
    print(f"  After - mean: {in_output[0, i].mean():.4f}, std: {in_output[0, i].std():.4f}")

# Visualize
show_images(images[:2], titles=["Original 1", "Original 2"])
show_images(in_output[:2], titles=["InstanceNorm 1", "InstanceNorm 2"])

print("\nCommon use cases:")
print("- Style transfer: Removes style information, keeps content")
print("- GANs: Each sample treated independently")
print("- Any task where batch statistics are not meaningful")



# More about normalizations here (with images and comparisons):
# https://isaac-the-man.dev/posts/normalization-strategies/


# %%
# =====================================================================
# 14. ACTIVATION FUNCTIONS - ReLU
# =====================================================================
"""
ReLU (Rectified Linear Unit) is the most common activation function.

Formula: f(x) = max(0, x)

How it works:
- Outputs x if x > 0, otherwise outputs 0
- Non-linear, allows networks to learn complex patterns
- Very simple and efficient

Purpose:
- Introduces non-linearity (without it, network would be linear)
- Solves vanishing gradient problem (gradients don't vanish for positive values)
- Sparse activations (negative values become 0)
- Fast computation

Parameters:
- inplace: If True, modifies input directly (default=False)

Advantages:
- Simple and fast
- Doesn't saturate for positive values
- Sparse activations

Disadvantages:
- "Dying ReLU" problem: neurons can get stuck at 0
- Not zero-centered
"""

print("=" * 70)
print("ACTIVATION FUNCTION - ReLU")
print("=" * 70)

relu = nn.ReLU()
x = torch.linspace(-3, 3, 100)
y = relu(x)

plt.figure(figsize=(10, 4))
plt.plot(x.numpy(), y.numpy(), linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('ReLU Activation Function: f(x) = max(0, x)')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.show()

# Apply to images
images, _ = get_sample_images(num_images=2)
# First apply a transformation that creates negative values
conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
conv.bias.data.fill_(-1)  # Add negative bias
features = conv(images)
features_relu = relu(features)

print(f"Before ReLU - min: {features.min():.4f}, max: {features.max():.4f}")
print(f"After ReLU - min: {features_relu.min():.4f}, max: {features_relu.max():.4f}")
print("All negative values are clipped to 0")

# %%
# =====================================================================
# 15. ACTIVATION FUNCTIONS - Leaky ReLU
# =====================================================================
"""
Leaky ReLU allows small negative values instead of zero.

Formula: f(x) = x if x > 0, else alpha * x

How it works:
- Like ReLU, but instead of 0 for negative values, outputs alpha * x
- alpha is usually small (e.g., 0.01)
- Prevents "dying ReLU" problem

Purpose:
- Fixes dying ReLU problem
- All neurons can still contribute to learning
- Slightly better than ReLU in some cases

Parameters:
- negative_slope: Slope for negative values (default=0.01)
- inplace: If True, modifies input directly (default=False)
"""

print("=" * 70)
print("ACTIVATION FUNCTION - Leaky ReLU")
print("=" * 70)

leaky_relu = nn.LeakyReLU(negative_slope=0.1)
x = torch.linspace(-3, 3, 100)
y_relu = relu(x)
y_leaky = leaky_relu(x)

plt.figure(figsize=(10, 4))
plt.plot(x.numpy(), y_relu.numpy(), label='ReLU', linewidth=2)
plt.plot(x.numpy(), y_leaky.numpy(), label='Leaky ReLU (slope=0.1)', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('ReLU vs Leaky ReLU')
plt.legend()
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.show()

print("Key difference: Leaky ReLU has small gradient for negative values")
print("This prevents neurons from 'dying' (getting stuck at 0)")

# %%
# =====================================================================
# 16. ACTIVATION FUNCTIONS - ELU
# =====================================================================
"""
ELU (Exponential Linear Unit) uses exponential for negative values.

Formula: f(x) = x if x > 0, else alpha * (exp(x) - 1)

How it works:
- Linear for positive values
- Smooth exponential curve for negative values
- Approaches -alpha as x goes to negative infinity

Purpose:
- Smoother than ReLU and Leaky ReLU
- Mean activation closer to zero (better optimization)
- Reduces bias shift
- Can produce negative outputs (unlike ReLU)

Parameters:
- alpha: Scale for negative values (default=1.0)
- inplace: If True, modifies input directly (default=False)

Advantages:
- No dying ReLU problem
- Smoother gradient
- Mean activations closer to zero

Disadvantages:
- Slightly slower than ReLU (exponential computation)
"""

print("=" * 70)
print("ACTIVATION FUNCTION - ELU")
print("=" * 70)

elu = nn.ELU(alpha=1.0)
x = torch.linspace(-3, 3, 100)
y_relu = relu(x)
y_leaky = leaky_relu(x)
y_elu = elu(x)

plt.figure(figsize=(12, 4))
plt.plot(x.numpy(), y_relu.numpy(), label='ReLU', linewidth=2)
plt.plot(x.numpy(), y_leaky.numpy(), label='Leaky ReLU', linewidth=2)
plt.plot(x.numpy(), y_elu.numpy(), label='ELU', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Comparison of ReLU, Leaky ReLU, and ELU')
plt.legend()
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.show()

print("ELU characteristics:")
print("- Smooth curve for negative values")
print("- Approaches -1.0 as x → -∞")
print("- Can help with zero-centered activations")

# %%
# =====================================================================
# 17. ACTIVATION FUNCTIONS - Sigmoid
# =====================================================================
"""
Sigmoid squashes input to range (0, 1).

Formula: f(x) = 1 / (1 + exp(-x))

How it works:
- S-shaped curve
- Output always between 0 and 1
- Smooth and differentiable everywhere

Purpose:
- Binary classification (output layer)
- Gating mechanisms (LSTM, attention)
- When you need outputs in range [0, 1]

Parameters:
- None

Advantages:
- Smooth gradient
- Clear probabilistic interpretation
- Good for binary decisions

Disadvantages:
- Vanishing gradient problem (gradients very small for large |x|)
- Not zero-centered (can slow learning)
- Saturates on both ends
- Rarely used in hidden layers anymore (replaced by ReLU)
"""

print("=" * 70)
print("ACTIVATION FUNCTION - Sigmoid")
print("=" * 70)

sigmoid = nn.Sigmoid()
x = torch.linspace(-6, 6, 100)
y = sigmoid(x)

plt.figure(figsize=(10, 4))
plt.plot(x.numpy(), y.numpy(), linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Sigmoid Activation Function: f(x) = 1/(1+exp(-x))')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axhline(y=1, color='r', linestyle='--', alpha=0.3, label='y=1')
plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.3, label='y=0.5')
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.legend()
plt.ylim(-0.1, 1.1)
plt.show()

print("Key points:")
print("- Output always in (0, 1)")
print("- f(0) = 0.5")
print("- Used for binary classification and gates")
print("- Suffers from vanishing gradients for large |x|")

# %%
# =====================================================================
# 18. ACTIVATION FUNCTIONS - Tanh
# =====================================================================
"""
Tanh (Hyperbolic Tangent) squashes input to range (-1, 1).

Formula: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

How it works:
- S-shaped curve like sigmoid
- Output between -1 and 1
- Zero-centered (unlike sigmoid)

Purpose:
- Hidden layers (when not using ReLU)
- RNNs and LSTMs
- When zero-centered outputs are beneficial

Parameters:
- None

Advantages:
- Zero-centered (better than sigmoid)
- Stronger gradients than sigmoid
- Smooth and differentiable

Disadvantages:
- Still suffers from vanishing gradient problem
- Slower than ReLU
- Mostly replaced by ReLU in CNNs
"""

print("=" * 70)
print("ACTIVATION FUNCTION - Tanh")
print("=" * 70)

tanh = nn.Tanh()
x = torch.linspace(-6, 6, 100)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)

plt.figure(figsize=(12, 4))
plt.plot(x.numpy(), y_sigmoid.numpy(), label='Sigmoid', linewidth=2)
plt.plot(x.numpy(), y_tanh.numpy(), label='Tanh', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Sigmoid vs Tanh')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.legend()
plt.show()

print("Key differences from Sigmoid:")
print("- Range: (-1, 1) instead of (0, 1)")
print("- Zero-centered: f(0) = 0")
print("- Stronger gradients than sigmoid")
print("- Often preferred over sigmoid for hidden layers")

# %%
# =====================================================================
# 19. ACTIVATION FUNCTIONS - Hardswish
# =====================================================================
"""
Hardswish is a computationally efficient approximation of Swish.

Formula: f(x) = x * ReLU6(x + 3) / 6
Where ReLU6(x) = min(max(0, x), 6)

How it works:
- Smooth approximation of ReLU
- Similar to Swish but computationally cheaper
- Used in MobileNetV3

Purpose:
- Efficient activation for mobile/edge devices
- Better than ReLU in some architectures
- Good for resource-constrained environments

Parameters:
- inplace: If True, modifies input directly (default=False)

Advantages:
- More efficient than Swish
- Better than ReLU in some cases
- Good for mobile architectures

When to use:
- MobileNet-style architectures
- Resource-constrained deployments
"""

print("=" * 70)
print("ACTIVATION FUNCTION - Hardswish")
print("=" * 70)

hardswish = nn.Hardswish()
x = torch.linspace(-4, 4, 100)
y_relu = relu(x)
y_hardswish = hardswish(x)

plt.figure(figsize=(10, 4))
plt.plot(x.numpy(), y_relu.numpy(), label='ReLU', linewidth=2)
plt.plot(x.numpy(), y_hardswish.numpy(), label='Hardswish', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('ReLU vs Hardswish')
plt.legend()
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.show()

print("Hardswish characteristics:")
print("- Smooth activation (unlike ReLU)")
print("- Efficient to compute")
print("- Used in modern mobile architectures (MobileNetV3)")

# %%
# =====================================================================
# 20. TRANSPOSED CONVOLUTION (nn.ConvTranspose2d)
# =====================================================================
"""
ConvTranspose2d (also called Deconvolution) upsamples feature maps.

How it works:
- Reverse of convolution (in terms of spatial dimensions)
- Inserts zeros between input elements (controlled by stride)
- Applies convolution to produce larger output
- Learns how to upsample

Purpose:
- Upsampling in decoder networks
- Semantic segmentation
- GANs (generator networks)
- Autoencoders
- Super-resolution

Parameters:
- in_channels: Number of input channels
- out_channels: Number of output channels
- kernel_size: Size of the kernel
- stride: Controls upsampling factor
- padding: Padding to add
- output_padding: Additional size to output (for ambiguous output shapes)

Common use: stride=2 doubles spatial dimensions
"""

print("=" * 70)
print("TRANSPOSED CONVOLUTION (nn.ConvTranspose2d)")
print("=" * 70)

# Start with small feature maps
small_features = torch.randn(2, 64, 28, 28)
print(f"Small feature maps: {small_features.shape}")

# Upsample with transposed convolution
transpose_conv = nn.ConvTranspose2d(
    in_channels=64, 
    out_channels=32, 
    kernel_size=4, 
    stride=2, 
    padding=1
)
upsampled = transpose_conv(small_features)

print(f"After ConvTranspose2d(stride=2): {upsampled.shape}")
print(f"Spatial dimensions: 28x28 → {upsampled.shape[2]}x{upsampled.shape[3]}")

# Practical example with images
images, _ = get_sample_images(num_images=2)
print(f"\nOriginal images: {images.shape}")

# Downsample
downsample = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
small = downsample(images)
print(f"After downsampling: {small.shape}")

# Upsample back
upsample = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)
reconstructed = upsample(small)
print(f"After upsampling: {reconstructed.shape}")

# Visualize
show_images(images, titles=["Original 1", "Original 2"])
show_images(small, titles=["Downsampled 1", "Downsampled 2"])
show_images(reconstructed, titles=["Reconstructed 1", "Reconstructed 2"])

print("\nCommon architecture pattern:")
print("Encoder: Conv → Pool → Conv → Pool (downsample)")
print("Decoder: ConvTranspose → ConvTranspose (upsample)")

# %%
# =====================================================================
# 21. EMBEDDING LAYER (nn.Embedding)
# =====================================================================
"""
Embedding layer converts discrete tokens (integers) to dense vectors.

How it works:
- Maintains a lookup table of embeddings
- Each integer index maps to a learned dense vector
- Used to convert discrete data (words, categories) to continuous representations

Purpose:
- Word embeddings in NLP
- Categorical feature encoding
- User/item embeddings in recommendation systems
- Any discrete-to-continuous conversion

Parameters:
- num_embeddings: Size of the vocabulary (number of unique tokens)
- embedding_dim: Size of each embedding vector
- padding_idx: Index to use for padding (embedding stays at zero)

Example: Word embeddings
- Vocabulary size: 10,000 words
- Embedding dim: 300
- Each word → 300-dimensional vector
"""

print("=" * 70)
print("EMBEDDING LAYER (nn.Embedding)")
print("=" * 70)

# Example: Embedding for a small vocabulary
vocab_size = 10
embedding_dim = 4

embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# Example sentences as token indices
sentences = torch.LongTensor([
    [1, 3, 5, 2],  # Sentence 1
    [4, 1, 0, 7],  # Sentence 2
    [2, 8, 6, 1],  # Sentence 3
])

print(f"Input sentences (token indices): {sentences.shape}")
print(f"Sentences:\n{sentences}")

# Convert to embeddings
embedded = embedding(sentences)

print(f"\nEmbedded sentences: {embedded.shape}")
print(f"(batch_size=3, sequence_length=4, embedding_dim=4)")

print(f"\nFirst sentence, first token (index {sentences[0, 0].item()}):")
print(embedded[0, 0])

print(f"\nFirst sentence, second token (index {sentences[0, 1].item()}):")
print(embedded[0, 1])

print("\nEach unique token has its own learned embedding vector")
print("Same token → Same embedding vector")

# Practical example: Category embedding for waste classes
num_classes = len(final_classes)
class_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=8)

class_indices = torch.LongTensor([0, 1, 2, 3])  # organic, battery, glass, metal
class_vectors = class_embedding(class_indices)

print(f"\nWaste class embeddings:")
print(f"Class indices: {class_indices}")
print(f"Embedded vectors shape: {class_vectors.shape}")
print("\nEach waste class → 8-dimensional learned vector")

# %%
# =====================================================================
# 22. FLATTEN LAYER (nn.Flatten)
# =====================================================================
"""
Flatten reshapes multi-dimensional tensors into 2D (batch, features).

How it works:
- Keeps batch dimension
- Flattens all other dimensions into a single feature dimension
- Essential for connecting convolutional layers to fully connected layers

Purpose:
- Transition from CNN layers to fully connected layers
- Prepare spatial features for classification
- Convert 4D tensors (batch, channels, height, width) to 2D (batch, features)

Parameters:
- start_dim: First dimension to flatten (default=1, keeps batch)
- end_dim: Last dimension to flatten (default=-1, all remaining)
"""

print("=" * 70)
print("FLATTEN LAYER (nn.Flatten)")
print("=" * 70)

# Example with feature maps from CNN
feature_maps = torch.randn(4, 64, 7, 7)  # Batch=4, Channels=64, H=7, W=7
print(f"CNN output (feature maps): {feature_maps.shape}")

flatten = nn.Flatten()
flattened = flatten(feature_maps)

print(f"After Flatten: {flattened.shape}")
print(f"64 × 7 × 7 = {64*7*7} features per sample")

# This can now go through a linear layer
linear = nn.Linear(in_features=64*7*7, out_features=10)
output = linear(flattened)
print(f"After Linear layer: {output.shape}")

# Typical CNN architecture ending
print("\nTypical CNN architecture:")
print("Conv2d → ReLU → MaxPool → ... → Conv2d → ReLU → AdaptiveAvgPool")
print("→ Flatten → Linear → Output")

# %%
# =====================================================================
# 23. LAZY LAYERS (nn.LazyLinear, nn.LazyConv2d)
# =====================================================================
"""
Lazy layers automatically infer input dimensions during first forward pass.

How it works:
- You don't specify input dimensions when creating the layer
- First time data passes through, layer infers input size
- Creates weights based on actual input shape
- Very convenient for prototyping

Purpose:
- Easier prototyping (don't need to calculate dimensions)
- Flexible architectures
- Less error-prone for complex dimension chains

Available lazy layers:
- nn.LazyLinear: Lazy version of Linear
- nn.LazyConv2d: Lazy version of Conv2d
- nn.LazyBatchNorm2d: Lazy version of BatchNorm2d
- And more...

Note: After first forward pass, lazy layer becomes a regular layer
"""

print("=" * 70)
print("LAZY LAYERS (nn.LazyLinear, nn.LazyConv2d)")
print("=" * 70)

# Example 1: LazyLinear
print("Example 1: LazyLinear")
print("-" * 50)

# Don't need to specify in_features!
lazy_linear = nn.LazyLinear(out_features=10)
print(f"Created LazyLinear(out_features=10)")

# do not try to access weight yet, they are uninitialized, so we don't know them yet
print("Weight is uninitialized (will be created on first forward pass)")

# First forward pass
input_data = torch.randn(5, 20)  # Batch=5, Features=20
output = lazy_linear(input_data)

print(f"\nAfter first forward pass:")
print(f"Input shape: {input_data.shape}")
print(f"Output shape: {output.shape}")
print(f"Weight shape: {lazy_linear.weight.shape}")
print("LazyLinear automatically figured out in_features=20!")

# Example 2: LazyConv2d
print("\n\nExample 2: LazyConv2d")
print("-" * 50)

# Don't need to specify in_channels!
lazy_conv = nn.LazyConv2d(out_channels=32, kernel_size=3, padding=1)
print(f"Created LazyConv2d(out_channels=32, kernel_size=3)")

# First forward pass with images
images, _ = get_sample_images(num_images=2)
conv_output = lazy_conv(images)

print(f"\nAfter first forward pass:")
print(f"Input shape: {images.shape}")
print(f"Output shape: {conv_output.shape}")
print(f"Weight shape: {lazy_conv.weight.shape}")
print("LazyConv2d automatically figured out in_channels=3!")

# Example 3: Building a lazy network
print("\n\nExample 3: Lazy Network")
print("-" * 50)

class LazyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # No need to calculate dimensions!
        self.conv1 = nn.LazyConv2d(16, kernel_size=3, padding=1)
        self.conv2 = nn.LazyConv2d(32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(10)  # No need to know flattened size!
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

lazy_net = LazyNetwork()
test_input = torch.randn(2, 3, 64, 64)
output = lazy_net(test_input)

print(f"Input: {test_input.shape}")
print(f"Output: {output.shape}")
print("\nAll dimensions figured out automatically!")
print("\nNote: Lazy layers are great for prototyping but use")
print("regular layers in production for better control and clarity.")


# Question - if this is co convenient , why not always use lazy layers?
# Firstly - they add some overhead during the first forward pass (not a big problem)
# Secondly - in production code, it's often better to be explicit about dimensions
# as you can see we do not know the dimensions until we run the first forward pass
# which can make debugging and understanding the model harder.




# %%
"""
Summary

BASIC LAYERS:
- Linear: Fully connected, learns general relationships
- Conv2d: Learns spatial features, preserves spatial structure
- Flatten: Converts spatial features to 1D vector

POOLING LAYERS (Reduce spatial dimensions):
- MaxPool2d: Takes maximum, keeps strongest features
- AvgPool2d: Takes average, smoother than MaxPool
- AdaptiveAvgPool2d/AdaptiveMaxPool2d: Output size specified, input size flexible
- MaxUnpool2d: Reverses MaxPool using saved indices

NORMALIZATION LAYERS (Stabilize and speed up training):
- BatchNorm2d: Normalizes across batch, most common for CNNs
- LayerNorm: Normalizes across features, good for small batches/Transformers
- GroupNorm: Middle ground, divides channels into groups
- InstanceNorm2d: Normalizes each sample/channel independently, used in style transfer

REGULARIZATION:
- Dropout: Randomly zeros elements, prevents overfitting
- Dropout2d: Randomly zeros entire channels, better for CNNs

ACTIVATION FUNCTIONS (Introduce non-linearity):
- ReLU: Most common, fast, can have dying neurons
- LeakyReLU: Fixes dying ReLU, small negative slope
- ELU: Smooth, closer to zero mean, slower
- Sigmoid: Output in (0,1), for binary classification/gates
- Tanh: Output in (-1,1), zero-centered
- Hardswish: Efficient, used in mobile networks

SPECIAL LAYERS:
- ConvTranspose2d: Upsamples (decoder networks, GANs, segmentation)
- Embedding: Converts discrete indices to dense vectors (NLP, categories)
- LazyLinear/LazyConv2d: Auto-infer dimensions, great for prototyping

WHEN TO USE WHAT:

CNNs for Images:
Conv2d → BatchNorm2d → ReLU → MaxPool2d → ... → AdaptiveAvgPool2d → Flatten → Linear

Mobile/Edge:
Conv2d → BatchNorm2d → Hardswish → ...

Small Batches:
Conv2d → GroupNorm → ReLU → ...

Segmentation (Encoder-Decoder):
Encoder: Conv2d → Pool
Decoder: ConvTranspose2d (upsample)

Style Transfer:
Conv2d → InstanceNorm2d → ReLU → ...

Regularization:
- Add Dropout before final Linear layers
- Add Dropout2d after Conv2d layers

NLP/Sequences:
Embedding → ... → LayerNorm → ...
"""

# %%

