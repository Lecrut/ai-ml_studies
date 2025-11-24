# Lab 6: taking advantage of knowledge
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



#%% 

class WasteDataModuleMulti(LightningDataModule):
    def __init__(self, root_dir='datasets/combined_waste_dataset',
                 batch_size=32):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_classes = 9  # organic, battery, glass, metal, paper, cardboard, plastic, textiles, trash
        self.class_weights = None

    def setup(self, stage=None):
        self.train_dataset = CombinedWasteDatasetMulti(
            root_dir=self.root_dir, split='train', transform=train_transform
        )
        self.val_dataset = CombinedWasteDatasetMulti(
            root_dir=self.root_dir, split='val', transform=val_test_transform
        )
        self.test_dataset = CombinedWasteDatasetMulti(
            root_dir=self.root_dir, split='test', transform=val_test_transform
        )

        # Use 0 workers on Windows to avoid multiprocessing issues
        import platform
        if platform.system() == 'Windows':
            self.num_workers = 0
        else:
            self.num_workers = os.cpu_count() - 1


        if self.class_weights is not None:
            self.sample_weights = [self.class_weights[label].item() for _, label in self.train_dataset.data]

    def train_dataloader(self):
        if hasattr(self, 'sample_weights'):

            sampler = WeightedRandomSampler(self.sample_weights, len(self.train_dataset), replacement=True)
            
            return DataLoader(self.train_dataset, batch_size=self.batch_size,
                              sampler=sampler, num_workers=self.num_workers)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size,
                              shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)


# %%
'''
Now that we, hopefully, feel really comfortable with pytorch let's solve the task from the previous labs together.
We will try to see what is so special about those, "ResNets", whata are "Residual Connections" and how to take advantage of them.

Let's start by implementing a simple residual block. Not inside the model, just as a standalone block.

'''

#%%
# Short reminder on Residual Connections

# A "residual connection"/"skip connection" is a path letting the original
# input of a stack of layers bypass those layers and be added to the output.
# Instead of only computing y = F(x) (F could be Conv -> ReLU -> Conv), a
# residual block produces y = F(x) + x. Often we then apply
# ReLU: y = ReLU(F(x) + x) for non‑linearity and stability.
#
# Now why would we do that? Very deep networks can suffer from vanishing or exploding gradients. Forcing many
# layers to learn an exact identity mapping is hard. By rewriting the target as
# F(x) = x + R(x), the block just learns the residual R(x). If the best early
# behavior is "do nothing", R(x) -> 0 and F(x) approximates x easily.
#
# During training, the skip pathway carries gradients directly backwards,
# stabilizing optimization and enabling much deeper networks without accuracy
# degradation that appears in plain (non‑residual) architectures.
#
# One more thing to notice - Addition requires EXACT same shape (N, C, H, W). 
# And the main rule of AI is: "Shapes must match!" (TM)
# Therefore we need an operation, that will allow us to do that.
# We could use a Linear layer, but that would mean flattening, transforming, and reshaping back.
# That is too much work, and is computationally expensive.
# So we need a "trick", a "network in network", as the 1x1 convolution is sometimes called.
#
# DURING TRAINING: Initially weights are random so F(x)+x slightly perturbs x.
# As learning proceeds the residual R(x) extracts useful patterns (edges,
# textures, semantics) while preserving original information for later reuse.
#
# SUMMARY POINTS:
# 1. y = x + R(x). R easier to learn than full mapping.
# 2. Mitigates vanishing gradients; enables very deep nets.
# 3. Identity behavior is trivial (set residual near zero).
# 4. Preserves information flow (feature reuse).
# 5. Projection only when shapes differ.
#
# ----------------------------------------------------------------------------------


#%%
# let's start with a minimal residual block with two 3x3 conv layers


class SimplestResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # residual addition
        out = self.relu(out)
        return out







#%% Example 1: Same shape (identity skip)
# Demonstrates a block where input and output shapes match; skip is plain identity.
x_same = torch.randn(2, 16, 32, 32)  # (batch, channels, H, W)
block_same = SimplestResidualBlock(in_channels=16, out_channels=16, stride=1)
y_same = block_same(x_same)

print("Example 1 (Identity Skip):")
print("Input shape :", x_same.shape)
print("Output shape:", y_same.shape)




#%%

# what if we will need to change the shape?
# we will need a projection skip connection (1x1 conv with stride and/or channel change
# this is simple to do - we just need to add a condition


class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection used if spatial size or channel count changes.
        if stride != 1 or in_channels != out_channels: # can there be more conditions?
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, # this magical kernel size 1
                                                              # let's revise - do we really understand what we are doing and why?
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # residual addition
        out = self.relu(out)
        return out


#%% Example 2: Shape change (projection skip)
# Demonstrates a block that changes channels and spatial resolution; needs projection.
x_proj = torch.randn(2, 16, 32, 32)
block_proj = SimpleResidualBlock(in_channels=16, out_channels=32, stride=2)
y_proj = block_proj(x_proj)

print("Example 2 (Projection Skip):")
print("Input shape :", x_proj.shape)
print("Output shape:", y_proj.shape)
print("Skip type    : projection (1x1 conv with stride=2)")


#%% ResNet18 Architecture (from scratch, minimal)
# We now build a simple ResNet18 implementation
# We will not explain it for now, but it will be compatible with torchvision
# so that we might do one more fun trick later

class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

# a "factory" method, to make our lives easier
def _make_layer(block_cls, in_channels, out_channels, blocks, stride):
    layers = []
    layers.append(block_cls(in_channels, out_channels, stride=stride))
    for _ in range(1, blocks):
        layers.append(block_cls(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class SimpleResNet18(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.in_channels = 64  # initial configuration
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet18 layers configuration: [2,2,2,2]
        self.layer1 = _make_layer(ResNetBasicBlock, self.in_channels, self.in_channels, blocks=2, stride=1)
        self.layer2 = _make_layer(ResNetBasicBlock, self.in_channels, self.in_channels*2, blocks=2, stride=2)
        self.layer3 = _make_layer(ResNetBasicBlock, self.in_channels*2, self.in_channels*4, blocks=2, stride=2)
        self.layer4 = _make_layer(ResNetBasicBlock, self.in_channels*4, self.in_channels*8, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels*8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


#%% Lightning Module wrapping SimpleResNet18
# as simple as possible - we don't need anything fancy for now
class ResNet18Lightning(LightningModule):
    def __init__(self, num_classes=9, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleResNet18(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


#%% Config variables
EPOCHS = 1  # change as needed
LR = 1e-3  # just to start
BATCH_SIZE = 32 # change depending on GPU memory


#%% Training from scratch (single epoch demo)
if __name__ == '__main__':
    data_module = WasteDataModuleMulti(batch_size=BATCH_SIZE)
    data_module.setup()
    resnet_model = ResNet18Lightning(num_classes=data_module.num_classes, lr=LR)
    trainer = Trainer(max_epochs=EPOCHS, accelerator='auto')

    #%%
    # let's train
    trainer.fit(resnet_model, datamodule=data_module)

    #%%
    # and now testing
    trainer.test(resnet_model, datamodule=data_module)


    #%% Forward pass sanity check (no training) with random tensor
    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_logits = resnet_model(dummy_input)
    print("Sanity check logits shape (should be [2, 9]):", dummy_logits.shape)


    #%% Transfer Learning: load ImageNet pretrained weights, freeze backbone, new head

    '''
    Now the best part about using established techniques like ResNet - someone has already done it!
    We can take advantage of that and use the "pretrained weights". 
    What are those weights and why are they special? That is a multipart questions. 

    First of all - do we alwas have "enough" data to train a deep neural network from scratch?
    The answer is usually no. Deep neural networks have millions of parameters, and training them from scratch requires massive datasets.
    Can we then just collect more data? Usually also no - our problem might be too specific, too niche, too new OR we just might not have enough
    time or resources to do that. But big companies or research centres do, and they sometimes share their data.

    Secondly - compute resources. Even IF we have enough data, do we have enough compute power to process it? Probably also no.
    But again, others do, and they sometimes share their trained models.

    We can therefore try to borrow their knowledge by taking their trained models and adapting them to our specific task.
    This is called Transfer Learning..
    We will demonstrate it now by taking a ResNet18 model pretrained on ImageNet dataset.


    '''
    #%% 
    # let's start by importing those weights and model definition
    from torchvision.models import resnet18, ResNet18_Weights

    #%%
    # now we can try and extract them
    def load_pretrained_backbone(target_model: SimpleResNet18):
        weights = ResNet18_Weights.IMAGENET1K_V1
        pretrained = resnet18(weights=weights)
        # ramainder - what is "state_dict"?
        # A state_dict is a Python dictionary object that maps each layer to its parameter tensor.
        # It is used to save or load models in PyTorch.
        state_dict = pretrained.state_dict()
        # let's print the state_dict keys to understand its structure
        print("Pretrained state_dict keys:", state_dict.keys())


        # Remove final fc weights/bias
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')

        missing, unexpected = target_model.load_state_dict(state_dict, strict=False)
        print("Loaded pretrained backbone. Missing:", missing, "Unexpected:", unexpected)


    #%% 
    # and apply them to our model

    transfer_model = ResNet18Lightning(num_classes=data_module.num_classes, lr=LR)
    load_pretrained_backbone(transfer_model.model)


    #%%
    # we can now do one more thing - "freeze" some of the layers
    # they will become non-trainable
    # now why would we want to do that?
    # TODO: discuss

    # Freeze all except final fc
    for name, param in transfer_model.model.named_parameters():
        print("processing layer", name)
        if not name.startswith('fc'):
            print(" - freezing")
            param.requires_grad = False

    #%% let's create a new trainer
    trainer_tl = Trainer(max_epochs=EPOCHS, accelerator='auto')

    #%%
    # ok, now let's train our transfer learning model
    trainer_tl.fit(transfer_model, datamodule=data_module)
    #%%
    # and test what happens 
    trainer_tl.test(transfer_model, datamodule=data_module)

    # %%
    '''
    So, what was the result?
    Is it better than training from scratch? Was it faster?

    If this did not work, then it is easy to say why - our problem is totally different, right?
    But if this did work, then it becomes confusing, because our problem is totally different, right? Or maybe it isn't?


    What are the pros and cons of transfer learning?

    '''