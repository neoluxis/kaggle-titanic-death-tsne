import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import tqdm

class Neo_Conv(nn.Module):
    def __init__(self):
        super(Neo_Conv, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv1_s = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        return x
