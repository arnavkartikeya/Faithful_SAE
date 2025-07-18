import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class ColorMNISTCNN(nn.Module):
    def __init__(self, input_size=28):
        super(ColorMNISTCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        conv_output_size = (input_size // 8) * (input_size // 8) * 128
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.pool(F.relu(self.conv3(x))) 
        
        x = self.adaptive_pool(x)  
        x = x.view(x.size(0), -1)  
        
        # MLP layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  
        
        return x