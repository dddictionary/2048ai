import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape=(4, 4), num_actions=4):
        super(DQN, self).__init__()
        
        # Flatten input
        self.flatten = nn.Flatten()
        
        # Convolutional layers to extract features
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        
        # Fully connected layers
        def conv2d_size_out(size, kernel_size=2, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(4))
        linear_input_size = convw * convw * 64
        
        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        # Ensure input is a 4D tensor: [batch_size, channels, height, width]
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)