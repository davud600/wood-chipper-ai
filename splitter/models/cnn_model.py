import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import prev_pages_to_append, pages_to_append


class CNNModel(nn.Module):
    def __init__(self, image_size=(256, 256), dropout: float = 0.3):
        super().__init__()

        # Calculate number of channels based on context
        self.in_channels = prev_pages_to_append + 1 + pages_to_append

        # Lightweight CNN
        self.conv1 = nn.Conv2d(self.in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        # Dynamically determine flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, *image_size)
            dummy_out = self._forward_conv(dummy)
            self.flattened_dim = dummy_out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.dropout(x)

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)
