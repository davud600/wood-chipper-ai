import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import prev_pages_to_append, pages_to_append


class CNNModel(nn.Module):
    def __init__(self, image_size=(256, 256), dropout: float = 0.1):
        super().__init__()

        self.in_channels = prev_pages_to_append + 1 + pages_to_append

        # ⚖️ Medium complexity CNN
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 256 → 128
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 → 64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 → 32
            nn.Dropout2d(dropout),
        )

        # Dynamic flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, *image_size)
            dummy_out = self.conv_block(dummy)
            self.flattened_dim = dummy_out.view(1, -1).shape[1]

        # 💡 Smaller MLP head
        self.fc1 = nn.Linear(self.flattened_dim, 64)
        self.fc2 = nn.Linear(64, prev_pages_to_append + 1 + pages_to_append)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # (B, 3)
