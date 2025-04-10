import torch.nn as nn


class PageContextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(5, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
