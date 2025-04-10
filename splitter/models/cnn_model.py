import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class CNNModel(nn.Module):
    def __init__(self, image_size=(256, 256)):
        super().__init__()

        # Calculate number of channels based on context
        self.in_channels = config.prev_pages_to_append + 1 + config.pages_to_append

        # Lightweight CNN
        self.conv1 = nn.Conv2d(self.in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

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
        return self.fc2(x).squeeze(-1)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class CNNModel(nn.Module):
#     def __init__(self, in_channels=1, image_size=(256, 256)):
#         super().__init__()
#
#         # Lightweight CNN
#         self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.25)
#
#         # Determine flattened size dynamically
#         with torch.no_grad():
#             dummy = torch.zeros(1, in_channels, *image_size)
#             dummy = self._forward_conv(dummy)
#             self.flattened_dim = dummy.view(1, -1).shape[1]
#
#         self.fc1 = nn.Linear(self.flattened_dim, 64)
#         self.fc2 = nn.Linear(64, 1)
#
#     def _forward_conv(self, x):
#         x = self.pool(F.relu(self.conv1(x)))  # Downsample
#         x = self.pool(F.relu(self.conv2(x)))  # Downsample again
#         return self.dropout(x)
#
#     def forward(self, x):
#         x = self._forward_conv(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x).squeeze(-1)
