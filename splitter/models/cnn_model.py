import torch.nn.functional as F
import torch.nn as nn
import torch

from torchvision import transforms

from ..config import device
from ..utils import init_weights
from config.settings import prev_pages_to_append, pages_to_append

target_size = (512, 512)


class CNNModel(nn.Module):
    def __init__(self, image_size=(1024, 1024), dropout: float = 0.1):
        super().__init__()
        self.title = "cnn"
        self.in_channels = (prev_pages_to_append + 1 + pages_to_append) * 2
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        self.transform = transforms.Compose(
            [
                # 1) Resize first, so blur kernel is in target pixels
                transforms.Resize(target_size),
                # 2) Gaussian blur; tweak kernel_size & sigma to taste
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.ConvertImageDtype(torch.float16),
            ]
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, *image_size)
            dummy_out = self.conv_block(dummy)
            self.flattened_dim = dummy_out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self.apply(init_weights)

    def forward(self, data, loss_fn=None):
        x = self.conv_block(data["cnn_input"].to(device))
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

        logits = self.classifier(x)  # (b, 1)

        if loss_fn:
            return logits, loss_fn(logits, data["labels"].to(device))

        return logits
