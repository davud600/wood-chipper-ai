import torch
import torch.nn as nn

from .models.llm_model import ReaderModel
from .models.cnn_model import CNNModel


class FusionModel(nn.Module):
    """
    Multimodal fusion model combining LLM and CNN outputs.

    This model combines the predictions from a CNN (image-based) and an LLM
    (text-based) using a small feedforward neural network. It’s designed to
    classify whether a page is the first in a new document.

    Parameters
    ----------
    image_size : tuple of int, optional
        The expected input size for the CNN.

    cnn_model : nn.Module, optional
        A custom CNN model to use instead of the default.

    reader_model : nn.Module, optional
        A custom LLM model to use instead of the default.

    dropout : float, optional
        Dropout probability applied to the fusion MLP.

    Methods
    -------
    forward(input_ids, attention_mask, cnn_inputs)
        Computes the fused prediction from the text and image inputs.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (256, 256),
        cnn_model: nn.Module | None = None,
        reader_model: nn.Module | None = None,
        dropout: float = 0.2,
    ):
        super(FusionModel, self).__init__()
        self.reader_model = reader_model or ReaderModel()
        self.cnn_model = cnn_model or CNNModel(image_size=image_size)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(
        self, input_ids, attention_mask, cnn_inputs, distance, return_all_logits=False
    ):
        llm_logits = self.reader_model(input_ids, attention_mask, distance)  # (B, 1)
        cnn_logits = self.cnn_model(cnn_inputs, distance)  # (B, 1)

        if self.training:
            distance += torch.randn_like(distance) * 0.001

        # 🔥 MLP fusion CNN + LLM inputs & other inputs.
        cnn = torch.sigmoid(cnn_logits)
        llm = torch.sigmoid(llm_logits)
        abs_diff = abs(cnn - llm)
        stack = torch.cat(
            [
                cnn_logits,
                llm_logits,
                cnn,
                llm,
                abs_diff,
                distance,
            ],
            dim=1,
        )

        # stack = torch.cat(
        #     [
        #         torch.tanh(llm_logits),
        #         torch.tanh(cnn_logits),
        #         distance,
        #     ],
        #     dim=1,
        # )

        fused_logits = self.fusion_mlp(stack)

        if return_all_logits:
            return fused_logits, llm_logits, cnn_logits

        return fused_logits  # (B, 1)
