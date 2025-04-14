import torch
import torch.nn as nn

from config.settings import pages_to_append, prev_pages_to_append
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
        dropout: float = 0.1,
    ):
        super(FusionModel, self).__init__()
        self.reader_model = reader_model or ReaderModel()
        self.cnn_model = cnn_model or CNNModel(image_size=image_size)

        # 🔥 Spicy MLP: (prev + 1 + next) * 2 + 1(distance) input → 8 hidden → prev + 1 + next outputs
        self.fusion_mlp = nn.Sequential(
            nn.Linear((prev_pages_to_append + 1 + pages_to_append) * 2 + 1, 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, prev_pages_to_append + 1 + pages_to_append),
        )

    ## FUSED ###
    def forward(
        self, input_ids, attention_mask, cnn_inputs, distance, return_all_logits=False
    ):
        # LLM logits
        llm_logits = self.reader_model(
            input_ids=input_ids, attention_mask=attention_mask
        )  # (B,)

        # CNN logits
        cnn_logits = self.cnn_model(cnn_inputs)  # (B,)
        # print(
        #     f"[Fusion] LLM: {llm_logits.detach().cpu().numpy()}, CNN: {cnn_logits.detach().cpu().numpy()}"
        # )

        distance = distance.squeeze(-1)
        if self.training:
            distance += torch.randn_like(distance) * 0.05

        # Stack logits → (B, (prev + 1 + next) * 2 + 1(distance))
        stacked_logits = torch.cat(
            [llm_logits, cnn_logits, distance.unsqueeze(-1)], dim=1
        )

        # 🔥 MLP fusion
        fused_logits = self.fusion_mlp(stacked_logits).squeeze(-1)  # (B,)

        if return_all_logits:
            return fused_logits, llm_logits, cnn_logits

        return fused_logits
