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
        dropout: float = 0.3,
    ):
        super(FusionModel, self).__init__()
        self.reader_model = reader_model or ReaderModel()
        self.cnn_model = cnn_model or CNNModel(image_size=image_size)

        # 🔥 Spicy MLP: 2 inputs → 16 hidden → 1 output
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    ## FUSED ###
    def forward(self, input_ids, attention_mask, cnn_inputs, return_all_logits=False):
        # LLM logits
        llm_logits = self.reader_model(
            input_ids=input_ids, attention_mask=attention_mask
        )  # (B,)

        # CNN logits
        cnn_logits = self.cnn_model(cnn_inputs)  # (B,)
        # print(
        #     f"[Fusion] LLM: {llm_logits.detach().cpu().numpy()}, CNN: {cnn_logits.detach().cpu().numpy()}"
        # )

        # Stack logits → (B, 2)
        stacked_logits = torch.stack([llm_logits, cnn_logits], dim=1)

        # 🔥 MLP fusion
        fused_logits = self.fusion_mlp(stacked_logits).squeeze(-1)  # (B,)

        if return_all_logits:
            return fused_logits, llm_logits, cnn_logits

        return fused_logits

    # ### ONLY LLM ###
    # def forward(self, input_ids, attention_mask, cnn_inputs):
    #     # LLM logits
    #     llm_logits = self.reader_model(
    #         input_ids=input_ids, attention_mask=attention_mask
    #     )  # (B,)
    #
    #     return llm_logits

    # ### ONLY CNN ###
    # def forward(self, input_ids, attention_mask, cnn_inputs):
    #     # CNN logits
    #     cnn_logits = self.cnn_model(cnn_inputs)  # (B,)
    #
    #     return cnn_logits
