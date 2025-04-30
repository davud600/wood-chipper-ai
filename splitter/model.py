import torch
import torch.nn as nn

from .config import device
from .utils import init_weights
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
        tokenizer_len: int | None = None,
    ):
        super(FusionModel, self).__init__()
        self.title = "mlp"
        self.reader_model = reader_model or ReaderModel(tokenizer_len=tokenizer_len)
        # self.cnn_model = cnn_model or CNNModel(image_size=image_size)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4, 1),
        )

        self.apply(init_weights)

    def forward(self, data, loss_fn=None, warmup=False, return_all_logits=False):
        llm_logits = self.reader_model(data)  # (b, 1)
        # cnn_logits = self.cnn_model(data)  # (b, 1)
        distance = data["distance"]

        stack = torch.cat(
            [
                distance.to(device),
                # cnn_logits,
                llm_logits,
            ],
            dim=1,
        )
        logits = self.fusion_mlp(stack)

        if loss_fn:
            labels = data["labels"].to(device)

            if warmup:
                return logits, loss_fn(logits, labels)

            fused_loss = loss_fn(logits, labels)
            aux_llm_loss = loss_fn(llm_logits, labels)

            alpha = 0.5
            # loss = fused_loss + alpha * (aux_llm_loss + aux_cnn_loss)
            loss = fused_loss + (alpha * aux_llm_loss)

            # if return_all_logits:
            #     return logits, cnn_logits, llm_logits, loss

            if return_all_logits:
                return logits, llm_logits, loss

            return logits, loss

        return logits  # (b, 1)
