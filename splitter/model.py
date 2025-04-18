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
        dropout: float = 0.2,
    ):
        super(FusionModel, self).__init__()
        self.reader_model = reader_model or ReaderModel()
        self.cnn_model = cnn_model or CNNModel(image_size=image_size)
        # self.fusion_mlp = nn.Sequential(
        #     # nn.Linear(3, 4),
        #     nn.Linear(2, 4),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(4, 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(2, 1),
        # )

        # self.fusion_mlp = nn.Sequential(
        #     nn.Linear(2, 1),
        # )

        # self.fusion_mlp = nn.Sequential(
        #     nn.Linear(3, 8),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(8, 1),
        # )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(3, 1),
        )

        self.apply(init_weights)

    def forward(self, data, loss_fn=None, warmup=False):
        llm_logits = self.reader_model(data)  # (B, 1)
        cnn_logits = self.cnn_model(data)  # (B, 1)

        distance = data["distance"]
        # if self.training:
        #     distance += torch.randn_like(data["distance"]) * 0.001

        # 🔥 MLP fusion CNN + LLM inputs & other inputs.
        # cnn = torch.sigmoid(cnn_logits)
        # llm = torch.sigmoid(llm_logits)
        stack = torch.cat(
            [
                distance.to(device),
                cnn_logits,
                llm_logits,
                # cnn,
                # llm,
                # abs(cnn - llm),
            ],
            dim=1,
        )
        logits = self.fusion_mlp(stack)

        # # debug - start
        # fusion_pred_probs = torch.sigmoid(logits[:1]).detach().cpu().numpy()
        # print(
        #     f"[DEBUG MLP] label: {data['labels'].squeeze(1)[0]} - pred: {fusion_pred_probs.squeeze(1)[0]}"
        # )
        # # debug - end

        if loss_fn:
            labels = data["labels"].to(device)

            # return logits, loss_fn(logits, labels)

            # if warmup:
            #     return logits, loss_fn(logits, labels)
            #

            fused_loss = loss_fn(logits, labels)
            aux_llm_loss = loss_fn(llm_logits, labels)
            aux_cnn_loss = loss_fn(cnn_logits, labels)

            alpha = 0.75
            loss = fused_loss + alpha * (aux_llm_loss + aux_cnn_loss)

            return logits, loss

        return logits  # (B, 1)
