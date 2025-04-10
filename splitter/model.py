import torch
import torch.nn as nn
from .models.llm_model import ReaderModel
from .models.cnn_model import CNNModel


class FusionModel(nn.Module):
    def __init__(
        self,
        cnn_model: nn.Module | None = None,
        reader_model: nn.Module | None = None,
    ):
        super(FusionModel, self).__init__()
        self.reader_model = reader_model or ReaderModel()
        self.cnn_model = cnn_model or CNNModel()

        # 🔥 Spicy MLP: 2 inputs → 16 hidden → 1 output
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, input_ids, attention_mask, cnn_inputs):
        # LLM logits
        llm_logits = self.reader_model(
            input_ids=input_ids, attention_mask=attention_mask
        )  # (B,)

        # CNN logits
        cnn_logits = self.cnn_model(cnn_inputs)  # (B,)

        # Stack logits → (B, 2)
        stacked_logits = torch.stack([llm_logits, cnn_logits], dim=1)

        # 🔥 MLP fusion
        fused_logits = self.fusion_mlp(stacked_logits).squeeze(-1)  # (B,)
        return fused_logits


# import torch.nn as nn
#
# from .models.cnn_model import CNNModel
# from .models.llm_model import ReaderModel
#
#
# class FusionModel(nn.Module):
#     def __init__(
#         self, cnn_model: nn.Module | None = None, reader_model: nn.Module | None = None
#     ):
#         super(FusionModel, self).__init__()
#         self.reader_model = reader_model or ReaderModel()
#         self.cnn_model = cnn_model or CNNModel()
#
#     def forward(self, input_ids, attention_mask, cnn_inputs):
#         # LLM logits
#         llm_logits = self.reader_model(
#             input_ids=input_ids, attention_mask=attention_mask
#         )  # (B,)
#
#         # CNN logits
#         cnn_logits = self.cnn_model(cnn_inputs)
#
#         # Functional fusion: simple average
#         fused_logits = (llm_logits + cnn_logits) / 2.0
#         return fused_logits
