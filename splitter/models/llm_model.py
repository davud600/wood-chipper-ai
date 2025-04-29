from transformers import AutoModel, AutoConfig

import torch
import torch.nn as nn

from ..config import device
from ..utils import init_weights


class ReaderModel(nn.Module):
    def __init__(
        self,
        model_name: str = "allenai/longformer-base-4096",
        dropout: float = 0.1,
        tokenizer_len: int | None = None,
    ):
        super(ReaderModel, self).__init__()
        self.title = "llm"
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.backbone.gradient_checkpointing_enable()
        if tokenizer_len is not None:
            self.backbone.resize_token_embeddings(tokenizer_len)

        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self.apply(init_weights)

    def forward(self, data, loss_fn=None):
        outputs = self.backbone(
            input_ids=data["input_ids"].to(device),
            attention_mask=data["attention_mask"].to(device),
        )
        hidden_state = outputs.last_hidden_state
        cls_rep = torch.cat(
            [hidden_state[:, 0], hidden_state.mean(dim=1)],
            dim=-1,
        )

        logits = self.classifier(cls_rep)  # (b, 1)

        if loss_fn:
            return logits, loss_fn(logits, data["labels"].to(device))

        return logits
