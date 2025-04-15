from transformers import AutoModel, AutoConfig

import torch.nn as nn
import torch


class ReaderModel(nn.Module):
    def __init__(
        self, model_name: str = "distilbert-base-uncased", dropout: float = 0.2
    ):
        super(ReaderModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.backbone.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(dropout)
        # self.classifier = nn.Linear(self.config.hidden_size + 1, 1)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, distance):
        # Get [CLS] representation from DistilBERT (uses first token's output)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = (
            outputs.last_hidden_state
        )  # shape: (batch_size, seq_len, hidden_dim)
        cls_rep = hidden_state[:, 0]  # shape: (batch_size, hidden_dim)

        # x = torch.cat([cls_rep, distance], dim=1)
        x = torch.cat([cls_rep], dim=1)

        dropped = self.dropout(x)
        logits = self.classifier(dropped)  # shape: (batch_size, 1)

        return logits  # (B, 1)
