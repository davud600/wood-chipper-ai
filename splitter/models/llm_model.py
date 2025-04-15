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
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    def forward(self, data, loss_fn=None):
        # Get [CLS] representation from DistilBERT (uses first token's output)
        outputs = self.backbone(
            input_ids=data["input_ids"], attention_mask=data["attention_mask"]
        )
        hidden_state = outputs.last_hidden_state  # shape: (b, seq_len, hidden_dim)
        cls_rep = hidden_state[:, 0]  # shape: (b, hidden_dim)

        x = torch.cat([cls_rep], dim=1)

        dropped = self.dropout(x)
        logits = self.classifier(dropped)  # (b, 1)

        # debugging - start
        true_labels = data["labels"][:1].cpu().numpy()
        pred_probs = torch.sigmoid(logits[:1]).detach().cpu().numpy()
        print(f"[DEBUG] True labels: {true_labels.squeeze(1)}")
        print(f"[DEBUG] LLM pred: {pred_probs.squeeze(1)}")
        # debugging - start

        if loss_fn:
            return logits, loss_fn(logits, data["labels"])

        return logits
