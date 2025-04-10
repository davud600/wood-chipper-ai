import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerModel


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


class SplitterModel(nn.Module):
    def __init__(self, pos_weight: float = 1.0, use_focal_loss: bool = False):
        super().__init__()
        self.encoder = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        self.hidden_size = self.encoder.config.hidden_size
        self.use_focal_loss = use_focal_loss
        self.pos_weight = pos_weight

        # Classifier head
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(self.hidden_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.out.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.out.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        # Encoder
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0]  # [CLS]

        # Classification head
        x = self.norm(cls_output)
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(F.gelu(self.fc2(x)))
        logits = self.out(x).squeeze(-1)

        if labels is not None:
            labels = labels.float()
            if self.use_focal_loss:
                loss_fn = FocalLoss()
                loss = loss_fn(logits, labels)
            else:
                pos_weight_tensor = torch.tensor(
                    [self.pos_weight], device=logits.device
                )
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
                loss = loss_fn(logits, labels)

            return loss, torch.sigmoid(logits)

        return torch.sigmoid(logits)
