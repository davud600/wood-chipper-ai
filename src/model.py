import torch.nn.functional as F
import torch.nn as nn
import torch

from utils import max_length


class ClassifierModel(nn.Module):
    def __init__(
        self,
        input_size: int = max_length,
        page_output_size: int = 2,
        type_output_size: int = 12,
        hidden_size: int = 2048,
        vocab_size: int = 60000,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.fc = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size, hidden_size // 2)
        self.page_out = nn.Linear(hidden_size // 2, page_output_size)
        self.type_out = nn.Linear(hidden_size // 2, type_output_size)
        self.attn = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(0.2)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.hidden_2.weight)
        nn.init.xavier_uniform_(self.page_out.weight)
        nn.init.xavier_uniform_(self.type_out.weight)
        # nn.init.kaiming_uniform_(self.fc.weight)
        # nn.init.kaiming_uniform_(self.hidden.weight)
        # nn.init.kaiming_uniform_(self.hidden_2.weight)
        # nn.init.kaiming_uniform_(self.page_out.weight)
        # nn.init.kaiming_uniform_(self.type_out.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.hidden_2.bias)
        nn.init.zeros_(self.page_out.bias)
        nn.init.zeros_(self.type_out.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        page_labels: torch.Tensor | None = None,
        type_labels: torch.Tensor | None = None,
    ):
        embedded = self.embedding(inputs)
        attn_weights = F.softmax(self.attn(embedded), dim=1)
        pooled = (embedded * attn_weights).sum(dim=1)

        x = self.fc(pooled)
        x = self.dropout(F.relu(x))
        x = self.hidden(x)
        x = self.dropout(F.relu(x))
        x = self.hidden_2(x)
        x = self.dropout(F.relu(x))

        page_logits = self.page_out(x)
        type_logits = self.type_out(x)

        if page_labels is not None and type_labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            page_loss = loss_fn(page_logits, page_labels)
            type_loss = loss_fn(type_logits, type_labels)
            total_loss = page_loss + type_loss

            return (
                total_loss,
                page_logits,
                type_logits,
            )

        return page_logits, type_logits
