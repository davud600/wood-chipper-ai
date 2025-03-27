import torch.nn.functional as F
import torch.nn as nn
import torch

from src.utils import max_length


class SplitterModel(nn.Module):
    def __init__(
        self,
        input_size: int = max_length,
        output_size: int = 2,
        hidden_size: int = 2048,
        vocab_size: int = 60000,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.fc = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size, hidden_size)
        self.hidden_3 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(0.2)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.hidden_2.weight)
        nn.init.xavier_uniform_(self.hidden_3.weight)
        nn.init.xavier_uniform_(self.out.weight)
        # nn.init.kaiming_uniform_(self.fc.weight)
        # nn.init.kaiming_uniform_(self.hidden.weight)
        # nn.init.kaiming_uniform_(self.hidden_2.weight)
        # nn.init.kaiming_uniform_(self.page_out.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.hidden_2.bias)
        nn.init.zeros_(self.hidden_3.bias)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor | None = None,
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
        x = self.hidden_3(x)
        x = self.dropout(F.relu(x))

        logits = self.out(x)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

            return (
                loss,
                logits,
            )

        return logits
