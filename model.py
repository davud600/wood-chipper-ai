import torch.nn.functional as F
import torch.nn as nn
import torch

from utils import max_length


class PageClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = max_length,
        output_size: int = 2,
        hidden_size: int = 1024,
        vocab_size: int = 50001,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.fc = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size // 2)
        self.hidden_2 = nn.Linear(hidden_size // 2, output_size)
        self.attn = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(0.1)
        self._initialize_weights()

    def _initialize_weights(self):
        # nn.init.xavier_uniform_(self.fc.weight)
        # nn.init.xavier_uniform_(self.hidden.weight)
        # nn.init.xavier_uniform_(self.hidden_2.weight)
        nn.init.kaiming_uniform_(self.fc.weight)
        nn.init.kaiming_uniform_(self.hidden.weight)
        nn.init.kaiming_uniform_(self.hidden_2.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.hidden_2.bias)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor | None = None):
        embedded = self.embedding(inputs)

        # Attention mechanism
        attn_weights = F.softmax(self.attn(embedded), dim=1)
        pooled = (embedded * attn_weights).sum(dim=1)

        z = self.fc(pooled)
        z = self.dropout(F.relu(z))
        z = self.hidden(z)
        z = self.dropout(F.relu(z))
        z = self.hidden_2(z)

        return z
