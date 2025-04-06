import torch.nn.functional as F
import torch.nn as nn
import torch

from config import max_length, max_vocab_size


class SplitterModel(nn.Module):
    # def __init__(
    #     self,
    #     input_size: int = max_length,
    #     output_size: int = 1,
    #     hidden_size: int = 2048,
    #     vocab_size: int = max_vocab_size,
    #     dim_size: int = max_vocab_size,
    #     pos_weight: float = 1.0,
    # ) -> None:
    #     super().__init__()
    #     self.pos_weight = pos_weight
    #     self.embedding = nn.Embedding(vocab_size, input_size)
    #     self.fc = nn.Linear(input_size, hidden_size)
    #     self.hidden = nn.Linear(hidden_size, hidden_size)
    #     self.hidden_2 = nn.Linear(hidden_size, hidden_size // 2)
    #     self.hidden_3 = nn.Linear(hidden_size // 2, hidden_size // 2)
    #     self.out = nn.Linear(hidden_size // 2, output_size)
    #     self.attn = nn.Linear(input_size, 1)
    #     self.dropout = nn.Dropout(0.2)
    #     self._initialize_weights()
    def __init__(
        self,
        embedding_dim: int = 512,
        output_size: int = 1,
        hidden_size: int = 2048,
        vocab_size: int = max_vocab_size,
        pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # much smaller now
        self.fc = nn.Linear(embedding_dim, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size, hidden_size // 2)
        self.hidden_3 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.out = nn.Linear(hidden_size // 2, output_size)
        self.attn = nn.Linear(embedding_dim, 1)
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

        logits = self.out(x).squeeze(-1)

        if labels is not None:
            labels = labels.float()

            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([self.pos_weight], dtype=torch.float16).to(
                    "cuda"
                )
            )
            loss = loss_fn(logits, labels)

            return (
                loss,
                logits,
            )

        return logits
