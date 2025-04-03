import torch
import pickle
import re

from collections import Counter

from src.utils import special_tokens, max_vocab_size, CORPUS_FILE, TOKENIZER_PATH


class CustomTokenizer:
    def __init__(self, special_tokens=None):
        self.token_to_idx = {}
        self.idx_to_token = []
        self.special_tokens = special_tokens if special_tokens else []

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, corpus_file, max_vocab_size=None):
        with open(corpus_file, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = self.tokenize(text)
        token_counts = Counter(tokens)

        if max_vocab_size:
            most_common = token_counts.most_common(
                max_vocab_size - len(self.special_tokens)
            )
            vocab_tokens = [token for token, _ in most_common]
        else:
            vocab_tokens = list(token_counts.keys())

        self.idx_to_token = self.special_tokens + vocab_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def encode(self, text, max_length=4096):
        tokens = self.tokenize(text)
        tokens = tokens[:max_length]

        return [self.token_to_idx.get(token, 0) for token in tokens]

    def decode(self, indices):
        return " ".join([self.idx_to_token[idx] for idx in indices])

    def vocab_size(self):
        return len(self.token_to_idx)

    def __call__(self, batch_texts: list[str], max_length=4096, padding="max_length"):
        batch_ids = []

        for text in batch_texts:
            token_ids = self.encode(text, max_length=max_length)
            if padding == "max_length":
                token_ids += [0] * (max_length - len(token_ids))  # pad with 0
            batch_ids.append(token_ids)

        return CustomBatchOutput(torch.tensor(batch_ids, dtype=torch.long))


class CustomBatchOutput:
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


if __name__ == "__main__":
    tokenizer = CustomTokenizer(special_tokens=special_tokens)
    tokenizer.fit(CORPUS_FILE, max_vocab_size=max_vocab_size)

    with open(file=TOKENIZER_PATH, mode="wb") as file:
        pickle.dump(
            tokenizer, file, protocol=None, fix_imports=True, buffer_callback=None
        )
