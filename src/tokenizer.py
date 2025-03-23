from collections import Counter

# from torch.utils.data import Dataset, DataLoader
# import torch.optim as optim
# import torch.nn as nn
# import torch
# import re

corpus_file = "./corpus.txt"

# import csv

# content = ""
# with (
#     open(TRAINING_DATA_CSV, mode="r", encoding="utf-8") as train_file,
#     open(TESTING_DATA_CSV, mode="r", encoding="utf-8") as test_file,
# ):
#     train_reader = csv.reader(train_file)
#     test_reader = csv.reader(test_file)
#     for row in train_reader:
#         if row[0] == "content":  # skip headers.
#             continue
#
#         content += str(row[0])
#     for row in test_reader:
#         if row[0] == "content":  # skip headers.
#             continue
#
#         content += str(row[0])
#
# with open(corpus_file, mode="w", encoding="utf-8") as file:
#     file.write(content)


class CustomTokenizer:
    def __init__(self, special_tokens=None):
        self.token_to_idx = {}
        self.idx_to_token = []
        self.special_tokens = special_tokens if special_tokens else []

    def tokenize(self, text):
        import re

        # Basic word-level tokenization
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, corpus_file, max_vocab_size=None):
        with open(corpus_file, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = self.tokenize(text)

        # Count token frequencies
        token_counts = Counter(tokens)

        # Get the most common tokens
        if max_vocab_size:
            most_common = token_counts.most_common(
                max_vocab_size - len(self.special_tokens)
            )
            vocab_tokens = [token for token, _ in most_common]
        else:
            vocab_tokens = list(token_counts.keys())

        # Prepend special tokens to vocab if they're not already there
        self.idx_to_token = self.special_tokens + vocab_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def encode(self, text, max_length=4096):
        tokens = self.tokenize(text)
        tokens = tokens[:max_length]  # truncate if necessary
        # Map tokens to indices; use a default (e.g., 0) for unknown tokens
        return [self.token_to_idx.get(token, 0) for token in tokens]

    def decode(self, indices):
        return " ".join([self.idx_to_token[idx] for idx in indices])

    def vocab_size(self):
        return len(self.token_to_idx)


special_tokens = [
    "<curr_page>",
    "</curr_page>",
    "<next_page_1>",
    "</next_page_1>",
    "<next_page_2>",
    "</next_page_2>",
    "<next_page_3>",
    "</next_page_3>",
    "<next_page_4>",
    "</next_page_4>",
    "<next_page_5>",
    "</next_page_5>",
    "<next_page_6>",
    "</next_page_6>",
    "<next_page_7>",
    "</next_page_7>",
]
tokenizer = CustomTokenizer(special_tokens=special_tokens)
tokenizer.fit(corpus_file, max_vocab_size=50000)
