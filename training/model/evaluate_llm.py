from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from transformers import AutoTokenizer

import numpy as np
import torch

from config import SPLITTER_MODEL_PATH, TESTING_DATA_CSV, max_length
from .model import SplitterModel
from training.dataset import get_dataset
from utils import parse_formatted_content_batch_to_sections


def encode_input(prev, curr, next, tokenizer):
    full_input = prev + tokenizer.sep_token + curr + tokenizer.sep_token + next
    encoded = tokenizer(
        full_input,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return encoded["input_ids"].to("cuda"), encoded["attention_mask"].to("cuda")


if __name__ == "__main__":
    true = []
    pred = []

    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/longformer-base-4096", device="cuda"
    )

    testing_dataset, _, _ = get_dataset(
        path=TESTING_DATA_CSV,
        mini_batch_size=1,  # one sample per batch
    )

    model = SplitterModel().to("cuda")
    model.eval()
    model.load_state_dict(
        torch.load(SPLITTER_MODEL_PATH, weights_only=False, map_location="cuda")
    )

    with torch.no_grad():
        for mini_batch in testing_dataset:
            for feature, label in zip(mini_batch["features"], mini_batch["labels"]):
                prev, curr, next = parse_formatted_content_batch_to_sections(
                    str(feature)
                )
                input_ids, attention_mask = encode_input(prev, curr, next, tokenizer)

                with torch.amp.autocast_mode.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    output = model(input_ids=input_ids, attention_mask=attention_mask)
                    score = output.item()

                pred_label = int(score >= 0.3)
                true.append(int(label))
                pred.append(pred_label)

    true = np.array(true)
    pred = np.array(pred)

    print("\n=== Evaluation Results ===")
    print(f"first pages (true=1): {np.sum(true == 1)}")
    print(f"other pages (true=0): {np.sum(true == 0)}")
    print(f"pages correct: {np.sum(true == pred)} / {len(true)}")
    print(f"accuracy: {accuracy_score(true, pred):.4f}")
    print(f"precision: {precision_score(true, pred):.4f}")
    print(f"recall: {recall_score(true, pred):.4f}")
    print(f"f1 score: {f1_score(true, pred):.4f}")
    print("confusion matrix:")
    print(confusion_matrix(true, pred))
    print("==========================\n")
