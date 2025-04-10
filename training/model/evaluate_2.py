import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from transformers import AutoTokenizer

from config import SPLITTER_MODEL_PATH, TESTING_DATA_CSV, max_length
from .model import SplitterModel
from training.dataset import get_dataset

if __name__ == "__main__":
    true = []
    pred = []

    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")

    testing_dataset, _, _ = get_dataset(
        path=TESTING_DATA_CSV,
        mini_batch_size=1,
    )

    model = SplitterModel().to("cuda")
    model.load_state_dict(torch.load(SPLITTER_MODEL_PATH, map_location="cuda"))
    model.eval()

    with torch.no_grad():
        for mini_batch in testing_dataset:
            for feature, label in zip(mini_batch["features"], mini_batch["labels"]):
                # `feature` already contains <prev>, <curr>, <next>
                encoding = tokenizer(
                    feature,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )

                input_ids = encoding["input_ids"].to("cuda")
                attention_mask = encoding["attention_mask"].to("cuda")

                with torch.amp.autocast_mode.autocast("cuda"):
                    probs = model(input_ids, attention_mask).item()

                pred_label = int(probs >= 0.3)  # use best threshold if you want
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
