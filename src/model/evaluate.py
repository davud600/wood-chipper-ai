from sklearn.metrics import confusion_matrix, f1_score
from transformers import AutoTokenizer

import numpy as np
import torch

from src.model.model import SplitterModel
from src.utils import (
    SPLITTER_MODEL_PATH,
    TESTING_DATA_CSV,
    get_dataset,
    max_length,
)

if __name__ == "__main__":
    true = np.array([])
    pred = np.array([])

    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/longformer-base-4096", device="cuda"
    )

    testing_dataset, _, _ = get_dataset(path=TESTING_DATA_CSV, mini_batch_size=1)
    # testing_dataset, _, _ = get_dataset(path="../example.csv", mini_batch_size=1)

    model = SplitterModel().to("cuda")
    model.eval()
    model.load_state_dict(
        torch.load(SPLITTER_MODEL_PATH, weights_only=False, map_location="cuda")
    )

    for mini_batch in testing_dataset:
        tokenized = tokenizer(
            mini_batch["features"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        features = tokenized.input_ids.to("cuda")
        labels = torch.stack(
            [
                torch.tensor(1 if label == 1 else 0, dtype=torch.long)
                for label in mini_batch["labels"]
            ]
        ).to("cuda")

        with torch.amp.autocast_mode.autocast(device_type="cuda", dtype=torch.float16):
            loss, logit = model(features, labels)

            true = np.append(true, [labels.to("cpu").detach().numpy()[0]])
            pred = np.append(pred, [int(logit > 0)])

    print("\n")
    correct = np.sum(true == pred)
    print(f"first pages: {np.sum(true == 1)}")
    print(f"other pages: {np.sum(true != 1)}")
    print(f"pages correct: {correct} / {len(true)}")
    print(f"accuracy: {round(correct / len(true), 4)}")
    print(confusion_matrix(true, pred))
    f1 = f1_score(true, pred)
    print(f"F1 score: {round(f1, 4)}")
    print("\n")
