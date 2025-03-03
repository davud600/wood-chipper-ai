from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer
import numpy as np
import torch.nn as nn
import torch

from model import PageClassifier
from utils import (
    TESTING_DATA_CSV,
    get_dataset,
    max_length,
)

if __name__ == "__main__":
    y_true = np.array([])
    y_pred = np.array([])

    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/longformer-base-4096", device="cuda"
    )

    testing_dataset = get_dataset(path=TESTING_DATA_CSV, mini_batch_size=1)

    page_classifier = PageClassifier().to("cuda")
    page_classifier.eval()
    page_classifier.load_state_dict(
        torch.load(
            "./model/page_classifier.pth", weights_only=False, map_location="cuda"
        )
    )
    loss_fn = nn.CrossEntropyLoss()

    loss = None
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
            predicted_pages = page_classifier(features)
            loss = loss_fn(predicted_pages, labels)
            # print(f"loss - {loss}")

            y_true = np.append(y_true, [labels.to("cpu").detach().numpy()[0]])
            y_pred = np.append(
                y_pred, [torch.argmax(predicted_pages).to("cpu").detach().numpy()]
            )
            # print(f"y_true: {y_true[-1]} - y_pred: {y_pred[-1]}")

    correct = np.sum(y_true == y_pred)
    print(f"first pages: {np.sum(y_true == 1)}")
    print(f"other pages: {np.sum(y_true != 1)}")
    print(f"correct: {correct} / {len(y_true)}")
    print(f"accuracy: {round(correct / len(y_true), 4)}")
    print(confusion_matrix(y_true, y_pred))
