from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer
import numpy as np
import torch

from model import ClassifierModel
from utils import (
    TESTING_DATA_CSV,
    TYPES,
    get_dataset,
    max_length,
)

if __name__ == "__main__":
    pages_true = np.array([])
    pages_pred = np.array([])
    types_true = np.array([])
    types_pred = np.array([])

    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/longformer-base-4096", device="cuda"
    )

    testing_dataset = get_dataset(path=TESTING_DATA_CSV, mini_batch_size=1)

    model = ClassifierModel().to("cuda")
    model.eval()
    model.load_state_dict(
        torch.load("./model/model.pth", weights_only=False, map_location="cuda")
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
        page_labels = torch.stack(
            [
                torch.tensor(1 if label == 1 else 0, dtype=torch.long)
                for label in mini_batch["page_labels"]
            ]
        ).to("cuda")
        type_labels = torch.stack(
            [
                torch.tensor(label, dtype=torch.long)
                for label in mini_batch["type_labels"]
            ]
        ).to("cuda")

        with torch.amp.autocast_mode.autocast(device_type="cuda", dtype=torch.float16):
            loss, predicted_pages, predicted_types = model(
                features, page_labels, type_labels
            )
            # print(f"loss - {loss}")

            pages_true = np.append(
                pages_true, [page_labels.to("cpu").detach().numpy()[0]]
            )
            pages_pred = np.append(
                pages_pred, [torch.argmax(predicted_pages).to("cpu").detach().numpy()]
            )
            types_true = np.append(
                types_true, [type_labels.to("cpu").detach().numpy()[0]]
            )
            types_pred = np.append(
                types_pred, [torch.argmax(predicted_types).to("cpu").detach().numpy()]
            )
            # print(f"pages_true: {pages_true[-1]} - pages_pred: {pages_pred[-1]}")

    print("\n")
    pages_correct = np.sum(pages_true == pages_pred)
    print(f"first pages: {np.sum(pages_true == 1)}")
    print(f"other pages: {np.sum(pages_true != 1)}")
    print(f"pages correct: {pages_correct} / {len(pages_true)}")
    print(f"accuracy: {round(pages_correct / len(pages_true), 4)}")
    print(confusion_matrix(pages_true, pages_pred))
    print("\n")

    types_correct = np.sum(types_true == types_pred)
    for type in list(TYPES.keys()):
        print(f"{type}: {np.sum(types_true == TYPES[type])}")
    print(f"types correct: {types_correct} / {len(types_true)}")
    print(f"accuracy: {round(types_correct / len(types_true), 4)}")
    print(confusion_matrix(types_true, types_pred))
    print("\n")
