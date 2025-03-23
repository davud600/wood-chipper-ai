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
    # testing_dataset = get_dataset(path=TRAINING_DATA_CSV, mini_batch_size=1)
    # testing_dataset = get_dataset(path="../example.csv", mini_batch_size=1)

    model = ClassifierModel().to("cuda")
    model.eval()
    model.load_state_dict(
        torch.load("../model/model.pth", weights_only=False, map_location="cuda")
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

        # filter out non-first pages in type labels and features.
        first_page_indices = (page_labels == 1).nonzero(as_tuple=True)[0]
        if first_page_indices.numel() > 0:
            filtered_features = features[first_page_indices]
            filtered_page_labels = page_labels[first_page_indices]
            filtered_type_labels = type_labels[first_page_indices]
        else:
            filtered_features = torch.empty(
                (0, features.shape[1]), dtype=torch.long, device="cuda"
            )
            filtered_page_labels = torch.empty((0,), dtype=torch.long, device="cuda")
            filtered_type_labels = torch.empty((0,), dtype=torch.long, device="cuda")

        with torch.amp.autocast_mode.autocast(device_type="cuda", dtype=torch.float16):
            loss, predicted_pages, _ = model(features, page_labels, type_labels)

            pages_true = np.append(
                pages_true, [page_labels.to("cpu").detach().numpy()[0]]
            )
            pages_pred = np.append(
                pages_pred, [torch.argmax(predicted_pages).to("cpu").detach().numpy()]
            )

            if len(filtered_features) > 0:
                _, _, predicted_types = model(
                    filtered_features, filtered_page_labels, filtered_type_labels
                )

                types_true = np.append(
                    types_true, [filtered_type_labels.to("cpu").detach().numpy()[0]]
                )
                types_pred = np.append(
                    types_pred,
                    [torch.argmax(predicted_types).to("cpu").detach().numpy()],
                )

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
