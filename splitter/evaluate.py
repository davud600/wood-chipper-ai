import os
import torch
import torch.nn as nn
import argparse

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config.settings import (
    SPLITTER_MODEL_PATH,
    SPLITTER_MODEL_DIR,
    TESTING_DATA_CSV,
    DOCUMENT_TYPES,
    BEST_PERF_TYPES,
    image_output_size,
)
from .config import device, testing_mini_batch_size, use_all_types
from .utils import count_classes, evaluate, verify_alignment, load_best_weights
from .model import FusionModel
from .dataset.dataset import DocumentDataset

session_dirs = [
    int(d)
    for d in os.listdir(SPLITTER_MODEL_DIR)
    if os.path.isdir(os.path.join(SPLITTER_MODEL_DIR, d)) and d.isdigit()
]
session = max(session_dirs, default=0)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--by_types", type=bool, default=False)
    p.add_argument(
        "--testing_mini_batch_size", type=int, default=testing_mini_batch_size
    )

    return p.parse_args()


def main():
    """
    Runs evaluation on the test set and visual verification on sample predictions.

    Loads model and tokenizer, prepares test data, runs batch evaluation,
    prints metrics and confusion matrix, and performs sample-level alignment checks.
    """

    global session

    args = parse_args()

    print("[INFO] Initializing...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    print(f"[INFO] Loading model weights from session {session}...")
    model = FusionModel(image_size=image_output_size).to(device)
    load_best_weights(model, session, True)
    # model.load_state_dict(
    #     torch.load(SPLITTER_MODEL_PATH, weights_only=False, map_location="cuda")
    # )
    model.eval()

    if not args.by_types:
        test_dataset = DocumentDataset(
            TESTING_DATA_CSV,
            tokenizer,
            mode="test",
            image_size=image_output_size,
            doc_types=(None if use_all_types else BEST_PERF_TYPES),
        )
        # count_classes(test_dataset)

        test_loader = DataLoader(
            test_dataset,
            batch_size=testing_mini_batch_size * 2,
            num_workers=0,
        )
        loss_fn = nn.BCEWithLogitsLoss()

        print("[INFO] Starting testing...")
        _, acc, rec, prec, f1, cm = evaluate(model, test_loader, loss_fn)
        print(f"  F1: {f1:.4f} | Acc: {acc:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f}")
        print(f"  Confusion Matrix:\n{cm}\n")

        # for i in range(50):
        #     verify_alignment(model, tokenizer, test_dataset, i)

        return

    for doc_type in list(DOCUMENT_TYPES.values()):
        print(f"[{list(DOCUMENT_TYPES.keys())[doc_type]}]")
        test_dataset = DocumentDataset(
            TESTING_DATA_CSV,
            tokenizer,
            mode="test",
            image_size=image_output_size,
            doc_types=[doc_type],
        )
        # count_classes(test_dataset)

        test_loader = DataLoader(
            test_dataset,
            batch_size=testing_mini_batch_size * 2,
            num_workers=0,
        )
        loss_fn = nn.BCEWithLogitsLoss()

        print("[INFO] Starting testing...")
        _, acc, rec, prec, f1, cm = evaluate(model, test_loader, loss_fn)
        print(f"  F1: {f1:.4f} | Acc: {acc:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f}")
        print(f"  Confusion Matrix:\n{cm}\n")

        # for i in range(50):
        #     verify_alignment(model, tokenizer, test_dataset, i)


if __name__ == "__main__":
    main()
