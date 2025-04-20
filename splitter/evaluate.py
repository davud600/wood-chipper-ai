import os
import torch.nn as nn

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config.settings import (
    SPLITTER_MODEL_DIR,
    TESTING_DATA_CSV,
    image_output_size,
)
from .config import device, testing_mini_batch_size
from .utils import count_classes, evaluate, verify_alignment, load_best_weights
from .model import FusionModel
from .dataset.dataset import DocumentDataset

session_dirs = [
    int(d)
    for d in os.listdir(SPLITTER_MODEL_DIR)
    if os.path.isdir(os.path.join(SPLITTER_MODEL_DIR, d)) and d.isdigit()
]
session = max(session_dirs, default=0)


def main():
    """
    Runs evaluation on the test set and visual verification on sample predictions.

    Loads model and tokenizer, prepares test data, runs batch evaluation,
    prints metrics and confusion matrix, and performs sample-level alignment checks.
    """

    global session

    print("[INFO] Initializing...")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = FusionModel(image_size=image_output_size).to(device)
    model.eval()
    print(f"[INFO] Loading model weights from session {session}...")
    load_best_weights(model, session, True)

    print("[TESTING]")
    test_dataset = DocumentDataset(
        TESTING_DATA_CSV,
        tokenizer,
        mode="test",
        image_size=image_output_size,
    )
    count_classes(test_dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=testing_mini_batch_size * 2,
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
