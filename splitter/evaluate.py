import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config.settings import SPLITTER_MODEL_PATH, TESTING_DATA_CSV, image_output_size
from .config import device, testing_mini_batch_size
from .utils import count_classes, evaluate, verify_alignment
from .model import FusionModel
from .dataset.dataset import DocumentDataset


def main():
    """
    Runs evaluation on the test set and visual verification on sample predictions.

    Loads model and tokenizer, prepares test data, runs batch evaluation,
    prints metrics and confusion matrix, and performs sample-level alignment checks.
    """

    print("[INFO] Initializing...")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    model = FusionModel(image_size=image_output_size).to(device)
    model.eval()
    model.load_state_dict(
        torch.load(SPLITTER_MODEL_PATH, weights_only=False, map_location="cuda")
    )

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
        batch_size=testing_mini_batch_size,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    print("[INFO] Starting training...")
    model.train()
    for batch in test_loader:
        with torch.amp.autocast_mode.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(batch)

        pred_probs = torch.sigmoid(logits[:2]).detach().cpu().numpy()
        true_labels = batch["labels"][:2].cpu().numpy()
        print(f"[DEBUG] Predictions (first 2): {pred_probs.squeeze(1)}")
        print(f"[DEBUG] True labels (first 2): {true_labels.squeeze(1)}")

    _, acc, rec, prec, f1, cm = evaluate(model, test_loader, loss_fn)
    print(f"  F1: {f1:.4f} | Acc: {acc:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f}")
    print(f"  Confusion Matrix:\n{cm}\n")

    for i in range(50):
        verify_alignment(model, tokenizer, test_dataset, i)


if __name__ == "__main__":
    main()
