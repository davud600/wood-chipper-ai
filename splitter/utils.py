import numpy as np
import torch

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from .dataset.dataset import DocumentDataset


def count_classes(dataset: DocumentDataset) -> tuple[int, int]:
    """
    Counts class distribution in a dataset.

    Useful for analyzing imbalance between first-page and non-first-page samples.

    Parameters
    ----------
    dataset : DocumentDataset
        A dataset containing labeled document pages.

    Returns
    -------
    tuple of int
        Number of first pages and non-first pages.
    """

    first_pages = (dataset.data["page"] == 1).sum()
    non_first_pages = (dataset.data["page"] != 1).sum()
    print(f"[INFO] First pages: {first_pages}, Non-first pages: {non_first_pages}")

    return first_pages, non_first_pages


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the provided dataset.

    Runs inference on the test set, computes loss and classification metrics.

    Parameters
    ----------
    model : nn.Module
        The trained fusion model.

    dataloader : DataLoader
        DataLoader for the evaluation set.

    criterion : torch.nn.Module
        Loss function used to compute evaluation loss.

    device : torch.device
        Device to run on.

    Returns
    -------
    tuple
        A tuple containing:
        - Average loss
        - Accuracy
        - Recall
        - Precision
        - F1 score
        - Confusion matrix
    """

    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cnn_input = batch["cnn_input"].to(device)
            labels = batch["labels"].to(device)
            prev_first_page_distance = batch["prev_first_page_distance"].to(device)

            logits = model(
                input_ids, attention_mask, cnn_input, prev_first_page_distance
            )
            loss = criterion(logits, labels.to(torch.float16))
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds, zero_division=0.0)  # type: ignore
    prec = precision_score(all_labels, all_preds, zero_division=0.0)  # type: ignore
    f1 = f1_score(all_labels, all_preds, zero_division=0.0)  # type: ignore
    cm = confusion_matrix(all_labels, all_preds)

    return total_loss / len(dataloader), acc, rec, prec, f1, cm


def verify_alignment(
    model, tokenizer, dataset: DocumentDataset, idx: int, device: torch.device
):
    """
    Debugs a single data sample to verify model alignment.

    Displays LLM/CNN logits, fused logits, tokens, raw text, and CNN input as an image.

    Parameters
    ----------
    model : FusionModel
        The fusion model being evaluated.

    tokenizer : PreTrainedTokenizer
        Tokenizer used to convert text input.

    dataset : DocumentDataset
        Dataset to sample from.

    idx : int
        Index of the sample to evaluate visually.
    """

    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():

        # === 1. Get sample from dataset ===
        sample = dataset[idx]

        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        cnn_input = sample["cnn_input"].unsqueeze(0).to(device)
        label = sample["labels"].item()

        # === 2. Decode raw tokens ===
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        raw_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # === 3. Forward pass ===
        llm_logits = model.reader_model(input_ids, attention_mask)
        cnn_logits = model.cnn_model(cnn_input)
        fused_logits = model(input_ids, attention_mask, cnn_input)
        pred_prob = torch.sigmoid(fused_logits).item()

        # === 4. Display ===
        print(f"\n📄 Label: {label}")
        print(f"🧠 LLM logits: {llm_logits.item():.4f}")
        print(f"🖼️ CNN logits: {cnn_logits.item():.4f}")
        print(f"🔀 Fused logits: {fused_logits.item():.4f}")
        print(f"✅ Predicted Probability: {pred_prob:.4f}")
        print(f"📝 Raw text:\n{raw_text[:500]}")
        print(f"🔤 Tokens:\n{tokens[:30]} ...")
        print("[CNN] logits:", cnn_logits.detach().cpu().numpy())
        print(
            "[CNN] min/max:",
            cnn_input.min().item(),
            cnn_input.max().item(),
            cnn_input.mean().item(),
        )

        # === 5. Show image ===
        img_grid = sample["cnn_input"]  # shape (C, H, W)
        img_np = img_grid.permute(1, 2, 0).cpu().numpy()  # to HWC
        plt.imshow(img_np.squeeze(), cmap="gray")
        plt.title("Stacked CNN Input")
        plt.axis("off")
        plt.show()
