import torch

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from splitter.models.cnn_model import CNNModel

from .config import device
from .model import FusionModel
from .dataset.dataset import DocumentDataset
from config.settings import SPLITTER_MODEL_PATH


def eval_and_save(model, scheduler, step, loss_fn, test_loader, best_f1):
    print(f"\n[Eval @ step {step}]")
    eval_loss, acc, rec, prec, f1, cm = evaluate(model, test_loader, loss_fn)
    scheduler.step(eval_loss)
    print(
        f"  Loss: {eval_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f}"
    )
    print(f"  Confusion Matrix:\n{cm}\n")

    if f1 > best_f1:
        name = "fused" if isinstance(model, FusionModel) else "cnn"
        name = "cnn" if isinstance(model, CNNModel) else "llm"
        best_f1 = f1
        torch.save(
            model.state_dict(),
            f"{SPLITTER_MODEL_PATH}_{name}",
        )
        print(f"  ✅ Saved new best model (F1: {f1:.4f})")


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


def evaluate(model, dataloader, loss_fn):
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
            logits, loss = model.forward(batch, loss_fn)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds, zero_division=0.0)  # type: ignore
    prec = precision_score(all_labels, all_preds, zero_division=0.0)  # type: ignore
    f1 = f1_score(all_labels, all_preds, zero_division=0.0)  # type: ignore
    cm = confusion_matrix(all_labels, all_preds)

    return total_loss / len(dataloader), acc, rec, prec, f1, cm


def verify_alignment(model, tokenizer, dataset: DocumentDataset, idx: int):
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
