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
from config.settings import prev_pages_to_append, pages_to_append


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
    all_llm_logits, all_cnn_logits = [], []
    all_rows = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cnn_input = batch["cnn_input"].to(device)
            labels = batch["labels"].to(device)
            distance = batch["prev_first_page_distance"].to(device)
            all_rows.extend(batch["files_and_pages"])

            logits, llm_logits, cnn_logits = model(
                input_ids,
                attention_mask,
                cnn_input,
                distance,
                return_all_logits=True,
            )

            loss = criterion(logits, labels.to(torch.float32))
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.float().cpu().numpy())
            all_llm_logits.extend(llm_logits.cpu().numpy())
            all_cnn_logits.extend(cnn_logits.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall micro-averaged metrics
    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds, average="micro", zero_division=0.0)  # type: ignore
    prec = precision_score(all_labels, all_preds, average="micro", zero_division=0.0)  # type: ignore
    f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0.0)  # type: ignore

    # Per-position metrics
    pos_labels = (
        [f"prev_{prev_pages_to_append - i}" for i in range(prev_pages_to_append)]
        + ["curr"]
        + [f"next_{i + 1}" for i in range(pages_to_append)]
    )

    print("\n📊 Per-Position Evaluation:")
    print(
        f"{'Position':<10} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | TP / FP / FN"
    )
    print("-" * 60)

    cms = []
    for i, label in enumerate(pos_labels):
        y_true = all_labels[:, i]
        y_pred = all_preds[:, i]

        cm = confusion_matrix(y_true, y_pred)
        cms.append(cm)

        TP = cm[1, 1] if cm.shape == (2, 2) else 0
        FP = cm[0, 1] if cm.shape[1] > 1 else 0
        FN = cm[1, 0] if cm.shape[0] > 1 else 0

        acc_i = accuracy_score(y_true, y_pred)
        prec_i = precision_score(y_true, y_pred, zero_division=0.0)  # type: ignore
        rec_i = recall_score(y_true, y_pred, zero_division=0.0)  # type: ignore
        f1_i = f1_score(y_true, y_pred, zero_division=0.0)  # type: ignore

        print(
            f"{label:<10} | {acc_i:.4f} | {prec_i:.4f} | {rec_i:.4f} | {f1_i:.4f} | {TP} / {FP} / {FN}"
        )

    print("\n🔍 Sample Breakdown for First Batch:\n")

    sample_idx = 0
    sample_files = all_rows[sample_idx]
    sample_labels = all_labels[sample_idx]
    sample_preds = all_preds[sample_idx]
    sample_llm = all_llm_logits[sample_idx]
    sample_cnn = all_cnn_logits[sample_idx]

    # if not isinstance(sample_files, list):
    #     sample_files = [sample_files]
    #     print("⚠️ files_and_pages was not a list — patched manually")

    if len(sample_files) != len(pos_labels):
        ...
        # print(
        #     f"❌ Mismatch: {len(sample_files)} filenames vs {len(pos_labels)} expected positions"
        # )
        # print("→ sample_files:", sample_files)
        # print("→ Skipping sample breakdown.\n")
    else:
        for i in range(len(pos_labels)):
            print(
                # f"{pos_labels[i]:<7} | {sample_files[i]:<60} | "
                f"GT: {int(sample_labels[i])} | Pred: {int(sample_preds[i])} | "
                f"LLM: {sample_llm[i]:+.4f} | CNN: {sample_cnn[i]:+.4f}"
            )

    # print(f"   Labels: {all_labels[0]}")
    # print(f"🧠 LLM logits: {all_llm_logits[0]}")
    # print(f"🖼️ CNN logits: {all_cnn_logits[0]}")

    return total_loss / len(dataloader), acc, rec, prec, f1, cms

    # model.eval()
    # all_preds, all_labels = [], []
    # all_llm_logits, all_cnn_logits = [], []
    # total_loss = 0
    #
    # with torch.no_grad():
    #     for batch in dataloader:
    #         input_ids = batch["input_ids"].to(device)
    #         attention_mask = batch["attention_mask"].to(device)
    #         cnn_input = batch["cnn_input"].to(device)
    #         labels = batch["labels"].to(device)
    #         prev_first_page_distance = batch["prev_first_page_distance"].to(device)
    #
    #         logits, llm_logits, cnn_logits = model(
    #             input_ids,
    #             attention_mask,
    #             cnn_input,
    #             prev_first_page_distance,
    #             return_all_logits=True,
    #         )
    #
    #         # logits = model(
    #         #     input_ids,
    #         #     attention_mask,
    #         #     cnn_input,
    #         #     prev_first_page_distance,
    #         # )
    #
    #         loss = criterion(logits, labels.to(torch.float16))
    #         total_loss += loss.item()
    #
    #         preds = (torch.sigmoid(logits) > 0.5).long()
    #         all_preds.extend(preds.cpu().numpy())
    #         all_labels.extend(labels.float().cpu().numpy())
    #         all_llm_logits.extend(llm_logits.cpu().numpy())
    #         all_cnn_logits.extend(cnn_logits.cpu().numpy())
    #
    # all_labels = np.array(all_labels)
    # all_preds = np.array(all_preds)
    #
    # acc = accuracy_score(all_labels, all_preds)
    # rec = recall_score(all_labels, all_preds, average="micro", zero_division=0.0)  # type: ignore
    # prec = precision_score(all_labels, all_preds, average="micro", zero_division=0.0)  # type: ignore
    # f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0.0)  # type: ignore
    # cms = []
    # for i in range(prev_pages_to_append + 1 + pages_to_append):
    #     cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
    #     cms.append(cm)
    #
    # print(f"\n📄 Labels: {all_labels[0]}")
    # print(f"🧠 LLM logits: {all_llm_logits[0]}")
    # print(f"🖼️ CNN logits: {all_cnn_logits[0]}")
    #
    # return total_loss / len(dataloader), acc, rec, prec, f1, cms


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
        label = sample["label"].item()

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


def custom_collate_fn(batch):
    batch_dict = {key: [] for key in batch[0].keys()}
    for sample in batch:
        for key, value in sample.items():
            batch_dict[key].append(value)

    # Stack tensor-like things, leave lists alone
    for key in batch_dict:
        if key != "files_and_pages":
            if isinstance(batch_dict[key][0], torch.Tensor):
                batch_dict[key] = torch.stack(batch_dict[key])

    return batch_dict
