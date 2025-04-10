import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .model import FusionModel
from .models.cnn_model import CNNModel
from .dataset.dataset import DocumentDataset
import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataloader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cnn_input = batch["cnn_input"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask, cnn_input)
            loss = criterion(logits, labels.float())
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


def count_classes(dataset: DocumentDataset) -> tuple[int, int]:
    first_pages = (dataset.data["page"] == 1).sum()
    non_first_pages = (dataset.data["page"] != 1).sum()
    print(f"[INFO] First pages: {first_pages}, Non-first pages: {non_first_pages}")

    return first_pages, non_first_pages


def main():
    print("[INFO] Initializing...")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    cnn = CNNModel()
    model = FusionModel(cnn_model=cnn).to(device)

    train_dataset = DocumentDataset(config.TRAINING_DATA_CSV, tokenizer, mode="train")
    test_dataset = DocumentDataset(config.TESTING_DATA_CSV, tokenizer, mode="test")

    train_loader = DataLoader(
        train_dataset, batch_size=config.training_mini_batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=config.testing_mini_batch_size)

    print("Training")
    N1, N0 = count_classes(train_dataset)
    pos_weight = torch.tensor([(N0 / N1) ** 0.75], dtype=torch.float).to(device)
    print("Testing")
    N1, N0 = count_classes(test_dataset)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config.patience, factor=config.factor
    )
    scaler = torch.amp.grad_scaler.GradScaler()

    best_f1 = 0
    step = 0

    print("[INFO] Starting training...")
    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            step += 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cnn_input = batch["cnn_input"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast_mode.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                logits = model(input_ids, attention_mask, cnn_input)
                loss = criterion(logits, labels.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()

            if step % config.log_steps == 0:
                print(f"[STEP {step}] Epoch {epoch+1} | Loss: {loss.item():.4f}")

            if step % config.log_steps == 0:
                pred_probs = torch.sigmoid(logits[:2]).detach().cpu().numpy()
                true_labels = labels[:2].cpu().numpy()
                print(f"[DEBUG] Predictions (first 2): {pred_probs}")
                print(f"[DEBUG] True labels (first 2): {true_labels}")

            if step % config.eval_steps == 0:
                eval_loss, acc, rec, prec, f1, cm = evaluate(
                    model, test_loader, criterion
                )
                scheduler.step(eval_loss)
                print(f"\n[Eval @ step {step}]")
                print(
                    f"  Loss: {eval_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f}"
                )
                print(f"  Confusion Matrix:\n{cm}\n")

                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), config.SPLITTER_MODEL_PATH)
                    print(f"  ✅ Saved new best model (F1: {f1:.4f})")


if __name__ == "__main__":
    main()
