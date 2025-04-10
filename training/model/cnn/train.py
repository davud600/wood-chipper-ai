import numpy as np
import torch

import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from .dataset import PDFPageDataset
from .model import PageContextCNN

PDF_DIR = "data/dataset/pdfs"
TRAINING_MINI_BATCH_SIZE = 2
EVAL_MINI_BATCH_SIZE = 2
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0005
PATIENCE = 12
FACTOR = 0.5
EPOCHS = 500
LOG_STEPS = 10
EVAL_STEPS = 100
SPLITTER_MODEL_PATH = "page_classifier_best.pt"


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1, gamma: float = 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input, target):
        bce_loss = self.bce(input, target.float())
        pt = torch.exp(-bce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal.mean()


def find_best_threshold(pred_probs, true_labels, target_metric="f1"):
    best_thresh = 0.5
    best_score = 0
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (pred_probs >= t).astype(int)
        if target_metric == "f1":
            score = f1_score(true_labels, preds, zero_division=0.0)
        elif target_metric == "precision":
            score = precision_score(true_labels, preds, zero_division=0.0)
        else:
            raise ValueError("Unsupported metric")

        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score


full_dataset = PDFPageDataset(max_samples=500)
train_size = int(0.8 * len(full_dataset))
eval_size = len(full_dataset) - train_size
train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])

train_loader = DataLoader(
    train_dataset, batch_size=TRAINING_MINI_BATCH_SIZE, shuffle=True
)
eval_loader = DataLoader(eval_dataset, batch_size=EVAL_MINI_BATCH_SIZE, shuffle=False)

# Log dataset stats
train_labels = [label for _, label in train_dataset]
eval_labels = [label for _, label in eval_dataset]
print(
    f"Train dataset size: {len(train_dataset)} | First pages: {sum(train_labels)} | Non-first pages: {len(train_labels) - sum(train_labels)}"
)
print(
    f"Eval dataset size:  {len(eval_dataset)} | First pages: {sum(eval_labels)} | Non-first pages: {len(eval_labels) - sum(eval_labels)}"
)


def evaluate_on_loader(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    # criterion = FocalLoss(alpha=1.0, gamma=2.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to("cuda"))

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs)
            loss = criterion(outputs, targets.float())
            total_loss += loss.item()
            all_preds.extend(probs.detach().cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    best_thresh, _ = find_best_threshold(np.array(all_preds), np.array(all_labels))
    final_preds = (np.array(all_preds) >= best_thresh).astype(int)
    print("Sample probs:", np.round(all_preds[:10], 3))

    acc = accuracy_score(all_labels, final_preds)
    prec = precision_score(all_labels, final_preds, zero_division=0.0)
    rec = recall_score(all_labels, final_preds, zero_division=0.0)
    f1 = f1_score(all_labels, final_preds, zero_division=0.0)
    cm = confusion_matrix(all_labels, final_preds)

    print("Confusion Matrix:\n", cm)

    return total_loss / len(loader), acc, prec, rec, f1, best_thresh


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PageContextCNN().to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # criterion = FocalLoss(alpha=1.0, gamma=2.0)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, patience=PATIENCE, factor=FACTOR)
    scaler = torch.amp.grad_scaler.GradScaler()

    epoch_steps = 0
    steps = 0
    best_f1_score = 0

    print("starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for _, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            epoch_steps += 1
            steps += 1

            with torch.amp.autocast_mode.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                logits = model(inputs).squeeze(1)
                loss = criterion(logits, targets.float())
                probs = torch.sigmoid(logits)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item()

            if steps % LOG_STEPS == 0:
                print(
                    f"Step {steps} | Epoch {epoch + (epoch_steps / len(train_loader)):.2f} | Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

            if steps % EVAL_STEPS == 0:
                val_loss, acc, prec, rec, f1, best_thresh = evaluate_on_loader(
                    model, eval_loader, device
                )

                print(
                    f"Eval Step {steps}: loss: {val_loss:.4f} | acc: {acc:.4f} | prec: {prec:.4f} | rec: {rec:.4f} | f1: {f1:.4f} | best_thresh: {best_thresh:.2f}"
                )

                scheduler.step(val_loss)
                running_loss = 0.0

                if f1 > best_f1_score:
                    torch.save(model.state_dict(), SPLITTER_MODEL_PATH)
                    best_f1_score = f1
                    print(f"New best model saved with F1: {f1:.4f}")

    print("Training complete.")


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PageContextCNN().to(device)
    model.load_state_dict(torch.load(SPLITTER_MODEL_PATH))
    model.eval()

    val_loss, acc, prec, rec, f1, best_thresh = evaluate_on_loader(
        model, eval_loader, device
    )
    print(
        f"Evaluation Results | acc: {acc:.4f} | prec: {prec:.4f} | rec: {rec:.4f} | f1: {f1:.4f} | best_thresh: {best_thresh:.2f}"
    )


if __name__ == "__main__":
    train_model()
    # evaluate_model()
