import torch.nn as nn
import torch
import argparse
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from .utils import count_classes
from .dataset.dataset import DocumentDataset
from .models.llm_model import ReaderModel
from .models.cnn_model import CNNModel
from .config import device
from config.settings import TRAINING_DATA_CSV

# --- args / setup ---
p = argparse.ArgumentParser()
p.add_argument("--model", type=str, default="c")  # "c" or "l"
args = p.parse_args()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
batch_size = 16
image_size = (1024, 1024)

# full dataset
full_dataset = DocumentDataset(
    TRAINING_DATA_CSV, tokenizer, mode="train", image_size=image_size
)

# split out a small val‐set for quick grid search
# train_ds, val_ds = random_split(full_dataset, [int(len(full_dataset) * 0.8), -1])
n_total = len(full_dataset)
n_train = int(n_total * 0.1)
n_val = n_total - n_train
train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=3)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=3)

# class balancing
n1, n0 = count_classes(full_dataset)
pos_weight = torch.tensor([n0 / n1], dtype=torch.float32).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# hyper‑param grid
lr_list = [1e-5, 3e-5, 1e-4]
wd_list = [1e-4, 5e-4, 1e-3]
results = []


with torch.amp.autocast_mode.autocast(device_type=str(device), dtype=torch.float16):
    for lr, wd in itertools.product(lr_list, wd_list):
        # instantiate fresh model & optimizer
        model = CNNModel().to(device) if args.model == "c" else ReaderModel().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        # train for just a few epochs
        model.train()
        for epoch in range(2):  # 2 epochs each
            for batch in train_loader:
                opt.zero_grad()
                outputs = model(batch)
                loss = loss_fn(outputs, batch["labels"].to(device))
                loss.backward()
                opt.step()

        # eval on val set
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch)
                l = loss_fn(outputs, batch["labels"].to(device)).item()
                total_loss += l
                count += 1
        avg_val_loss = total_loss / count

        print(f"lr={lr:.1e}, wd={wd:.1e} → val_loss={avg_val_loss:.4f}")
        results.append({"lr": lr, "wd": wd, "val_loss": avg_val_loss})

# collate results
df = pd.DataFrame(results)
best = df.loc[df["val_loss"].idxmin()]
print(f"\n>> Best combo: lr={best.lr:.1e}, wd={best.wd:.1e} (loss={best.val_loss:.4f})")

# pivot into matrix for heatmap
heat = df.pivot("lr", "wd", "val_loss")

# … after building your `heat` pivot …

plt.figure(figsize=(6, 5))
plt.title("Val Loss over (lr, wd) Grid")
plt.imshow(
    heat,
    origin="lower",
    aspect="auto",
    extent=[min(wd_list), max(wd_list), min(lr_list), max(lr_list)],
)
plt.xscale("log")
plt.yscale("log")
plt.colorbar(label="Avg Val Loss")
plt.xlabel("Weight Decay")
plt.ylabel("Learning Rate")
plt.tight_layout()

out_path = "grid_search_lr_wd.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")  # writes file, no window needed
plt.close()

print(f"✅ Grid‑search heatmap saved to {out_path}")
