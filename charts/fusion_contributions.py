import os
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config.settings import (
    SPLITTER_MODEL_PATH,
    TESTING_DATA_CSV,
    image_output_size,
    project_root,
)
from splitter.model import FusionModel
from splitter.dataset.dataset import DocumentDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Setup ===
OUTPUT_PATH = os.path.join(project_root, "visualizations/fusion_contributions.png")
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("[INFO] Loading model and data...")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = FusionModel(image_size=image_output_size).to(device)
model.eval()
model.load_state_dict(
    torch.load(
        "/home/davud/wood-chipper-ai/data/models/splitter_llm.pth", map_location=device
    )
)

# Load test dataset
test_dataset = DocumentDataset(
    TESTING_DATA_CSV,
    tokenizer,
    mode="test",
    image_size=image_output_size,
)
test_loader = DataLoader(test_dataset, batch_size=1)

# === Inference + Logit Collection ===
llm_scores = []
cnn_scores = []
fused_scores = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cnn_input = batch["cnn_input"].to(device)

        fused_logits, llm_logits, cnn_logits = model(
            input_ids, attention_mask, cnn_input, return_all_logits=True
        )

        llm_scores.extend(torch.sigmoid(llm_logits).cpu().numpy().tolist())
        cnn_scores.extend(torch.sigmoid(cnn_logits).cpu().numpy().tolist())
        fused_scores.extend(torch.sigmoid(fused_logits).cpu().numpy().tolist())

# === Plot ===
x = list(range(len(fused_scores)))

plt.figure(figsize=(12, 6))
plt.plot(x, llm_scores, label="LLM", linewidth=1)
plt.plot(x, cnn_scores, label="CNN", linewidth=1)
plt.plot(x, fused_scores, label="Fused Output", linewidth=2, alpha=0.85)

plt.xlabel("Sample Index")
plt.ylabel("Confidence (Sigmoid Output)")
plt.title("Fusion Contributions per Sample")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PATH)

print(f"[📊] Saved fusion contributions plot to: {OUTPUT_PATH}")
