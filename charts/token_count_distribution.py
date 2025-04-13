import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from config.settings import TRAINING_DATA_CSV, TESTING_DATA_CSV, project_root
from utils.general import clean_text

# === Config ===
CSV_PATH = TRAINING_DATA_CSV
OUTPUT_PATH = os.path.join(project_root, "visualizations/token_count_distribution.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
max_chars = 1024

# === Load Data ===
df = pd.read_csv(CSV_PATH)

# === Init Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


df["clean_text"] = df["content"].astype(str).apply(clean_text)
df["clean_text"] = df["clean_text"].apply(lambda x: x[:max_chars])

# === Token Count ===
df["token_count"] = df["clean_text"].apply(
    lambda x: len(tokenizer.encode(x, add_special_tokens=True))
)

# === Plot ===
plt.figure(figsize=(10, 6))
plt.hist(df["token_count"], bins=50, color="mediumseagreen", edgecolor="black")
plt.xlabel("Token Count per Page")
plt.ylabel("Number of Pages")
plt.title("Distribution of Token Count per Page")
plt.tight_layout()

# === Save ===
plt.savefig(OUTPUT_PATH)
print(f"[📊] Saved plot to: {OUTPUT_PATH}")
