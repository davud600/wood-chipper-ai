import os
import pandas as pd
import matplotlib.pyplot as plt

from config.settings import (
    DOCUMENT_TYPES,
    TRAINING_DATA_CSV,
    TESTING_DATA_CSV,
    project_root,
)

# === Config ===
CSV_PATH = TRAINING_DATA_CSV
OUTPUT_PATH = os.path.join(project_root, "visualizations/type_distribution.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# === Load CSV ===
df = pd.read_csv(CSV_PATH)

# === Group by document (file) and get type from first page ===
docs_df = df.sort_values("page").groupby("file").first().reset_index()

# === Count number of docs per type ===
type_counts = docs_df["type"].value_counts().sort_index()

# Map int codes back to type names
inv_doc_types = {v: k for k, v in DOCUMENT_TYPES.items()}
labels = [inv_doc_types.get(t, f"type_{t}") for t in type_counts.index]
counts = type_counts.values

# === Plot ===
plt.figure(figsize=(10, 6))
plt.barh(labels, counts, color="steelblue")
plt.xlabel("Number of Documents")
plt.title("Document Type Distribution (by File)")
plt.tight_layout()

# === Save ===
plt.savefig(OUTPUT_PATH)
print(f"[📊] Saved plot to: {OUTPUT_PATH}")
