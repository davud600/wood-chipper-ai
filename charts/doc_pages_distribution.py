import os
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from config.settings import PDF_DIR, project_root

# === Config ===
OUTPUT_PATH = os.path.join(project_root, "visualizations/doc_pages_distribution.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# === Count pages per PDF ===
page_counts = []

print("[🔍] Scanning PDF files in:", PDF_DIR)
for filename in os.listdir(PDF_DIR):
    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(PDF_DIR, filename)
    try:
        with fitz.open(pdf_path) as doc:
            page_counts.append(len(doc))
    except Exception as e:
        print(f"[⚠️] Failed to read {filename}: {e}")

# === Define bins ===
doc_length_bins = [5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 800]
bin_edges = [0] + doc_length_bins + [float("inf")]

# === Count documents in each bin ===
bin_counts = [0 for _ in range(len(doc_length_bins))]

for count in page_counts:
    for i in range(len(doc_length_bins)):
        if bin_edges[i] < count <= bin_edges[i + 1]:
            bin_counts[i] += 1
            break

total = sum(bin_counts)
doc_length_weights = [round(c / total, 6) for c in bin_counts]

# === Print arrays for hardcoding ===
print("doc_length_bins =", doc_length_bins)
print("doc_length_weights =", doc_length_weights)

# === Plot ===
plt.figure(figsize=(10, 6))
plt.hist(page_counts, bins=50, color="mediumseagreen", edgecolor="black")

max_pages = max(page_counts) if page_counts else 100
plt.xlim(0, max_pages + 10)
plt.xticks(np.arange(0, max_pages + 10, max(1, (max_pages // 20))))

plt.xlabel("Number of Pages per Document")
plt.ylabel("Frequency")
plt.title("Distribution of Document Lengths (pages)")
plt.tight_layout()

# === Save ===
plt.savefig(OUTPUT_PATH)
print(f"[📊] Saved plot to: {OUTPUT_PATH}")
