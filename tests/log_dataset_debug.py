import os
from transformers import AutoTokenizer
from splitter.dataset import DocumentDataset
from config.settings import TRAINING_DATA_CSV

from PIL import Image
import numpy as np
import torch

# CONFIG
TOKENIZER_NAME = "distilbert-base-uncased"
MAX_SAMPLES = None
DEBUG_DIR = "debug_cnn"


def save_context_images(cnn_input, files_and_pages, sample_idx):
    out_dir = os.path.join(DEBUG_DIR, str(sample_idx))
    os.makedirs(out_dir, exist_ok=True)

    num_channels = cnn_input.shape[0]
    assert num_channels % 2 == 0, "CNN input should contain 2 channels per context page"
    num_pages = num_channels // 2

    for i in range(num_pages):
        img = cnn_input[i * 2].numpy()  # take grayscale image channel
        tag, fname, page = files_and_pages[i]

        if fname == "empty":
            continue

        img = (img * 255).clip(0, 255).astype(np.uint8)
        page_str = f"{page:03d}" if page >= 0 else "---"
        filename = f"{i}_{tag}_{fname}_page_{page_str}.png"
        image_path = os.path.join(out_dir, filename)

        Image.fromarray(img, mode="L").save(image_path)


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"[INFO] Initializing dataset from {TRAINING_DATA_CSV}...")
    dataset = DocumentDataset(
        csv_path=TRAINING_DATA_CSV, tokenizer=tokenizer, mode="train"
    )

    print(f"[INFO] Dataset loaded with {len(dataset)} samples.")
    print("=" * 60)

    os.makedirs(DEBUG_DIR, exist_ok=True)

    for idx in range(min(len(dataset), MAX_SAMPLES or len(dataset))):
        sample = dataset[idx]

        print(f"📝 Sample #{idx}")
        for tag, fname, page in sample["files_and_pages"]:
            page_disp = f"{page-1:03d}" if page >= 0 else "---"
            print(f"  {tag:12s} → {fname}_page_{page_disp}")
        print("-" * 60)

        save_context_images(sample["cnn_input"], sample["files_and_pages"], idx)


if __name__ == "__main__":
    main()

# from transformers import AutoTokenizer
# from splitter.dataset import DocumentDataset
#
# from config.settings import TRAINING_DATA_CSV
#
# # CONFIG
# TOKENIZER_NAME = "distilbert-base-uncased"
# MAX_SAMPLES = None
#
#
# def main():
#     tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
#
#     print(f"[INFO] Initializing dataset from {TRAINING_DATA_CSV}...")
#     dataset = DocumentDataset(
#         csv_path=TRAINING_DATA_CSV, tokenizer=tokenizer, mode="train"
#     )
#
#     print(f"[INFO] Dataset loaded with {len(dataset)} samples.")
#     print("=" * 60)
#
#     for idx in range(min(len(dataset), MAX_SAMPLES or len(dataset))):
#         sample = dataset[idx]
#         print(f"📝 Sample #{idx}")
#         for tag, fname, page in sample["files_and_pages"]:
#             page_disp = f"{page-1:03d}" if page >= 0 else "---"
#             print(f"  {tag:12s} → {fname}_page_{page_disp}")
#         print("-" * 60)
#
#
# if __name__ == "__main__":
#     main()
