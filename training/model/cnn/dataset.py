# Efficient, disk-backed PDF context dataset builder

from random import random
import os
import csv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
from config import get_cnn_data_csv, get_cnn_image_dir, DOCUMENT_TYPES
from utils import get_document_type

# Constants
IMAGE_OUTPUT_SIZE = (1300, 1600)
PAGES_TO_APPEND = 2
PREV_PAGES_TO_APPEND = 2
EXCLUDED_DOCUMENT_TYPES = {
    DOCUMENT_TYPES["unknown"],
    DOCUMENT_TYPES["proprietary-lease"],
    DOCUMENT_TYPES["tenant-correspondence"],
    DOCUMENT_TYPES["transfer-of-title"],
    DOCUMENT_TYPES["purchase-application"],
    DOCUMENT_TYPES["closing-document"],
    DOCUMENT_TYPES["alteration-document"],
    DOCUMENT_TYPES["renovation-document"],
    DOCUMENT_TYPES["refinance-document"],
    DOCUMENT_TYPES["transfer-document"],
}


def format_image_to_shape(img, width, height):
    return np.array(Image.fromarray(img).resize((width, height)))


def apply_clahe(img):
    return img  # stub


def denoise(img):
    return img  # stub


def convert_pdf_page_to_image(
    file: str, page: int, doc: fitz.Document, out_size=IMAGE_OUTPUT_SIZE
) -> np.ndarray:
    try:
        mat = fitz.Matrix(2, 2)
        pix = doc.load_page(page).get_pixmap(matrix=mat, colorspace=fitz.csGRAY)  # type: ignore
        img = Image.open(BytesIO(pix.tobytes("jpg")))
        img = np.array(img)
        img = format_image_to_shape(img, out_size[0], out_size[1])
        img = apply_clahe(img)
        img = denoise(img)
        return img
    except Exception as e:
        print(f"Failed to convert page {page} of {file}:", e)
        return np.zeros(out_size, dtype=np.uint8)


def preprocess_pdfs_to_disk(pdf_dir: str, max_pages: int = 20):
    image_dir = get_cnn_image_dir(IMAGE_OUTPUT_SIZE)
    os.makedirs(image_dir, exist_ok=True)

    csv_path = get_cnn_data_csv(IMAGE_OUTPUT_SIZE)
    csv_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not csv_exists:
            writer.writerow(["file", "page", "label"])

        pdf_list = sorted(os.listdir(pdf_dir))
        for i, file in enumerate(pdf_list):
            print(f"{i+1}/{len(pdf_list)} - {file}")
            if not file.endswith(".pdf"):
                continue

            path = os.path.join(pdf_dir, file)
            doc = fitz.open(path)
            num_pages = min(len(doc), max_pages)
            for page_idx in range(num_pages):
                image_path = os.path.join(image_dir, f"{file}_page_{page_idx:03}.png")
                if os.path.exists(image_path):
                    continue

                img = convert_pdf_page_to_image(file, page_idx, doc)
                Image.fromarray(img).save(image_path)
                label = 1 if page_idx == 0 else 0
                writer.writerow([file, page_idx, label])
            doc.close()


class PDFPageDataset(Dataset):
    def __init__(
        self,
        image_output_size=IMAGE_OUTPUT_SIZE,
        prev=PREV_PAGES_TO_APPEND,
        next=PAGES_TO_APPEND,
        max_samples: int | None = None,
    ):
        self.image_output_size = image_output_size
        self.image_dir = get_cnn_image_dir(image_output_size)
        self.index_path = get_cnn_data_csv(image_output_size)
        self.prev = prev
        self.next = next

        if not os.path.exists(self.index_path):
            print(
                f"[INFO] Index CSV not found at {self.index_path}, creating empty one."
            )
            pd.DataFrame(columns=["file", "page", "label"]).to_csv(  # type: ignore
                self.index_path, index=False
            )

        # self.entries = pd.read_csv(self.index_path)
        self.entries = pd.read_csv(self.index_path)
        self.entries = self.entries.sort_values(by=["file", "page"]).reset_index(
            drop=True
        )

        # Filter by document type
        self.entries = self.entries[
            self.entries["file"].apply(
                lambda f: get_document_type(f) not in EXCLUDED_DOCUMENT_TYPES
            )
        ]

        balance_ratio = 0.1
        filtered_entries = []

        for _, row in self.entries.iterrows():
            if row["label"] == 1:
                filtered_entries.append(row)
            elif row["label"] == 0 and random() < balance_ratio:
                filtered_entries.append(row)

        self.entries = pd.DataFrame(filtered_entries).reset_index(drop=True)
        if max_samples is not None:
            self.entries = self.entries.iloc[:max_samples].reset_index(drop=True)

    def __len__(self):
        return len(self.entries)

    # def _load_img(self, file, page_idx):
    #     image_path = os.path.join(self.image_dir, f"{file}_page_{page_idx:03}.png")
    #     if os.path.exists(image_path):
    #         img = Image.open(image_path)
    #         return np.array(img)
    #     return np.zeros(self.image_output_size, dtype=np.uint8)

    def _load_img(self, file_name, page_idx):
        image_path = os.path.join(self.image_dir, f"{file_name}_page_{page_idx:03}.png")
        if os.path.exists(image_path):
            img = Image.open(image_path).convert("L")  # grayscale
            img = img.resize(self.image_output_size)  # force shape match here
            return np.array(img)
        return np.zeros(self.image_output_size, dtype=np.uint8)

    def __getitem__(self, idx):
        entry = self.entries.iloc[idx]
        context = []

        # prev pages (can include previous doc)
        for offset in range(self.prev, 0, -1):
            i = idx - offset
            if i >= 0:
                prev_entry = self.entries.iloc[i]
                context.append(self._load_img(prev_entry.file, prev_entry.page))
            else:
                context.append(np.zeros(self.image_output_size, dtype=np.uint8))

        # current page
        context.append(self._load_img(entry.file, entry.page))

        # next pages (only if same doc)
        for offset in range(1, self.next + 1):
            i = idx + offset
            if i < len(self.entries) and self.entries.iloc[i].file == entry.file:
                next_entry = self.entries.iloc[i]
                context.append(self._load_img(next_entry.file, next_entry.page))
            else:
                context.append(np.zeros(self.image_output_size, dtype=np.uint8))

        for i, img in enumerate(context):
            if img.shape != self.image_output_size:
                # print(f"[ERROR] Shape mismatch at idx={idx} context[{i}] → {img.shape}")
                context[i] = np.zeros(self.image_output_size, dtype=np.uint8)

        x = torch.tensor(np.stack(context), dtype=torch.float32).unsqueeze(1) / 255.0
        x = x.permute(1, 0, 2, 3).contiguous()  # [1, context, H, W]
        y = torch.tensor(entry.label, dtype=torch.long)
        return x, y
