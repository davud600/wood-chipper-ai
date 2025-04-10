import os
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import config
from config.settings import IMAGES_DIR

max_chars = {
    "curr_page": 1024,
    "prev_page": 512,
    "next_page": 512,
}


class DocumentDataset(Dataset):
    def __init__(
        self, csv_path, tokenizer, mode="train", image_dir=None, image_size=(256, 256)
    ):
        super().__init__()
        self.verbose_indices = set(range(5)) if mode == "train" else set()
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.mode = mode
        self.image_dir = image_dir or os.path.join(
            IMAGES_DIR, f"{image_size[0]}x{image_size[1]}"
        )

        self.prev_n = getattr(config, "prev_pages_to_append", 2)
        self.next_n = getattr(config, "pages_to_append", 2)

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )

        raw_data = pd.read_csv(csv_path)
        valid_rows = []

        for _, row in raw_data.iterrows():
            file_id = row["file"]
            page_num = int(row["page"])
            doc_type = int(row["type"])
            img_filename = f"{file_id}_page_{(page_num - 1):03d}.png"
            img_path = os.path.join(self.image_dir, img_filename)
            print(f"page {page_num} - {file_id}")

            # if doc_type > 7 or doc_type == 0:
            #     continue

            if os.path.exists(img_path):
                valid_rows.append(row)

        # self.data = pd.DataFrame(valid_rows).reset_index(drop=True)
        self.all_data = pd.DataFrame(valid_rows).reset_index(drop=True)

        # Apply sampling for training rows
        non_first_pages_prob = 0.5
        if mode == "train":
            sampled_rows = []
            for _, row in self.all_data.iterrows():
                page_num = int(row["page"])
                if page_num == 1 or random.random() < non_first_pages_prob:
                    sampled_rows.append(row)
            self.data = pd.DataFrame(sampled_rows).reset_index(drop=True)
        else:
            self.data = self.all_data

        print(f"[INFO] Loaded {len(self.data)} valid rows (with existing images)")

    def __len__(self):
        return len(self.data)

    def _load_image_tensor(self, file_id, page_num):
        img_filename = f"{file_id}_page_{page_num - 1:03d}.png"
        img_path = os.path.join(self.image_dir, img_filename)

        if os.path.exists(img_path):
            image = Image.open(img_path)
            return self.transform(image)  # shape: (1, H, W)
        else:
            return torch.zeros((1, *self.image_size), dtype=torch.float32)

    def _get_context_text(self, file_id, center_page, use_random_fallback=True):
        texts = []
        files = self.all_data["file"].unique().tolist()
        fallback_pages = {}  # Stores random fallback pages for reproducibility

        for offset in range(-self.prev_n, self.next_n + 1):
            page_idx = center_page + offset

            # Tag for this position
            if offset == 0:
                tag = "curr_page"
            elif offset < 0:
                tag = f"prev_page_{-offset}"
            else:
                tag = f"next_page_{offset}"

            # Character limits per tag
            if tag == "curr_page":
                char_limit = max_chars["curr_page"]
            elif "prev_page" in tag:
                char_limit = max_chars["prev_page"]
            elif "next_page" in tag:
                char_limit = max_chars["next_page"]
            else:
                char_limit = 512

            # === Normal context (same doc) ===
            match = self.all_data[
                (self.all_data["file"] == file_id) & (self.all_data["page"] == page_idx)
            ]

            if len(match) > 0:
                text = str(match.iloc[0]["content"])
                texts.append(f"<{tag}>{text[:char_limit]}</{tag}>")
                continue

            # === Backward context missing: use last page of previous or random doc ===
            if offset < 0:
                current_index = files.index(file_id)
                candidates = files[:current_index] if current_index > 0 else []

                if use_random_fallback and not candidates:
                    candidates = files.copy()
                    candidates.remove(file_id)

                if candidates:
                    if tag not in fallback_pages:
                        fallback_pages[tag] = random.choice(candidates)
                    prev_file = fallback_pages[tag]

                    prev_pages = self.all_data[self.all_data["file"] == prev_file]
                    if not prev_pages.empty:
                        last_row = prev_pages.sort_values("page").iloc[-1]  # type: ignore
                        text = str(last_row["content"])
                        texts.append(f"<{tag}>{text[:char_limit]}</{tag}>")
                        continue

            # === Forward context missing: use first page of random doc ===
            if offset > 0:
                candidates = [f for f in files if f != file_id]
                if candidates:
                    if tag not in fallback_pages:
                        fallback_pages[tag] = random.choice(candidates)
                    next_file = fallback_pages[tag]

                    next_pages = self.all_data[self.all_data["file"] == next_file]
                    if not next_pages.empty:
                        first_row = next_pages.sort_values("page").iloc[0]  # type: ignore
                        text = str(first_row["content"])
                        texts.append(f"<{tag}>{text[:char_limit]}</{tag}>")
                        continue

            # === Fallback: pad with empty or dummy ===
            texts.append(f"<{tag}></{tag}>")

        if hasattr(self, "verbose_tag_debug") and self.verbose_tag_debug:
            print(f"\n[📝 TEXT CONTEXT for {file_id} page {center_page}]")
            for tag, file in fallback_pages.items():
                print(f"  ⏪ {tag} used fallback file: {file}")

        return " ".join(texts), fallback_pages

    def _get_context_images(self, file_id, center_page, fallback_pages=None):
        images = []
        files = self.all_data["file"].unique().tolist()

        for offset in range(-self.prev_n, self.next_n + 1):
            if offset == 0:
                tag = "curr_page"
            elif offset < 0:
                tag = f"prev_page_{-offset}"
            else:
                tag = f"next_page_{offset}"

            is_fallback = False
            current_file = file_id
            page_idx = center_page + offset - 1

            match = self.all_data[
                (self.all_data["file"] == file_id)
                & (self.all_data["page"] == center_page + offset)
            ]

            if len(match) > 0:
                image_tensor = self._load_image_tensor(file_id, page_idx)
            elif fallback_pages and tag in fallback_pages:
                is_fallback = True
                current_file = fallback_pages[tag]
                fallback_df = self.all_data[self.all_data["file"] == current_file]

                if offset < 0:
                    fallback_row = fallback_df.sort_values("page").iloc[-1]
                else:
                    fallback_row = fallback_df.sort_values("page").iloc[0]

                page_idx = int(fallback_row["page"]) - 1
                image_tensor = self._load_image_tensor(current_file, page_idx)
            else:
                image_tensor = torch.zeros((1, *self.image_size), dtype=torch.float32)

            images.append(image_tensor)

            if hasattr(self, "verbose_tag_debug") and self.verbose_tag_debug:
                fallback_note = (
                    "fallback"
                    if is_fallback
                    else ("original" if len(match) > 0 else "empty fallback")
                )
                print(
                    f"  🖼️ {tag}: file='{current_file}', page={page_idx + 1} ({fallback_note})"
                )

        return torch.cat(images, dim=0)  # shape: (C, H, W)

    def __getitem__(self, idx):
        if idx in self.verbose_indices:
            self.verbose_tag_debug = True
        else:
            self.verbose_tag_debug = False

        row = self.data.iloc[idx]
        file_id = row["file"]
        page_num = int(row["page"])
        label = 1 if page_num == 1 else 0

        # === Build OCR context ===
        full_text, fallback_df = self._get_context_text(file_id, page_num)

        encoding = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # === Build image context ===
        cnn_input = self._get_context_images(
            file_id, page_num, fallback_df
        )  # (C, H, W)

        if idx in self.verbose_indices:
            print(f"\n[🔍 DEBUG - Dataset Sample {idx}]")
            print(f"  File: {file_id}, Page: {page_num}, Label: {label}")
            print(f"  Text context:\n  {full_text!r}")
            print(f"  input_ids[:10]: {input_ids[:10].tolist()}")
            print(f"  CNN input shape: {cnn_input.shape}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cnn_input": cnn_input,
            "label": torch.tensor(label, dtype=torch.float),
        }


# import os
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# import pandas as pd
# import config
# from config.settings import IMAGES_DIR
#
#
# class DocumentDataset(Dataset):
#     def __init__(
#         self, csv_path, tokenizer, mode="train", image_dir=None, image_size=(256, 256)
#     ):
#         super().__init__()
#         self.data = pd.read_csv(csv_path)
#         self.tokenizer = tokenizer
#         self.mode = mode
#         self.image_dir = image_dir or os.path.join(
#             IMAGES_DIR, f"{image_size[0]}x{image_size[1]}"
#         )
#         self.image_size = image_size
#
#         self.transform = transforms.Compose(
#             [
#                 transforms.Grayscale(num_output_channels=1),
#                 transforms.Resize(self.image_size),
#                 transforms.ToTensor(),  # output shape: (C, H, W), normalized to [0,1]
#             ]
#         )
#
#         # Load and filter the dataset rows based on image existence
#         raw_data = pd.read_csv(csv_path)
#         valid_rows = []
#
#         for _, row in raw_data.iterrows():
#             file_id = row["file"]
#             page_num = int(row["page"])
#             doc_type = int(row["type"])
#             img_filename = f"{file_id}_page_{(page_num - 1):03d}.png"
#             img_path = os.path.join(self.image_dir, img_filename)
#
#             if doc_type > 6 or doc_type == 0:
#                 continue
#
#             if os.path.exists(img_path):
#                 valid_rows.append(row)
#
#         self.data = pd.DataFrame(valid_rows).reset_index(drop=True)
#         print(f"[INFO] Loaded {len(self.data)} valid rows (with existing images)")
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         file_id = row["file"]  # e.g., "mydoc.pdf"
#         page_num = int(row["page"])
#         label = 1 if page_num == 1 else 0
#         text = str(row["content"])
#
#         # === Tokenize text ===
#         encoding = self.tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=config.max_length,
#             return_tensors="pt",
#         )
#         input_ids = encoding["input_ids"].squeeze(0)
#         attention_mask = encoding["attention_mask"].squeeze(0)
#
#         # === Load image ===
#         img_filename = f"{file_id}_page_{(page_num - 1):03d}.png"
#         img_path = os.path.join(self.image_dir, img_filename)
#
#         try:
#             image = Image.open(img_path)
#         except FileNotFoundError:
#             raise FileNotFoundError(f"Image not found: {img_path}")
#
#         cnn_input = self.transform(image)
#
#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "cnn_input": cnn_input,
#             "label": torch.tensor(label, dtype=torch.float),
#         }
