import os
import torch
import random
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from ..config import device
from config.settings import (
    IMAGES_DIR,
    max_chars,
    prev_pages_to_append,
    pages_to_append,
    max_length,
)
from utils.general import clean_text


doc_length_bins = [5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 800]
doc_length_weights = [
    0.228141,
    0.137013,
    0.134446,
    0.100112,
    0.0884,
    0.075084,
    0.114552,
    0.037061,
    0.038184,
    0.020857,
    0.016044,
    0.009145,
    0.000963,
]


class DocumentDataset(Dataset):
    """
    PyTorch Dataset for multimodal document data.

    Loads data from a CSV file containing document metadata, and returns
    tokenized text and image tensors, along with binary labels indicating
    whether the page is the first in a document.

    Parameters
    ----------
    csv_path : str
        Path to the dataset CSV.

    tokenizer : PreTrainedTokenizer
        Tokenizer used to convert text into input IDs.

    mode : str, optional
        Either "train" or "test", used to control sampling behavior.

    image_dir : str, optional
        Path to the directory where preprocessed page images are stored.

    image_size : tuple, optional
        Image size used for resizing CNN inputs.
    """

    def __init__(
        self, csv_path, tokenizer, mode="train", image_dir=None, image_size=(256, 256)
    ):
        super().__init__()
        self.verbose_indices = set(range(1)) if mode == "train" else set()
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.mode = mode
        self.image_dir = image_dir or os.path.join(
            IMAGES_DIR, f"{image_size[0]}x{image_size[1]}"
        )

        self.prev_n = prev_pages_to_append
        self.next_n = pages_to_append

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
            img_filename = f"{file_id}_page_{(page_num - 1):03d}.png"
            img_path = os.path.join(self.image_dir, img_filename)

            doc_type = int(row["type"])
            if doc_type == 0 or doc_type > 6:
                continue
            # print(f"page {page_num} - {file_id}")

            if os.path.exists(img_path):
                valid_rows.append(row)

        self.all_data = pd.DataFrame(valid_rows).reset_index(drop=True)

        max_files = None
        all_files = self.all_data["file"].unique()

        if max_files is not None:
            sampled_files = np.random.choice(  # type: ignore
                all_files, size=min(max_files, len(all_files)), replace=False  # type: ignore
            )

            self.all_data = self.all_data[
                self.all_data["file"].isin(sampled_files)
            ].reset_index(drop=True)

        max_pages_per_doc = 7
        sampled_rows = []
        for _, row in self.all_data.iterrows():
            page_num = int(row["page"])
            if page_num < max_pages_per_doc:
                sampled_rows.append(row)

        self.data = pd.DataFrame(sampled_rows).reset_index(drop=True)

        # if mode == "train":
        #     sampled_rows = []
        #     for _, row in self.all_data.iterrows():
        #         page_num = int(row["page"])
        #         if page_num < max_pages_per_doc:
        #             sampled_rows.append(row)
        #     self.data = pd.DataFrame(sampled_rows).reset_index(drop=True)
        # else:
        #     self.data = self.all_data

        print(f"[INFO] Loaded {len(self.data)} valid rows (with existing images)")

    def __len__(self):
        return len(self.data)

    def _load_image_tensor(self, file: str, page_num: int):
        """
        Load an image tensor from disk or return a zero tensor if the image does not exist.

        Parameters
        ----------
        file : str
            File name.
        page_num : int
            Zero-based index of the page number to load. The corresponding image
            file is expected to be named as '{file}_page_{page_num:03d}.png'.

        Returns
        -------
        torch.Tensor
            A transformed image tensor with shape (1, H, W) if the file exists,
            otherwise a zero tensor of shape `self.image_size` and dtype float16.
        """

        img_filename = f"{file}_page_{page_num:03d}.png"
        img_path = os.path.join(self.image_dir, img_filename)

        if os.path.exists(img_path):
            image = Image.open(img_path)
            image = self.transform(image)
            # print("image.shape:", image.shape)  # expecting shape: (1, H, W)

            return image
        else:
            return torch.zeros(self.image_size, dtype=torch.float16)

    def _get_context_text(self, file_id, center_page, use_random_fallback=True):
        texts = []
        files = self.all_data["file"].unique().tolist()  # type: ignore
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
                text = clean_text(text)
                texts.append(f"<{tag}>{text[:char_limit]}</{tag}>")
                continue

            # === Backward context missing: use last page of previous or random doc ===
            if offset < 0:
                current_index = files.index(file_id)
                candidates = files[:current_index] if current_index > 0 else []

                if use_random_fallback and not candidates:
                    candidates = files.copy()  # type: ignore
                    candidates.remove(file_id)

                if candidates:
                    if tag not in fallback_pages:
                        fallback_pages[tag] = random.choice(candidates)
                    prev_file = fallback_pages[tag]

                    prev_pages = self.all_data[self.all_data["file"] == prev_file]
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
                    first_row = next_pages.sort_values("page").iloc[0]  # type: ignore
                    text = str(first_row["content"])
                    texts.append(f"<{tag}>{text[:char_limit]}</{tag}>")
                    continue

            # === Fallback: pad with empty or dummy ===
            texts.append(f"<{tag}></{tag}>")

        # # debug - start
        # if hasattr(self, "verbose_tag_debug") and self.verbose_tag_debug:
        #     print(f"\n[📝 TEXT CONTEXT for {file_id} page {center_page}]")
        #     for tag, file in fallback_pages.items():
        #         print(f"  ⏪ {tag} used fallback file: {file}")
        # # debug - end

        return "".join(texts), fallback_pages

    # def _get_context_images(self, file_id, center_page, fallback_pages=None):
    #     images = []
    #
    #     for offset in range(-self.prev_n, self.next_n + 1):
    #         if offset == 0:
    #             tag = "curr_page"
    #         elif offset < 0:
    #             tag = f"prev_page_{-offset}"
    #         else:
    #             tag = f"next_page_{offset}"
    #
    #         # is_fallback = False
    #         current_file = file_id
    #         page_idx = center_page + offset - 1
    #
    #         match = self.all_data[
    #             (self.all_data["file"] == file_id)
    #             & (self.all_data["page"] == center_page + offset)
    #         ]
    #
    #         if len(match) > 0:
    #             image_tensor = self._load_image_tensor(file_id, page_idx)
    #         elif fallback_pages and tag in fallback_pages:
    #             # is_fallback = True
    #             current_file = fallback_pages[tag]
    #             fallback_df = self.all_data[self.all_data["file"] == current_file]
    #
    #             # print(f"\n[📄 Fallback file: {current_file}]")
    #             # print(fallback_df.sort_values("page")[["file", "page"]])
    #
    #             if offset < 0:
    #                 fallback_row = fallback_df.sort_values("page").iloc[-1]  # type: ignore
    #             else:
    #                 fallback_row = fallback_df.sort_values("page").iloc[0]  # type: ignore
    #
    #             page_idx = int(fallback_row["page"]) - 1
    #             image_tensor = self._load_image_tensor(current_file, page_idx)
    #         else:
    #             image_tensor = torch.zeros(self.image_size, dtype=torch.float16)
    #
    #         images.append(image_tensor)
    #
    #         # # debug - start
    #         # if hasattr(self, "verbose_tag_debug") and self.verbose_tag_debug:
    #         #     fallback_note = (
    #         #         "fallback"
    #         #         if is_fallback
    #         #         else ("original" if len(match) > 0 else "empty fallback")
    #         #     )
    #         #     print(
    #         #         f"  🖼️ {tag}: file='{current_file}', page={page_idx + 1} ({fallback_note})"
    #         #     )
    #         # # debug - end
    #
    #     return torch.cat(images, dim=0)  # shape: (C, H, W)

    def _get_context_images(self, file_id, center_page, fallback_pages=None):
        images = []

        for i, offset in enumerate(range(-self.prev_n, self.next_n + 1)):
            if offset == 0:
                tag = "curr_page"
            elif offset < 0:
                tag = f"prev_page_{-offset}"
            else:
                tag = f"next_page_{offset}"

            current_file = file_id
            page_idx = center_page + offset - 1

            match = self.all_data[
                (self.all_data["file"] == file_id)
                & (self.all_data["page"] == center_page + offset)
            ]

            if len(match) > 0:
                image_tensor = self._load_image_tensor(file_id, page_idx)
            elif fallback_pages and tag in fallback_pages:
                current_file = fallback_pages[tag]
                fallback_df = self.all_data[self.all_data["file"] == current_file]
                if offset < 0:
                    fallback_row = fallback_df.sort_values("page").iloc[-1]
                else:
                    fallback_row = fallback_df.sort_values("page").iloc[0]

                page_idx = int(fallback_row["page"]) - 1
                image_tensor = self._load_image_tensor(current_file, page_idx)
            else:
                image_tensor = torch.zeros(self.image_size, dtype=torch.float16)

            # 📌 ADD POSITION MASK
            pos_mask = torch.full_like(image_tensor, 0.2)
            if offset == 0:  # only current page gets 1s
                pos_mask.fill_(1.0)

            # Combine image + mask → (2, H, W)
            combined = torch.stack([image_tensor, pos_mask], dim=0)

            # images.append(combined)
            image_tensor = image_tensor.squeeze(0)  # (H, W)
            pos_mask = torch.full_like(image_tensor, 0.3)
            if offset == 0:
                pos_mask.fill_(1.0)

            combined = torch.stack([image_tensor, pos_mask], dim=0)  # (2, H, W)
            images.append(combined)

        return torch.cat(images, dim=0)  # (C=2*3, H, W) if 3 pages

    # def _get_context_labels(
    #     self, file_id, center_page, fallback_pages=None
    # ) -> torch.Tensor:
    #     """
    #     Returns Tensor (prev + 1 + next,)
    #     """
    #
    #     labels = []
    #
    #     for offset in range(-self.prev_n, self.next_n + 1):
    #         if offset == 0:
    #             tag = "curr_page"
    #         elif offset < 0:
    #             tag = f"prev_page_{-offset}"
    #         else:
    #             tag = f"next_page_{offset}"
    #
    #         current_file = file_id
    #         page_idx = center_page + offset - 1
    #
    #         match = self.all_data[
    #             (self.all_data["file"] == file_id)
    #             & (self.all_data["page"] == center_page + offset)
    #         ]
    #
    #         if len(match) > 0:
    #             label = self._get_item_label(page_idx)
    #         elif fallback_pages and tag in fallback_pages:
    #             current_file = fallback_pages[tag]
    #             fallback_df = self.all_data[self.all_data["file"] == current_file]
    #
    #             if offset < 0:
    #                 fallback_row = fallback_df.sort_values("page").iloc[-1]  # type: ignore
    #             else:
    #                 fallback_row = fallback_df.sort_values("page").iloc[0]  # type: ignore
    #
    #             page_idx = int(fallback_row["page"]) - 1
    #             label = self._get_item_label(page_idx)
    #         else:
    #             label = torch.tensor(0, dtype=torch.float16)
    #
    #         labels.append(label)
    #
    #     return torch.cat(labels, dim=0)  # shape: (C,)

    def _get_context_labels(
        self, file_id, center_page, fallback_pages=None
    ) -> torch.Tensor:
        """
        Returns Tensor (1,)
        """

        labels = [int(center_page == 1)]

        return torch.tensor(labels, dtype=torch.float16)  # shape: (C,)

    def _get_item_label(self, idx: int) -> torch.Tensor:
        return torch.tensor(
            [int(self.data.iloc[idx]["page"] == 1)], dtype=torch.float16
        )

    def _get_context_row_data(
        self,
        file_id,
        center_page,
        fallback_pages=None,
    ) -> list[str]:
        """
        Returns list of strings (prev + 1 + next,) (logging)
        """

        row_data = []

        for offset in range(-self.prev_n, self.next_n + 1):
            if offset == 0:
                tag = "curr_page"
            elif offset < 0:
                tag = f"prev_page_{-offset}"
            else:
                tag = f"next_page_{offset}"

            current_file = file_id
            page_idx = center_page + offset - 1

            match = self.all_data[
                (self.all_data["file"] == file_id)
                & (self.all_data["page"] == center_page + offset)
            ]

            row = "[MISSING]"
            if len(match) > 0:
                row = f"{current_file}_page_{page_idx:03d}.png"
            elif fallback_pages and tag in fallback_pages:
                current_file = fallback_pages[tag]
                fallback_df = self.all_data[self.all_data["file"] == current_file]

                if offset < 0:
                    fallback_row = fallback_df.sort_values("page").iloc[-1]  # type: ignore
                else:
                    fallback_row = fallback_df.sort_values("page").iloc[0]  # type: ignore

                page_idx = int(fallback_row["page"]) - 1
                row = f"{fallback_row['file']}_page_{page_idx:03d}.png"

            row_data.append(row)

        return row_data

    def __getitem__(self, idx):
        if idx in self.verbose_indices:
            self.verbose_tag_debug = True
        else:
            self.verbose_tag_debug = False

        row = self.data.iloc[idx]
        file_id = row["file"]
        page_num = int(row["page"])
        doc_type = int(row["type"])

        # === Build text context ===
        full_text, fallback_df = self._get_context_text(file_id, page_num)

        encoding = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # === Build image context ===
        cnn_input = self._get_context_images(
            file_id, page_num, fallback_df
        )  # (C, H, W)

        # === Labels ===
        labels = self._get_context_labels(file_id, page_num, fallback_df)  # (C,)
        labels = labels.half()

        if page_num == 1:
            distance = random.choices(doc_length_bins, weights=doc_length_weights, k=1)[
                0
            ]
        else:
            distance = page_num - 1

        distance = distance / max(doc_length_bins)
        files_and_pages = self._get_context_row_data(file_id, page_num, fallback_df)

        # # debugging
        # if idx in self.verbose_indices:
        #     print(f"\n[🔍 DEBUG - Dataset Sample {idx}]")
        #     print(
        #         f"  File: {file_id}, Page: {page_num}, Labels: {labels}, Distance: {distance}"
        #     )
        #     print(f"  Text context:\n  {full_text[:25]!r}...")
        #     print(f"  input_ids[:10]: {input_ids[:10].tolist()}")
        #     print(f"  CNN input shape: {cnn_input.shape}")  # -> (3, H, W)
        #
        #     # save all 3 images of cnn input
        #     for i, img in enumerate(cnn_input):
        #         img = img.numpy()
        #         # print("img shape:", img.shape)  # should be -> (H, W)
        #
        #         dir_path = f"/home/davud/wood-chipper-ai/debug_cnn/{idx}"
        #
        #         # image_path = f"{dir_path}/{file_id}_page_{page_num + offset}.png"
        #         image_path = f"{dir_path}/{i}.png"
        #
        #         img = img.squeeze()  # (1, H, W) -> (H, W)
        #         img = (img * 255).clip(0, 255).astype(np.uint8)
        #         Image.fromarray(img, mode="L").save(image_path)
        #         os.makedirs(dir_path, exist_ok=True)

        return {
            "files_and_pages": files_and_pages,
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "cnn_input": cnn_input.to(device),
            "labels": labels.to(device),
            "doc_type": torch.tensor(doc_type, dtype=torch.int).to(device),
            "distance": torch.tensor([distance], dtype=torch.float16).to(device),
        }
