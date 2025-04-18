import os
import torch
import random
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

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
        self, csv_path, tokenizer, mode="train", image_dir=None, image_size=(1024, 1024)
    ):
        super().__init__()
        self.verbose_indices = set(range(3)) if mode == "train" else set()
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
                # transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float16),
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

        # debug - start
        max_files = 200
        all_files = self.all_data["file"].unique()
        if max_files is not None:
            sampled_files = np.random.choice(  # type: ignore
                all_files, size=min(max_files, len(all_files)), replace=False  # type: ignore
            )

            self.all_data = self.all_data[
                self.all_data["file"].isin(sampled_files)
            ].reset_index(drop=True)
        # debug - end

        max_pages_per_doc = 10
        sampled_rows = []
        num_augmented = 5

        # for _, row in self.all_data.iterrows():
        #     page_num = int(row["page"])
        #
        #     if page_num < max_pages_per_doc:
        #         if page_num == 1 and mode == "train":
        #             for i in range(num_augmented - 1):
        #                 sampled_rows.append(row)
        #
        #             continue
        #
        #         sampled_rows.append(row)

        if mode == "train":
            sampled_rows = []
            for _, row in self.all_data.iterrows():
                page_num = int(row["page"])

                if page_num < max_pages_per_doc:
                    if page_num == 1 and mode == "train":
                        for i in range(num_augmented - 1):
                            sampled_rows.append(row)

                        continue

                    sampled_rows.append(row)
            self.data = pd.DataFrame(sampled_rows).reset_index(drop=True)
        else:
            self.data = self.all_data

        # self.data = pd.DataFrame(sampled_rows).reset_index(drop=True)

        print(f"[INFO] Loaded {len(self.data)} valid rows (with existing images)")

    def __len__(self):
        return len(self.data)

    def _load_image_tensor(self, file: str, page_num: int) -> torch.Tensor:
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

            return image  # type: ignore
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

        return "".join(texts), fallback_pages

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
                    fallback_row = fallback_df.sort_values("page").iloc[-1]  # type: ignore
                else:
                    fallback_row = fallback_df.sort_values("page").iloc[0]  # type: ignore

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

    def _get_context_row_data(self, file_id, center_page, fallback_pages=None):
        context_info = []

        for offset in range(-self.prev_n, self.next_n + 1):
            tag = (
                "curr_page"
                if offset == 0
                else f"prev_page_{-offset}" if offset < 0 else f"next_page_{offset}"
            )

            page_idx = center_page + offset
            current_file = file_id

            match = self.all_data[
                (self.all_data["file"] == file_id) & (self.all_data["page"] == page_idx)
            ]

            if len(match) > 0:
                context_info.append((tag, current_file, page_idx))
            elif fallback_pages and tag in fallback_pages:
                current_file = fallback_pages[tag]
                fallback_df = self.all_data[self.all_data["file"] == current_file]

                if offset < 0:
                    fallback_row = fallback_df.sort_values("page").iloc[-1]  # type: ignore
                else:
                    fallback_row = fallback_df.sort_values("page").iloc[0]  # type: ignore

                context_info.append((tag, current_file, int(fallback_row["page"])))
            else:
                context_info.append((tag, "empty", -1))  # Placeholder for empty input

        return context_info

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

        return {
            "files_and_pages": files_and_pages,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cnn_input": cnn_input,
            "labels": labels,
            "doc_type": torch.tensor(doc_type, dtype=torch.int),
            "distance": torch.tensor([distance], dtype=torch.float16),
        }
