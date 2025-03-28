import torch.nn as nn
import numpy as np
import easyocr
import torch
import fitz
import os

from transformers import PreTrainedTokenizer
from PIL import Image
from io import BytesIO

from src.utils import (
    DOWNLOADS_DIR,
    SPLIT_DOCUMENTS_DIR,
    clean_text,
    max_length,
)

reader = easyocr.Reader(["en"], gpu=True)


def is_first_page(
    tokenizer: PreTrainedTokenizer, model: nn.Module, content: str
) -> bool:
    if "newdocumentseparator" in content.split("</curr_page>")[0]:
        return True

    tokenized = tokenizer(
        [content],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    # print(tokenized.input_ids)

    features = tokenized.input_ids.to("cuda")

    with torch.amp.autocast_mode.autocast(device_type="cuda", dtype=torch.float16):
        logit = model(features)
        page_class = int(logit > 0)

        if page_class == 1:
            return True

    return False


def convert_pdf_page_to_image(file_name: str, page: int) -> np.ndarray | None:
    """
    file_name -> file name of pdf inside downloads dir.
    page -> 0-based.

    return -> grayscale image as np array [w, h].
    """

    try:
        file_path = f"{DOWNLOADS_DIR}/{file_name}"
        doc = fitz.open(file_path)

        pix = doc.load_page(page).get_pixmap(  # type: ignore
            matrix=fitz.Matrix(2, 2), colorspace=fitz.csGRAY
        )
        img = Image.open(BytesIO(pix.tobytes("jpg")))

        return np.array(img)
    except Exception as e:
        print(f"failed to convert page to image {file_name}, page {page}:", e)


def get_image_contents(file: np.ndarray) -> str:
    """
    file -> grayscale image as np array [w, h].
    """

    try:
        results = reader.readtext(file, detail=0, paragraph=False, decoder="greedy")
        content = " ".join(results)
        content = clean_text(content)

        return content.replace("\n", " ")
    except Exception as e:
        print(f"failed to get image conents:", e)

        return ""


def delete_pdf_images(file_name: str):
    """
    file_name -> original file name of pdf.
    """

    dir = f"{DOWNLOADS_DIR}/{file_name.replace('.pdf', '')}"

    os.remove(dir)


def create_sub_document(file_name: str, start_page: int, end_page: int, id: int) -> str:
    file_path = os.path.join(DOWNLOADS_DIR, file_name)
    output_path = os.path.join(SPLIT_DOCUMENTS_DIR, f"{id}.pdf")

    src_doc = fitz.open(file_path)
    sub_doc = fitz.open()

    sub_doc.insert_pdf(src_doc, from_page=start_page, to_page=end_page)
    sub_doc.save(output_path)

    return output_path

    # images_dir = os.path.join(PROCESSING_IMAGES_DIR, file_name.replace(".pdf", ""))
    # split_doc_path = os.path.join(SPLIT_DOCUMENTS_DIR, file_name, f"{id}.pdf")
    #
    # os.makedirs(os.path.dirname(split_doc_path), exist_ok=True)
    #
    # image_files = sorted(os.listdir(images_dir))
    # selected_images = image_files[start_page:end_page]
    #
    # image_paths = [os.path.join(images_dir, img) for img in selected_images]
    # images = [Image.open(path) for path in image_paths]
    #
    # if not images:
    #     raise ValueError("No images found to generate PDF.")
    #
    # images[0].save(split_doc_path, save_all=True, append_images=images[1:])
    #
    # return split_doc_path
