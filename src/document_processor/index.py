import pytesseract
import torch.nn as nn
import torch
import fitz
import os

from transformers import PreTrainedTokenizer
from PIL import Image

from src.utils import (
    DOWNLOADS_DIR,
    PROCESSING_IMAGES_DIR,
    SPLIT_DOCUMENTS_DIR,
    tesseract_config,
    clean_text,
    max_length,
    render_and_preprocess_page_in_memory,
)


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
        logits = model(features)
        page_class = torch.argmax(logits, dim=1).item()

        # print("page_logits.shape:", page_logits.shape)
        # print("page_logits:", page_logits)

        if page_class == 1:
            return True

    return False


def convert_pdf_page_to_image(file_name: str, page: int) -> str:
    """
    file_name -> file name of pdf inside downloads dir.
    page -> 0-based.
    """

    images_dir = f"{PROCESSING_IMAGES_DIR}/{file_name.replace('.pdf', '')}"
    file_path = f"{DOWNLOADS_DIR}/{file_name}"

    doc = fitz.open(file_path)
    page_img = render_and_preprocess_page_in_memory(doc, page)

    page_file_name = f"{page}.png"
    page_path = os.path.join(images_dir, page_file_name)
    os.makedirs(os.path.dirname(page_path), exist_ok=True)

    page_img.save(page_path, "PNG")
    image_path = os.path.join(images_dir, f"{page}.png")

    return image_path


def delete_pdf_images(file_name: str):
    """
    file_name -> original file name of pdf.
    """

    dir = f"{DOWNLOADS_DIR}/{file_name.replace('.pdf', '')}"

    os.remove(dir)


def get_image_contents(file_path: str) -> str:
    """
    file_path -> path of image file.
    """

    content = pytesseract.image_to_string(file_path, config=tesseract_config).strip()
    content = clean_text(content)

    return content.replace("\n", " ")


def create_sub_document(file_name: str, start_page: int, end_page: int, id: int) -> str:
    images_dir = os.path.join(PROCESSING_IMAGES_DIR, file_name.replace(".pdf", ""))
    split_doc_path = os.path.join(SPLIT_DOCUMENTS_DIR, file_name, f"{id}.pdf")

    os.makedirs(os.path.dirname(split_doc_path), exist_ok=True)

    image_files = sorted(os.listdir(images_dir))
    selected_images = image_files[start_page:end_page]

    image_paths = [os.path.join(images_dir, img) for img in selected_images]
    images = [Image.open(path) for path in image_paths]

    if not images:
        raise ValueError("No images found to generate PDF.")

    images[0].save(split_doc_path, save_all=True, append_images=images[1:])

    return split_doc_path
