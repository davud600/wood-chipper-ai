import torch.nn as nn
import numpy as np
import easyocr
import torch

# import redis
import fitz
import nltk
import os

from transformers import PreTrainedTokenizer
from PIL import Image
from io import BytesIO


from src.utils import (
    DOWNLOADS_DIR,
    SPLIT_DOCUMENTS_DIR,
    clean_text,
    light_autocorrect,
    max_length,
)

# light auto correct setup.
nltk.download("words")
from nltk.corpus import words

english_words = set(w.lower() for w in words.words())

reader = easyocr.Reader(["en"], gpu=True)


def is_first_page(
    tokenizer: PreTrainedTokenizer, model: nn.Module, content: str
) -> tuple[bool, int]:
    """
    returns -> bool (is first page), offset (int)
    """

    if "newdocumentseparator" in content.split("</curr_page>")[0]:
        return True, 1

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
            return True, 0

    return False, 0


def convert_pdf_page_to_image(
    file_name: str, page: int, doc: fitz.open
) -> np.ndarray | None:
    """
    file_name -> file name of pdf inside downloads dir.
    page -> 0-based.

    return -> grayscale image as np array [w, h].
    """

    try:
        pix = doc.load_page(page).get_pixmap(  # type: ignore
            matrix=fitz.Matrix(1, 1), colorspace=fitz.csGRAY
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
        content = content.replace("\n", " ")

        return light_autocorrect(english_words, content)
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
    """
    file_name -> file name of parent document inside downloads dir.
    start_page -> prev first page, NOT included in sub document.
    end_page -> detected first page, included in sub document.
    """

    file_path = os.path.join(DOWNLOADS_DIR, file_name)
    output_path = os.path.join(SPLIT_DOCUMENTS_DIR, f"{id}.pdf")

    src_doc = fitz.open(file_path)
    sub_doc = fitz.open()

    sub_doc.insert_pdf(src_doc, from_page=start_page + 1, to_page=end_page)
    sub_doc.save(output_path)

    return output_path


# def get_formatted_page_content_from_file_or_redis(
#     document_id: int,
#     file_name: str,
#     page: int,  # 0-based.
#     page_content: str,
#     pages_to_append: int,
#     r: redis.Redis,
#     document_pages: int = 10,
#     doc: fitz.open | None = None,
#     check_redis: bool = False,
# ) -> str:
#     content = f"<curr_page>{page_content}</curr_page>"
#
#     for j in range(1, min(pages_to_append + 1, document_pages - page), 1):
#         next_page_content = None
#
#         if check_redis:
#             next_page_content = r.get(f"page_content:{document_id}:{page + j}")
#
#         if next_page_content is None and doc is not None:
#             next_page_image = convert_pdf_page_to_image(file_name, page + j, doc)
#
#             next_page_content = ""
#             if next_page_image is not None:
#                 next_page_content = get_image_contents(next_page_image)
#                 r.set(f"page_content:{document_id}:{page + j}", next_page_content)
#
#         content += f"<next_page_{j}>{next_page_content}</next_page_{j}>"
#
#     return content
