import numpy as np
import easyocr
import fitz

# import time
import os


from PIL import Image
from io import BytesIO

from config.settings import DOWNLOADS_DIR, SPLIT_DOCUMENTS_DIR, image_output_size
from utils.general import clean_text
from .images import apply_clahe, format_image_to_shape, denoise


reader = easyocr.Reader(["en"], gpu=True)


def convert_pdf_page_to_image(
    file_name: str,
    page: int,
    doc: fitz.open,
    keep_original_size=False,
    out_size: tuple[int, int] = image_output_size,
) -> np.ndarray | None:
    """
    file_name -> file name of pdf inside downloads dir.
    page -> 0-based.

    return -> grayscale image as np array [w, h].
    """

    try:
        mat = fitz.Matrix(2, 2)
        pix = doc.load_page(page).get_pixmap(matrix=mat, colorspace=fitz.csGRAY)  # type: ignore

        img = Image.open(BytesIO(pix.tobytes("jpg")))
        img = np.array(img)

        if not keep_original_size:
            img = format_image_to_shape(img, out_size[0], out_size[1])

        img = apply_clahe(img)
        img = denoise(img)
        # img = binarize(img)

        return img

    except Exception as e:
        print(f"failed to convert page to image {file_name}, page {page}:", e)


def get_image_batch_contents(images: list[np.ndarray]) -> list[str]:
    """
    image -> grayscale image as np array [w, h].
    """

    try:
        # Prepare a list of non-empty images for OCR processing.
        non_empty_images = []
        for img in images:
            if img.size != 0:
                non_empty_images.append(img)

        # Run OCR only on non-empty images.
        results = reader.readtext_batched(
            non_empty_images, detail=0, paragraph=False, decoder="greedy"
        )

        contents = []
        non_empty_index = 0  # Index for the results list.
        for img in images:
            if img.size == 0:
                # If image is empty, append an empty string.
                contents.append("")
            else:
                # Process OCR result: join tokens and clean newline characters.
                text = " ".join(results[non_empty_index])
                text = clean_text(text)
                text = text.replace("\n", " ")
                contents.append(text)
                non_empty_index += 1

        return contents

    except Exception as e:
        print(f"failed to get image conents:", e)

        return []


def get_image_contents(image: np.ndarray) -> str:
    """
    image -> grayscale image as np array [w, h].
    """

    try:
        # t0 = time.time()
        results = reader.readtext(image, detail=0, paragraph=False, decoder="greedy")
        # t1 = time.time()
        # print(f"ocr: {t1 - t0}")
        content = " ".join(results)
        content = clean_text(content)
        content = content.replace("\n", " ")

        return content

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
    start_page -> prev first page, included in sub document.
    end_page -> detected first page, included in sub document.
    """

    file_path = os.path.join(DOWNLOADS_DIR, file_name)
    output_path = os.path.join(SPLIT_DOCUMENTS_DIR, f"{id}.pdf")

    src_doc = fitz.open(file_path)
    sub_doc = fitz.open()

    sub_doc.insert_pdf(src_doc, from_page=start_page, to_page=end_page)
    sub_doc.save(output_path)

    return output_path
