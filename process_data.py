import os
import csv
import random
import fitz  # PyMuPDF
import pytesseract

from utils import (
    IMAGE_DIR,
    PDF_DIR,
    TESTING_DATA_CSV,
    TRAINING_DATA_CSV,
    TRAINING_PERCENTAGE,
    TYPES,
    DocumentType,
    clean_text,
    get_doc_typ_from_name,
    render_and_preprocess_page_in_memory,
)


type DatasetRow = tuple[str, int, DocumentType, str, int]

tesseract_config = (
    r"--oem 1 --psm 3"  # Use LSTM mode (1) and Default Page Segmentation Mode (3)
)

os.makedirs(IMAGE_DIR, exist_ok=True)
training_write_header = not os.path.exists(TRAINING_DATA_CSV)
testing_write_header = not os.path.exists(TESTING_DATA_CSV)


def get_data_from_pdf(page: int, doc) -> DatasetRow:
    """
    page -> 0-based page numbering
    """

    page_img = render_and_preprocess_page_in_memory(doc, page)
    page_filename = f"{os.path.splitext(pdf_file)[0]}_page_{page}.png"
    page_path = os.path.join(IMAGE_DIR, page_filename)
    page_img.save(page_path, "PNG")

    content = pytesseract.image_to_string(page_path, config=tesseract_config).strip()
    content = clean_text(content)
    content = content.replace("\n", " ")
    row = (content, page == 0, doc_type, pdf_file, page)

    return row


with (
    open(TRAINING_DATA_CSV, mode="a", encoding="utf-8", newline="") as train_file,
    open(TESTING_DATA_CSV, mode="a", encoding="utf-8", newline="") as test_file,
):
    train_writer = csv.writer(train_file)
    test_writer = csv.writer(test_file)

    headers = ["text", "is_first_page", "type", "file_name", "page_number"]
    if training_write_header:
        train_writer.writerow(headers)
    if testing_write_header:
        test_writer.writerow(headers)

    pdf_list = os.listdir(PDF_DIR)
    for i, pdf_file in enumerate(pdf_list):
        if not pdf_file.lower().endswith(".pdf"):
            continue

        doc_type = get_doc_typ_from_name(pdf_file)
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        print(
            f"Processing {list(TYPES.keys())[doc_type]} {i} / {len(pdf_list)}: {pdf_file}"
        )

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening {pdf_path}: {e}")
            continue

        if len(doc) == 0:
            print(f"No pages found in {pdf_path}")
            doc.close()
            continue

        first_page_row = get_data_from_pdf(0, doc)
        training = random.random() < TRAINING_PERCENTAGE
        if training:
            train_writer.writerow(first_page_row)
        else:
            test_writer.writerow(first_page_row)

        # recursively find a page that isn't blank
        found = False
        if len(doc) > 1:
            other_page_row = None
            max_iterations = 10
            i = 0
            while not found:
                i += 1
                random_page_num = random.randint(2, len(doc)) - 1
                print(f"random_page_num: {random_page_num}")
                other_page_row = get_data_from_pdf(random_page_num, doc)

                if (
                    other_page_row[0] != ""
                    and other_page_row[0] != None
                    or i >= max_iterations
                ):
                    found = True
                else:
                    # delete image file
                    os.remove(
                        os.path.join(
                            IMAGE_DIR,
                            f"{os.path.splitext(pdf_file)[0]}_page_{random_page_num}.png",
                        )
                    )

                print(f"found: {found}")

            if other_page_row is None:
                continue

            if training:
                train_writer.writerow(other_page_row)
            else:
                test_writer.writerow(other_page_row)

        doc.close()

print("Processing completed.")
