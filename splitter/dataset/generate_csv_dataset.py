import numpy as np
import fitz  # PyMuPDF
import random
import csv
import os

from type_defs.shared import DatasetRow
from config.settings import (
    PDF_DIR,
    TRAINING_DATA_CSV,
    TESTING_DATA_CSV,
    TRAINING_PERCENTAGE,
    DOCUMENT_TYPES,
)
from utils.general import get_document_type
from lib.doc_tools import get_image_contents, convert_pdf_page_to_image


def extract_row_from_pdf_page(
    page: int, document_type: int, doc: fitz.open
) -> DatasetRow:
    """
    Extracts OCR content and metadata from a single PDF page.

    Parameters
    ----------
    page : int
        Page number (1-based).

    document_type : int
        Numerical ID representing the document type.

    doc : fitz.Document
        Open PyMuPDF document.

    Returns
    -------
    DatasetRow
        A tuple of (content, page number, document type, file name).
    """

    img = convert_pdf_page_to_image("", page - 1, doc, keep_original_size=True)
    content = get_image_contents(np.array(img))
    row = (
        content,
        page,
        document_type,
        file,
    )

    return row


if __name__ == "__main__":
    """
    Command-line entry point for extracting text from pdfs using OCR
    and creating training & testing datasets csv.
    """

    print(f"running process: p-1")

    training_write_header = not os.path.exists(TRAINING_DATA_CSV)
    testing_write_header = not os.path.exists(TESTING_DATA_CSV)

    training_dataset_files, testing_dataset_files = [], []
    if not training_write_header and not testing_write_header:
        with open(TRAINING_DATA_CSV, "r", encoding="utf-8") as train_file:
            training_dataset_files = [row[3] for row in csv.reader(train_file)][1:]
        with open(TESTING_DATA_CSV, "r", encoding="utf-8") as test_file:
            testing_dataset_files = [row[3] for row in csv.reader(test_file)][1:]

    type_counters = [0 for _ in DOCUMENT_TYPES]
    open_errors = [0 for _ in DOCUMENT_TYPES]
    no_pages = [0 for _ in DOCUMENT_TYPES]
    max_count_per_type = 1000
    max_pages_per_doc = 30

    with open(
        TRAINING_DATA_CSV, "a", encoding="utf-8", newline="\n"
    ) as train_file, open(
        TESTING_DATA_CSV, "a", encoding="utf-8", newline="\n"
    ) as test_file:

        train_writer, test_writer = csv.writer(train_file), csv.writer(test_file)
        headers = ["content", "page", "type", "file"]
        if training_write_header:
            train_writer.writerow(headers)
        if testing_write_header:
            test_writer.writerow(headers)

        pdf_list = os.listdir(PDF_DIR)
        print(f"p-1 - starting with {len(pdf_list)} files...")

        for i, file in enumerate(pdf_list):
            if not file.lower().endswith(".pdf"):
                continue

            document_type = get_document_type(file)
            if type_counters[document_type] >= max_count_per_type:
                continue

            if file in training_dataset_files or file in testing_dataset_files:
                print(f"p-1 skipping already processed: {file}")
                type_counters[document_type] += 1
                continue

            training = random.random() < TRAINING_PERCENTAGE
            pdf_path = os.path.join(PDF_DIR, file)
            print(f"p-1 processing {i+1}/{len(pdf_list)}: {file}")

            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                print(f"p-1 error opening {file}: {e}")
                open_errors[document_type] += 1
                continue

            if len(doc) == 0:
                print(f"p-1 no pages found in {file}")
                no_pages[document_type] += 1
                continue

            for page in range(0, min(len(doc), max_pages_per_doc)):
                row = extract_row_from_pdf_page(page + 1, document_type, doc)
                (train_writer if training else test_writer).writerow(row)

            doc.close()
            type_counters[document_type] += 1

    print(f"p-1 processing completed.")
    for t, doc_type in enumerate(DOCUMENT_TYPES.keys()):
        print(f"p-1 count {doc_type}: {type_counters[t]}")
        print(f"p-1 open errors {doc_type}: {open_errors[t]}")
        print(f"p-1 no pages {doc_type}: {no_pages[t]}")
