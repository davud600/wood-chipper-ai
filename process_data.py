from Levenshtein import ratio
from typing import Dict, TypedDict
import pytesseract
import fitz  # PyMuPDF
import random
import csv
import os
import re


from utils import (
    EDGE_CASES_FILE_PATH,
    PAGE_SIMILARITY_THRESHOLD,
    IMAGE_DIR,
    PDF_DIR,
    TESTING_DATA_CSV,
    TRAINING_DATA_CSV,
    TRAINING_PERCENTAGE,
    TYPES,
    DocumentType,
    EdgeCases,
    PageType,
    clean_text,
    render_and_preprocess_page_in_memory,
)


type DatasetRow = tuple[str, int, int, str, int]


class EdgeCaseFile(TypedDict):
    type: int
    case: str


type EdgeCaseFiles = Dict[str, EdgeCaseFile]

tesseract_config = r"--oem 1 --psm 3"
# tesseract_config = r"--oem 0 --psm 3 --dpi 72"

os.makedirs(IMAGE_DIR, exist_ok=True)
training_write_header = not os.path.exists(TRAINING_DATA_CSV)
testing_write_header = not os.path.exists(TESTING_DATA_CSV)


def get_data_from_pdf(page: int, page_type: int, type: int, doc) -> DatasetRow:
    """page -> 0-based page numbering."""

    page_img = render_and_preprocess_page_in_memory(doc, page)
    page_filename = f"{os.path.splitext(pdf_file)[0]}_page_{page}.png"
    page_path = os.path.join(IMAGE_DIR, page_filename)
    page_img.save(page_path, "PNG")

    content = pytesseract.image_to_string(page_path, config=tesseract_config).strip()
    content = clean_text(content)
    content = content.replace("\n", " ")
    # type is 0 (unknown) if not the first page.
    row = (
        content,
        page_type,
        DocumentType.UNKNOWN.value if page_type == PageType.OTHER else type,
        pdf_file,
        page,
    )

    return row


def get_other_page_data_from_pdf(
    doc, original_page_row: DatasetRow, alias_page_row: DatasetRow | None
) -> DatasetRow | None:
    """recursively find a page that isn't blank and similar to original or alias page."""

    other_page_row = None
    max_iterations = 5
    j = 0
    found = False
    while not found:
        j += 1
        random_page_num = random.randint(2, len(doc)) - 1
        other_page_row = get_data_from_pdf(
            random_page_num, PageType.OTHER.value, DocumentType.UNKNOWN.value, doc
        )
        similarity_with_original_page = ratio(other_page_row[0], original_page_row[0])
        similarity_with_alias_page = (
            ratio(other_page_row[0], alias_page_row[0])
            if alias_page_row is not None
            else 0
        )

        if (
            other_page_row[0] != ""
            and other_page_row[0] != None
            and similarity_with_original_page <= PAGE_SIMILARITY_THRESHOLD
            and similarity_with_alias_page <= PAGE_SIMILARITY_THRESHOLD
        ) or j >= max_iterations:
            found = True
        else:
            # delete image file
            os.remove(
                os.path.join(
                    IMAGE_DIR,
                    f"{os.path.splitext(pdf_file)[0]}_page_{random_page_num}.png",
                )
            )

    return other_page_row


def get_edge_cases(edge_cases_file_path: str) -> EdgeCaseFiles:
    edge_cases_file = open(edge_cases_file_path, "r")
    edge_cases = edge_cases_file.read().split("\n")
    edge_case_type = DocumentType.UNKNOWN.value
    edge_case_files = {}
    for edge_case in edge_cases:
        if edge_case is None or ":" not in edge_case:
            continue

        if "type:" in edge_case:
            edge_case_type = TYPES[edge_case.replace("type:", "")]
            continue

        [case, file_name] = edge_case.split(":")
        edge_case_files[file_name] = {"type": edge_case_type, "case": case}

    return edge_case_files


if __name__ == "__main__":
    training_dataset_files = None
    testing_dataset_files = None
    if not training_write_header and not testing_write_header:
        with (
            open(
                TRAINING_DATA_CSV, mode="r", encoding="utf-8", newline=""
            ) as train_file,
            open(TESTING_DATA_CSV, mode="r", encoding="utf-8", newline="") as test_file,
        ):
            training_dataset_files = [
                file_name for [_, _, _, file_name, _] in csv.reader(train_file)
            ][1:]
            testing_dataset_files = [
                file_name for [_, _, _, file_name, _] in csv.reader(test_file)
            ][1:]

    type_counters = [0 for _ in range(len(TYPES.keys()))]
    type_errors = [0 for _ in list(TYPES.keys())]
    open_errors = [0 for _ in list(TYPES.keys())]
    no_pages = [0 for _ in list(TYPES.keys())]
    max_count_per_type = 300
    edge_case_files = get_edge_cases(edge_cases_file_path=EDGE_CASES_FILE_PATH)
    with (
        open(TRAINING_DATA_CSV, mode="a", encoding="utf-8", newline="") as train_file,
        open(TESTING_DATA_CSV, mode="a", encoding="utf-8", newline="") as test_file,
    ):
        train_writer = csv.writer(train_file)
        test_writer = csv.writer(test_file)

        headers = ["text", "page", "type", "file_name", "page_number"]
        if training_write_header:
            train_writer.writerow(headers)
        if testing_write_header:
            test_writer.writerow(headers)

        for doc_type in TYPES.values():
            if doc_type != 1 and doc_type != 2:
                continue

            pdf_list = os.listdir(f"{PDF_DIR}/{doc_type}")
            for i, pdf_file in enumerate(pdf_list):
                if (
                    not pdf_file.lower().endswith(".pdf")
                    or type_counters[doc_type] >= max_count_per_type
                ):
                    continue

                if (training_dataset_files and testing_dataset_files) and (
                    pdf_file in training_dataset_files
                    or pdf_file in testing_dataset_files
                ):
                    print("File already in dataset, skipping: ", pdf_file)
                    continue

                edge_case = edge_case_files.get(pdf_file)
                if edge_case is not None and (
                    edge_case["case"] == EdgeCases.DELETE.value
                    or edge_case["case"] == EdgeCases.AGREEMENT.value
                    or edge_case["case"] == EdgeCases.SUBLEASE.value
                ):
                    continue

                training = random.random() < TRAINING_PERCENTAGE
                pdf_path = os.path.join(PDF_DIR, str(doc_type), pdf_file)
                print(
                    f"Processing {list(TYPES.keys())[doc_type]} {i} / {max_count_per_type}: {pdf_file}"
                )

                try:
                    doc = fitz.open(pdf_path)
                except Exception as e:
                    print(f"Error opening {pdf_path}: {e}")
                    open_errors[doc_type] += 1
                    continue

                if len(doc) == 0:
                    print(f"No pages found in {pdf_path}")
                    doc.close()
                    no_pages[doc_type] += 1
                    continue

                original_page_num = 0
                alias_page_num = None
                if edge_case is not None:
                    if "start" in edge_case["case"]:
                        match = re.search(EdgeCases.START.value, edge_case["case"])
                        if match is not None:
                            original_page_num = int(match.group(1)) - 1
                    elif "alias" in edge_case["case"]:
                        match = re.search(EdgeCases.ALIAS.value, edge_case["case"])
                        if match is not None:
                            alias_page_num = int(match.group(1)) - 1

                original_page_row = get_data_from_pdf(
                    original_page_num, PageType.ORIGINAL.value, doc_type, doc
                )
                (train_writer if training else test_writer).writerow(original_page_row)

                if len(doc) <= 1:
                    print(f"Only 1 page found in {pdf_path}")
                    doc.close()
                    type_counters[doc_type] += 1
                    continue

                alias_page_row = None
                if alias_page_num is not None:
                    alias_page_row = get_data_from_pdf(
                        alias_page_num, PageType.ALIAS.value, doc_type, doc
                    )
                    (train_writer if training else test_writer).writerow(alias_page_row)

                other_page_row = get_other_page_data_from_pdf(
                    doc,
                    original_page_row,
                    alias_page_row,
                )
                if other_page_row is not None:
                    (train_writer if training else test_writer).writerow(other_page_row)

                doc.close()
                type_counters[doc_type] += 1

    print("Processing completed.")
