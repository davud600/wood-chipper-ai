import numpy as np
import fitz  # PyMuPDF
import random
import csv
import os
import re

from lib.doc_tools import get_image_contents, convert_pdf_page_to_image
from type_defs import (
    DatasetRow,
    EdgeCaseFiles,
    EdgeCases,
)

from config import (
    EDGE_CASES_FILE_PATH,
    PDF_DIR,
    TESTING_DATA_CSV,
    TRAINING_DATA_CSV,
    TRAINING_PERCENTAGE,
    DOCUMENT_TYPES,
)

from utils import (
    get_document_type,
)


# process_name = sys.argv[1] if len(sys.argv) > 1 else "default"
# total_processes = int(sys.argv[2]) if len(sys.argv) > 2 else 1
process_name = 1
total_processes = 1
print(f"running process: p-{process_name}")

# training_data_csv = f"{TRAINING_DATA_CSV.replace('.csv', '')}-{process_name}.csv"
# testing_data_csv = f"{TESTING_DATA_CSV.replace('.csv', '')}-{process_name}.csv"
training_data_csv = TRAINING_DATA_CSV
testing_data_csv = TESTING_DATA_CSV

training_write_header = not os.path.exists(training_data_csv)
testing_write_header = not os.path.exists(testing_data_csv)


def get_data_from_pdf(page: int, document_type: int, doc: fitz.open) -> DatasetRow:
    """page -> 1-based page numbering."""

    img = convert_pdf_page_to_image("", page - 1, doc)
    content = get_image_contents(np.array(img))
    row = (
        content,
        page,
        document_type,
        file,
    )

    return row


def get_edge_cases(edge_cases_file_path: str) -> EdgeCaseFiles:
    edge_cases_file = open(edge_cases_file_path, "r")
    edge_cases = edge_cases_file.read().split("\n")
    edge_case_files = {}

    for edge_case in edge_cases:
        if edge_case is None or ":" not in edge_case:
            continue

        [case, file] = edge_case.split(":")
        edge_case_files[file] = {"case": case}

    return edge_case_files


if __name__ == "__main__":
    training_dataset_files = None
    testing_dataset_files = None

    if not training_write_header and not testing_write_header:
        with (
            open(
                training_data_csv, mode="r", encoding="utf-8", newline=""
            ) as train_file,
            open(testing_data_csv, mode="r", encoding="utf-8", newline="") as test_file,
        ):
            training_dataset_files = [
                file for [_, _, _, file] in csv.reader(train_file)
            ][1:]
            testing_dataset_files = [file for [_, _, _, file] in csv.reader(test_file)][
                1:
            ]

    type_counters = [0 for _ in range(len(DOCUMENT_TYPES.keys()))]
    open_errors = [0 for _ in list(DOCUMENT_TYPES.keys())]
    no_pages = [0 for _ in list(DOCUMENT_TYPES.keys())]
    max_count_per_type = 1000
    max_pages_per_doc = 30
    edge_case_files = get_edge_cases(edge_cases_file_path=EDGE_CASES_FILE_PATH)

    with (
        open(training_data_csv, mode="a", encoding="utf-8", newline="\n") as train_file,
        open(testing_data_csv, mode="a", encoding="utf-8", newline="\n") as test_file,
    ):
        train_writer = csv.writer(train_file)
        test_writer = csv.writer(test_file)

        headers = ["content", "page", "type", "file"]
        if training_write_header:
            train_writer.writerow(headers)
        if testing_write_header:
            test_writer.writerow(headers)

        pdf_list = os.listdir(PDF_DIR)
        # chunks = split_into_n_chunks(pdf_list, total_processes)
        # pdf_list = chunks[int(process_name) - 1]
        print(f"p-{process_name} - {pdf_list[:3]}...")

        for i, file in enumerate(pdf_list):
            document_type = get_document_type(file)

            if (
                not file.lower().endswith(".pdf")
                or type_counters[document_type] >= max_count_per_type
            ):
                continue

            if (training_dataset_files and testing_dataset_files) and (
                file in training_dataset_files or file in testing_dataset_files
            ):
                # print("file already in dataset, skipping: ", file)
                type_counters[document_type] += 1
                continue

            edge_case = edge_case_files.get(file)
            if edge_case is not None and (edge_case["case"] == EdgeCases.DELETE.value):
                continue

            training = random.random() < TRAINING_PERCENTAGE
            pdf_path = os.path.join(PDF_DIR, file)
            print(
                f"p-{process_name} processing {sum(type_counters) + 1} / {len(pdf_list)} - {file}"
            )

            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                print(f"error opening {pdf_path}: {e}")
                open_errors[document_type] += 1
                continue

            if len(doc) == 0:
                print(f"no pages found in {pdf_path}")
                doc.close()
                no_pages[document_type] += 1
                continue

            start = 0
            if edge_case is not None:
                if "start" in edge_case["case"]:
                    match = re.search(EdgeCases.START.value, edge_case["case"])
                    if match is not None:
                        start = int(match.group(1)) - 1

            for page in range(start, min(len(doc), max_pages_per_doc)):
                row = get_data_from_pdf(page + 1, document_type, doc)
                (train_writer if training else test_writer).writerow(row)

            doc.close()
            type_counters[document_type] += 1

    print(f"p-{process_name} processing completed.")

    for t, document_type in enumerate(list(DOCUMENT_TYPES.keys())):
        print(f"p-{process_name} count {document_type}: {type_counters[t]}")
        print(f"p-{process_name} open errors {document_type}: {open_errors[t]}")
        print(f"p-{process_name} no pages {document_type}: {no_pages[t]}")
