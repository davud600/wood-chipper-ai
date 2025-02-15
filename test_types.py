import os
import fitz  # PyMuPDF

from utils import TYPES, get_doc_type_from_name

PDF_DIR = "/home/davud/bombonjero-ai/pdfs"


type_errors = [0 for _ in list(TYPES.keys())]
open_errors = [0 for _ in list(TYPES.keys())]
no_pages = [0 for _ in list(TYPES.keys())]
# print(len(type_errors), type_errors)
# print(len(list(TYPES.keys())), list(TYPES.keys()))

for type in TYPES.values():
    pdf_list = os.listdir(f"{PDF_DIR}/{type}")
    for i, pdf_file in enumerate(pdf_list):
        if not pdf_file.lower().endswith(".pdf"):
            continue

        doc_type = get_doc_type_from_name(pdf_file)
        if doc_type != type:
            # print(f"incorrect {type_errors}: {i} / {len(pdf_list)} '{pdf_file.lower()}'")
            if type == 10:
                print(
                    f"incorrect {type_errors[type]}: {i} / {len(pdf_list)} '{pdf_file.lower()}'"
                )
            type_errors[type] += 1

        pdf_path = os.path.join(PDF_DIR, pdf_file)

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            # print(f"Error opening {pdf_path}: {e}")
            open_errors[type] += 1
            continue

        if len(doc) == 0:
            # print(f"No pages found in {pdf_path}")
            no_pages[type] += 1
            doc.close()


# print(len(type_errors), type_errors)
# print(len(list(TYPES.keys())), list(TYPES.keys()))

print(f"Processing completed. {sum(type_errors)} type_errors")
for type in range(len(type_errors)):
    print(f"{type} - {list(TYPES.keys())[type]}: {type_errors[type]}")
