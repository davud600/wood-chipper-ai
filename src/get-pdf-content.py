import pytesseract
import fitz
import csv
import os

from utils import (
    PROCESSING_IMAGES_DIR,
    clean_text,
    render_and_preprocess_page_in_memory,
)


tesseract_config = r"--oem 1 --psm 3"

file_name = "merged.pdf"
file_path = os.path.join("../", file_name)

doc = fitz.open(file_path)

contents: list[str] = []
for page in range(0, len(doc)):
    print(f"processing page {page}...")
    page_img = render_and_preprocess_page_in_memory(doc, page)
    page_file_name = f"{file_name.replace('.pdf', '')}_page_{page + 1}.png"
    page_path = os.path.join(
        PROCESSING_IMAGES_DIR, file_name.replace(".pdf", ""), page_file_name
    )
    os.makedirs(os.path.dirname(page_path), exist_ok=True)
    page_img.save(page_path, "PNG")

    content = pytesseract.image_to_string(page_path, config=tesseract_config).strip()
    content = clean_text(content)
    content = content.replace("\n", " ")
    contents += [content]

with open(
    f"../{file_name.replace('.pdf', '')}.txt", mode="w", encoding="utf-8"
) as file:
    writer = csv.writer(file)
    for i, content in enumerate(contents):
        writer.writerow([content, i + 1, 1, "merged.pdf"])
