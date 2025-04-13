import os
import csv
import fitz
from PIL import Image

from config.settings import (
    image_output_size,
    get_cnn_image_dir,
    get_cnn_data_csv,
    PDF_DIR,
)
from lib.doc_tools import convert_pdf_page_to_image


def extract_page_images_to_disk(pdf_dir: str, max_pages: int = 30):
    """
    Extracts images from PDF pages and saves them to disk.

    Saves one PNG per page, and writes corresponding metadata (file, page, label) to a CSV.

    Parameters
    ----------
    pdf_dir : str
        Directory containing input PDFs.

    max_pages : int, optional
        Maximum number of pages to extract per PDF.
    """

    image_dir = get_cnn_image_dir(image_output_size)
    os.makedirs(image_dir, exist_ok=True)

    csv_path = get_cnn_data_csv(image_output_size)
    csv_exists = os.path.exists(csv_path)

    # Open the CSV file in append mode
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # If the file is new, write the header
        if not csv_exists:
            writer.writerow(["file", "page", "label"])

        pdf_list = sorted([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])
        for i, file in enumerate(pdf_list):
            print(f"[{i+1}/{len(pdf_list)}] Processing: {file}")
            path = os.path.join(pdf_dir, file)

            try:
                doc = fitz.open(path)
            except Exception as e:
                print(f"  ⚠️ Failed to open {file}: {e}")
                continue

            for page_idx in range(min(len(doc), max_pages)):
                image_filename = f"{file}_page_{page_idx:03}.png"
                image_path = os.path.join(image_dir, image_filename)

                if os.path.exists(image_path):
                    print(f"  ⏩ Skipping page {page_idx}, image exists.")
                    continue

                try:
                    img = convert_pdf_page_to_image(file, page_idx, doc)
                    Image.fromarray(img).save(image_path)

                    label = 1 if page_idx == 0 else 0
                    writer.writerow([file, page_idx, label])
                    print(f"  ✅ Saved page {page_idx} as {image_filename}")
                except Exception as e:
                    print(f"  ❌ Error processing page {page_idx} of {file}: {e}")

            doc.close()


if __name__ == "__main__":
    """
    Command-line entry point for generating CNN-compatible images from PDFs.
    """

    extract_page_images_to_disk(PDF_DIR)
