import os
import csv
import random
import fitz  # PyMuPDF
import pytesseract
import io
import cv2
import numpy as np
import re
from PIL import Image

# Directories
PDF_DIR = "pdfs"
IMAGE_DIR = "pdf_images"
TRAINING_DATA_CSV = "training_data.csv"
TESTING_DATA_CSV = "testing_data.csv"
TRAINING_PERCENTAGE = 0.8  # Percentage of data to save in training data
TYPES = {
    "lease_renewal": 0,
    "closing_documents": 1,
    "sublease": 2
}

tesseract_config = r'--oem 1 --psm 3' # Use LSTM mode (1) and Default Page Segmentation Mode (3)
pymupdf_dpi = 300

# Ensure image directory exists
os.makedirs(IMAGE_DIR, exist_ok=True)

# Check if CSV exists to determine if we need a header
training_write_header = not os.path.exists(TRAINING_DATA_CSV)
testing_write_header = not os.path.exists(TESTING_DATA_CSV)

def get_doc_typ_from_name(file_name):
    if "lease renewal" in file_name.lower():
        return TYPES["lease_renewal"]
    elif "closing documents" in file_name.lower():
        return TYPES["closing_documents"]
    elif "sublease" in file_name.lower():
        return TYPES["sublease"]
    else:
        return TYPES["sublease"]

def normalize_image(image):
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def correct_skew(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Threshold the image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect lines using Hough Line Transform
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def scale_image(image, ppi=300):
    dpi_scale = ppi / 72
    new_size = (int(image.width * dpi_scale), int(image.height * dpi_scale))
    return image.resize(new_size, Image.LANCZOS)

def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def convert_to_grayscale(image):
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image has 3 channels
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # Return the image as-is if it's already grayscale

def binarize_image(image):
    gray = convert_to_grayscale(image)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return binary

def clean_text(text):
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove extraneous punctuation
    text = re.sub(r'[^a-zA-Z0-9.,;:\'"\-\s]', '', text)
    return text

# Open CSV files for training and testing
with open(TRAINING_DATA_CSV, mode='a', encoding='utf-8', newline='') as train_file, \
     open(TESTING_DATA_CSV, mode='a', encoding='utf-8', newline='') as test_file:

    train_writer = csv.writer(train_file)
    test_writer = csv.writer(test_file)

    # Write headers if necessary
    headers = ["text", "is_first_page", "type", "file_name", "page_number"]
    if training_write_header:
        train_writer.writerow(headers)
    if testing_write_header:
        test_writer.writerow(headers)

    # Iterate over PDFs
    pdf_list = os.listdir(PDF_DIR)
    for i, pdf_file in enumerate(pdf_list):
        if not pdf_file.lower().endswith(".pdf") or i > 100:
            continue

        pdf_path = os.path.join(PDF_DIR, pdf_file)
        print(f"Processing {i} / {len(pdf_list)}: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening {pdf_path}: {e}")
            continue

        if len(doc) == 0:
            print(f"No pages found in {pdf_path}")
            doc.close()
            continue

        # Function to render and preprocess a page to an in-memory image
        def render_and_preprocess_page(doc, page_num):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=pymupdf_dpi)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))

            # Convert to OpenCV format
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Preprocessing steps
            image = normalize_image(image)
            # image = correct_skew(image)
            # image = remove_noise(image)
            image = binarize_image(image)

            return Image.fromarray(image)

        # Render and preprocess the first page
        first_page_img = render_and_preprocess_page(doc, 0)
        first_page_filename = f"{os.path.splitext(pdf_file)[0]}_page1.png"
        first_page_path = os.path.join(IMAGE_DIR, first_page_filename)
        first_page_img.save(first_page_path, "PNG")

        # OCR for the first page
        first_page_text = pytesseract.image_to_string(first_page_path, config=tesseract_config).strip()
        # Post-process the extracted text
        first_page_text = clean_text(first_page_text)
        # Replace newlines with spaces
        first_page_text = first_page_text.replace("\n", " ")
        # Label = 1 for the first page
        row = [first_page_text, 1, get_doc_typ_from_name(pdf_file), pdf_file, 1]

        # Randomly assign to training or testing
        if random.random() < TRAINING_PERCENTAGE:
            train_writer.writerow(row)
        else:
            test_writer.writerow(row)

        # If there's more than one page, pick a random other page
        if len(doc) > 1:
            random_page_num = random.randint(2, len(doc))  # 1-based page numbering
            random_page_img = render_and_preprocess_page(doc, random_page_num - 1)
            random_page_filename = f"{os.path.splitext(pdf_file)[0]}_page{random_page_num}.png"
            random_page_path = os.path.join(IMAGE_DIR, random_page_filename)
            random_page_img.save(random_page_path, "PNG")

            # OCR for the random page
            random_page_text = pytesseract.image_to_string(random_page_path).strip()
            # Post-process the extracted text
            random_page_text = clean_text(random_page_text)
            # Replace newlines with spaces
            random_page_text = random_page_text.replace("\n", " ")
            # Label = 0 for the random page
            row = [random_page_text, 0, get_doc_typ_from_name(pdf_file), pdf_file, random_page_num]

            # Randomly assign to training or testing
            if random.random() < TRAINING_PERCENTAGE:
                train_writer.writerow(row)
            else:
                test_writer.writerow(row)

        doc.close()

print("Processing completed.")

