to do:

text post-processing:

- dictionary-based correction (autocorrect)
- rule-based correction e.g. replace "rn" with "m", norm special chars like “—” to “-”

config:

- use Tesseract's LSTM-based mode (--oem 1) for better accuracy. - done
- use higher DPI for image rendering in PyMuPDF (fitz) (e.g., 300 DPI for better OCR results). - done
- extract bounding boxes for layout features and integrate them into your dataset for richer feature engineering (hOCR). - done

features:

- avg word ocr conf (or word conf list)
- avg bbox_stats (derived stats from hocr, word width, height and density) (or list)
- font_size (avg or individual words list)
- font_size_max (avg or individual words list)
- baseline_skew (avg or individual words list)
- has_table
- has_date

Option 1: Missing Features for Certain Training Data
Concept:

Train the model without the page_number feature when classifying whether it's the first page.
Include page_number during training for type classification once the first page is identified.

Option 2: No Page Number for First Page Detection
Concept:

Remove the page_number feature entirely and classify both first-page status and document type directly from the page content and layout.

problem: misclassified pages are usually middle pages that look very similar to first pages
solutions:

- rule-based system (e.g. min amount of pages per doc)
- sequential-information

scratched idea for hocr, it's taking too long so i'll look into that after trained on more data and different types.
if there are issues then i can compromise time to do hocr.
