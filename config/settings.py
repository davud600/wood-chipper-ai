from dotenv import load_dotenv

import os

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = os.getenv("API_URL")

max_length = 512
prev_pages_to_append = 1
pages_to_append = 1
max_vocab_size = 60000
img_workers = 1
ocr_workers = 1
inf_workers = 1  # this can not be bigger than 1.
ocr_batch_size = 4
image_output_size = (1024, 1024)


max_chars = {
    "curr_page": 1024,
    "prev_page": 512,
    "next_page": 512,
}

special_tokens = [
    "<curr_page>",
    "</curr_page>",
]

for i in range(1, pages_to_append + 1, 1):
    special_tokens += [f"<prev_page_{i}>", f"</prev_page_{i}>"]

for i in range(1, pages_to_append + 1, 1):
    special_tokens += [f"<next_page_{i}>", f"</next_page_{i}>"]

DELETE_REDIS_KEYS_TIMEOUT = 60

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CORPUS_FILE = os.path.join(project_root, "data", "dataset", "corpus.txt")
TOKENIZER_PATH = os.path.join(project_root, "models", "tokenizer.pkl")
SPLITTER_MODEL_PATH = os.path.join(
    project_root, "data", "models", "splitter_all_types_2.pth"
)
# SPLITTER_MODEL_PATH = os.path.join(
#     project_root, "data", "models", "splitter_all_types.pth"
# )
# SPLITTER_MODEL_PATH = os.path.join(
#     project_root, "data", "models", "splitter_best_types.pth"
# )
SPLITTER_MODEL_DIR = os.path.join(project_root, "data", "models")

DOWNLOADS_DIR = os.path.join(project_root, "data", "downloads")
SPLIT_DOCUMENTS_DIR = os.path.join(project_root, "data", "split_documents")
IMAGES_DIR = os.path.join(project_root, "data", "dataset", "images")
TRAINING_DATA_CSV = os.path.join(project_root, "data", "dataset", "training.csv")
TESTING_DATA_CSV = os.path.join(project_root, "data", "dataset", "testing.csv")
PDF_DIR = os.path.join(project_root, "data", "dataset", "pdfs")
EDGE_CASES_FILE_PATH = os.path.join(project_root, "data", "dataset", "bad_files.txt")


def get_cnn_data_csv(output_size: tuple[int, int]):
    return os.path.join(
        project_root,
        "data",
        "dataset",
        "images",
        f"{output_size[0]}x{output_size[1]}.csv",
    )


def get_cnn_image_dir(output_size: tuple[int, int]):
    return os.path.join(
        project_root, "data", "dataset", "images", f"{output_size[0]}x{output_size[1]}"
    )


PAGE_SIMILARITY_THRESHOLD = 0.7
TRAINING_PERCENTAGE = 0.75
DOCUMENT_TYPES = {
    "unknown": 0,
    "lease": 1,
    "lease-agreement": 2,
    "lease-renewal": 3,
    "sublease": 4,
    "sublease-agreement": 5,
    "sublease-renewal": 6,
    "proprietary-lease": 7,
    "tenant-correspondence": 8,
    "transfer-of-title": 9,
    "purchase-application": 10,
    "closing-document": 11,
    "alteration-document": 12,
    "renovation-document": 13,
    "refinance-document": 14,
    "transfer-document": 15,
}

BEST_PERF_TYPES = [0, 1, 2, 3, 4, 5, 6, 7, 11]
