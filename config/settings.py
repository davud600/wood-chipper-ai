from dotenv import load_dotenv

import os

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = os.getenv("API_URL")

# max_length = 3064
max_length = 512
prev_pages_to_append = 1
pages_to_append = 1
max_vocab_size = 60000
training_mini_batch_size = 42
testing_mini_batch_size = 42
learning_rate = 0.00005
weight_decay = 0.0001
patience = 15
factor = 0.5
epochs = 5
log_steps = 10
eval_steps = 50
img_workers = 2
ocr_workers = 1
inf_workers = 1  # this can not be bigger than 1.
ocr_batch_size = 2
image_output_size = (900, 1000)  # upscale this when processing dataset.


special_tokens = [
    "<curr_page>",
    "</curr_page>",
]

for i in range(1, pages_to_append + 1, 1):
    special_tokens += [f"<next_page_{i}>", f"</next_page_{i}>"]

DELETE_REDIS_KEYS_TIMEOUT = 60

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CORPUS_FILE = os.path.join(project_root, "data", "dataset", "corpus.txt")
TOKENIZER_PATH = os.path.join(project_root, "models", "tokenizer.pkl")
SPLITTER_MODEL_PATH = os.path.join(project_root, "models", "splitter.pth")
DOWNLOADS_DIR = os.path.join(project_root, "data", "downloads")
SPLIT_DOCUMENTS_DIR = os.path.join(project_root, "data", "split_documents")
IMAGES_DIR = os.path.join(project_root, "data", "dataset", "images")
TRAINING_DATA_CSV = os.path.join(project_root, "data", "dataset", "training.csv")
TESTING_DATA_CSV = os.path.join(project_root, "data", "dataset", "testing.csv")
PDF_DIR = os.path.join(project_root, "data", "dataset", "pdfs")
EDGE_CASES_FILE_PATH = os.path.join(project_root, "data", "dataset", "bad_files.txt")

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
