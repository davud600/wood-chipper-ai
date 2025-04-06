import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
OPENAI_API_KEY = (
    "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"  # dude env pls...
)


api_url = "http://localhost:3001"
max_length = 3064
pages_to_append = 4
max_vocab_size = 60000
training_mini_batch_size = 16
testing_mini_batch_size = 16
learning_rate = 0.000075
weight_decay = 0.005
patience = 10
factor = 0.5
epochs = 1
log_steps = 10
eval_steps = 50
pymupdf_dpi = 300
img_workers = 2
ocr_workers = 1
inf_workers = 1  # this can not be bigger than 1.
ocr_batch_size = 1
image_output_size = (720, 1280)


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
TRAINING_DATA_CSV = os.path.join(project_root, "data", "dataset", "training_data.csv")
TESTING_DATA_CSV = os.path.join(project_root, "data", "dataset", "testing_data.csv")
PDF_DIR = os.path.join(project_root, "data", "dataset", "pdfs")
EDGE_CASES_FILE_PATH = os.path.join(project_root, "data", "dataset", "bad_files.txt")

# pymupdf_dpi = 72
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
