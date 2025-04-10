# preprocess.py
from training.model.cnn.dataset import preprocess_pdfs_to_disk

if __name__ == "__main__":
    preprocess_pdfs_to_disk("data/dataset/pdfs", max_pages=30)
