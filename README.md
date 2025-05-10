# 🪓 Wood Chipper AI — Multimodal Document Splitter

**Wood Chipper AI** is an intelligent document processing pipeline designed to automatically detect and segment documents using both **textual** and **visual (image-based)** signals. It combines a **FusionModel** (LLM + CNN), Redis-backed multiprocessing, and a Flask API interface for seamless integration into backend systems.

---

## 🔧 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [How It Works](#how-it-works)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Training](#2-model-training)
  - [3. Inference Pipeline](#3-inference-pipeline)
  - [4. Flask API](#4-flask-api)
- [Modules Overview](#modules-overview)
- [Usage](#usage)
- [Dataset Format](#dataset-format)
- [Environment Variables](#environment-variables)
- [Future Enhancements](#future-enhancements)

---

## 📘 Project Overview

The goal of this project is to **intelligently split multi-document PDF files** into individual documents by analyzing both their **textual content** (via LLM) and **visual layout** (via CNN). This is especially useful in enterprise, legal, or financial document flows where long files are scanned or bulk-uploaded with little structure.

It supports:

- Text + image-based classification of **first pages**
- Parallelized processing via **multiprocessing + Redis queues**
- Document segmentation & re-upload to **AWS S3**
- Modular & testable design with **scipy-style docs** throughout

---

## ⚙️ Architecture

```text
[PDF Upload]
    ↓
[Flask API]
    ↓
[Split or Process]
    ↓
[Image Producers] ← fitz
    ↓
[OCR Workers] ← batch OCR
    ↓
[Inference Workers] ← FusionModel (LLM + CNN)
    ↓
[First Page Detection]
    ↓
[Sub-Doc Creation + S3 Upload]
```

- Shared queues are Redis-backed and synchronized via keys.
- Image, OCR, and Inference processes are fully parallelized.

---

## 📦 Installation

```bash
# 1. Clone repo
git clone https://github.com/your-org/wood-chipper-ai.git
cd wood-chipper-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables (see below)

# 4. Start the Flask API
python server.py
```

You will also need:

- Redis server running (`redis-server`)
- Access to AWS S3 (credentials in environment)
- Pretrained model weights in `config.SPLITTER_MODEL_PATH`

---

## 🧠 How It Works

### 1. Data Preprocessing

```bash
# Step 1: Create labeled CSVs
python dataset/generate_csv_dataset.py

# Step 2: Extract and save page images
python dataset/extract_page_images.py
```

This creates:

- `training.csv`, `testing.csv`
- PNGs per page, saved by filename convention
- CSV file for indexing images.

### 2. Model Training

```bash
python splitter/train.py
```

- Loads `DocumentDataset`
- Trains the `FusionModel` on both text and images
- Logs F1 score and saves best model

### 3. Inference Pipeline

Called via the `/split` endpoint:

- PDF is downloaded from S3
- Images are created and pushed to Redis
- OCR workers extract text
- Inference workers predict first pages
- Detected sub-docs are:
  - Merged using PyMuPDF
  - Saved to disk
  - Uploaded back to S3

### 4. Flask API

| Endpoint   | Method | Description                   |
| ---------- | ------ | ----------------------------- |
| `/split`   | POST   | Start document splitting      |
| `/process` | POST   | Start content processing only |

Example request:

```json
{
  "token": "abc123",
  "transaction_id": 42,
  "document_id": 7,
  "signed_get_url": "https://s3.aws.com/mydoc.pdf"
}
```

---

## 📁 Modules Overview

### `server.py`

Flask API entrypoint. Validates JSON and triggers background threading jobs.

### `api/`

Contains `split_request()` and `process_request()` entrypoints run in background threads.

### `core/`

Manages the full pipeline of:

- Image producers
- OCR workers
- Inference workers

### `splitter/`

Home of the `FusionModel` and training/inference scripts.

- `FusionModel` combines LLM + CNN logits via an MLP.
- `inference.py` determines first-page classification.

### `dataset/`

Handles all data prep:

- Loading page data
- Constructing CNN input (multi-page)
- Tokenizing context text

### `services/`

Multiprocessing backends:

- `img_producers.py`: Convert pages to images
- `ocr_workers.py`: Batch OCR
- `inf_workers.py`: Run the trained model
- Shared via Redis queues

---

## 🗂️ Dataset Format

Each row in the CSV contains:

| Column  | Type   | Description         |
| ------- | ------ | ------------------- |
| content | string | OCR text            |
| page    | int    | 1-based page number |
| type    | int    | Document type/class |
| file    | str    | Filename of PDF     |

Images are saved in:
`<IMAGES_DIR>/<WIDTH>x<HEIGHT>/<file>_page_<###>.png`

---

## 🔑 Environment Variables & Configs

Stored in `config/settings.py`

- `PDF_DIR` – Directory with source PDFs
- `IMAGES_DIR` – Where PNGs are stored
- `SPLITTER_MODEL_PATH` – Path to `.pt` model file
- `pages_to_append`, `prev_pages_to_append` – Context window size
- `max_chars` – Char limit per context type
- `AWS` credentials for `signed_get_url` + upload

---

## 🤝 Credits

Built by @davud600 (github).
Model inspired by hybrid NLP + CV document understanding techniques.
