"""
Document Processing Services

This module launches subprocesses for parallel image generation,
OCR, and model inference, using shared Redis-backed queues.

Functions
---------
start_img_producers(...)
    Starts parallel processes for generating images from PDF pages.

start_ocr_workers(...)
    Starts OCR worker processes for batched image-to-text extraction.

start_inf_workers(...)
    Starts inference worker processes for document boundary detection.
"""

from .img_producers import start_img_producers
from .ocr_workers import start_ocr_workers
from .inf_workers import start_inf_workers
