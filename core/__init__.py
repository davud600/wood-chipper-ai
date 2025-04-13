"""
Core Processing Pipeline

This module provides the core document processing pipeline that
coordinates image extraction, OCR, and inference steps using
concurrent multiprocessing workers.

Functions
---------
process_pages_pipeline(pages, document_context, ...)
    Launches and coordinates the image, OCR, and inference workers to process document pages.
"""

from .pipeline import process_pages_pipeline
