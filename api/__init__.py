"""
API Module

This module provides entry points for background document processing
tasks: splitting and processing documents using contextual metadata.

Functions
---------
split_request(document_context)
    Splits the PDF document and initiates downstream processing for each page.

process_request(document_context)
    Processes the document contents, either from cached Redis content or by extracting it directly from the PDF.
"""

from .split import split_request
from .process import process_request
