"""Shared pipeline utilities."""

from .docling_batch_converter import BatchDoclingConverter, convert_folder
from .document_fetcher import DocumentFetcher, FetchStats
from .image_description_workflow import ImageDescriptionWorkflow, UploadSummary
from .docling_markdown_downloader import (
    DoclingMarkdownDownloader,
    MarkdownDownloadResult,
)
from .supabase import (
    SupabaseConfig,
    fetch_existing_representations,
    fetch_doc_ids_missing_docling_img,
)

from .gemini import validate_environment, init_client

__all__ = [
    "BatchDoclingConverter",
    "convert_folder",
    "DocumentFetcher",
    "FetchStats",
    "ImageDescriptionWorkflow",
    "UploadSummary",
    "DoclingMarkdownDownloader",
    "MarkdownDownloadResult",
    "SupabaseConfig",
    "fetch_existing_representations",
    "fetch_doc_ids_missing_docling_img",
    "validate_environment",
    "init_client",
]
