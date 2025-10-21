"""High-level Docling batch conversion utilities."""

from __future__ import annotations
import logging
import multiprocessing
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set

from src.docling_converter import DoclingConverter
from src.processing_logger import ProcessingLogger
from src.pipeline.supabase import SupabaseConfig, fetch_existing_representations
from src.pipeline.paths import DATA_DIR, LOGS_DIR, STATE_DIR

__all__ = [
    "BatchDoclingConverter",
    "convert_folder",
]


def wrap_long_lines(text: str, max_line_length: int = 10000) -> str:
    lines = text.split("\n")
    wrapped_lines: List[str] = []

    for line in lines:
        if len(line) <= max_line_length:
            wrapped_lines.append(line)
            continue

        remaining = line
        while len(remaining) > max_line_length:
            break_point = max_line_length
            space_idx = remaining.rfind(" ", 0, max_line_length)
            if space_idx > max_line_length * 0.8:
                break_point = space_idx + 1
            wrapped_lines.append(remaining[:break_point])
            remaining = remaining[break_point:]
        if remaining:
            wrapped_lines.append(remaining)

    return "\n".join(wrapped_lines)


def clean_repeating_characters(text: str) -> str:
    text = re.sub(r" {10,}", " " * 9, text)

    def replace_long_repeats(match: re.Match[str]) -> str:
        char = match.group(1)
        return char * 49

    return re.sub(r"([^\s])\1{49,}", replace_long_repeats, text)


def _process_single_pdf_worker(
    pdf_file_path: str,
    output_path: str,
    raw_output_path: str,
    images_output_dir: str,
    use_gpu: bool,
    table_mode: str,
    images_scale: float,
    do_cell_matching: bool,
    ocr_confidence_threshold: float,
    add_page_numbers: bool,
    chunk_page_limit: int,
    batch_size: int = 2,
    images_only: bool = False,
) -> Tuple[str, str, bool, str, int, int, float]:
    import os
    import subprocess
    import time
    from pathlib import Path

    from src.gpu_memory_manager import (
        clear_gpu_cache,
        is_oom_error,
        log_gpu_memory_stats,
        set_process_gpu_memory_fraction,
        wait_for_gpu_memory,
    )

    if use_gpu:
        from src.gpu_memory_manager import setup_gpu_memory_limits

        memory_fraction = 0.8 / batch_size
        setup_gpu_memory_limits(memory_fraction=memory_fraction, max_split_size_mb=128)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    pdf_file = Path(pdf_file_path)

    try:
        result = subprocess.run(
            ["file", "--mime-type", "-b", str(pdf_file)],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        mime_type = result.stdout.strip()

        if mime_type.startswith("text/"):
            error_msg = f"Skipping {pdf_file.name} - detected as text file, not a PDF"
            try:
                pdf_file.unlink()
                error_msg += " [DELETED]"
            except Exception as exc:
                error_msg += f" [DELETE FAILED: {exc}]"
            return (pdf_file_path, "", False, error_msg, 0, 0, 0.0)

        if not mime_type.startswith("application/pdf"):
            error_msg = f"Skipping {pdf_file.name} - not a PDF (detected as {mime_type})"
            try:
                pdf_file.unlink()
                error_msg += " [DELETED]"
            except Exception as exc:
                error_msg += f" [DELETE FAILED: {exc}]"
            return (pdf_file_path, "", False, error_msg, 0, 0, 0.0)
    except subprocess.TimeoutExpired:
        return (pdf_file_path, "", False, f"Timeout detecting file type for {pdf_file.name}", 0, 0, 0.0)
    except Exception:
        pass

    if images_only and Path(output_path).exists():
        try:
            from docling.models import ImageExtractionConfig
            from docling.pipeline import Pipeline

            start_time = time.time()
            doc_id = Path(pdf_file_path).stem.replace("doc_", "")
            images_dir = Path(images_output_dir) / f"doc_{doc_id}"
            images_dir.mkdir(parents=True, exist_ok=True)

            pipeline = Pipeline(
                steps=[
                    (
                        "extract_images",
                        ImageExtractionConfig(output_dir=str(images_dir)),
                    )
                ]
            )

            result = pipeline.run(pdf_file_path)
            image_artifacts = result.artifacts.get("extract_images", [])
            image_count = len(image_artifacts) if image_artifacts else 0
            document = getattr(result, "document", None)
            page_count = len(list(document.pages)) if document else 0
            processing_time = time.time() - start_time
            return (
                pdf_file_path,
                output_path,
                True,
                "",
                image_count,
                page_count,
                processing_time,
            )
        except Exception as exc:
            return (pdf_file_path, output_path, False, f"Image extraction failed: {exc}", 0, 0, 0.0)

    from src.gpu_memory_manager import (
        clear_gpu_cache,
        is_oom_error,
        log_gpu_memory_stats,
        set_process_gpu_memory_fraction,
        wait_for_gpu_memory,
    )

    max_retries = 3

    try:
        converter = DoclingConverter(
            artifacts_path=None,
            add_page_numbers=add_page_numbers,
            use_gpu=use_gpu,
            table_mode=table_mode,
            images_scale=images_scale,
            do_cell_matching=do_cell_matching,
            ocr_confidence_threshold=ocr_confidence_threshold,
        )

        if use_gpu:
            memory_fraction = 0.9 / batch_size
            set_process_gpu_memory_fraction(memory_fraction)
            log_gpu_memory_stats()

        try:
            doc_id = pdf_file.stem
            if doc_id.startswith("doc_"):
                doc_id = doc_id[4:]

            chunk_paths: List[Path] = [pdf_file]
            page_offsets: List[int] = [0]
            chunked = False
            try:
                chunk_page_limit = max(0, int(chunk_page_limit))
            except (TypeError, ValueError):
                chunk_page_limit = 0

            chunk_temp_dir: Optional[Path] = None

            if chunk_page_limit > 0:
                try:
                    from pypdf import PdfReader, PdfWriter

                    reader = PdfReader(str(pdf_file))
                    total_pages = len(reader.pages)
                    if total_pages > chunk_page_limit:
                        chunked = True
                        chunk_paths = []
                        page_offsets = []
                        chunk_temp_dir = Path(tempfile.mkdtemp(prefix="pdf_chunks_"))

                        for start in range(0, total_pages, chunk_page_limit):
                            writer = PdfWriter()
                            end = min(start + chunk_page_limit, total_pages)
                            for page_idx in range(start, end):
                                writer.add_page(reader.pages[page_idx])

                            writer.add_blank_page(width=612, height=792)

                            chunk_file = chunk_temp_dir / f"{pdf_file.stem}_part_{len(chunk_paths) + 1}.pdf"
                            with open(chunk_file, "wb") as chunk_fp:
                                writer.write(chunk_fp)

                            chunk_paths.append(chunk_file)
                            page_offsets.append(start)
                except ImportError:
                    return (
                        pdf_file_path,
                        "",
                        False,
                        f"pypdf not available; cannot chunk {pdf_file.name}",
                        0,
                        0,
                        0.0,
                    )
                except Exception as chunk_error:
                    return (
                        pdf_file_path,
                        "",
                        False,
                        f"Failed to chunk {pdf_file.name} ({chunk_error}); chunking required",
                        0,
                        0,
                        0.0,
                    )

            combined_markdown_parts: List[str] = []
            total_page_count = 0
            total_image_count = 0
            total_processing_time = 0.0

            for idx, chunk_path in enumerate(chunk_paths):
                page_offset = page_offsets[idx]
                chunk_label = f"part {idx + 1}" if chunked else "full document"

                retry_count = 0
                while retry_count <= max_retries:
                    try:
                        if use_gpu and retry_count > 0:
                            clear_gpu_cache()
                            if not wait_for_gpu_memory(required_mb=2000, timeout_seconds=180):
                                raise RuntimeError("Timeout waiting for GPU memory")

                        start_time = time.time()
                        markdown, document, page_count = converter.convert_pdf(
                            chunk_path,
                            page_offset=page_offset,
                            strip_last_page_from_output=chunked,
                        )
                        chunk_processing_time = time.time() - start_time

                        image_count = converter.extract_images(
                            document,
                            Path(images_output_dir),
                            doc_id,
                            page_offset=page_offset,
                        )

                        total_processing_time += chunk_processing_time
                        total_page_count += page_count
                        total_image_count += image_count

                        if use_gpu:
                            clear_gpu_cache()

                        break

                    except Exception as chunk_error:
                        if is_oom_error(chunk_error):
                            retry_count += 1
                            if retry_count <= max_retries:
                                logging.warning(
                                    "GPU OOM for %s %s, retry %s/%s",
                                    pdf_file.name,
                                    chunk_label,
                                    retry_count,
                                    max_retries,
                                )
                                clear_gpu_cache()
                                time.sleep(5 * retry_count)
                                continue
                            raise RuntimeError(
                                f"GPU out of memory after {max_retries} retries"
                            ) from chunk_error
                        raise

                combined_markdown_parts.append(markdown)
                del markdown, document

                if chunked:
                    import gc

                    gc.collect()
                    logging.debug("Memory cleanup after %s", chunk_label)

            combined_markdown = "".join(combined_markdown_parts)

            if chunk_temp_dir and chunk_temp_dir.exists():
                import shutil

                shutil.rmtree(chunk_temp_dir, ignore_errors=True)

            images_location = DATA_DIR / "images" / doc_id
            metadata_lines = [
                "---",
                f"**Document:** {pdf_file.name}",
                f"**Pages:** {total_page_count}",
                f"**Images Extracted:** {total_image_count}",
                f"**Images Location:** {images_location}",
                f"**Processing Time:** {total_processing_time:.2f} seconds",
            ]
            if chunked:
                metadata_lines.append(
                    f"**Chunks:** {len(chunk_paths)} (limit {chunk_page_limit} pages)"
                )
            metadata_lines.append(
                f"**Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            metadata_lines.append("---\n\n")

            metadata_header = "\n".join(metadata_lines)
            raw_markdown = metadata_header + combined_markdown

            cleaned_markdown = clean_repeating_characters(raw_markdown)
            cleaned_markdown = wrap_long_lines(cleaned_markdown)

            Path(raw_output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(raw_output_path, "w", encoding="utf-8") as fh:
                fh.write(raw_markdown)

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(cleaned_markdown)

            return (
                pdf_file_path,
                output_path,
                True,
                "",
                total_image_count,
                total_page_count,
                total_processing_time,
            )
        finally:
            converter.cleanup()
            if use_gpu:
                clear_gpu_cache()
                log_gpu_memory_stats()

    except Exception as exc:
        if use_gpu:
            clear_gpu_cache()
        return (pdf_file_path, output_path, False, f"Failed to convert {pdf_file.name}: {exc}", 0, 0, 0.0)


class BatchDoclingConverter:
    def __init__(
        self,
        input_folder: Union[str, Path],
        output_folder: Union[str, Path],
        batch_size: int = 1,
        artifacts_path: Optional[str] = None,
        add_page_numbers: bool = False,
        remove_processed: bool = True,
        use_gpu: bool = True,
        log_level: int = logging.INFO,
        table_mode: str = "accurate",
        images_scale: float = 3.0,
        do_cell_matching: bool = True,
        ocr_confidence_threshold: float = 0.05,
        upload_enabled: bool = False,
        upload_api_url: Optional[str] = None,
        upload_api_key: Optional[str] = None,
        upload_ticker: Optional[str] = None,
        upload_document_type: str = "FILING",
        doc_type: str = "both",
        extract_images: bool = False,
        chunk_page_limit: int = 30,
        max_docs: Optional[int] = None,
        min_doc_id: Optional[int] = None,
        max_doc_id: Optional[int] = None,
    ) -> None:
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.batch_size = batch_size
        self.artifacts_path = artifacts_path
        self.add_page_numbers = add_page_numbers
        self.remove_processed = remove_processed
        self.use_gpu = use_gpu
        self.table_mode = table_mode
        self.images_scale = images_scale
        self.do_cell_matching = do_cell_matching
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.upload_enabled = upload_enabled
        self.upload_api_url = upload_api_url
        self.upload_api_key = upload_api_key
        self.upload_ticker = upload_ticker
        self.upload_document_type = upload_document_type
        self.doc_type = doc_type.lower()
        self.extract_images = extract_images
        self.chunk_page_limit = chunk_page_limit
        self.max_docs = max_docs
        self.min_doc_id = min_doc_id
        self.max_doc_id = max_doc_id

        self.processed_tracker_file = STATE_DIR / "processed_documents.txt"
        self.processed_doc_ids = self._load_processed_documents()

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "skipped_files": 0,
            "removed_files": 0,
            "uploaded_files": 0,
            "upload_failed_files": 0,
            "skipped_already_processed": 0,
            "skipped_has_both_reps": 0,
            "total_images_extracted": 0,
        }

        self.proc_logger = ProcessingLogger()
        self.supabase_config = SupabaseConfig.from_env()
        self.existing_representations: Dict[int, Set[str]] = {}
        self.session_doc_ids: Set[int] = set()

    def _load_processed_documents(self) -> Set[str]:
        if not self.processed_tracker_file.exists():
            return set()
        try:
            with open(self.processed_tracker_file, "r", encoding="utf-8") as fh:
                return {line.strip() for line in fh if line.strip()}
        except Exception as exc:
            logging.warning("Could not load processed documents list: %s", exc)
            return set()

    def _extract_doc_id_from_filename(self, pdf_file: Path) -> Optional[str]:
        stem = pdf_file.stem
        if stem.startswith("doc_"):
            return stem[4:]
        return None

    def _has_both_representations(self, pdf_file: Path) -> bool:
        doc_id_str = self._extract_doc_id_from_filename(pdf_file)
        if doc_id_str is None:
            return False
        try:
            doc_id = int(doc_id_str)
        except ValueError:
            return False
        reps = self.existing_representations.get(doc_id, set())
        return "DOCLING" in reps and "DOCLING_IMG" in reps

    def _is_already_processed(self, pdf_file: Path) -> bool:
        doc_id_str = self._extract_doc_id_from_filename(pdf_file)
        return doc_id_str is not None and doc_id_str in self.processed_doc_ids

    def _mark_as_processed(self, pdf_file: Path) -> None:
        doc_id_str = self._extract_doc_id_from_filename(pdf_file)
        if not doc_id_str:
            return
        self.processed_doc_ids.add(doc_id_str)
        try:
            self.session_doc_ids.add(int(doc_id_str))
        except ValueError:
            pass
        with open(self.processed_tracker_file, "a", encoding="utf-8") as fh:
            fh.write(f"{doc_id_str}\n")

    def _get_output_path(self, input_file: Path) -> Path:
        relative_path = input_file.relative_to(self.input_folder)
        return self.output_folder / relative_path.with_suffix(".md")

    async def _fetch_existing_representations(self, document_ids: List[int]) -> Dict[int, Set[str]]:
        return await fetch_existing_representations(self.supabase_config, document_ids)

    def _filter_input_files(self, pdf_files: List[Path]) -> List[Path]:
        filtered = []
        for pdf_file in pdf_files:
            # Filter by doc type
            if self.doc_type != "both":
                name_lower = pdf_file.name.lower()
                if self.doc_type == "filings" and "filing" not in name_lower:
                    continue
                elif self.doc_type == "slides" and "slide" not in name_lower:
                    continue

            # Filter by doc ID range
            if self.min_doc_id is not None or self.max_doc_id is not None:
                doc_id_str = self._extract_doc_id_from_filename(pdf_file)
                if doc_id_str:
                    try:
                        doc_id = int(doc_id_str)
                        if self.min_doc_id is not None and doc_id < self.min_doc_id:
                            continue
                        if self.max_doc_id is not None and doc_id > self.max_doc_id:
                            continue
                    except ValueError:
                        pass  # Skip files with invalid doc IDs

            filtered.append(pdf_file)
        return filtered

    def _get_pdf_files(self) -> List[Path]:
        pdf_files: List[Path] = []
        seen: Set[Path] = set()

        for file_path in self.input_folder.glob("*.pdf"):
            if not file_path.is_file():
                continue

            resolved = file_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            if self._is_already_processed(file_path):
                self.stats["skipped_already_processed"] += 1
                continue

            pdf_files.append(file_path)

        pdf_files.sort()
        self.logger.info(f"Found {len(pdf_files)} PDF files before filtering in %s", self.input_folder)
        filtered_files = self._filter_input_files(pdf_files)
        self.logger.info(f"After filtering by doc ID range ({self.min_doc_id}-{self.max_doc_id}): {len(filtered_files)} files")
        if filtered_files and len(filtered_files) <= 20:
            self.logger.info(f"Files to process: {[f.name for f in filtered_files]}")
        return filtered_files

    async def convert_all(self) -> Dict[str, int]:
        self.logger.info(
            "Starting batch conversion from %s to %s (workers=%d)",
            self.input_folder,
            self.output_folder,
            self.batch_size,
        )

        pdf_files = self._get_pdf_files()
        self.stats["total_files"] = len(pdf_files)

        if not pdf_files:
            self.logger.warning("No PDF files found in %s", self.input_folder)
            return self.stats

        doc_ids = []
        for pdf_file in pdf_files:
            doc_id_str = self._extract_doc_id_from_filename(pdf_file)
            if doc_id_str:
                try:
                    doc_ids.append(int(doc_id_str))
                except ValueError:
                    continue

        if doc_ids:
            self.existing_representations = await self._fetch_existing_representations(doc_ids)
            filtered_files = []
            for pdf_file in pdf_files:
                if self._has_both_representations(pdf_file):
                    self.stats["skipped_has_both_reps"] += 1
                    self.logger.info(
                        "Skipping %s - both representations already exist in Supabase",
                        pdf_file.name,
                    )
                else:
                    filtered_files.append(pdf_file)
            pdf_files = filtered_files

        if self.max_docs is not None and len(pdf_files) > self.max_docs:
            pdf_files = pdf_files[: self.max_docs]

        if not pdf_files:
            self.logger.info("All files already processed; nothing to do")
            return self.stats

        if self.batch_size == 1:
            results = self._process_batch_sequential(pdf_files)
        else:
            results = self._process_batch_parallel(pdf_files, max_workers=self.batch_size)

        for (
            input_path,
            _output_path,
            success,
            error_msg,
            image_count,
            _page_count,
            _processing_time,
        ) in results:
            if not success:
                self.logger.error("Failed to convert %s: %s", input_path.name, error_msg)
                self.stats["failed_files"] += 1
                continue

            if image_count > 0:
                self.stats["total_images_extracted"] += image_count

            self.stats["processed_files"] += 1
            self._mark_as_processed(input_path)

            if self.remove_processed and input_path.exists():
                try:
                    input_path.unlink()
                    self.stats["removed_files"] += 1
                except OSError as exc:
                    self.logger.warning("Failed to delete %s: %s", input_path.name, exc)

        self.stats["doc_ids_processed"] = sorted(self.session_doc_ids)

        return self.stats

    def _process_batch_parallel(
        self,
        pdf_files: List[Path],
        max_workers: int,
    ) -> List[Tuple[Path, Path, bool, str, int, int, float]]:
        results = []
        total = len(pdf_files)
        completed = 0
        worker = partial(
            _process_single_pdf_worker,
            output_path="",
            raw_output_path="",
            images_output_dir=str(DATA_DIR / "images"),
            use_gpu=self.use_gpu,
            table_mode=self.table_mode,
            images_scale=self.images_scale,
            do_cell_matching=self.do_cell_matching,
            ocr_confidence_threshold=self.ocr_confidence_threshold,
            add_page_numbers=self.add_page_numbers,
            chunk_page_limit=self.chunk_page_limit,
            batch_size=self.batch_size,
            images_only=self.extract_images and not self.remove_processed,
        )

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            future_to_file = {
                executor.submit(
                    worker,
                    str(pdf_file),
                    str(self._get_output_path(pdf_file)),
                    str(self._get_output_path(pdf_file).with_suffix(".raw.md")),
                    str((DATA_DIR / "images") / f"doc_{pdf_file.stem.replace('doc_', '')}"),
                ): pdf_file
                for pdf_file in pdf_files
            }

            for future in as_completed(future_to_file):
                pdf_file = future_to_file[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    self.logger.error("Error processing %s: %s", pdf_file.name, exc)
                    results.append((pdf_file, self._get_output_path(pdf_file), False, str(exc), 0, 0, 0.0))
                finally:
                    completed += 1
                    self.logger.info(
                        "Conversion progress: %s/%s (%s)",
                        completed,
                        total,
                        pdf_file.name,
                    )

        return results

    def _process_batch_sequential(
        self,
        pdf_files: List[Path],
    ) -> List[Tuple[Path, Path, bool, str, int, int, float]]:
        results = []
        total = len(pdf_files)
        completed = 0
        for pdf_file in pdf_files:
            result = _process_single_pdf_worker(
                str(pdf_file),
                str(self._get_output_path(pdf_file)),
                str(self._get_output_path(pdf_file).with_suffix(".raw.md")),
                str((DATA_DIR / "images") / f"doc_{pdf_file.stem.replace('doc_', '')}"),
                self.use_gpu,
                self.table_mode,
                self.images_scale,
                self.do_cell_matching,
                self.ocr_confidence_threshold,
                self.add_page_numbers,
                self.chunk_page_limit,
                self.batch_size,
                self.extract_images and not self.remove_processed,
            )
            pdf_path = Path(result[0]) if isinstance(result[0], str) else result[0]
            results.append((pdf_path, *result[1:]))
            completed += 1
            self.logger.info(
                "Conversion progress: %s/%s (%s)",
                completed,
                total,
                pdf_file.name,
            )
        return results

    def cleanup(self) -> None:
        if self.artifacts_path and os.path.exists(self.artifacts_path):
            try:
                import shutil

                shutil.rmtree(self.artifacts_path, ignore_errors=True)
                self.logger.info("Cleaned up artifacts directory: %s", self.artifacts_path)
            except Exception as exc:
                self.logger.warning("Failed to clean up artifacts directory: %s", exc)


async def convert_folder(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    batch_size: int = 1,
    add_page_numbers: bool = False,
    remove_processed: bool = True,
    use_gpu: bool = True,
    log_level: int = logging.INFO,
    table_mode: str = "accurate",
    images_scale: float = 3.0,
    do_cell_matching: bool = True,
    ocr_confidence_threshold: float = 0.05,
    upload_enabled: bool = False,
    upload_api_url: Optional[str] = None,
    upload_api_key: Optional[str] = None,
    upload_ticker: Optional[str] = None,
    upload_document_type: str = "FILING",
    doc_type: str = "both",
    extract_images: bool = False,
    chunk_page_limit: int = 30,
    max_docs: Optional[int] = None,
    min_doc_id: Optional[int] = None,
    max_doc_id: Optional[int] = None,
) -> Dict[str, int]:
    converter = BatchDoclingConverter(
        input_folder=input_folder,
        output_folder=output_folder,
        batch_size=batch_size,
        add_page_numbers=add_page_numbers,
        remove_processed=remove_processed,
        use_gpu=use_gpu,
        log_level=log_level,
        table_mode=table_mode,
        images_scale=images_scale,
        do_cell_matching=do_cell_matching,
        ocr_confidence_threshold=ocr_confidence_threshold,
        upload_enabled=upload_enabled,
        upload_api_url=upload_api_url,
        upload_api_key=upload_api_key,
        upload_ticker=upload_ticker,
        upload_document_type=upload_document_type,
        doc_type=doc_type,
        extract_images=extract_images,
        chunk_page_limit=chunk_page_limit,
        max_docs=max_docs,
        min_doc_id=min_doc_id,
        max_doc_id=max_doc_id,
    )

    try:
        return await converter.convert_all()
    finally:
        converter.cleanup()
