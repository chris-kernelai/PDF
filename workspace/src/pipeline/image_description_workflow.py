"""End-to-end Vertex Gemini image description workflow."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set
from uuid import uuid4

from google.cloud import storage
from google.genai.types import CreateBatchJobConfig

from src.pipeline.gemini import init_client, validate_environment
from src.pipeline.paths import DATA_DIR, WORKSPACE_ROOT
from src.standalone_upload_representations import DocumentRepresentationUploader

_INTEGRATOR_SPEC = importlib.util.spec_from_file_location(
    "image_description_integrator",
    Path(__file__).resolve().parents[2] / "scripts" / "legacy_pipeline" / "5_integrate_descriptions.py",
)
if _INTEGRATOR_SPEC is None or _INTEGRATOR_SPEC.loader is None:
    raise RuntimeError("Unable to load ImageDescriptionIntegrator module")
_integrator_module = importlib.util.module_from_spec(_INTEGRATOR_SPEC)
_INTEGRATOR_SPEC.loader.exec_module(_integrator_module)
ImageDescriptionIntegrator = _integrator_module.ImageDescriptionIntegrator

logger = logging.getLogger(__name__)


@dataclass
class BatchPreparationResult:
    session_id: str
    batch_files: List[Path]
    batches_dir: Path
    metadata_path: Path
    doc_ids: Set[int]
    total_images: int
    total_documents: int


@dataclass
class UploadJob:
    batch_file: Path
    job_name: str
    timestamp: str


@dataclass
class UploadResult:
    tracking_file: Path
    jobs: List[UploadJob]


@dataclass
class DownloadResult:
    descriptions_dir: Path
    description_files: List[Path]
    total_descriptions: int


@dataclass
class IntegrationResult:
    processed_files: int
    skipped_files: int
    failed_files: int
    descriptions_added: int


@dataclass
class UploadSummary:
    uploaded: int = 0
    skipped: int = 0
    failed: int = 0
    errors: List[str] = field(default_factory=list)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _encode_image(image_path: Path, max_size: int = 1024, quality: int = 85) -> str:
    from PIL import Image
    from io import BytesIO

    img = Image.open(image_path)

    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _create_batch_request(
    image_path: Path,
    doc_id: str,
    page_number: int,
    image_index: int,
    system_instruction: Optional[str] = None,
) -> Dict:
    image_base64 = _encode_image(image_path)
    request_key = f"{doc_id}_page_{page_number:03d}_img_{image_index:02d}"

    user_prompt = """
Provide a terse, factual description of this image as structured JSON.

Rules:
- Report only what is directly visible.
- Do not interpret or analyze.
- Use one JSON object per distinct data type present (text, table, chart, etc.).
- Each object should contain only fields relevant to that type.
- Include visible text, labels, numbers, and units.
- Keys must be self-explanatory and values literal.
- If data is financially relevant, include "financially_relevant": true.
- Output only JSON objects separated by newlines. No prose.
"""

    batch_request = {
        "key": request_key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64,
                            }
                        },
                        {"text": user_prompt},
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.0},
        },
    }

    if system_instruction:
        batch_request["request"]["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }

    return batch_request


def _parse_image_filename(filename: str) -> Optional[tuple[int, int]]:
    match = re.match(r"page_(\d+)_img_(\d+)\.png$", filename)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


class ImageDescriptionWorkflow:
    """Runs the Gemini image description pipeline end-to-end."""

    def __init__(
        self,
        *,
        images_dir: Path = DATA_DIR / "images",
        processed_markdown_dir: Path = DATA_DIR / "processed",
        enhanced_markdown_dir: Path = DATA_DIR / "processed_images",
        generated_root: Path = WORKSPACE_ROOT / ".generated",
        gemini_model: str = "gemini-2.0-flash-001",
        gcs_input_prefix: str = "gemini_batches/input",
        gcs_output_prefix: str = "gemini_batches/output",
        batch_prefix: str = "image_description_batches",
        image_format: str = "detailed",
    ) -> None:
        self.images_dir = images_dir
        self.processed_markdown_dir = processed_markdown_dir
        self.enhanced_markdown_dir = enhanced_markdown_dir
        self.generated_root = generated_root
        self.model = gemini_model
        self.gcs_input_prefix = gcs_input_prefix
        self.gcs_output_prefix = gcs_output_prefix
        self.batch_prefix = batch_prefix
        self.image_format = image_format

    async def run(
        self,
        *,
        session_id: Optional[str] = None,
        batch_size: int = 100,
        system_instruction: Optional[str] = None,
        wait_seconds: int = 120,
        max_retries: int = 60,
        upload: bool = True,
    ) -> UploadSummary:
        """Run the entire workflow. Returns upload summary."""
        validate_environment()
        session_id = session_id or uuid4().hex[:8]
        logger.info("Starting image description workflow (session=%s)", session_id)

        prep_result = await asyncio.to_thread(
            self._prepare_batches,
            session_id,
            batch_size,
            system_instruction,
        )

        if not prep_result.batch_files:
            logger.warning("No batches prepared; skipping image workflow")
            return UploadSummary(skipped=len(prep_result.doc_ids))

        client = init_client()
        upload_result = await asyncio.to_thread(
            self._upload_batches,
            client,
            prep_result,
        )

        await self._monitor_batches(client, upload_result.jobs, wait_seconds, max_retries)

        gcs_client = storage.Client()
        download_result = await asyncio.to_thread(
            self._download_results,
            client,
            gcs_client,
            prep_result.session_id,
            upload_result.jobs,
        )

        integration_result = await asyncio.to_thread(
            self._integrate_descriptions,
            download_result.descriptions_dir,
        )

        summary = UploadSummary()

        if upload:
            summary = await self._upload_documents(prep_result.doc_ids)

        logger.info(
            "Image workflow complete: %s files processed, %s descriptions added",
            integration_result.processed_files,
            integration_result.descriptions_added,
        )
        return summary

    def _prepare_batches(
        self,
        session_id: str,
        batch_size: int,
        system_instruction: Optional[str],
    ) -> BatchPreparationResult:
        images_dir = self.images_dir
        processed_dir = self.processed_markdown_dir
        batches_dir = _ensure_dir(
            self.generated_root / self.batch_prefix
        )

        if not images_dir.exists():
            raise FileNotFoundError(f"Images folder not found: {images_dir}")

        doc_folders = [
            d for d in images_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
        if not doc_folders:
            logger.warning("No document folders found in %s", images_dir)
            return BatchPreparationResult(
                session_id,
                [],
                batches_dir,
                batches_dir / "batch_metadata.json",
                set(),
                0,
                0,
            )

        valid_folders: List[Path] = []
        doc_ids: Set[int] = set()
        for folder in doc_folders:
            doc_name = folder.name
            md_candidates = [processed_dir / f"{doc_name}.md", processed_dir / f"doc_{doc_name}.md"]
            if any(candidate.exists() for candidate in md_candidates):
                valid_folders.append(folder)
                try:
                    doc_ids.add(int(doc_name.replace("doc_", "")))
                except ValueError:
                    pass
            else:
                logger.debug("Skipping %s (no markdown found)", doc_name)

        if not valid_folders:
            logger.warning("No image folders have matching markdown files")
            return BatchPreparationResult(
                session_id,
                [],
                batches_dir,
                batches_dir / "batch_metadata.json",
                set(),
                0,
                0,
            )

        all_requests: List[Dict] = []
        for folder in sorted(valid_folders):
            doc_name = folder.name
            image_files = sorted(folder.glob("*.png"))
            for image_file in image_files:
                parsed = _parse_image_filename(image_file.name)
                if parsed is None:
                    logger.debug("Skipping invalid image filename: %s", image_file.name)
                    continue
                page_num, image_idx = parsed
                request = _create_batch_request(
                    image_file,
                    doc_name,
                    page_num,
                    image_idx,
                    system_instruction,
                )
                all_requests.append(request)

        if not all_requests:
            raise RuntimeError("No images found to prepare batches")

        batch_files: List[Path] = []
        for i in range(0, len(all_requests), batch_size):
            batch_num = (i // batch_size) + 1
            chunk = all_requests[i : i + batch_size]
            batch_filename = (
                f"image_description_batch_{batch_num:03d}_imgs_{len(chunk):04d}_{session_id}.jsonl"
            )
            batch_path = batches_dir / batch_filename
            with open(batch_path, "w", encoding="utf-8") as fh:
                for request in chunk:
                    fh.write(json.dumps(request) + "\n")
            batch_files.append(batch_path)
            logger.info("Created batch file %s", batch_filename)

        metadata = {
            "created_at": datetime.now().isoformat(),
            "session_id": session_id,
            "total_documents": len(valid_folders),
            "total_images": len(all_requests),
            "total_batches": len(batch_files),
            "batch_size": batch_size,
            "batch_files": [str(p.name) for p in batch_files],
        }
        metadata_path = batches_dir / "batch_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        return BatchPreparationResult(
            session_id=session_id,
            batch_files=batch_files,
            batches_dir=batches_dir,
            metadata_path=metadata_path,
            doc_ids=doc_ids,
            total_images=len(all_requests),
            total_documents=len(valid_folders),
        )

    def _upload_batches(
        self,
        client,
        prep_result: BatchPreparationResult,
    ) -> UploadResult:
        batch_files = prep_result.batch_files
        if not batch_files:
            return UploadResult(prep_result.batches_dir / "batch_jobs_tracking.json", [])

        bucket_name = os.environ.get("GCS_BUCKET")
        if not bucket_name:
            raise RuntimeError("GCS_BUCKET environment variable must be set")

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        jobs: List[UploadJob] = []
        tracking_file = prep_result.batches_dir / "batch_jobs_tracking.json"

        for batch_file in batch_files:
            job_time = datetime.now().isoformat()
            blob_path = f"{self.gcs_input_prefix}/{batch_file.name}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(batch_file)
            logger.info("Uploaded %s to gs://%s/%s", batch_file.name, bucket_name, blob_path)

            gcs_output_uri = (
                f"gs://{bucket_name}/{self.gcs_output_prefix}/{prep_result.session_id}/{batch_file.stem}/"
            )

            job = client.batches.create(
                model=self.model,
                src=f"gs://{bucket_name}/{blob_path}",
                config=CreateBatchJobConfig(dest=gcs_output_uri),
            )

            jobs.append(
                UploadJob(
                    batch_file=batch_file,
                    job_name=job.name,
                    timestamp=job_time,
                )
            )
            logger.info("Created batch job %s", job.name)

        tracking_payload = {
            "session_id": prep_result.session_id,
            "session_start": datetime.now().isoformat(),
            "jobs": [
                {
                    "batch_file": str(job.batch_file),
                    "job_name": job.job_name,
                    "timestamp": job.timestamp,
                }
                for job in jobs
            ],
        }
        with open(tracking_file, "w", encoding="utf-8") as fh:
            json.dump(tracking_payload, fh, indent=2)

        return UploadResult(tracking_file=tracking_file, jobs=jobs)

    async def _monitor_batches(
        self,
        client,
        jobs: Sequence[UploadJob],
        wait_seconds: int,
        max_retries: int,
    ) -> None:
        if not jobs:
            logger.info("No jobs to monitor")
            return

        logger.info("Monitoring %s Gemini batch job(s)", len(jobs))
        attempts = 0
        while attempts < max_retries:
            attempts += 1
            all_complete = True
            any_failed = False

            for job in jobs:
                job_state = client.batches.get(name=job.job_name)
                state_str = str(job_state.state).upper()
                logger.info("Job %s status: %s", job.job_name, state_str)
                if any(tag in state_str for tag in ["FAILED", "CANCELLED"]):
                    any_failed = True
                if not any(tag in state_str for tag in ["SUCCEEDED", "COMPLETED"]):
                    all_complete = False

            if any_failed:
                raise RuntimeError("One or more Gemini batch jobs failed. See logs above.")
            if all_complete:
                logger.info("All Gemini batch jobs completed")
                return

            logger.info(
                "Jobs still running; waiting %s seconds before retry %s/%s",
                wait_seconds,
                attempts,
                max_retries,
            )
            await asyncio.sleep(wait_seconds)

        raise TimeoutError(
            f"Timeout waiting for Gemini batch jobs after {wait_seconds * max_retries / 60:.1f} minutes"
        )

    def _download_results(
        self,
        client,
        gcs_client: storage.Client,
        session_id: str,
        jobs: Sequence[UploadJob],
    ) -> DownloadResult:
        descriptions_dir = _ensure_dir(self.generated_root / "image_descriptions")
        session_outputs: List[Path] = []
        total_descriptions = 0

        for job in jobs:
            job_info = client.batches.get(name=job.job_name)
            dest = getattr(job_info, "dest", None)
            gcs_uri = getattr(dest, "gcs_uri", None) if dest else None
            if not gcs_uri:
                logger.warning("Job %s has no gcs_uri output", job.job_name)
                continue

            bucket_name, prefix = self._split_gcs_uri(gcs_uri)
            bucket = gcs_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix))
            if not blobs:
                logger.warning("No blobs found for job %s output %s", job.job_name, gcs_uri)
                continue

            for blob in blobs:
                if not blob.name.endswith(".jsonl"):
                    continue
                content = blob.download_as_text()
                records = [json.loads(line) for line in content.strip().split("\n") if line.strip()]
                total_descriptions += len(records)

                output_file = descriptions_dir / f"{Path(blob.name).stem}_{session_id}.json"
                with open(output_file, "w", encoding="utf-8") as fh:
                    json.dump(records, fh, indent=2)
                session_outputs.append(output_file)
                logger.info("Downloaded %s records to %s", len(records), output_file)

        if not session_outputs:
            raise RuntimeError("No description files downloaded from Gemini outputs")

        self._update_uuid_tracking(descriptions_dir, session_outputs, session_id)
        return DownloadResult(
            descriptions_dir=descriptions_dir,
            description_files=session_outputs,
            total_descriptions=total_descriptions,
        )

    def _split_gcs_uri(self, uri: str) -> tuple[str, str]:
        if not uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {uri}")
        path = uri[5:]
        parts = path.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix

    def _update_uuid_tracking(
        self,
        descriptions_dir: Path,
        description_files: Sequence[Path],
        session_id: str,
    ) -> None:
        tracking_file = descriptions_dir / "uuid_tracking.json"
        tracking_data: Dict[str, Dict] = {}
        if tracking_file.exists():
            try:
                tracking_data = json.loads(tracking_file.read_text())
            except json.JSONDecodeError:
                logger.warning("Existing uuid_tracking.json is invalid; replacing")

        for file in description_files:
            with open(file, "r", encoding="utf-8") as fh:
                records = json.load(fh)
            for record in records:
                doc_id = record.get("document_id") or record.get("key", "").split("_")[0]
                batch_uuid = record.get("batch_uuid", session_id)
                if not doc_id:
                    continue
                entry = tracking_data.setdefault(
                    doc_id,
                    {"uuids": [], "latest_uuid": batch_uuid, "updated_at": datetime.now().isoformat()},
                )
                if batch_uuid not in entry["uuids"]:
                    entry["uuids"].append(batch_uuid)
                entry["latest_uuid"] = batch_uuid
                entry["updated_at"] = datetime.now().isoformat()

        tracking_file.write_text(json.dumps(tracking_data, indent=2))
        logger.info("Updated UUID tracking at %s", tracking_file)

    def _integrate_descriptions(self, descriptions_dir: Path) -> IntegrationResult:
        integrator = ImageDescriptionIntegrator(
            markdown_dir=self.processed_markdown_dir,
            output_dir=self.enhanced_markdown_dir,
            descriptions_dir=descriptions_dir,
            image_format=self.image_format,
            overwrite=True,
        )
        descriptions = integrator.load_all_descriptions()
        integrator.process_all(descriptions)
        stats = integrator.stats
        integrator.print_summary()
        return IntegrationResult(
            processed_files=stats.get("processed_files", 0),
            skipped_files=stats.get("skipped_files", 0),
            failed_files=stats.get("failed_files", 0),
            descriptions_added=stats.get("total_descriptions_added", 0),
        )

    async def _upload_documents(self, doc_ids: Iterable[int]) -> UploadSummary:
        doc_ids = {doc_id for doc_id in doc_ids if isinstance(doc_id, int)}
        if not doc_ids:
            logger.warning("No document IDs available for upload")
            return UploadSummary()

        uploader = DocumentRepresentationUploader()
        await uploader.initialize()
        summary = UploadSummary()

        try:
            existing = await uploader.get_existing_representations(list(doc_ids))
            for idx, doc_id in enumerate(sorted(doc_ids), start=1):
                docling_path = self.processed_markdown_dir / f"doc_{doc_id}.md"
                if not docling_path.exists():
                    docling_path = self.processed_markdown_dir / f"{doc_id}.md"
                docling_img_path = self.enhanced_markdown_dir / f"doc_{doc_id}.md"
                if not docling_img_path.exists():
                    docling_img_path = self.enhanced_markdown_dir / f"{doc_id}.md"

                reps = existing.get(doc_id, set())
                docling_file = str(docling_path) if docling_path.exists() and "DOCLING" not in reps else None
                docling_img_file = (
                    str(docling_img_path)
                    if docling_img_path.exists() and "DOCLING_IMG" not in reps
                    else None
                )

                if not docling_file and not docling_img_file:
                    summary.skipped += 1
                    continue

                result = await uploader.upload_representations(
                    document_id=doc_id,
                    docling_file=docling_file,
                    docling_img_file=docling_img_file,
                    docling_filename=f"doc_{doc_id}.txt" if docling_file else None,
                    docling_img_filename=f"doc_{doc_id}.txt" if docling_img_file else None,
                )

                if result["errors"]:
                    summary.failed += 1
                    summary.errors.extend(result["errors"])
                else:
                    summary.uploaded += 1
        finally:
            await uploader.close()

        return summary


__all__ = ["ImageDescriptionWorkflow", "BatchPreparationResult", "UploadSummary"]
