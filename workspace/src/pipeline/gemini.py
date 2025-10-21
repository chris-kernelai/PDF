"""Vertex-only Gemini helper utilities."""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def _have_active_gcloud_account(timeout: int = 5) -> bool:
    try:
        result = subprocess.run(
            [
                "gcloud",
                "auth",
                "list",
                "--filter=status:ACTIVE",
                "--format=value(account)",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return bool(result.stdout.strip())
    except FileNotFoundError:
        logger.debug("gcloud CLI not found when checking authentication")
        return False
    except subprocess.SubprocessError as exc:
        logger.warning("Failed to verify gcloud authentication: %s", exc)
        return False


def validate_environment() -> None:
    """Ensure Vertex AI environment variables and authentication are available."""
    errors: list[str] = []

    if not os.environ.get("GCP_PROJECT"):
        errors.append("GCP_PROJECT not set (required for Vertex mode)")
    if not os.environ.get("GCP_LOCATION"):
        errors.append("GCP_LOCATION not set (required for Vertex mode)")
    if not os.environ.get("GCS_BUCKET"):
        errors.append("GCS_BUCKET not set (required for Vertex mode)")

    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    has_service_account = bool(credentials_path and os.path.exists(credentials_path))
    has_gcloud_auth = False if has_service_account else _have_active_gcloud_account()

    if not has_service_account and not has_gcloud_auth:
        errors.append(
            "No Google Cloud authentication found. Either run 'gcloud auth login' or set GOOGLE_APPLICATION_CREDENTIALS."
        )

    if errors:
        header = "\n" + "=" * 60 + "\nENVIRONMENT VALIDATION FAILED\n" + "=" * 60
        footer = "\nSet the required environment variables and authenticate before retrying.\n" + "=" * 60 + "\n"
        error_block = "\n".join(f"❌ {error}" for error in errors)
        raise RuntimeError(f"{header}\n{error_block}{footer}")

    logger.info("Environment validation passed for Vertex mode")


def init_client():
    """Create a Vertex Gemini client."""
    from google import genai  # Imported lazily to keep module lightweight

    project = os.environ.get("GCP_PROJECT")
    location = os.environ.get("GCP_LOCATION")
    if not project or not location:
        raise RuntimeError("GCP_PROJECT and GCP_LOCATION must be set for Vertex mode")

    logger.info("Using Gemini Vertex AI mode (project=%s, location=%s)", project, location)
    return genai.Client(vertexai=True, project=project, location=location)


__all__ = ["validate_environment", "init_client"]
