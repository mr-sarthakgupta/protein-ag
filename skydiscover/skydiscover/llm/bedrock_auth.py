"""Credential helpers shared by native Bedrock and Bedrock Mantle."""

from __future__ import annotations

import configparser
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("skydiscover.llm")


def bedrock_api_key_from_aws_credentials(
    profile: Optional[str] = None,
    credentials_path: Optional[Path] = None,
) -> Optional[str]:
    """Return an ABSK Bedrock API key stored as ``aws_session_token``."""
    credentials_path = credentials_path or Path(
        os.environ.get("AWS_SHARED_CREDENTIALS_FILE", Path.home() / ".aws" / "credentials")
    )
    if not credentials_path.exists():
        return None

    parser = configparser.RawConfigParser()
    parser.read(credentials_path)
    section = profile or os.environ.get("AWS_PROFILE") or "default"
    if not parser.has_section(section):
        return None

    token = parser.get(section, "aws_session_token", fallback="").strip()
    return token if token.startswith("ABSK") else None


def resolve_bedrock_bearer_token(profile: Optional[str] = None) -> Optional[str]:
    """Resolve Bedrock bearer auth without exposing the credential in logs."""
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if token:
        return token

    token = bedrock_api_key_from_aws_credentials(profile)
    if not token:
        return None

    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = token
    logger.info("Loaded Bedrock bearer credential from the AWS credentials profile")
    return token


def ensure_bedrock_bearer_token(profile: Optional[str] = None) -> bool:
    """Install an ABSK credential in the environment; return whether it was installed."""
    had_token = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))
    return bool(resolve_bedrock_bearer_token(profile)) and not had_token
