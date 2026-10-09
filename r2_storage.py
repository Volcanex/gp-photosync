"""R2 storage backend for photo-sync-server.

Single-file module. Drops into main.py as a replacement for filesystem IO.
All keys live under the prefix `data/photos/<raw|masters|compressed>/<filename>`
to match the layout of the existing R2 backup.
"""
from __future__ import annotations

import hashlib
import io
import os
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


R2_BUCKET = os.environ["R2_BUCKET"]
R2_ENDPOINT = os.environ.get("R2_ENDPOINT") or f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
R2_PREFIX = os.environ.get("R2_PREFIX", "photos")  # top-level prefix
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

# Public base URL (custom domain or .r2.dev): set if you want direct photo URLs.
R2_PUBLIC_BASE = os.environ.get("R2_PUBLIC_BASE", "").rstrip("/")


_thread_local = threading.local()


def _client():
    c = getattr(_thread_local, "s3", None)
    if c is None:
        c = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name="auto",
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 5, "mode": "standard"},
            ),
        )
        _thread_local.s3 = c
    return c


def check():
    """Cheap liveness check for R2: one head_bucket round-trip.
    Returns (ok, detail): verifies creds are valid and the bucket is reachable."""
    try:
        _client().head_bucket(Bucket=R2_BUCKET)
        return True, f"bucket {R2_BUCKET} reachable"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _key(category: str, filename: str) -> str:
    return f"{R2_PREFIX}/{category}/{filename}"


@dataclass
class ObjectInfo:
    key: str
    filename: str
    size: int
    last_modified: str
    etag: str


def exists(category: str, filename: str) -> bool:
    try:
        _client().head_object(Bucket=R2_BUCKET, Key=_key(category, filename))
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def head(category: str, filename: str) -> Optional[dict]:
    try:
        return _client().head_object(Bucket=R2_BUCKET, Key=_key(category, filename))
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def put_bytes(category: str, filename: str, data: bytes, content_type: str = "application/octet-stream") -> dict:
    """Upload bytes. Computes sha256 alongside. Returns {key, size, etag, sha256}."""
    sha = hashlib.sha256(data).hexdigest()
    key = _key(category, filename)
    _client().put_object(
        Bucket=R2_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
        Metadata={"sha256": sha},
    )
    return {"key": key, "size": len(data), "sha256": sha}


def put_fileobj(category: str, filename: str, fileobj, content_type: str = "application/octet-stream") -> dict:
    """Upload a file-like object, using multipart as needed."""
    key = _key(category, filename)
    # boto3 upload_fileobj handles multipart automatically for >8MB by default
    _client().upload_fileobj(
        fileobj, R2_BUCKET, key,
        ExtraArgs={"ContentType": content_type},
    )
    return {"key": key}


def get_bytes(category: str, filename: str) -> Optional[bytes]:
    try:
        obj = _client().get_object(Bucket=R2_BUCKET, Key=_key(category, filename))
        return obj["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def get_stream(category: str, filename: str):
    """Return a streaming body (boto3's StreamingBody) or None if missing."""
    try:
        obj = _client().get_object(Bucket=R2_BUCKET, Key=_key(category, filename))
        return obj["Body"], obj.get("ContentLength", 0), obj.get("ContentType", "application/octet-stream")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def delete(category: str, filename: str) -> bool:
    try:
        _client().delete_object(Bucket=R2_BUCKET, Key=_key(category, filename))
        return True
    except ClientError:
        return False


def list_category(category: str) -> list[ObjectInfo]:
    prefix = f"{R2_PREFIX}/{category}/"
    results: list[ObjectInfo] = []
    paginator = _client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = key[len(prefix):]
            if not fname or "/" in fname:  # skip pseudo-subdirs if any
                continue
            results.append(ObjectInfo(
                key=key,
                filename=fname,
                size=obj["Size"],
                last_modified=obj["LastModified"].isoformat(),
                etag=obj["ETag"].strip('"'),
            ))
    return results


def public_url(category: str, filename: str) -> Optional[str]:
    """If R2_PUBLIC_BASE is set, return a direct public URL; else None."""
    if not R2_PUBLIC_BASE:
        return None
    return f"{R2_PUBLIC_BASE}/{_key(category, filename)}"


def presigned_url(category: str, filename: str, expires: int = 3600) -> str:
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET, "Key": _key(category, filename)},
        ExpiresIn=expires,
    )
