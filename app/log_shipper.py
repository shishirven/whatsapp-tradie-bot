"""Copy model-call records (`llm_calls`) to S3 as gzipped JSON Lines.

Each upload holds the rows added since the last successful upload, one JSON object per
customer message (the same shape as `python -m app.llm_log export`), under

    s3://<LOG_BUCKET>/llm_calls/dt=YYYY-MM-DD/<first id>-<last id>.jsonl.gz

The position of the last uploaded row is kept in the database and moves only after S3
accepts the file, so a failed upload is retried next time and nothing is skipped. Object
names come from the row ids, so a retry overwrites the same file rather than duplicating it.

Turned on by LOG_BUCKET. Credentials come from AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
(the tradie-bot IAM user, which can only reach that bucket); AWS_REGION defaults to Sydney.
The rows include customers' words and the replies sent, so the bucket must stay private.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import threading
from datetime import datetime, timezone

logger = logging.getLogger("tradie.log_shipper")

MARK = "llm_calls_s3"
DEFAULT_PREFIX = "llm_calls"
DEFAULT_REGION = "ap-southeast-2"


def s3_client(region: str | None = None):
    import boto3

    return boto3.client("s3", region_name=region or os.getenv("AWS_REGION") or DEFAULT_REGION)


def ship_once(db, client, bucket: str, *, prefix: str = DEFAULT_PREFIX, batch: int = 1000) -> dict | None:
    """Upload the next batch of new rows. Returns {"key", "rows", "first_id", "last_id"}, or None if nothing is new."""
    from app.llm_log import export_rows

    after = db.get_ship_mark(MARK)
    calls = db.list_llm_calls(after_id=after, limit=batch)
    if not calls:
        return None
    first_id, last_id = calls[0]["id"], calls[-1]["id"]
    day = (calls[0].get("created_at") or "")[:10] or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"{prefix.strip('/')}/dt={day}/{first_id:08d}-{last_id:08d}.jsonl.gz"
    lines = []
    for call, row in zip(calls, export_rows(calls)):
        lines.append(json.dumps({"id": call["id"], **row}, ensure_ascii=False))
    body = gzip.compress(("\n".join(lines) + "\n").encode("utf-8"))
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/gzip",
    )
    db.set_ship_mark(MARK, last_id)
    logger.info("shipped %s llm_calls rows (%s-%s) to s3://%s/%s", len(calls), first_id, last_id, bucket, key)
    return {"key": key, "rows": len(calls), "first_id": first_id, "last_id": last_id}


def ship_all(db, client, bucket: str, *, prefix: str = DEFAULT_PREFIX, batch: int = 1000) -> list[dict]:
    """Upload everything new, a batch at a time."""
    shipped = []
    while True:
        result = ship_once(db, client, bucket, prefix=prefix, batch=batch)
        if result is None:
            return shipped
        shipped.append(result)


def check_access(client, bucket: str, *, prefix: str = DEFAULT_PREFIX) -> str:
    """Write, read back and delete a small test object. Returns its key; raises on any failure."""
    key = f"{prefix.strip('/')}/_connection_test.txt"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = f"whatsapp-tradie-bot connection test {stamp}\n".encode("utf-8")
    client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="text/plain")
    read_back = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    if read_back != payload:
        raise RuntimeError("the test object came back different from what was written")
    client.delete_object(Bucket=bucket, Key=key)
    return key


class LogShipper:
    """Background thread that uploads new rows every `interval` seconds. Errors are logged, never raised."""

    def __init__(self, db, bucket: str, *, interval: int = 300, client=None, prefix: str = DEFAULT_PREFIX) -> None:
        self.db = db
        self.bucket = bucket
        self.interval = max(30, int(interval))
        self.prefix = prefix
        self._client = client
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="llm-calls-s3", daemon=True)
        self._thread.start()
        logger.info("uploading llm_calls to s3://%s/%s every %ss", self.bucket, self.prefix, self.interval)

    def stop(self) -> None:
        self._stop.set()

    def run_once(self) -> list[dict]:
        try:
            if self._client is None:
                self._client = s3_client()
            return ship_all(self.db, self._client, self.bucket, prefix=self.prefix)
        except Exception as exc:
            logger.warning("llm_calls upload to s3://%s failed, will retry: %s", self.bucket, _short(exc))
            return []

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.run_once()


def _short(exc: Exception) -> str:
    return f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"
