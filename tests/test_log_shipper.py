"""llm_calls rows are copied to S3. No network: an in-memory S3, plus botocore's Stubber to
check the request against the real S3 API definition."""

import gzip
import io
import json
import logging
from argparse import Namespace

import pytest

from app.business import load_business
from app.db import Database
from app.llm import RuleBasedQualifier
from app.llm_log import _s3_command
from app.log_shipper import MARK, LogShipper, check_access, ship_all, ship_once
from app.messaging import Inbound, SimulatedGateway
from app.notify import ListNotifier
from app.service import handle_incoming
from evals.harness import EVAL_NOW

BUCKET = "tradie-bot-data-7k3q-771904882715-ap-southeast-2-an"


class FakeS3:
    def __init__(self, fail_puts=0):
        self.objects = {}
        self.puts = []
        self.fail_puts = fail_puts

    def put_object(self, Bucket, Key, Body, **kwargs):
        if self.fail_puts:
            self.fail_puts -= 1
            raise ConnectionError("network down")
        self.objects[(Bucket, Key)] = Body
        self.puts.append({"Bucket": Bucket, "Key": Key, **kwargs})
        return {"ETag": '"x"'}

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)
        return {}


def _rows(body: bytes) -> list[dict]:
    return [json.loads(line) for line in gzip.decompress(body).decode("utf-8").splitlines()]


def _send(db, text, n):
    handle_incoming(
        db, RuleBasedQualifier(load_business()), SimulatedGateway(), ListNotifier(),
        Inbound(wa_id=f"6141299630{n}", text=text, message_type="text", provider_message_id=f"ship-{n}", source="simulator"),
        business=load_business(), now=EVAL_NOW,
    )


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "ship.db"))


def test_new_rows_are_uploaded_once_as_gzipped_json_lines(db):
    s3 = FakeS3()
    _send(db, "need a new power point", 1)
    _send(db, "live wire hanging nearby", 2)

    first = ship_once(db, s3, BUCKET)
    assert first["rows"] == 2
    assert first["key"].startswith("llm_calls/dt=")
    assert first["key"].endswith("/00000001-00000002.jsonl.gz")
    rows = _rows(s3.objects[(BUCKET, first["key"])])
    assert [row["customer"] for row in rows] == ["need a new power point", "live wire hanging nearby"]
    assert rows[1]["hazard_final"] == "wire" and rows[1]["sent_reply"]
    assert db.get_ship_mark(MARK) == 2

    assert ship_once(db, s3, BUCKET) is None  # nothing new: no empty files

    _send(db, "fire alarma keeps activating", 3)
    third = ship_once(db, s3, BUCKET)
    assert third["key"].endswith("00000003-00000003.jsonl.gz")
    assert [row["customer"] for row in _rows(s3.objects[(BUCKET, third["key"])])] == ["fire alarma keeps activating"]


def test_a_failed_upload_is_retried_without_skipping_rows(db, caplog):
    s3 = FakeS3(fail_puts=1)
    _send(db, "need a new power point", 1)
    shipper = LogShipper(db, BUCKET, client=s3)
    with caplog.at_level(logging.WARNING, logger="tradie.log_shipper"):
        assert shipper.run_once() == []
    assert "will retry" in caplog.text
    assert db.get_ship_mark(MARK) == 0  # not moved on
    [retried] = shipper.run_once()
    assert retried["first_id"] == 1 and db.get_ship_mark(MARK) == 1


def test_large_backlogs_go_in_batches(db):
    s3 = FakeS3()
    for n in range(5):
        _send(db, f"need power point number {n}", n)
    shipped = ship_all(db, s3, BUCKET, batch=2)
    assert [item["rows"] for item in shipped] == [2, 2, 1]


def test_connection_check_writes_reads_back_and_cleans_up():
    s3 = FakeS3()
    key = check_access(s3, BUCKET)
    assert key == "llm_calls/_connection_test.txt"
    assert s3.objects == {}  # deleted afterwards


def test_upload_request_matches_the_real_s3_api(db):
    import boto3
    from botocore.stub import ANY, Stubber

    client = boto3.client(
        "s3", region_name="ap-southeast-2", aws_access_key_id="test", aws_secret_access_key="test"
    )
    _send(db, "need a new power point", 1)
    with Stubber(client) as stub:
        stub.add_response(
            "put_object",
            {"ETag": '"abc"'},
            {"Bucket": BUCKET, "Key": ANY, "Body": ANY, "ContentType": "application/gzip"},
        )
        result = ship_once(db, client, BUCKET)
        stub.assert_no_pending_responses()
    assert result["rows"] == 1


def test_commands_report_success_and_failure(db, capsys):
    s3 = FakeS3()
    _send(db, "need a new power point", 1)
    assert _s3_command(Namespace(command="s3-check"), db, BUCKET, client=s3) == 0
    assert "OK: wrote, read back and deleted" in capsys.readouterr().out
    assert _s3_command(Namespace(command="ship"), db, BUCKET, client=s3) == 0
    assert "Uploaded 1 rows to s3://" in capsys.readouterr().out
    assert _s3_command(Namespace(command="ship"), db, BUCKET, client=s3) == 0
    assert "Nothing new to upload" in capsys.readouterr().out
    assert _s3_command(Namespace(command="ship"), db, "", client=s3) == 2
    broken = FakeS3(fail_puts=1)
    assert _s3_command(Namespace(command="s3-check"), db, BUCKET, client=broken) == 1
    assert "FAILED" in capsys.readouterr().err
