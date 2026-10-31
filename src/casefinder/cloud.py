"""AWS pieces used when the API runs on Lambda (deploy/aws). Nothing here is needed locally or on Render.

- load_ssm_secrets(): copies encrypted SSM parameters (Gemini and Langfuse keys) into the environment
  at start-up, so no secret sits in the Lambda configuration or the Terraform state.
- DynamoDailyCap: the daily cap on live reports, shared by every Lambda instance. The in-memory
  DailyCap in api.py counts per process, and Lambda runs several processes at once.

boto3 is part of the Lambda Python runtime; locally it comes with the [aws] extra.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

QUOTA_TZ = ZoneInfo("America/Los_Angeles")   # Gemini's free-tier daily quota resets at midnight Pacific


def load_ssm_secrets(prefix: str | None = None, client=None) -> list[str]:
    """Copy every parameter under `prefix` (e.g. /findxray/) into os.environ, named after the last path part.
    Variables that are already set win, so a local .env still works. Returns the names loaded."""
    prefix = prefix or os.environ.get("FINDXRAY_SSM_PREFIX")
    if not prefix:
        return []
    if client is None:
        import boto3
        client = boto3.client("ssm")
    loaded, token = [], None
    while True:
        kw = {"Path": prefix, "WithDecryption": True, "Recursive": False}
        if token:
            kw["NextToken"] = token
        page = client.get_parameters_by_path(**kw)
        for p in page["Parameters"]:
            name = p["Name"].rsplit("/", 1)[-1]
            if name not in os.environ:
                os.environ[name] = p["Value"]
                loaded.append(name)
        token = page.get("NextToken")
        if not token:
            return loaded


class DynamoDailyCap:
    """Same interface as api.DailyCap, but the count lives in DynamoDB, so all Lambda instances share it.

    One item per Pacific-time day: {"day": "2026-10-30", "used": 17, "expires_at": <epoch>}. take() is a
    single conditional increment, so two simultaneous requests can never both get the last slot.
    Items expire through DynamoDB TTL a few days later. limit 0 = unlimited (DynamoDB is not touched).
    """

    def __init__(self, table: str, limit: int, client=None, clock=lambda: datetime.now(QUOTA_TZ).date()):
        self.table, self.limit, self.clock = table, limit, clock
        if client is None and limit:
            import boto3
            client = boto3.client("dynamodb")
        self.client = client

    def _key(self, day: date) -> dict:
        return {"day": {"S": day.isoformat()}}

    def take(self) -> bool:
        if not self.limit:
            return True
        day = self.clock()
        expires = int(time.mktime((day + timedelta(days=3)).timetuple()))
        try:
            self.client.update_item(
                TableName=self.table, Key=self._key(day),
                UpdateExpression="ADD used :one SET expires_at = :exp",
                ConditionExpression="attribute_not_exists(used) OR used < :limit",
                ExpressionAttributeValues={":one": {"N": "1"}, ":limit": {"N": str(self.limit)},
                                           ":exp": {"N": str(expires)}})
            return True
        except self.client.exceptions.ConditionalCheckFailedException:
            return False

    def remaining(self) -> int | None:
        if not self.limit:
            return None
        item = self.client.get_item(TableName=self.table, Key=self._key(self.clock()),
                                    ConsistentRead=True).get("Item")
        used = int(item["used"]["N"]) if item and "used" in item else 0
        return max(0, self.limit - used)
