"""Publish the local prompt texts (src/casefinder/agent/prompts.py) to Langfuse as the
"production" version. Creates a new version only when the text actually changed.

Usage:  python scripts/push_prompts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from casefinder.agent.model import langfuse_configured  # noqa: E402
from casefinder.agent.prompts import LOCAL_PROMPTS  # noqa: E402


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not langfuse_configured():
        raise SystemExit("Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL in .env first")
    from langfuse import get_client

    client = get_client()
    if not client.auth_check():
        raise SystemExit("Langfuse rejected the keys; check them and LANGFUSE_BASE_URL")
    for name, text in LOCAL_PROMPTS.items():
        try:
            current = client.get_prompt(name, label="production", type="text", cache_ttl_seconds=0)
            if current.prompt == text:
                print(f"{name}: unchanged (version {current.version})")
                continue
        except Exception:
            current = None  # prompt does not exist yet
        created = client.create_prompt(name=name, prompt=text, type="text", labels=["production"],
                                       tags=["findxray"])
        print(f"{name}: published version {created.version}" + ("" if current else " (new prompt)"))
    client.flush()


if __name__ == "__main__":
    main()
