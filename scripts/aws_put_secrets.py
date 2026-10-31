"""Copy the API's secrets from .env into AWS SSM Parameter Store (encrypted SecureString), once.

Reading them from .env keeps the keys out of your shell history and out of the Terraform state.
The Lambda loads every parameter under /findxray/ at start-up (casefinder.cloud.load_ssm_secrets).

Usage:  python scripts/aws_put_secrets.py            (uses your AWS CLI profile: credentials and region)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["GEMINI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="/findxray/")
    ap.add_argument("--region", default=None, help="default: your AWS CLI profile's region")
    args = ap.parse_args()
    env = dotenv_values(ROOT / ".env")
    ssm = boto3.client("ssm", region_name=args.region)
    for name in NAMES:
        value = (env.get(name) or "").strip()
        if not value:
            print(f"skipped {name}: not set in .env")
            continue
        ssm.put_parameter(Name=f"{args.prefix}{name}", Value=value, Type="SecureString", Overwrite=True)
        print(f"stored {args.prefix}{name} (SecureString, {len(value)} characters)")


if __name__ == "__main__":
    main()
