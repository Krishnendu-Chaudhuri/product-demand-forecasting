"""Upload datasets or model artifacts to S3-compatible storage."""

from __future__ import annotations

import argparse
from pathlib import Path


def _upload_path(local_path: Path, bucket: str, prefix: str) -> None:
    try:
        import boto3
    except ImportError as exc:
        raise SystemExit("Install boto3 to use this script: pip install boto3") from exc

    client = boto3.client("s3")
    if local_path.is_file():
        key = f"{prefix}/{local_path.name}".lstrip("/")
        client.upload_file(str(local_path), bucket, key)
        print(f"uploaded {local_path} -> s3://{bucket}/{key}")
        return

    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(local_path)
            key = f"{prefix}/{relative.as_posix()}".lstrip("/")
            client.upload_file(str(file_path), bucket, key)
            print(f"uploaded {file_path} -> s3://{bucket}/{key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload artifacts to S3-compatible storage")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument(
        "--target",
        choices=["models", "data"],
        required=True,
        help="Artifact target to upload",
    )
    parser.add_argument("--version", help="Model version directory under models/")
    parser.add_argument("--prefix", default="", help="Optional key prefix inside the bucket")
    args = parser.parse_args()

    if args.target == "models":
        if not args.version:
            raise SystemExit("--version is required when --target=models")
        local_path = Path("models") / args.version
    else:
        local_path = Path("assets/data.csv")

    if not local_path.exists():
        raise SystemExit(f"Path not found: {local_path}")

    prefix = args.prefix or args.target
    if args.target == "models":
        prefix = f"{prefix}/{args.version}"
    _upload_path(local_path, args.bucket, prefix)


if __name__ == "__main__":
    main()
