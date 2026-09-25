#!/usr/bin/env python3
"""Fetch and verify OpenFOAM benchmark mesh release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = REPOSITORY_ROOT / "mesh_dataset.json"
CHUNK_SIZE = 1024 * 1024
USER_AGENT = "SmootherTest-mesh-fetcher/1"


class FetchError(RuntimeError):
    """A user-facing mesh fetch or validation failure."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise FetchError(f"cannot read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise FetchError(f"JSON root in {path} must be an object")
    return value


def load_config(path: Path) -> dict[str, str]:
    root = read_json(path)
    dataset = root.get("dataset")
    if not isinstance(dataset, dict):
        raise FetchError(f"missing object 'dataset' in {path}")
    result: dict[str, str] = {}
    for key in ("repository", "release", "catalog", "catalog_ref"):
        value = dataset.get(key)
        if not isinstance(value, str) or not value:
            raise FetchError(f"missing non-empty dataset.{key} in {path}")
        result[key] = value
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", result["repository"]):
        raise FetchError("dataset.repository must have the form owner/repository")
    if Path(result["catalog"]).name != result["catalog"]:
        raise FetchError("dataset.catalog must be a plain filename")
    return result


def request_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise FetchError(f"download failed for {url}: {error}") from error


def cached_catalog_path(
    cache_dir: Path, release: str, catalog_filename: str
) -> Path:
    safe_release = re.sub(r"[^A-Za-z0-9_.-]", "_", release)
    safe_catalog = re.sub(
        r"[^A-Za-z0-9_.-]", "_", Path(catalog_filename).stem
    )
    return cache_dir / f".dataset-{safe_release}-{safe_catalog}.json"


def validate_manifest(value: dict[str, Any], source: str) -> dict[str, Any]:
    archives = value.get("archives")
    if not isinstance(archives, list) or not archives:
        raise FetchError(f"manifest {source} has no non-empty 'archives' array")
    seen: set[tuple[str, str]] = set()
    for entry in archives:
        if not isinstance(entry, dict):
            raise FetchError(f"manifest {source} contains a non-object archive")
        mesh = entry.get("mesh")
        variant = entry.get("variant")
        filename = entry.get("filename")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if not all(isinstance(item, str) and item for item in (mesh, variant, filename)):
            raise FetchError(f"manifest {source} has incomplete archive identity")
        if Path(filename).name != filename:
            raise FetchError(f"unsafe archive filename in manifest: {filename!r}")
        if not isinstance(size, int) or size < 0:
            raise FetchError(f"invalid size for {filename}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise FetchError(f"invalid SHA-256 for {filename}")
        key = (mesh.lower(), variant.lower())
        if key in seen:
            raise FetchError(f"duplicate mesh/variant in manifest: {mesh} {variant}")
        seen.add(key)
    return value


def load_catalog(config: dict[str, str], cache_dir: Path) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cached_catalog_path(
        cache_dir, config["release"], config["catalog"]
    )
    if cached.is_file():
        return validate_manifest(read_json(cached), str(cached))

    catalog_url = (
        f"https://raw.githubusercontent.com/{config['repository']}/"
        f"{quote(config['catalog_ref'], safe='')}/"
        f"{quote(config['catalog'], safe='')}"
    )
    print(f"Fetching dataset catalog: {catalog_url}")
    payload = request_bytes(catalog_url)
    try:
        catalog = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FetchError(f"invalid dataset catalog from {catalog_url}: {error}") from error
    if not isinstance(catalog, dict):
        raise FetchError(f"dataset catalog from {catalog_url} is not an object")
    validate_manifest(catalog, catalog_url)

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=cached.name + ".", suffix=".tmp",
            dir=cache_dir, delete=False
        ) as stream:
            stream.write(payload)
            temporary = Path(stream.name)
        os.replace(temporary, cached)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return catalog


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, entry: dict[str, Any]) -> None:
    expected_size = entry["size_bytes"]
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise FetchError(
            f"cached archive has wrong size: {path}\n"
            f"expected {expected_size} bytes, found {actual_size}; "
            "remove the file and retry"
        )
    actual_digest = sha256_file(path)
    expected_digest = entry["sha256"].lower()
    if actual_digest != expected_digest:
        raise FetchError(
            f"cached archive checksum mismatch: {path}\n"
            f"expected {expected_digest}, found {actual_digest}; "
            "remove the file and retry"
        )


def download_archive(
    entry: dict[str, Any], config: dict[str, str], cache_dir: Path
) -> Path:
    destination = cache_dir / entry["filename"]
    if destination.exists():
        if not destination.is_file():
            raise FetchError(f"archive cache path is not a file: {destination}")
        verify_archive(destination, entry)
        print(f"Using cached archive: {destination}")
        return destination

    url = (
        f"https://github.com/{config['repository']}/releases/download/"
        f"{quote(config['release'], safe='')}/{quote(entry['filename'], safe='')}"
    )
    print(f"Downloading {entry['mesh']} ({entry['variant']}): {url}")
    temporary: Path | None = None
    digest = hashlib.sha256()
    size = 0
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=entry["filename"] + ".", suffix=".tmp",
            dir=cache_dir, delete=False
        ) as output:
            temporary = Path(output.name)
            try:
                with urlopen(request, timeout=30) as response:
                    while chunk := response.read(CHUNK_SIZE):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                raise FetchError(
                    f"download failed for {url}: {error}\n"
                    f"expected local path: {destination}"
                ) from error

        if size != entry["size_bytes"]:
            raise FetchError(
                f"downloaded size mismatch for {entry['filename']}: "
                f"expected {entry['size_bytes']}, received {size}"
            )
        actual_digest = digest.hexdigest()
        if actual_digest != entry["sha256"].lower():
            raise FetchError(
                f"downloaded checksum mismatch for {entry['filename']}: "
                f"expected {entry['sha256'].lower()}, received {actual_digest}"
            )
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    print(f"Cached verified archive: {destination}")
    return destination


def select_entries(
    catalog: dict[str, Any], mesh: str | None, variant: str | None, all_meshes: bool
) -> list[dict[str, Any]]:
    archives: list[dict[str, Any]] = catalog["archives"]
    if all_meshes:
        return archives
    assert mesh is not None
    matches = [entry for entry in archives if entry["mesh"].lower() == mesh.lower()]
    if not matches:
        available = ", ".join(sorted({entry["mesh"] for entry in archives}))
        raise FetchError(f"unknown mesh {mesh!r}; available meshes: {available}")
    if variant is not None:
        matches = [entry for entry in matches if entry["variant"].lower() == variant.lower()]
        if not matches:
            available = ", ".join(
                sorted(entry["variant"] for entry in archives if entry["mesh"].lower() == mesh.lower())
            )
            raise FetchError(
                f"mesh {mesh!r} has no variant {variant!r}; available: {available}"
            )
    elif len(matches) != 1:
        available = ", ".join(sorted(entry["variant"] for entry in matches))
        raise FetchError(f"mesh {mesh!r} requires a variant: {available}")
    return matches


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify OpenFOAM benchmark mesh archives"
    )
    parser.add_argument("mesh", nargs="?", help="mesh name from the dataset catalog")
    parser.add_argument("variant", nargs="?", help="original or renumbered")
    parser.add_argument("--all", action="store_true", help="fetch every dataset archive")
    parser.add_argument(
        "--cache-dir", type=Path, default=SCRIPT_DIR,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=argparse.SUPPRESS
    )
    arguments = parser.parse_args()
    if arguments.all and (arguments.mesh is not None or arguments.variant is not None):
        parser.error("--all cannot be combined with a mesh or variant")
    if not arguments.all and arguments.mesh is None:
        parser.error("specify a mesh or --all")
    if arguments.mesh is None and arguments.variant is not None:
        parser.error("a variant requires a mesh")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    config: dict[str, str] = {}
    expected_paths: list[Path] = []
    requested = "all meshes" if arguments.all else " ".join(
        item for item in (arguments.mesh, arguments.variant) if item
    )
    try:
        config = load_config(arguments.config)
        cache_dir = arguments.cache_dir.resolve()
        catalog = load_catalog(config, cache_dir)
        entries = select_entries(
            catalog, arguments.mesh, arguments.variant, arguments.all
        )
        expected_paths = [cache_dir / entry["filename"] for entry in entries]
        for entry in entries:
            download_archive(entry, config, cache_dir)
        return 0
    except FetchError as error:
        repository = config.get("repository", "dmishura/OpenFOAM-Benchmark-Meshes")
        release = config.get("release", "dataset-v1")
        print(f"ERROR: required dataset: {requested}", file=sys.stderr)
        print(f"ERROR: local cache: {arguments.cache_dir.resolve()}", file=sys.stderr)
        for path in expected_paths:
            print(f"ERROR: expected local path: {path}", file=sys.stderr)
        print(f"ERROR: {error}", file=sys.stderr)
        print(
            f"Recovery: download the matching asset from "
            f"https://github.com/{repository}/releases/tag/{release} "
            f"into {arguments.cache_dir.resolve()}, then run this command again.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
