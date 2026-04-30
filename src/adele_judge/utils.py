from __future__ import annotations

from contextlib import contextmanager
import json
import os
import random
import subprocess
import sys
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator, TextIO

import numpy as np


class Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@contextmanager
def tee_output(path: str | Path, mode: str = "w") -> Iterator[Path]:
    path = Path(path)
    ensure_dir(path.parent)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with path.open(mode, encoding="utf-8") as handle:
        sys.stdout = Tee(original_stdout, handle)  # type: ignore[assignment]
        sys.stderr = Tee(original_stderr, handle)  # type: ignore[assignment]
        try:
            yield path
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonable(data: Any) -> Any:
    if isinstance(data, dict):
        return {str(key): jsonable(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [jsonable(value) for value in data]
    if isinstance(data, set):
        return sorted(jsonable(value) for value in data)
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, np.generic):
        return data.item()
    if type(data).__module__ == "torch":
        return str(data)
    return data


def stable_json_hash(data: Any) -> str:
    encoded = json.dumps(jsonable(data), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def package_versions() -> dict[str, str]:
    packages = [
        "python",
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "peft",
        "trl",
        "unsloth",
        "deepspeed",
    ]
    versions = {"python": sys.version.split()[0]}
    for package in packages:
        if package == "python":
            continue
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def project_output_dir(config: dict[str, Any]) -> Path:
    return Path(config["project"]["output_dir"])


def prepared_dir(config: dict[str, Any]) -> Path:
    explicit = config.get("data", {}).get("prepared_dir")
    if explicit:
        return Path(explicit)
    return project_output_dir(config) / "prepared"
