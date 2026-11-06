import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with (ROOT / "config.json").open("r", encoding="utf-8") as stream:
        return json.load(stream)


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.write_text(content, encoding="utf-8", newline="\n")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_verified(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"hash verification failed for {Path(path).name}")
    return actual
