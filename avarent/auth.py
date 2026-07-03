"""
auth.py
API key authentication with optional tier tagging.

.env format:
    API_KEYS=key1:pro,key2:free,key3:pro

If no tier is specified the key defaults to "pro".
verify_key() returns a dict: {"client_id": <key>, "tier": <tier>}
"""

import os
from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()

raw_keys = os.getenv("API_KEYS", "")

VALID_KEYS: dict[str, dict] = {}
for _entry in raw_keys.split(","):
    _entry = _entry.strip()
    if not _entry:
        continue
    _parts = _entry.split(":", 1)
    _key  = _parts[0].strip()
    _tier = _parts[1].strip() if len(_parts) > 1 else "pro"
    if _key:
        VALID_KEYS[_key] = {"client_id": _key, "tier": _tier} #now we have seperated the key and their tier


def verify_key(x_api_key: str = Header(...)) -> dict:
    """FastAPI dependency. Returns {"client_id": ..., "tier": ...}."""
    if x_api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return VALID_KEYS[x_api_key] #verification of api key