import os
from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()

raw_keys = os.getenv("API_KEYS", "")
VALID_KEYS = {k.strip(): "client" for k in raw_keys.split(",") if k.strip()}

def verify_key(x_api_key: str = Header(...)):
    if x_api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return VALID_KEYS[x_api_key]