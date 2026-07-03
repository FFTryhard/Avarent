"""
Avarent API — Example Client Script

Demonstrates both modes the API supports:

  Mode A (hosted)  — poll /v1/latest to get the shared ETF signal.
                     The server feeds its own CSV data into the engine.
                     Every client gets the same output.

  Mode B (BYOD)    — POST your own data rows to /v1/ingest.
                     The server creates a private engine just for you.
                     Your asset universe, your signal.

Setup:
    pip install requests

Usage:
    python test.py
"""

import requests
import time

# --- CONFIG ------------------------------------------------------------------
API_BASE = "http://api.avarent.co"      # swap for your VPS IP / domain
API_KEY  = "owner_htts"                    # must match API_KEYS env var on server

HEADERS  = {"x-api-key": API_KEY}
# -----------------------------------------------------------------------------


# =============================================================================
# MODE A — Poll the hosted signal
# =============================================================================

def mode_a():
    """Read the shared hosted ETF regime signal."""

    print("\n--- Mode A: Hosted Signal ---")
    print(f"Polling {API_BASE}/v1/latest every 2s. Ctrl+C to stop.\n")

    last_ts = None

    while True:
        try:
            r = requests.get(f"{API_BASE}/v1/latest", headers=HEADERS, timeout=5)

            if r.status_code == 401:
                print("ERROR 401: Invalid API key.")
                break

            data = r.json()

            if data.get("status") == "warming_up":
                print("  [engine warming up...]")
                time.sleep(2)
                continue

            ts = data.get("timestamp")
            if ts == last_ts:       # nothing new since last poll
                time.sleep(2)
                continue

            last_ts = ts
            print_snapshot(data)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  ERROR: {e}")

        time.sleep(2)


# =============================================================================
# MODE B — Push your own data (BYOD)
# =============================================================================

def mode_b():
    """
    Demonstrate BYOD: push synthetic log-return rows and read your private signal.

    In production you would replace FAKE_ROW with real log returns from your
    own data source (Bloomberg, Refinitiv, your broker, etc.).
    """

    print("\n--- Mode B: BYOD Signal ---")

    # Your asset universe — define it once, then send it with every row.
    MY_ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                 "TSLA", "META", "BRK-B", "JPM",  "V"]

    import random
    import math
    from datetime import datetime, timedelta
    random.seed(42)

    print("(Engine needs ~250 rows before producing output)\n")

    start_date = datetime(2020, 1, 1)
    i = 0

    while True:
        fake_returns = [random.gauss(0, 0.01) for _ in MY_ASSETS]
        
        current_date = start_date + timedelta(days=i)

        payload = {
            "timestamp": current_date.strftime("%Y-%m-%d"),
            "headers":   MY_ASSETS,
            "values":    fake_returns,
        }

        r = requests.post(f"{API_BASE}/v1/ingest", headers=HEADERS, json=payload, timeout=5)

        if r.status_code == 401:
            print("ERROR 401: Invalid API key.")
            return

        if r.status_code == 404:
            print(f"ERROR 404: Endpoint not found — is the server up to date? ({r.url})")
            return

        if r.status_code != 200:
            print(f"ERROR {r.status_code}: {r.text}")
            return

        data = r.json()

        i = i + 1

        if i == 300:
             print("\nDone. Engine is now ready for output.")
             print(f"You can now poll {API_BASE}/v1/client/latest for your private signal.")

        #if data.get("status") == "warming_up":
        if (
         data.get("status") == "warming_up"
         or data.get("regime_current") is None
        ):
            if i % 50 == 0 and i < 300:
                print(f"  row {i}/300 — warming up...")
            continue

        # Engine is warm — every subsequent ingest returns a live snapshot
        print(f"\n  [Row {i}] Engine output ready:")
        print_snapshot(data)

        time.sleep(0.5)

# =============================================================================
# Shared: Snapshot printer
# =============================================================================

def print_snapshot(data: dict):
    sep = "-" * 60

    source   = data.get("source",           "?")
    ts       = data.get("timestamp",        "?")
    regime   = data.get("regime",           "?")          # was: regime_current
    conf     = data.get("confidence") or 0.0              # was: regime_confidence
    stab     = data.get("regime_stability") or 0.0
    age      = data.get("regime_age",       0)
    rationale= data.get("rationale",        "?")          # was: decision_action
    geo      = data.get("geometry",         0.0)
    ergo     = data.get("ergodicity",       0.0)
    drivers  = data.get("drivers",          {})

    print(sep)
    print(f"  [{source.upper()}]  {ts}")
    print(sep)
    print(f"  Regime    {regime:<22} confidence {conf:.0%}")
    print(f"  Stability {stab:.3f}               age {age} bars")
    print(f"  Geometry  {geo:.3f}               Ergodicity {ergo:.3f}")
    print(f"  Rationale {rationale[:55]}")
    if drivers:
        top = "  ".join(f"{k}({v:+.3f})" for k, v in list(drivers.items())[:3])
        print(f"  Drivers   {top}")
    print(sep)

# =============================================================================
# Health check
# =============================================================================

def health_check():
    try:
        r = requests.get(f"{API_BASE}/v1/health", timeout=5)
        h = r.json()
        print(f"  server: {h['server']}  |  feed: {h.get('feed_mode','?')}  "
              f"|  engine: {'running' if h['engine_running'] else 'stopped'}  "
              f"|  output: {'ready' if h['has_output'] else 'warming up'}")
        return True
    except Exception as e:
        print(f"  Cannot reach {API_BASE}: {e}")
        return False


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print(f"Avarent API Client  |  {API_BASE}")
    print("Checking server health...")
    if not health_check():
        exit(1)

    print("\nSelect mode:")
    print("  A — Poll the hosted ETF signal (shared)")
    print("  B — Push your own data (private per-client engine)")

    choice = input("\nMode [A/B]: ").strip().upper()

    if choice == "A":
        mode_a()
    elif choice == "B":
        mode_b()
    else:
        print("Unknown mode. Run again and pick A or B.")
