"""
otx_check.py — AlienVault OTX Threat Intelligence
Downloads malicious IPs, domains, and file hashes
from OTX and stores them in a local SQLite database.
Runs a background refresh thread every 15 minutes.
Pre-ML intelligence layer — known bad IPs blocked instantly.
"""

import sqlite3
import threading
import time
import requests
import logging
from datetime import datetime
from typing import Optional
import config

logger = logging.getLogger(__name__)

# ─── SQLite database path ───────────────────────────────────
DB_PATH = "otx_intel.db"

# ─── OTX API endpoints ──────────────────────────────────────
OTX_BASE      = "https://otx.alienvault.com/api/v1"
OTX_IPS_URL   = f"{OTX_BASE}/indicators/export?type=IPv4"
OTX_DOMS_URL  = f"{OTX_BASE}/indicators/export?type=domain"
OTX_HASH_URL  = f"{OTX_BASE}/indicators/export?type=FileHash-MD5"

# ─── Global state ───────────────────────────────────────────
_last_sync: Optional[datetime] = None
_sync_lock  = threading.Lock()
_stop_event = threading.Event()


def _get_connection() -> sqlite3.Connection:
    """
    Open and return a SQLite connection to the intel DB.
    Creates the database file if it does not exist.

    Returns:
        sqlite3.Connection object
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def init_db() -> None:
    """
    Create the SQLite tables for threat intel storage.
    Safe to call multiple times — uses IF NOT EXISTS.
    Called once at startup before first sync.
    """
    try:
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bad_ips (
                ip TEXT PRIMARY KEY,
                added_at TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bad_domains (
                domain TEXT PRIMARY KEY,
                added_at TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bad_hashes (
                hash TEXT PRIMARY KEY,
                added_at TEXT
            )
        """)
        conn.commit()
        conn.close()
        logger.info("[OTX] Database initialized")
    except Exception as e:
        logger.error(f"[OTX] Database init error: {e}")


def _fetch_otx(url: str) -> list:
    """
    Fetch indicator list from OTX API endpoint.
    Returns empty list on any error — never crashes.

    Args:
        url: OTX API endpoint URL

    Returns:
        List of indicator strings
    """
    if config.OTX_API_KEY == "PENDING_VERIFICATION":
        return []

    try:
        headers  = {"X-OTX-API-KEY": config.OTX_API_KEY}
        response = requests.get(
            url, headers=headers, timeout=30)

        if response.status_code != 200:
            logger.warning(
                f"[OTX] API returned {response.status_code}"
                f" for {url}")
            return []

        indicators = []
        for line in response.text.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                indicators.append(line)
        return indicators

    except requests.exceptions.Timeout:
        logger.warning("[OTX] Request timed out")
        return []
    except Exception as e:
        logger.error(f"[OTX] Fetch error: {e}")
        return []


def sync_otx() -> dict:
    """
    Download latest threat intel from OTX and store
    in local SQLite database. Thread-safe with lock.

    Returns:
        Dictionary with counts of ips, domains, hashes synced
    """
    global _last_sync

    with _sync_lock:
        result = {"ips": 0, "domains": 0, "hashes": 0}

        try:
            now      = datetime.utcnow().isoformat()
            conn     = _get_connection()
            cur      = conn.cursor()

            # Sync malicious IPs
            ips = _fetch_otx(OTX_IPS_URL)
            if ips:
                cur.executemany(
                    "INSERT OR IGNORE INTO bad_ips"
                    " (ip, added_at) VALUES (?, ?)",
                    [(ip, now) for ip in ips]
                )
                result["ips"] = len(ips)

            # Sync malicious domains
            domains = _fetch_otx(OTX_DOMS_URL)
            if domains:
                cur.executemany(
                    "INSERT OR IGNORE INTO bad_domains"
                    " (domain, added_at) VALUES (?, ?)",
                    [(d, now) for d in domains]
                )
                result["domains"] = len(domains)

            # Sync malicious file hashes
            hashes = _fetch_otx(OTX_HASH_URL)
            if hashes:
                cur.executemany(
                    "INSERT OR IGNORE INTO bad_hashes"
                    " (hash, added_at) VALUES (?, ?)",
                    [(h, now) for h in hashes]
                )
                result["hashes"] = len(hashes)

            conn.commit()
            conn.close()
            _last_sync = datetime.utcnow()
            logger.info(
                f"[OTX] Sync complete — "
                f"IPs:{result['ips']} "
                f"Domains:{result['domains']} "
                f"Hashes:{result['hashes']}"
            )

        except Exception as e:
            logger.error(f"[OTX] Sync error: {e}")

        return result


def _background_sync_loop() -> None:
    """
    Background thread that refreshes OTX intel every
    15 minutes. Runs until stop_background_sync() called.
    """
    logger.info("[OTX] Background sync thread started")
    while not _stop_event.is_set():
        try:
            sync_otx()
        except Exception as e:
            logger.error(f"[OTX] Background sync error: {e}")
        # Wait 15 minutes or until stop event
        _stop_event.wait(timeout=config.OTX_REFRESH_INTERVAL)
    logger.info("[OTX] Background sync thread stopped")


def start_background_sync() -> threading.Thread:
    """
    Start the OTX background refresh thread.
    Call this once at FastAPI startup.

    Returns:
        The background thread object
    """
    init_db()
    t = threading.Thread(
        target=_background_sync_loop,
        daemon=True,
        name="otx-sync"
    )
    t.start()
    return t


def stop_background_sync() -> None:
    """
    Signal the background sync thread to stop.
    Call this at FastAPI shutdown.
    """
    _stop_event.set()


def check_ip(ip: str) -> bool:
    """
    Check if an IP address is in the malicious IPs table.

    Args:
        ip: IP address string to check

    Returns:
        True if IP is known malicious, False otherwise
    """
    try:
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute(
            "SELECT 1 FROM bad_ips WHERE ip = ? LIMIT 1",
            (ip,)
        )
        found = cur.fetchone() is not None
        conn.close()
        return found
    except Exception as e:
        logger.error(f"[OTX] check_ip error: {e}")
        return False


def check_domain(domain: str) -> bool:
    """
    Check if a domain is in the malicious domains table.

    Args:
        domain: Domain string to check

    Returns:
        True if domain is known malicious, False otherwise
    """
    try:
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute(
            "SELECT 1 FROM bad_domains"
            " WHERE domain = ? LIMIT 1",
            (domain,)
        )
        found = cur.fetchone() is not None
        conn.close()
        return found
    except Exception as e:
        logger.error(f"[OTX] check_domain error: {e}")
        return False


def check_hash(file_hash: str) -> bool:
    """
    Check if a file hash is in the malicious hashes table.

    Args:
        file_hash: MD5 hash string to check

    Returns:
        True if hash is known malicious, False otherwise
    """
    try:
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute(
            "SELECT 1 FROM bad_hashes"
            " WHERE hash = ? LIMIT 1",
            (file_hash,)
        )
        found = cur.fetchone() is not None
        conn.close()
        return found
    except Exception as e:
        logger.error(f"[OTX] check_hash error: {e}")
        return False


def get_last_sync_time() -> Optional[str]:
    """
    Return the timestamp of the last successful OTX sync.

    Returns:
        ISO format timestamp string or None if never synced
    """
    if _last_sync is None:
        return None
    return _last_sync.isoformat()


def get_intel_counts() -> dict:
    """
    Return count of all stored threat indicators.

    Returns:
        Dictionary with ips, domains, hashes counts
    """
    counts = {"ips": 0, "domains": 0, "hashes": 0}
    try:
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM bad_ips")
        counts["ips"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM bad_domains")
        counts["domains"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM bad_hashes")
        counts["hashes"] = cur.fetchone()[0]
        conn.close()
    except Exception as e:
        logger.error(f"[OTX] Count error: {e}")
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Initializing OTX database...")
    init_db()

    print("Testing check functions on empty database...")
    print(f"check_ip('1.2.3.4')     : {check_ip('1.2.3.4')}")
    print(f"check_domain('evil.com'): {check_domain('evil.com')}")
    print(f"check_hash('abc123')    : {check_hash('abc123')}")

    print("\nInserting test data manually...")
    conn = _get_connection()
    cur  = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO bad_ips (ip, added_at)"
        " VALUES (?, ?)",
        ("1.2.3.4", datetime.utcnow().isoformat())
    )
    cur.execute(
        "INSERT OR IGNORE INTO bad_domains (domain, added_at)"
        " VALUES (?, ?)",
        ("evil.com", datetime.utcnow().isoformat())
    )
    cur.execute(
        "INSERT OR IGNORE INTO bad_hashes (hash, added_at)"
        " VALUES (?, ?)",
        ("abc123def456", datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

    print("\nTesting check functions with test data...")
    print(f"check_ip('1.2.3.4')        : {check_ip('1.2.3.4')}")
    print(f"check_ip('9.9.9.9')        : {check_ip('9.9.9.9')}")
    print(f"check_domain('evil.com')   : {check_domain('evil.com')}")
    print(f"check_domain('google.com') : {check_domain('google.com')}")
    print(f"check_hash('abc123def456') : {check_hash('abc123def456')}")
    print(f"check_hash('cleanfile')    : {check_hash('cleanfile')}")

    counts = get_intel_counts()
    print(f"\nIntel counts: {counts}")
    print(f"Last sync   : {get_last_sync_time()}")
    print("\notx_check.py OK")