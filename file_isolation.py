"""
file_isolation.py — File Isolation and Recovery System
Monitors protected directories, detects tampering via
SHA-256 hashing, quarantines corrupted files, and
restores clean backups. Runs as a background thread.
Never blocks the packet processing pipeline.
"""

import os
import json
import shutil
import hashlib
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Callable
import config
import gpio_controller

logger = logging.getLogger(__name__)

# ─── Global state ───────────────────────────────────────────
_baseline      : dict = {}
_scan_results  : dict = {
    "monitored"  : 0,
    "scanned"    : 0,
    "quarantined": 0,
    "restored"   : 0,
    "last_scan"  : None,
    "files"      : []
}
_scan_lock     = threading.Lock()
_ws_callback   : Optional[Callable] = None


def set_websocket_callback(callback: Callable) -> None:
    """
    Register a WebSocket broadcast callback.
    Called by main.py to hook live updates.

    Args:
        callback: Async function that broadcasts
                  a message dict to all WS clients
    """
    global _ws_callback
    _ws_callback = callback


def _sha256(path: str) -> Optional[str]:
    """
    Compute SHA-256 hash of a file.
    Returns None if file cannot be read.

    Args:
        path: Full file path

    Returns:
        Hex string hash or None on error
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(
                    lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.error(
            f"[ISOLATION] Hash error {path}: {e}")
        return None


def _walk_protected() -> list:
    """
    Walk all protected directories and return
    list of all file paths found.

    Returns:
        List of absolute file path strings
    """
    files = []
    for directory in config.PROTECTED_DIRS:
        directory = directory.strip()
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(
                    f"[ISOLATION] Created protected"
                    f" dir: {directory}")
            except Exception:
                continue

        for root, dirs, filenames in os.walk(directory):
            # Skip quarantine directory
            if "quarantine" in root:
                continue
            for fname in filenames:
                files.append(
                    os.path.join(root, fname))
    return files


def build_baseline() -> dict:
    """
    Phase 0 — Walk all protected directories and
    compute SHA-256 hash for every file.
    Saves baseline to baseline.json.
    Called once at startup.

    Returns:
        Baseline dictionary mapping path to hash
    """
    global _baseline
    logger.info("[ISOLATION] Building baseline...")

    baseline = {}
    files    = _walk_protected()

    for path in files:
        h = _sha256(path)
        if h:
            baseline[path] = h

    _baseline = baseline

    # Save to disk
    try:
        with open(config.BASELINE_PATH, "w") as f:
            json.dump(baseline, f, indent=2)
        logger.info(
            f"[ISOLATION] Baseline built — "
            f"{len(baseline)} files")
    except Exception as e:
        logger.error(
            f"[ISOLATION] Baseline save error: {e}")

    with _scan_lock:
        _scan_results["monitored"] = len(baseline)

    return baseline


def load_baseline() -> dict:
    """
    Load existing baseline from disk.
    Falls back to building new baseline if not found.

    Returns:
        Baseline dictionary mapping path to hash
    """
    global _baseline

    if os.path.exists(config.BASELINE_PATH):
        try:
            with open(config.BASELINE_PATH, "r") as f:
                _baseline = json.load(f)
            logger.info(
                f"[ISOLATION] Baseline loaded — "
                f"{len(_baseline)} files")
            with _scan_lock:
                _scan_results["monitored"] = \
                    len(_baseline)
            return _baseline
        except Exception as e:
            logger.error(
                f"[ISOLATION] Baseline load error: {e}")

    return build_baseline()


def _quarantine_file(path: str) -> bool:
    """
    Phase 4 — Move a corrupted file to quarantine.
    Adds timestamp prefix to filename.
    Sets chmod 000 on Linux (skipped on Windows).

    Args:
        path: Full path of file to quarantine

    Returns:
        True if quarantined successfully
    """
    try:
        os.makedirs(config.QUARANTINE_DIR,
                    exist_ok=True)

        timestamp = datetime.now(
            timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename  = os.path.basename(path)
        dest      = os.path.join(
            config.QUARANTINE_DIR,
            f"{timestamp}_{filename}"
        )

        shutil.move(path, dest)

        # chmod 000 — Linux only
        if os.name != "nt":
            os.chmod(dest, 0o000)

        logger.warning(
            f"[ISOLATION] Quarantined: "
            f"{path} → {dest}")
        return True

    except Exception as e:
        logger.error(
            f"[ISOLATION] Quarantine error "
            f"{path}: {e}")
        return False


def _restore_file(path: str) -> bool:
    """
    Phase 5 — Restore a clean file from backup.
    Verifies SHA-256 matches baseline after restore.

    Args:
        path: Full path of file to restore

    Returns:
        True if restored and verified successfully
    """
    try:
        # Build backup path
        backup_path = os.path.join(
            config.BACKUP_PARTITION,
            os.path.relpath(path, "/")
            if os.name != "nt"
            else path.replace(":", "").lstrip("\\")
        )

        if not os.path.exists(backup_path):
            logger.warning(
                f"[ISOLATION] No backup for: {path}")
            return False

        # Restore
        os.makedirs(
            os.path.dirname(path), exist_ok=True)
        shutil.copy2(backup_path, path)

        # Verify hash matches baseline
        restored_hash = _sha256(path)
        expected_hash = _baseline.get(path)

        if restored_hash == expected_hash:
            logger.info(
                f"[ISOLATION] Restored verified: "
                f"{path}")
            return True
        else:
            logger.error(
                f"[ISOLATION] Restore hash mismatch:"
                f" {path}")
            return False

    except Exception as e:
        logger.error(
            f"[ISOLATION] Restore error {path}: {e}")
        return False


def run_scan() -> dict:
    """
    Phases 3-7 — Run full forensic scan of all
    protected directories. Compares current hashes
    to baseline. Quarantines corrupted files.
    Attempts restore from backup partition.

    Returns:
        Scan results dictionary with file details
    """
    global _scan_results

    logger.info("[ISOLATION] Starting forensic scan...")
    gpio_controller.system_healing()

    scan_time    = datetime.now(
        timezone.utc).isoformat()
    files        = _walk_protected()
    quarantined  = []
    restored     = []
    suspicious   = []
    clean        = []

    for path in files:
        current_hash = _sha256(path)
        if current_hash is None:
            continue

        if path not in _baseline:
            # New file not in baseline — suspicious
            suspicious.append(path)
            logger.warning(
                f"[ISOLATION] Suspicious new file:"
                f" {path}")
            _quarantine_file(path)
            quarantined.append(path)

        elif current_hash != _baseline[path]:
            # Hash mismatch — corrupted
            logger.warning(
                f"[ISOLATION] Corrupted: {path}")
            _quarantine_file(path)
            quarantined.append(path)

            # Attempt restore
            if _restore_file(path):
                restored.append(path)
        else:
            clean.append(path)

    result = {
        "monitored"  : len(_baseline),
        "scanned"    : len(files),
        "quarantined": len(quarantined),
        "restored"   : len(restored),
        "suspicious" : len(suspicious),
        "clean"      : len(clean),
        "last_scan"  : scan_time,
        "files"      : {
            "quarantined": quarantined,
            "restored"   : restored,
            "suspicious" : suspicious,
        }
    }

    with _scan_lock:
        _scan_results.update(result)

    # Push to WebSocket if callback registered
    if _ws_callback:
        try:
            import asyncio
            asyncio.create_task(
                _ws_callback({
                    "type"       : "file_isolation",
                    "scan_result": result
                })
            )
        except Exception:
            pass

    logger.info(
        f"[ISOLATION] Scan complete — "
        f"scanned={len(files)} "
        f"quarantined={len(quarantined)} "
        f"restored={len(restored)}"
    )

    # Return to safe state if no threats found
    if len(quarantined) == 0:
        gpio_controller.system_safe()

    return result


def trigger_isolation(blocking: bool = False) -> None:
    """
    Trigger file isolation scan as background thread.
    Called automatically after every BLOCK or ISOLATE.
    Never blocks the packet processing pipeline.

    Args:
        blocking: If True run synchronously (for testing)
                  If False run as background thread
    """
    if blocking:
        run_scan()
        return

    t = threading.Thread(
        target=run_scan,
        daemon=True,
        name="file-isolation"
    )
    t.start()
    logger.info(
        "[ISOLATION] Scan thread started")


def get_scan_report() -> dict:
    """
    Return latest scan results.
    Used by GET /file-isolation/report endpoint.

    Returns:
        Latest scan results dictionary
    """
    with _scan_lock:
        return dict(_scan_results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing file isolation...")

    # Create test protected directory
    import tempfile
    test_dir = os.path.join(
        tempfile.gettempdir(), "aegisai_test")
    os.makedirs(test_dir, exist_ok=True)

    # Override config for testing
    config.PROTECTED_DIRS  = [test_dir]
    config.QUARANTINE_DIR  = os.path.join(
        tempfile.gettempdir(), "aegisai_quarantine")
    config.BASELINE_PATH   = os.path.join(
        tempfile.gettempdir(), "aegisai_baseline.json")
    config.BACKUP_PARTITION = os.path.join(
        tempfile.gettempdir(), "aegisai_backup")

    # Create test files
    test_file1 = os.path.join(test_dir, "clean.txt")
    test_file2 = os.path.join(test_dir, "will_corrupt.txt")

    with open(test_file1, "w") as f:
        f.write("This file is clean")
    with open(test_file2, "w") as f:
        f.write("Original content")

    print("\nTest 1 — Build baseline")
    baseline = build_baseline()
    print(f"Baseline files: {len(baseline)}")

    print("\nTest 2 — Scan clean files")
    result = run_scan()
    print(f"Scanned: {result['scanned']}")
    print(f"Quarantined: {result['quarantined']}")
    print(f"Clean: {result['clean']}")

    print("\nTest 3 — Corrupt a file and rescan")
    with open(test_file2, "w") as f:
        f.write("CORRUPTED BY RANSOMWARE")

    result = run_scan()
    print(f"Scanned: {result['scanned']}")
    print(f"Quarantined: {result['quarantined']}")

    print("\nTest 4 — Add suspicious new file")
    suspicious_file = os.path.join(
        test_dir, "suspicious_new.exe")
    with open(suspicious_file, "w") as f:
        f.write("suspicious content")

    result = run_scan()
    print(f"Suspicious: {result['suspicious']}")
    print(f"Quarantined: {result['quarantined']}")

    print("\nTest 5 — Get scan report")
    report = get_scan_report()
    print(f"Report: {report}")

    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)
    shutil.rmtree(config.QUARANTINE_DIR,
                  ignore_errors=True)

    print("\nfile_isolation.py OK")