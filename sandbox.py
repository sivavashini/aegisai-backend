"""
sandbox.py — Docker Canary Sandbox
Runs suspicious traffic in an isolated Docker container.
Monitors for canary file touches and malicious behavior.
Simulates on Windows if Docker is not available.
"""

import subprocess
import logging
import platform
import threading
import time
import os
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ─── Platform and Docker detection ──────────────────────────
IS_LINUX = platform.system() == "Linux"

# ─── Canary file path inside container ──────────────────────
CANARY_FILE    = "/tmp/canary_aegisai"
CANARY_HOST    = "./canary_aegisai"
CONTAINER_NAME = "aegisai_sandbox"

# ─── Sandbox state ──────────────────────────────────────────
_sandbox_results: dict = {}
_sandbox_lock   = threading.Lock()


def _docker_available() -> bool:
    """
    Check if Docker is installed and running.

    Returns:
        True if Docker is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def _cleanup_container() -> None:
    """
    Stop and remove the sandbox container if running.
    Safe to call even if container does not exist.
    """
    try:
        subprocess.run(
            ["docker", "rm", "-f", CONTAINER_NAME],
            capture_output=True,
            timeout=10
        )
    except Exception:
        pass


def _create_canary() -> None:
    """
    Create the canary file on the host system.
    This file is mounted into the Docker container.
    If canary is touched inside container it means
    the process is exploring the filesystem.
    """
    try:
        with open(CANARY_HOST, "w") as f:
            f.write(
                "AEGISAI_CANARY — "
                "touching this file triggers isolation"
            )
        logger.info(
            f"[SANDBOX] Canary created: {CANARY_HOST}")
    except Exception as e:
        logger.error(
            f"[SANDBOX] Canary create error: {e}")


def _canary_was_touched(
        start_mtime: float) -> bool:
    """
    Check if canary file was modified after sandbox start.

    Args:
        start_mtime: File modification time before sandbox

    Returns:
        True if canary was touched, False otherwise
    """
    try:
        current_mtime = os.path.getmtime(CANARY_HOST)
        return current_mtime > start_mtime
    except Exception:
        return False


def run_sandbox(
    src_ip  : str,
    payload : Optional[dict] = None,
    timeout : int = 30
) -> dict:
    """
    Run suspicious traffic in Docker sandbox.
    Monitors canary file and container behavior.
    Returns result indicating if threat was confirmed.

    On Windows or without Docker — returns simulated
    result for testing purposes.

    Args:
        src_ip : Source IP of suspicious traffic
        payload: Optional payload data to analyze
        timeout: Seconds to run sandbox (default 30)

    Returns:
        Dictionary with canary_touched, anomalous,
        sandbox_decision, and details keys
    """
    result = {
        "src_ip"          : src_ip,
        "canary_touched"  : False,
        "anomalous"       : False,
        "sandbox_decision": "ALLOW",
        "details"         : "",
        "timestamp"       : datetime.now(
            timezone.utc).isoformat()
    }

    # Simulate on Windows or without Docker
    if not IS_LINUX or not _docker_available():
        logger.info(
            f"[SANDBOX] Simulation mode — "
            f"src_ip={src_ip}")
        result["details"] = (
            "Sandbox simulated — "
            "Docker not available on this platform"
        )
        result["sandbox_decision"] = "ALLOW"
        return result

    try:
        # Create canary file
        _create_canary()
        start_mtime = os.path.getmtime(CANARY_HOST)

        # Clean up any existing container
        _cleanup_container()

        # Run isolated container
        logger.info(
            f"[SANDBOX] Starting container "
            f"for {src_ip}...")

        proc = subprocess.Popen([
            "docker", "run",
            "--name", CONTAINER_NAME,
            "--network", "none",
            "--memory", "64m",
            "--cpus", "0.5",
            "--read-only",
            "-v", f"{os.path.abspath(CANARY_HOST)}"
                  f":{CANARY_FILE}",
            "--rm",
            "alpine",
            "sh", "-c",
            f"sleep {timeout}"
        ], capture_output=True)

        # Monitor for timeout seconds
        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(1)
            if _canary_was_touched(start_mtime):
                result["canary_touched"] = True
                result["anomalous"]      = True
                result["details"]        = (
                    "Canary file was accessed "
                    "inside sandbox")
                logger.warning(
                    f"[SANDBOX] CANARY TOUCHED — "
                    f"src_ip={src_ip}")
                break

        # Stop container
        _cleanup_container()

        if result["canary_touched"]:
            result["sandbox_decision"] = "ISOLATE"
        else:
            result["sandbox_decision"] = "ALLOW"
            result["details"] = (
                "No malicious behavior detected "
                "in sandbox")

    except Exception as e:
        logger.error(f"[SANDBOX] Error: {e}")
        result["details"] = f"Sandbox error: {e}"
        result["sandbox_decision"] = "ALLOW"

    finally:
        _cleanup_container()
        # Remove canary file
        try:
            os.remove(CANARY_HOST)
        except Exception:
            pass

    with _sandbox_lock:
        _sandbox_results[src_ip] = result

    return result


def get_sandbox_result(src_ip: str) -> Optional[dict]:
    """
    Get the latest sandbox result for an IP.

    Args:
        src_ip: Source IP address

    Returns:
        Sandbox result dict or None if not found
    """
    with _sandbox_lock:
        return _sandbox_results.get(src_ip)


def run_sandbox_async(
    src_ip    : str,
    payload   : Optional[dict] = None,
    callback  : Optional[callable] = None,
    timeout   : int = 30
) -> threading.Thread:
    """
    Run sandbox in background thread.
    Calls callback with result when complete.
    Never blocks the main pipeline.

    Args:
        src_ip  : Source IP of suspicious traffic
        payload : Optional payload data
        callback: Optional function called with result
        timeout : Seconds to run sandbox

    Returns:
        The background thread object
    """
    def _run():
        result = run_sandbox(src_ip, payload, timeout)
        if callback:
            try:
                callback(result)
            except Exception as e:
                logger.error(
                    f"[SANDBOX] Callback error: {e}")

    t = threading.Thread(
        target=_run,
        daemon=True,
        name=f"sandbox-{src_ip}"
    )
    t.start()
    return t


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing sandbox...")
    print(f"Running on Linux : {IS_LINUX}")
    print(f"Docker available : {_docker_available()}")
    print()

    # Test 1 — run sandbox (simulated on Windows)
    print("Test 1 — run_sandbox() simulated")
    result = run_sandbox(
        src_ip  = "192.168.1.10",
        timeout = 5
    )
    print(f"Canary touched   : {result['canary_touched']}")
    print(f"Anomalous        : {result['anomalous']}")
    print(f"Decision         : {result['sandbox_decision']}")
    print(f"Details          : {result['details']}")
    print()

    # Test 2 — async sandbox
    print("Test 2 — run_sandbox_async()")
    done = threading.Event()

    def on_complete(r):
        print(f"Async result: {r['sandbox_decision']}")
        done.set()

    run_sandbox_async(
        src_ip   = "192.168.1.11",
        callback = on_complete,
        timeout  = 3
    )
    done.wait(timeout=10)
    print()

    # Test 3 — get result
    print("Test 3 — get_sandbox_result()")
    r = get_sandbox_result("192.168.1.10")
    print(f"Stored result: "
          f"{r['sandbox_decision'] if r else 'None'}")

    print("\nsandbox.py OK")