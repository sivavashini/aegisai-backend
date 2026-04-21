"""
responder.py — Threat Response Orchestrator
Executes response actions based on threat decisions.
Combines iptables (on Pi) + GPIO + CEF logging.
Simulates iptables on Windows with print statements.
"""

import subprocess
import logging
import platform
from typing import Optional
import gpio_controller
import cef_writer

logger = logging.getLogger(__name__)

# ─── Platform detection ─────────────────────────────────────
IS_LINUX = platform.system() == "Linux"

# ─── Track blocked IPs to avoid duplicate rules ─────────────
_blocked_ips: set = set()


def _run_iptables(args: list) -> bool:
    """
    Run an iptables command on Linux.
    Prints simulation message on Windows.

    Args:
        args: List of iptables arguments

    Returns:
        True if command succeeded, False otherwise
    """
    cmd = ["iptables"] + args
    if not IS_LINUX:
        print(f"[IPTABLES SIM] {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            logger.error(
                f"[RESPONDER] iptables error: "
                f"{result.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(
            f"[RESPONDER] iptables exception: {e}")
        return False


def block_ip(src_ip: str) -> bool:
    """
    Block all traffic from a source IP using iptables.
    Adds DROP rule to INPUT chain.
    Skips if IP is already blocked.

    Args:
        src_ip: Source IP address to block

    Returns:
        True if blocked successfully, False otherwise
    """
    if src_ip in _blocked_ips:
        logger.info(
            f"[RESPONDER] {src_ip} already blocked")
        return True

    ok = _run_iptables([
        "-I", "INPUT", "-s", src_ip, "-j", "DROP"
    ])

    if ok:
        _blocked_ips.add(src_ip)
        logger.info(
            f"[RESPONDER] Blocked IP: {src_ip}")

    return ok


def unblock_ip(src_ip: str) -> bool:
    """
    Remove iptables block rule for a source IP.
    Safe to call even if IP is not currently blocked.

    Args:
        src_ip: Source IP address to unblock

    Returns:
        True if unblocked successfully, False otherwise
    """
    if src_ip not in _blocked_ips:
        logger.info(
            f"[RESPONDER] {src_ip} was not blocked")
        return True

    ok = _run_iptables([
        "-D", "INPUT", "-s", src_ip, "-j", "DROP"
    ])

    if ok:
        _blocked_ips.discard(src_ip)
        logger.info(
            f"[RESPONDER] Unblocked IP: {src_ip}")

    return ok


def respond_block(
    src_ip   : str,
    dst_ip   : str,
    label    : str,
    score    : float,
    mitre    : dict,
    layer    : str
) -> dict:
    """
    Execute BLOCK response — highest severity action.
    1. Block IP with iptables
    2. Fire relay to cut network
    3. Turn on red LED
    4. Write CEF alert

    Args:
        src_ip : Source IP to block
        dst_ip : Destination IP
        label  : Attack label
        score  : Threat score 0-100
        mitre  : MITRE ATT&CK dict
        layer  : Detection layer name

    Returns:
        Response result dictionary
    """
    logger.warning(
        f"[RESPONDER] BLOCK — {src_ip} "
        f"label={label} score={score:.2f}")

    results = {}

    # Step 1 — iptables block
    results["iptables"] = block_ip(src_ip)

    # Step 2 — fire relay
    gpio_controller.fire_relay()
    results["relay"] = True

    # Step 3 — red LED
    gpio_controller.threat_detected()
    results["led"] = "red"

    # Step 4 — CEF alert
    results["cef"] = cef_writer.write_alert(
        src_ip   = src_ip,
        dst_ip   = dst_ip,
        label    = label,
        score    = score,
        decision = "BLOCK",
        mitre    = mitre,
        layer    = layer
    )

    results["action"]  = "BLOCK"
    results["src_ip"]  = src_ip
    results["success"] = results["iptables"]
    return results


def respond_isolate(
    src_ip   : str,
    dst_ip   : str,
    label    : str,
    score    : float,
    mitre    : dict,
    layer    : str
) -> dict:
    """
    Execute ISOLATE response — high severity action.
    1. Block IP with iptables
    2. Fire relay to cut network
    3. Turn on red LED
    4. Write CEF alert
    Note: File isolation is triggered separately
          from main.py as a parallel thread.

    Args:
        src_ip : Source IP to isolate
        dst_ip : Destination IP
        label  : Attack label
        score  : Threat score 0-100
        mitre  : MITRE ATT&CK dict
        layer  : Detection layer name

    Returns:
        Response result dictionary
    """
    logger.warning(
        f"[RESPONDER] ISOLATE — {src_ip} "
        f"label={label} score={score:.2f}")

    results = {}

    # Step 1 — iptables block
    results["iptables"] = block_ip(src_ip)

    # Step 2 — fire relay
    gpio_controller.fire_relay()
    results["relay"] = True

    # Step 3 — red LED
    gpio_controller.threat_detected()
    results["led"] = "red"

    # Step 4 — CEF alert
    results["cef"] = cef_writer.write_alert(
        src_ip   = src_ip,
        dst_ip   = dst_ip,
        label    = label,
        score    = score,
        decision = "ISOLATE",
        mitre    = mitre,
        layer    = layer
    )

    results["action"]  = "ISOLATE"
    results["src_ip"]  = src_ip
    results["success"] = results["iptables"]
    return results


def respond_sandbox(
    src_ip   : str,
    dst_ip   : str,
    label    : str,
    score    : float,
    mitre    : dict,
    layer    : str
) -> dict:
    """
    Execute SANDBOX response — medium severity action.
    1. Orange LED — scanning in progress
    2. Write CEF alert
    Note: Docker sandbox is triggered separately
          from main.py.

    Args:
        src_ip : Source IP
        dst_ip : Destination IP
        label  : Attack label
        score  : Threat score 0-100
        mitre  : MITRE ATT&CK dict
        layer  : Detection layer name

    Returns:
        Response result dictionary
    """
    logger.info(
        f"[RESPONDER] SANDBOX — {src_ip} "
        f"label={label} score={score:.2f}")

    results = {}

    # Orange LED — scanning
    gpio_controller.system_healing()
    results["led"] = "orange"

    # CEF alert
    results["cef"] = cef_writer.write_alert(
        src_ip   = src_ip,
        dst_ip   = dst_ip,
        label    = label,
        score    = score,
        decision = "SANDBOX",
        mitre    = mitre,
        layer    = layer
    )

    results["action"]  = "SANDBOX"
    results["src_ip"]  = src_ip
    results["success"] = True
    return results


def respond_allow(
    src_ip   : str,
    dst_ip   : str,
    score    : float
) -> dict:
    """
    Execute ALLOW response — normal traffic.
    1. Green LED — system safe
    2. Write CEF allow entry

    Args:
        src_ip : Source IP
        dst_ip : Destination IP
        score  : Threat score 0-100

    Returns:
        Response result dictionary
    """
    gpio_controller.system_safe()

    cef_writer.write_allow(
        src_ip = src_ip,
        dst_ip = dst_ip,
        score  = score
    )

    return {
        "action" : "ALLOW",
        "src_ip" : src_ip,
        "led"    : "green",
        "success": True
    }


def respond_suspicious(
    src_ip   : str,
    dst_ip   : str,
    label    : str,
    score    : float,
    mitre    : dict,
    layer    : str
) -> dict:
    """
    Execute SUSPICIOUS response — log only.
    No network block. No relay. Orange LED.
    Write CEF alert for SIEM review.

    Args:
        src_ip : Source IP
        dst_ip : Destination IP
        label  : Attack label
        score  : Threat score 0-100
        mitre  : MITRE ATT&CK dict
        layer  : Detection layer name

    Returns:
        Response result dictionary
    """
    logger.info(
        f"[RESPONDER] SUSPICIOUS — {src_ip} "
        f"label={label} score={score:.2f}")

    gpio_controller.system_healing()

    cef_writer.write_alert(
        src_ip   = src_ip,
        dst_ip   = dst_ip,
        label    = label,
        score    = score,
        decision = "SUSPICIOUS",
        mitre    = mitre,
        layer    = layer
    )

    return {
        "action" : "SUSPICIOUS",
        "src_ip" : src_ip,
        "led"    : "orange",
        "success": True
    }


def execute_response(
    decision : str,
    src_ip   : str,
    dst_ip   : str,
    label    : str,
    score    : float,
    mitre    : dict,
    layer    : str
) -> dict:
    """
    Main entry point — route to correct response
    function based on decision string.
    Called by main.py for every processed event.

    Args:
        decision : BLOCK/ISOLATE/SANDBOX/SUSPICIOUS/ALLOW
        src_ip   : Source IP address
        dst_ip   : Destination IP address
        label    : Attack label from ML
        score    : Threat score 0-100
        mitre    : MITRE ATT&CK dict
        layer    : Detection layer name

    Returns:
        Response result dictionary
    """
    decision = decision.upper()

    if decision == "BLOCK":
        return respond_block(
            src_ip, dst_ip, label,
            score, mitre, layer)

    if decision == "ISOLATE":
        return respond_isolate(
            src_ip, dst_ip, label,
            score, mitre, layer)

    if decision == "SANDBOX":
        return respond_sandbox(
            src_ip, dst_ip, label,
            score, mitre, layer)

    if decision == "SUSPICIOUS":
        return respond_suspicious(
            src_ip, dst_ip, label,
            score, mitre, layer)

    # Default — ALLOW
    return respond_allow(src_ip, dst_ip, score)


def get_blocked_ips() -> list:
    """
    Return list of currently blocked IP addresses.
    Used by GET /status endpoint.

    Returns:
        List of blocked IP address strings
    """
    return list(_blocked_ips)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing responder...")
    print(f"Running on Linux: {IS_LINUX}")
    print()

    # Test 1 — BLOCK
    print("Test 1 — BLOCK response")
    result = execute_response(
        decision = "BLOCK",
        src_ip   = "192.168.1.10",
        dst_ip   = "192.168.1.20",
        label    = "PortScan",
        score    = 85.5,
        mitre    = {
            "id"    : "T1046",
            "tactic": "Discovery",
            "name"  : "Network Service Discovery"
        },
        layer    = "SentinelX"
    )
    print(f"Result: {result}")
    print()

    # Test 2 — ISOLATE
    print("Test 2 — ISOLATE response")
    result = execute_response(
        decision = "ISOLATE",
        src_ip   = "192.168.1.10",
        dst_ip   = "192.168.1.20",
        label    = "Infiltration",
        score    = 61.7,
        mitre    = {
            "id"    : "T1078",
            "tactic": "Defense Evasion",
            "name"  : "Valid Accounts"
        },
        layer    = "PhantomNet"
    )
    print(f"Result: {result}")
    print()

    # Test 3 — ALLOW
    print("Test 3 — ALLOW response")
    result = execute_response(
        decision = "ALLOW",
        src_ip   = "192.168.1.30",
        dst_ip   = "192.168.1.20",
        label    = "Normal",
        score    = 12.3,
        mitre    = {"id": "NONE",
                    "tactic": "None",
                    "name": "No Threat"},
        layer    = "SentinelX"
    )
    print(f"Result: {result}")
    print()

    # Test 4 — blocked IPs list
    print(f"Blocked IPs: {get_blocked_ips()}")

    print("\nresponder.py OK")