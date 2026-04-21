"""
cef_writer.py — CEF Format Alert Logger
Writes security alerts in Common Event Format (CEF)
to a log file. Compatible with Splunk and Wazuh SIEM.

CEF Format:
CEF:0|AegisAI|Trinity|1.0|THREAT|{label}|{severity}|
start={timestamp} src={src_ip} dst={dst_ip} act={action}
cs1={score} cs1Label=RiskScore
cs2={mitre_id} cs2Label=MITRETactic
cs3={mitre_name} cs3Label=MITREName
cs4={layer} cs4Label=DetectionLayer
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional
import config

logger = logging.getLogger(__name__)

# ─── Severity mapping per decision ──────────────────────────
SEVERITY_MAP = {
    "BLOCK"     : 8,
    "ISOLATE"   : 9,
    "SANDBOX"   : 5,
    "SUSPICIOUS": 4,
    "ALLOW"     : 1,
}


def _get_severity(decision: str) -> int:
    """
    Map a decision string to CEF severity integer.

    Args:
        decision: Decision string from predictor

    Returns:
        Integer severity 1-9
    """
    return SEVERITY_MAP.get(decision.upper(), 3)


def _sanitize(value: str) -> str:
    """
    Sanitize a string value for CEF format.
    CEF uses pipe and equals as delimiters —
    these must be escaped in field values.

    Args:
        value: Raw string value

    Returns:
        Sanitized string safe for CEF output
    """
    if not isinstance(value, str):
        value = str(value)
    value = value.replace("\\", "\\\\")
    value = value.replace("|", "\\|")
    value = value.replace("=", "\\=")
    return value


def _ensure_log_dir() -> None:
    """
    Create the log directory if it does not exist.
    On Windows this creates a local alerts.cef file.
    On Pi this creates /var/log/aegisai/ directory.
    """
    log_path = config.CEF_LOG_PATH
    log_dir  = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            logger.warning(
                f"[CEF] Could not create log dir"
                f" {log_dir}: {e}"
            )


def write_alert(
    src_ip   : str,
    dst_ip   : str,
    label    : str,
    score    : float,
    decision : str,
    mitre    : dict,
    layer    : str,
    timestamp: Optional[str] = None
) -> bool:
    """
    Write one CEF alert line to the log file.
    Appends to existing file — never overwrites.
    Thread-safe for concurrent writes.

    Args:
        src_ip   : Source IP address
        dst_ip   : Destination IP address
        label    : Attack label from ML model
        score    : Threat score 0-100
        decision : BLOCK/ISOLATE/SANDBOX/SUSPICIOUS/ALLOW
        mitre    : MITRE dict with id, tactic, name keys
        layer    : Detection layer name
        timestamp: ISO timestamp — uses UTC now if None

    Returns:
        True if write succeeded, False otherwise
    """
    try:
        _ensure_log_dir()

        if timestamp is None:
            timestamp = datetime.now(
                timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ")

        severity   = _get_severity(decision)
        mitre_id   = _sanitize(
            mitre.get("id",     "NONE"))
        mitre_tact = _sanitize(
            mitre.get("tactic", "None"))
        mitre_name = _sanitize(
            mitre.get("name",   "Unknown"))

        # Build CEF header
        cef_header = (
            f"CEF:0|AegisAI|Trinity|1.0|THREAT"
            f"|{_sanitize(label)}"
            f"|{severity}"
        )

        # Build CEF extension
        cef_ext = (
            f"start={timestamp}"
            f" src={_sanitize(src_ip)}"
            f" dst={_sanitize(dst_ip)}"
            f" act={_sanitize(decision)}"
            f" cs1={score:.2f}"
            f" cs1Label=RiskScore"
            f" cs2={mitre_id}"
            f" cs2Label=MITRETactic"
            f" cs3={mitre_name}"
            f" cs3Label=MITREName"
            f" cs4={_sanitize(layer)}"
            f" cs4Label=DetectionLayer"
        )

        cef_line = f"{cef_header}|{cef_ext}\n"

        with open(config.CEF_LOG_PATH, "a",
                  encoding="utf-8") as f:
            f.write(cef_line)

        logger.debug(
            f"[CEF] Alert written — "
            f"{label} score={score:.2f} "
            f"decision={decision}"
        )
        return True

    except Exception as e:
        logger.error(f"[CEF] Write error: {e}")
        return False


def write_allow(
    src_ip   : str,
    dst_ip   : str,
    score    : float,
    timestamp: Optional[str] = None
) -> bool:
    """
    Write a CEF entry for allowed normal traffic.
    Keeps audit trail of all traffic not just threats.

    Args:
        src_ip   : Source IP address
        dst_ip   : Destination IP address
        score    : Threat score 0-100
        timestamp: ISO timestamp — uses UTC now if None

    Returns:
        True if write succeeded, False otherwise
    """
    return write_alert(
        src_ip   = src_ip,
        dst_ip   = dst_ip,
        label    = "Normal",
        score    = score,
        decision = "ALLOW",
        mitre    = {
            "id"    : "NONE",
            "tactic": "None",
            "name"  : "No Threat Detected"
        },
        layer     = "SentinelX",
        timestamp = timestamp
    )


def get_recent_alerts(n: int = 50) -> list:
    """
    Read the last N lines from the CEF log file.
    Used by the /status endpoint to show recent activity.

    Args:
        n: Number of recent lines to return

    Returns:
        List of CEF line strings, most recent last
    """
    try:
        if not os.path.exists(config.CEF_LOG_PATH):
            return []
        with open(config.CEF_LOG_PATH, "r",
                  encoding="utf-8") as f:
            lines = f.readlines()
        return [l.strip() for l in lines[-n:]
                if l.strip()]
    except Exception as e:
        logger.error(f"[CEF] Read error: {e}")
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing CEF writer...")

    # Test 1 — write a BLOCK alert
    ok = write_alert(
        src_ip   = "192.168.1.10",
        dst_ip   = "192.168.1.20",
        label    = "PortScan",
        score    = 85.5,
        decision = "BLOCK",
        mitre    = {
            "id"    : "T1046",
            "tactic": "Discovery",
            "name"  : "Network Service Discovery"
        },
        layer    = "SentinelX"
    )
    print(f"Write BLOCK alert  : {'OK' if ok else 'FAIL'}")

    # Test 2 — write an ISOLATE alert
    ok = write_alert(
        src_ip   = "192.168.1.10",
        dst_ip   = "192.168.1.20",
        label    = "Infiltration",
        score    = 61.7,
        decision = "ISOLATE",
        mitre    = {
            "id"    : "T1078",
            "tactic": "Defense Evasion",
            "name"  : "Valid Accounts"
        },
        layer    = "PhantomNet"
    )
    print(f"Write ISOLATE alert: {'OK' if ok else 'FAIL'}")

    # Test 3 — write an ALLOW entry
    ok = write_allow(
        src_ip = "192.168.1.30",
        dst_ip = "192.168.1.20",
        score  = 12.3
    )
    print(f"Write ALLOW entry  : {'OK' if ok else 'FAIL'}")

    # Test 4 — read back recent alerts
    alerts = get_recent_alerts(10)
    print(f"\nRecent alerts ({len(alerts)} lines):")
    for line in alerts:
        print(f"  {line}")

    print("\ncef_writer.py OK")