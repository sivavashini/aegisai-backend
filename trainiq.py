"""
trainiq.py — UEBA User Training Module
Monitors behavioral signals per user IP.
Auto-assigns training modules when risk score hits 60.
Score decays 5 points per day. Caps at 100.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional
import config

logger = logging.getLogger(__name__)

# ─── Behavioral signal point values ─────────────────────────
SIGNAL_POINTS = {
    "failed_login"        : 10,
    "unusual_access_time" : 15,
    "phishing_domain"     : 25,
    "sensitive_file"      : 20,
    "privilege_escalation": 30,
}

# ─── Module auto-assignment rules ───────────────────────────
SIGNAL_MODULE_MAP = {
    "phishing_domain"     : "phishing_awareness",
    "failed_login"        : "password_hygiene",
    "unusual_access_time" : "safe_browsing",
    "sensitive_file"      : "safe_browsing",
    "privilege_escalation": "password_hygiene",
}

# ─── Module descriptions ────────────────────────────────────
MODULES = {
    "phishing_awareness": {
        "name"       : "Phishing Awareness",
        "description": "Teaches email threat "
                       "identification and safe "
                       "link practices",
        "score_reduction": 20
    },
    "password_hygiene": {
        "name"       : "Password Hygiene",
        "description": "Teaches password strength, "
                       "MFA setup, and credential "
                       "management",
        "score_reduction": 20
    },
    "safe_browsing": {
        "name"       : "Safe Browsing",
        "description": "Teaches HTTPS verification, "
                       "VPN usage, and USB security",
        "score_reduction": 20
    },
}

# ─── Auto-assign threshold ───────────────────────────────────
AUTO_ASSIGN_THRESHOLD = 60
SCORE_CAP             = 100
DAILY_DECAY           = 5

# ─── In-memory user state ───────────────────────────────────
# Keyed by IP address
_users: dict = {}
_users_lock  = threading.Lock()
_decay_thread: Optional[threading.Thread] = None
_stop_decay   = threading.Event()


def _get_user(ip: str) -> dict:
    """
    Get or create user state for an IP address.
    Must be called with _users_lock held.

    Args:
        ip: User IP address string

    Returns:
        User state dictionary
    """
    if ip not in _users:
        _users[ip] = {
            "ip"               : ip,
            "score"            : 0,
            "signals"          : {},
            "assigned_modules" : [],
            "completed_modules": [],
            "flagged_behaviors": [],
            "last_updated"     : datetime.now(
                timezone.utc).isoformat(),
            "created_at"       : datetime.now(
                timezone.utc).isoformat(),
        }
    return _users[ip]


def record_signal(ip: str, signal: str) -> dict:
    """
    Record a behavioral signal for a user IP.
    Adds points to risk score.
    Auto-assigns training module if score hits 60.
    Score is capped at 100.

    Args:
        ip    : User IP address
        signal: Signal name from SIGNAL_POINTS keys

    Returns:
        Updated user state dictionary
    """
    if signal not in SIGNAL_POINTS:
        logger.warning(
            f"[TRAINIQ] Unknown signal: {signal}")
        return {}

    points = SIGNAL_POINTS[signal]

    with _users_lock:
        user = _get_user(ip)

        # Add points — cap at 100
        old_score    = user["score"]
        user["score"] = min(
            user["score"] + points, SCORE_CAP)

        # Track signal counts
        user["signals"][signal] = \
            user["signals"].get(signal, 0) + 1

        # Track flagged behaviors
        user["flagged_behaviors"].append({
            "signal"   : signal,
            "points"   : points,
            "timestamp": datetime.now(
                timezone.utc).isoformat()
        })

        # Keep only last 50 behaviors
        if len(user["flagged_behaviors"]) > 50:
            user["flagged_behaviors"] = \
                user["flagged_behaviors"][-50:]

        user["last_updated"] = datetime.now(
            timezone.utc).isoformat()

        # Auto-assign module if threshold crossed
        if (old_score < AUTO_ASSIGN_THRESHOLD and
                user["score"] >= AUTO_ASSIGN_THRESHOLD):
            module = SIGNAL_MODULE_MAP.get(signal)
            if (module and
                    module not in
                    user["assigned_modules"] and
                    module not in
                    user["completed_modules"]):
                user["assigned_modules"].append(module)
                logger.info(
                    f"[TRAINIQ] Auto-assigned "
                    f"{module} to {ip}")

        logger.info(
            f"[TRAINIQ] Signal recorded — "
            f"ip={ip} signal={signal} "
            f"points=+{points} "
            f"score={user['score']}")

        return dict(user)


def complete_module(ip: str,
                    module: str) -> dict:
    """
    Mark a training module as completed for a user.
    Reduces risk score by 20 points.
    Moves module from assigned to completed list.

    Args:
        ip    : User IP address
        module: Module name string

    Returns:
        Updated user state dictionary or empty dict
        if module not found or not assigned
    """
    if module not in MODULES:
        logger.warning(
            f"[TRAINIQ] Unknown module: {module}")
        return {}

    with _users_lock:
        user = _get_user(ip)

        if module not in user["assigned_modules"]:
            logger.warning(
                f"[TRAINIQ] Module {module} not "
                f"assigned to {ip}")
            return dict(user)

        # Move from assigned to completed
        user["assigned_modules"].remove(module)
        if module not in user["completed_modules"]:
            user["completed_modules"].append(module)

        # Reduce score by module reduction amount
        reduction = MODULES[module]["score_reduction"]
        user["score"] = max(
            user["score"] - reduction, 0)

        user["last_updated"] = datetime.now(
            timezone.utc).isoformat()

        logger.info(
            f"[TRAINIQ] Module completed — "
            f"ip={ip} module={module} "
            f"score={user['score']}")

        return dict(user)


def get_risk_score(ip: str) -> dict:
    """
    Get current risk score and state for a user IP.
    Used by GET /trainiq/risk-score endpoint.

    Args:
        ip: User IP address

    Returns:
        User state dictionary with score and modules
    """
    with _users_lock:
        user = _get_user(ip)
        return dict(user)


def get_all_users() -> list:
    """
    Get risk scores for all tracked users.
    Used by GET /trainiq/all-users endpoint.

    Returns:
        List of all user state dictionaries
    """
    with _users_lock:
        return [dict(u) for u in _users.values()]


def _apply_daily_decay() -> None:
    """
    Background thread that applies score decay.
    Reduces each user score by 5 points every 24 hours.
    Runs until stop_decay() is called.
    """
    logger.info("[TRAINIQ] Decay thread started")
    while not _stop_decay.is_set():
        # Wait 24 hours
        _stop_decay.wait(timeout=86400)
        if _stop_decay.is_set():
            break

        with _users_lock:
            for ip, user in _users.items():
                if user["score"] > 0:
                    old = user["score"]
                    user["score"] = max(
                        user["score"] - DAILY_DECAY, 0)
                    logger.info(
                        f"[TRAINIQ] Decay applied — "
                        f"ip={ip} "
                        f"{old} → {user['score']}")

    logger.info("[TRAINIQ] Decay thread stopped")


def start_decay_thread() -> threading.Thread:
    """
    Start the daily score decay background thread.
    Call once at FastAPI startup.

    Returns:
        The background thread object
    """
    global _decay_thread
    _stop_decay.clear()
    _decay_thread = threading.Thread(
        target=_apply_daily_decay,
        daemon=True,
        name="trainiq-decay"
    )
    _decay_thread.start()
    return _decay_thread


def stop_decay_thread() -> None:
    """
    Stop the decay background thread.
    Call at FastAPI shutdown.
    """
    _stop_decay.set()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing TrainIQ UEBA...")
    print()

    test_ip = "192.168.1.10"

    # Test 1 — record signals
    print("Test 1 — Record behavioral signals")
    state = record_signal(test_ip, "failed_login")
    print(f"After failed_login    : score={state['score']}")

    state = record_signal(test_ip, "unusual_access_time")
    print(f"After unusual_access  : score={state['score']}")

    state = record_signal(test_ip, "sensitive_file")
    print(f"After sensitive_file  : score={state['score']}")

    # Test 2 — trigger auto-assign
    print("\nTest 2 — Trigger auto module assignment")
    state = record_signal(test_ip, "phishing_domain")
    print(f"After phishing_domain : score={state['score']}")
    print(f"Assigned modules      : "
          f"{state['assigned_modules']}")

    # Test 3 — complete module
    print("\nTest 3 — Complete assigned module")
    if state["assigned_modules"]:
        module = state["assigned_modules"][0]
        state  = complete_module(test_ip, module)
        print(f"After completing {module}:")
        print(f"  Score              : {state['score']}")
        print(f"  Assigned modules   : "
              f"{state['assigned_modules']}")
        print(f"  Completed modules  : "
              f"{state['completed_modules']}")

    # Test 4 — get risk score
    print("\nTest 4 — Get risk score")
    score_data = get_risk_score(test_ip)
    print(f"Score     : {score_data['score']}")
    print(f"Signals   : {score_data['signals']}")

    # Test 5 — score cap
    print("\nTest 5 — Score cap at 100")
    for _ in range(5):
        record_signal(test_ip, "privilege_escalation")
    state = get_risk_score(test_ip)
    print(f"Score after 5x privilege_escalation: "
          f"{state['score']} (max 100)")

    # Test 6 — all users
    print("\nTest 6 — Get all users")
    record_signal("192.168.1.20", "failed_login")
    all_users = get_all_users()
    print(f"Total users tracked: {len(all_users)}")
    for u in all_users:
        print(f"  {u['ip']:15} score={u['score']}")

    print("\ntrainiq.py OK")