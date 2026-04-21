"""
gpio_controller.py — GPIO and Relay Controller
Controls LEDs and relay module on Raspberry Pi.
Simulates all actions on Windows with print statements.
Never crashes on Windows — IS_PI flag handles platform.
"""

import logging
import time

logger = logging.getLogger(__name__)

# ─── Platform detection ─────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    IS_PI = True
    logger.info("[GPIO] RPi.GPIO loaded — running on Pi")
except (ImportError, RuntimeError):
    IS_PI = False
    logger.info("[GPIO] RPi.GPIO not found — simulation mode")

# ─── GPIO Pin assignments ───────────────────────────────────
import config
PIN_RELAY      = config.GPIO_RELAY       # 17
PIN_RED_LED    = config.GPIO_RED_LED     # 27
PIN_GREEN_LED  = config.GPIO_GREEN_LED   # 22
PIN_ORANGE_LED = config.GPIO_ORANGE_LED  # 23

# ─── Current state tracking ─────────────────────────────────
_state = {
    "relay"      : False,
    "red_led"    : False,
    "green_led"  : False,
    "orange_led" : False,
}


def setup_gpio() -> None:
    """
    Initialize GPIO pins as outputs.
    Call once at FastAPI startup.
    Safe to call on Windows — does nothing in sim mode.
    """
    if not IS_PI:
        print("[GPIO SIM] GPIO setup complete")
        return

    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(PIN_RELAY,      GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(PIN_RED_LED,    GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(PIN_GREEN_LED,  GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(PIN_ORANGE_LED, GPIO.OUT, initial=GPIO.LOW)
        logger.info("[GPIO] Pins initialized")
    except Exception as e:
        logger.error(f"[GPIO] Setup error: {e}")


def cleanup_gpio() -> None:
    """
    Release all GPIO pins.
    Call at FastAPI shutdown to leave pins clean.
    Safe to call on Windows — does nothing in sim mode.
    """
    if not IS_PI:
        print("[GPIO SIM] GPIO cleanup complete")
        return

    try:
        GPIO.cleanup()
        logger.info("[GPIO] Cleanup complete")
    except Exception as e:
        logger.error(f"[GPIO] Cleanup error: {e}")


def _set_pin(pin: int, state: bool,
             pin_name: str) -> None:
    """
    Set a single GPIO pin HIGH or LOW.
    Prints simulation message on Windows.

    Args:
        pin     : GPIO pin number
        state   : True for HIGH, False for LOW
        pin_name: Human readable pin name for logs
    """
    level = "HIGH" if state else "LOW"
    if not IS_PI:
        print(f"[GPIO SIM] Pin {pin} ({pin_name}) → {level}")
        return

    try:
        GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
        logger.debug(
            f"[GPIO] Pin {pin} ({pin_name}) → {level}")
    except Exception as e:
        logger.error(
            f"[GPIO] Pin {pin} error: {e}")


def system_safe() -> None:
    """
    Signal system is safe — green LED on.
    Turns off red and orange LEDs.
    Called when threat score is below threshold.
    """
    global _state
    _set_pin(PIN_GREEN_LED,  True,  "GREEN_LED")
    _set_pin(PIN_RED_LED,    False, "RED_LED")
    _set_pin(PIN_ORANGE_LED, False, "ORANGE_LED")
    _state["green_led"]  = True
    _state["red_led"]    = False
    _state["orange_led"] = False
    logger.info("[GPIO] System safe — green LED on")


def threat_detected() -> None:
    """
    Signal threat detected — red LED on.
    Turns off green and orange LEDs.
    Called on BLOCK or ISOLATE decision.
    """
    global _state
    _set_pin(PIN_RED_LED,    True,  "RED_LED")
    _set_pin(PIN_GREEN_LED,  False, "GREEN_LED")
    _set_pin(PIN_ORANGE_LED, False, "ORANGE_LED")
    _state["red_led"]    = True
    _state["green_led"]  = False
    _state["orange_led"] = False
    logger.info("[GPIO] Threat detected — red LED on")


def system_healing() -> None:
    """
    Signal system is healing — orange LED on.
    Turns off red and green LEDs.
    Called during file isolation and restore process.
    """
    global _state
    _set_pin(PIN_ORANGE_LED, True,  "ORANGE_LED")
    _set_pin(PIN_RED_LED,    False, "RED_LED")
    _set_pin(PIN_GREEN_LED,  False, "GREEN_LED")
    _state["orange_led"] = True
    _state["red_led"]    = False
    _state["green_led"]  = False
    logger.info("[GPIO] System healing — orange LED on")


def fire_relay() -> None:
    """
    Fire the relay — cuts Pi 1 ethernet connection.
    GPIO pin 17 HIGH → relay clicks → network cut.
    Called immediately on BLOCK decision.
    """
    global _state
    _set_pin(PIN_RELAY, True, "RELAY")
    _state["relay"] = True
    logger.info("[GPIO] Relay FIRED — network cut")


def reset_relay() -> None:
    """
    Reset the relay — restores Pi 1 ethernet.
    GPIO pin 17 LOW → relay releases → network restored.
    Called after threat is resolved.
    """
    global _state
    _set_pin(PIN_RELAY, False, "RELAY")
    _state["relay"] = False
    logger.info("[GPIO] Relay RESET — network restored")


def get_state() -> dict:
    """
    Return current state of all GPIO pins.
    Used by GET /status endpoint.

    Returns:
        Dictionary with relay, red_led, green_led,
        orange_led boolean states
    """
    return dict(_state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing GPIO controller...")
    print(f"Running on Pi: {IS_PI}")
    print()

    print("Setting up GPIO...")
    setup_gpio()

    print("\nTest 1 — system_safe()")
    system_safe()
    print(f"State: {get_state()}")

    print("\nTest 2 — threat_detected()")
    threat_detected()
    print(f"State: {get_state()}")

    print("\nTest 3 — system_healing()")
    system_healing()
    print(f"State: {get_state()}")

    print("\nTest 4 — fire_relay()")
    fire_relay()
    print(f"State: {get_state()}")

    print("\nTest 5 — reset_relay()")
    reset_relay()
    print(f"State: {get_state()}")

    print("\nTest 6 — cleanup")
    cleanup_gpio()

    print("\ngpio_controller.py OK")