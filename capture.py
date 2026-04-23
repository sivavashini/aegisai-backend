"""
capture.py — Packet Capture Daemon
Captures live packets from network interface using Scapy.
Extracts features and feeds them through the ML pipeline.
Runs as a background thread — never blocks FastAPI.
Simulates on Windows without Npcap installed.
"""

import logging
import threading
import time
import platform
from typing import Optional, Callable
from datetime import datetime, timezone
import config

logger = logging.getLogger(__name__)

# ─── Platform detection ─────────────────────────────────────
IS_LINUX = platform.system() == "Linux"

# ─── Capture state ──────────────────────────────────────────
_capturing       : bool = False
_capture_thread  : Optional[threading.Thread] = None
_stop_event      = threading.Event()
_packet_count    : int = 0
_event_callback  : Optional[Callable] = None
_stats = {
    "packets_captured" : 0,
    "packets_processed": 0,
    "packets_skipped"  : 0,
    "started_at"       : None,
    "last_packet_at"   : None,
}


def set_event_callback(callback: Callable) -> None:
    """
    Register callback for processed packet events.
    Called by main.py to hook into WebSocket broadcast.

    Args:
        callback: Function called with event dict
                  for every processed packet
    """
    global _event_callback
    _event_callback = callback
    


def _process_packet(packet) -> None:
    """
    Process a single captured packet through the
    full AegisAI pipeline.
    Called for every packet by Scapy sniff().

    Args:
        packet: Scapy packet object
    """
    global _packet_count
    _packet_count += 1
    _stats["packets_captured"] += 1
    _stats["last_packet_at"] = datetime.now(
        timezone.utc).isoformat()

    try:
        # Import here to avoid circular imports
        from features import extract_features
        from predictor import predict
        from responder import execute_response
        from otx_check import check_ip
        from scapy.layers.inet import IP as _IP

        # Whitelist — never block management traffic
        WHITELIST_IPS = {
            config.PI1_IP,
            config.PI2_IP,
            config.PI3_IP,
            config.LAPTOP_IP,
            "127.0.0.1",
        }
        if packet.haslayer(_IP):
            _src = packet[_IP].src
            _dst = packet[_IP].dst
            if _src in WHITELIST_IPS or _dst in WHITELIST_IPS:
                return

        # Extract feature vector
        feature_vec = extract_features(packet)
        if feature_vec is None:
            _stats["packets_skipped"] += 1
            return

        # Get IP addresses
        from scapy.layers.inet import IP
        if not packet.haslayer(IP):
            _stats["packets_skipped"] += 1
            return

        src_ip = packet[IP].src
        dst_ip = packet[IP].dst

        # Pre-ML OTX check — known bad IPs
        if check_ip(src_ip):
            logger.warning(
                f"[CAPTURE] OTX hit — "
                f"blocking known bad IP: {src_ip}")
            from responder import respond_block
            from mitre_map import get_mitre
            respond_block(
                src_ip = src_ip,
                dst_ip = dst_ip,
                label  = "OTX_BLACKLIST",
                score  = 100.0,
                mitre  = get_mitre("UNKNOWN"),
                layer  = "OTX"
            )
            _stats["packets_processed"] += 1
            return

        # Skip flows with less than 3 packets — unreliable stats
        from features import _flow_table, _flow_key
        from scapy.layers.inet import IP as _IP2, TCP, UDP
        if packet.haslayer(_IP2):
            _ip = packet[_IP2]
            _sp = packet[TCP].sport if packet.haslayer(TCP) else (packet[UDP].sport if packet.haslayer(UDP) else 0)
            _dp = packet[TCP].dport if packet.haslayer(TCP) else (packet[UDP].dport if packet.haslayer(UDP) else 0)
            _key = _flow_key(_ip.src, _ip.dst, _sp, _dp, _ip.proto)
            if _flow_table.get(_key, {}).get("spkts", 0) < 3:
                return

        # Build feature dict from vector
        from predictor import _models
        feature_cols = _models.get("feature_cols1", [])
        interaction_cols = {
            "sbytes_per_spkt", "dbytes_per_dpkt",
            "load_ratio", "smean_x_rate",
            "ttl_diff", "jit_ratio",
            "sbytes_per_dur", "rate_x_dur",
            "sloss_ratio", "port_reuse_ratio",
            "ack_syn_ratio", "pkt_symmetry",
            "sbytes_per_srv", "srv_src_dst_ratio",
        }
        base_cols = [c for c in feature_cols
                     if c not in interaction_cols]

        feature_dict = {}
        for i, col in enumerate(base_cols):
            if i < len(feature_vec):
                feature_dict[col] = float(
                    feature_vec[i])

        feature_dict["src_ip"] = src_ip
        feature_dict["dst_ip"] = dst_ip

        # Run ML pipeline
        prediction = predict(feature_dict)

        decision = prediction.get("decision", "ALLOW")
        label    = prediction.get("label",    "Normal")
        score    = prediction.get("score",    0.0)
        mitre    = prediction.get("mitre",    {})
        layer    = prediction.get("layer",
                                  "SentinelX")

        # Execute response
        execute_response(
            decision = decision,
            src_ip   = src_ip,
            dst_ip   = dst_ip,
            label    = label,
            score    = score,
            mitre    = mitre,
            layer    = layer
        )

        # Trigger file isolation on block/isolate
        if decision in ("BLOCK", "ISOLATE"):
            from file_isolation import (
                trigger_isolation)
            trigger_isolation(blocking=False)

        # Fire event callback for WebSocket
        if _event_callback:
            event = {
                "type"      : "packet_event",
                "timestamp" : datetime.now(
                    timezone.utc).isoformat(),
                "src_ip"    : src_ip,
                "dst_ip"    : dst_ip,
                "label"     : label,
                "score"     : score,
                "decision"  : decision,
                "mitre"     : mitre,
                "layer"     : layer,
            }
            try:
                _event_callback(event)
            except Exception:
                pass

        _stats["packets_processed"] += 1

        logger.info(
            f"[CAPTURE] Processed — "
            f"src={src_ip} dst={dst_ip} "
            f"label={label} score={score:.2f} "
            f"decision={decision}"
        )

    except Exception as e:
        logger.error(
            f"[CAPTURE] Process error: {e}")
        _stats["packets_skipped"] += 1


def _capture_loop(interface: Optional[str] = None,
                  packet_count: int = 0) -> None:
    """
    Main capture loop using Scapy sniff().
    Runs until stop_capture() is called.

    Args:
        interface   : Network interface name
                      None uses default interface
        packet_count: Max packets to capture
                      0 means capture forever
    """
    logger.info(
        f"[CAPTURE] Starting capture on "
        f"interface={interface or 'default'}")

    try:
        from scapy.all import sniff
        from features import clear_flow_table
        _last_clear = time.time()

        def _prn_with_clear(packet):
            nonlocal _last_clear
            if time.time() - _last_clear > 300:
                clear_flow_table()
                _last_clear = time.time()
                logger.info("[CAPTURE] Flow table cleared")
            _process_packet(packet)

        sniff(
            iface    = interface,
            prn      = _prn_with_clear,
            store    = False,
            stop_filter=lambda p: _stop_event.is_set(),
            count    = packet_count
        )
    except Exception as e:
        logger.error(
            f"[CAPTURE] Sniff error: {e}")
    finally:
        logger.info("[CAPTURE] Capture loop ended")


def _simulate_capture_loop() -> None:
    """
    Simulate packet capture on Windows for testing.
    Generates synthetic packet events every 2 seconds.
    Runs until stop_capture() is called.
    """
    logger.info(
        "[CAPTURE] Simulation mode started")

    # Import predict here to avoid startup delay
    from predictor import predict

    synthetic_flows = [
        {
            "dur": 0.1, "proto": "tcp",
            "service": "http", "state": "FIN",
            "spkts": 5, "dpkts": 3,
            "sbytes": 500, "dbytes": 300,
            "rate": 50.0, "sttl": 64, "dttl": 64,
            "sload": 40000.0, "dload": 24000.0,
            "sloss": 0, "dloss": 0,
            "sinpkt": 0.02, "dinpkt": 0.03,
            "sjit": 0.001, "djit": 0.001,
            "swin": 255, "stcpb": 0, "dtcpb": 0,
            "dwin": 255, "tcprtt": 0.01,
            "synack": 0.005, "ackdat": 0.005,
            "smean": 100, "dmean": 100,
            "trans_depth": 1,
            "response_body_len": 200,
            "ct_srv_src": 3, "ct_state_ttl": 2,
            "ct_dst_ltm": 2, "ct_src_dport_ltm": 1,
            "ct_dst_sport_ltm": 1,
            "ct_dst_src_ltm": 2,
            "is_ftp_login": 0, "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": 0, "ct_src_ltm": 2,
            "ct_srv_dst": 3, "is_sm_ips_ports": 0,
            "src_ip": config.PI1_IP,
            "dst_ip": config.PI2_IP,
        },
    ]

    flow_idx = 0
    _last_clear = time.time()
    while not _stop_event.is_set():
        try:
            if time.time() - _last_clear > 300:
                from features import clear_flow_table
                clear_flow_table()
                _last_clear = time.time()
                logger.info("[CAPTURE] Flow table cleared")
            flow = dict(
                synthetic_flows[
                    flow_idx % len(synthetic_flows)])
            flow_idx += 1

            prediction = predict(flow)
            decision   = prediction.get(
                "decision", "ALLOW")
            label      = prediction.get(
                "label", "Normal")
            score      = prediction.get("score", 0.0)
            mitre      = prediction.get("mitre",  {})
            layer      = prediction.get(
                "layer", "SentinelX")

            _stats["packets_captured"]  += 1
            _stats["packets_processed"] += 1
            _stats["last_packet_at"] = datetime.now(
                timezone.utc).isoformat()

            if _event_callback:
                event = {
                    "type"     : "packet_event",
                    "timestamp": datetime.now(
                        timezone.utc).isoformat(),
                    "src_ip"   : flow["src_ip"],
                    "dst_ip"   : flow["dst_ip"],
                    "label"    : label,
                    "score"    : score,
                    "decision" : decision,
                    "mitre"    : mitre,
                    "layer"    : layer,
                    "simulated": True,
                }
                try:
                    _event_callback(event)
                except Exception:
                    pass

            logger.info(
                f"[CAPTURE SIM] "
                f"src={flow['src_ip']} "
                f"label={label} "
                f"score={score:.2f} "
                f"decision={decision}"
            )

        except Exception as e:
            logger.error(
                f"[CAPTURE SIM] Error: {e}")

        _stop_event.wait(timeout=2)

    logger.info("[CAPTURE] Simulation loop ended")


def start_capture(
    interface   : Optional[str] = None,
    simulate    : bool = False,
    packet_count: int = 0
) -> threading.Thread:
    """
    Start packet capture as a background thread.
    Uses real Scapy capture on Linux Pi.
    Uses simulation loop on Windows.
    Call once at FastAPI startup.

    Args:
        interface   : Network interface name
        simulate    : Force simulation mode
        packet_count: Max packets (0 = forever)

    Returns:
        The background capture thread
    """
    global _capture_thread, _capturing
    _stop_event.clear()
    _capturing = True
    _stats["started_at"] = datetime.now(
        timezone.utc).isoformat()

    use_sim = simulate or not IS_LINUX

    target = (_simulate_capture_loop
              if use_sim
              else lambda: _capture_loop(
                  interface, packet_count))

    _capture_thread = threading.Thread(
        target=target,
        daemon=True,
        name="packet-capture"
    )
    _capture_thread.start()
    logger.info(
        f"[CAPTURE] Thread started — "
        f"sim={use_sim}")
    return _capture_thread


def stop_capture() -> None:
    """
    Stop the packet capture thread gracefully.
    Call at FastAPI shutdown.
    """
    global _capturing
    _stop_event.set()
    _capturing = False
    logger.info("[CAPTURE] Stop signal sent")


def get_capture_stats() -> dict:
    """
    Return current capture statistics.
    Used by GET /status endpoint.

    Returns:
        Dictionary with packet counts and timestamps
    """
    return dict(_stats)


def is_capturing() -> bool:
    """
    Return whether capture is currently active.

    Returns:
        True if capture thread is running
    """
    return _capturing


if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)
    print("Testing capture daemon...")
    print(f"Running on Linux: {IS_LINUX}")
    print()

    # Load models first
    print("Loading models...")
    from predictor import load_all_models
    load_all_models()
    print()

    # Test — start simulation capture for 6 seconds
    print("Starting simulated capture for 6 seconds...")
    events_received = []

    def on_event(event):
        events_received.append(event)

    set_event_callback(on_event)
    start_capture(simulate=True)
    time.sleep(6)
    stop_capture()
    time.sleep(1)

    stats = get_capture_stats()
    print(f"\nCapture stats:")
    print(f"  Packets captured  : "
          f"{stats['packets_captured']}")
    print(f"  Packets processed : "
          f"{stats['packets_processed']}")
    print(f"  Events received   : "
          f"{len(events_received)}")

    if events_received:
        last = events_received[-1]
        print(f"\nLast event:")
        print(f"  Label    : {last['label']}")
        print(f"  Score    : {last['score']:.2f}")
        print(f"  Decision : {last['decision']}")

    print("\ncapture.py OK")