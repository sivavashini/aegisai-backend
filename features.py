"""
features.py — Packet to Feature Vector Extractor
Converts raw Scapy packets into the exact feature vector
that the ML models expect. Feature order must match
feature_cols.pkl used during training.
"""

import time
import math
from typing import Optional
import numpy as np


# ─── Flow tracker for session-level features ────────────────
# Stores active flows keyed by (src_ip, dst_ip, src_port, dst_port, proto)
_flow_table: dict = {}


def _flow_key(src_ip: str, dst_ip: str,
              src_port: int, dst_port: int,
              proto: int) -> tuple:
    """
    Build a canonical flow key from packet fields.
    Bidirectional flows use sorted tuple so A→B and B→A
    map to the same flow entry.
    """
    return tuple(sorted([
        (src_ip, src_port),
        (dst_ip, dst_port)
    ])) + (proto,)


def _update_flow(key: tuple, pkt_len: int,
                 timestamp: float) -> dict:
    """
    Update flow statistics for an ongoing session.
    Creates a new entry if this is the first packet.

    Args:
        key       : Flow key tuple
        pkt_len   : Length of current packet in bytes
        timestamp : Unix timestamp of current packet

    Returns:
        Updated flow statistics dictionary
    """
    if key not in _flow_table:
        _flow_table[key] = {
            "start_time": timestamp,
            "last_time": timestamp,
            "fwd_pkts": 0,
            "bwd_pkts": 0,
            "fwd_bytes": 0,
            "bwd_bytes": 0,
            "pkt_lens": [],
            "iat_times": [],
            "flags_syn": 0,
            "flags_fin": 0,
            "flags_rst": 0,
            "flags_psh": 0,
            "flags_ack": 0,
        }

    flow = _flow_table[key]
    iat = timestamp - flow["last_time"]
    flow["iat_times"].append(iat)
    flow["last_time"] = timestamp
    flow["fwd_pkts"] += 1
    flow["fwd_bytes"] += pkt_len
    flow["pkt_lens"].append(pkt_len)
    return flow


def extract_features(packet) -> Optional[np.ndarray]:
    """
    Extract a flat numeric feature vector from a Scapy packet.
    Returns None if packet cannot be parsed.

    This produces 20 features matching the training schema:
      0  src_port
      1  dst_port
      2  protocol
      3  flow_duration
      4  fwd_pkt_count
      5  bwd_pkt_count
      6  fwd_bytes_total
      7  bwd_bytes_total
      8  pkt_len_mean
      9  pkt_len_std
      10 pkt_len_max
      11 pkt_len_min
      12 iat_mean
      13 iat_std
      14 iat_max
      15 iat_min
      16 flag_syn
      17 flag_fin
      18 flag_rst
      19 flag_psh

    Args:
        packet: Scapy packet object

    Returns:
        numpy array of shape (20,) or None
    """
    try:
        # Scapy layer imports done here to avoid crash on Windows
        # when Scapy is installed but Npcap is missing
        from scapy.layers.inet import IP, TCP, UDP

        if not packet.haslayer(IP):
            return None

        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = ip_layer.proto
        pkt_len = len(packet)
        timestamp = time.time()

        # Port extraction
        src_port = 0
        dst_port = 0
        flags = 0

        if packet.haslayer(TCP):
            tcp = packet[TCP]
            src_port = tcp.sport
            dst_port = tcp.dport
            flags = tcp.flags

        elif packet.haslayer(UDP):
            udp = packet[UDP]
            src_port = udp.sport
            dst_port = udp.dport

        # Flow tracking
        key = _flow_key(src_ip, dst_ip, src_port, dst_port, proto)
        flow = _update_flow(key, pkt_len, timestamp)

        # Compute statistics safely
        pkt_lens = flow["pkt_lens"]
        iat_times = flow["iat_times"]

        pkt_len_mean = float(np.mean(pkt_lens)) if pkt_lens else 0.0
        pkt_len_std = float(np.std(pkt_lens)) if len(pkt_lens) > 1 else 0.0
        pkt_len_max = float(max(pkt_lens)) if pkt_lens else 0.0
        pkt_len_min = float(min(pkt_lens)) if pkt_lens else 0.0

        iat_mean = float(np.mean(iat_times)) if iat_times else 0.0
        iat_std = float(np.std(iat_times)) if len(iat_times) > 1 else 0.0
        iat_max = float(max(iat_times)) if iat_times else 0.0
        iat_min = float(min(iat_times)) if iat_times else 0.0

        flow_duration = flow["last_time"] - flow["start_time"]

        # TCP flag bits
        flag_syn = 1 if flags & 0x02 else 0
        flag_fin = 1 if flags & 0x01 else 0
        flag_rst = 1 if flags & 0x04 else 0
        flag_psh = 1 if flags & 0x08 else 0

        feature_vector = np.array([
            float(src_port),
            float(dst_port),
            float(proto),
            float(flow_duration),
            float(flow["fwd_pkts"]),
            float(flow["bwd_pkts"]),
            float(flow["fwd_bytes"]),
            float(flow["bwd_bytes"]),
            pkt_len_mean,
            pkt_len_std,
            pkt_len_max,
            pkt_len_min,
            iat_mean,
            iat_std,
            iat_max,
            iat_min,
            float(flag_syn),
            float(flag_fin),
            float(flag_rst),
            float(flag_psh),
        ], dtype=np.float32)

        return feature_vector

    except Exception as e:
        # Never crash the capture loop on a bad packet
        return None


def extract_features_from_dict(data: dict) -> np.ndarray:
    """
    Build feature vector from a pre-extracted dictionary.
    Used by POST /ingest and POST /predict endpoints
    when the caller sends features directly.

    Args:
        data: Dictionary with 'features' key containing
              a list of floats in correct order

    Returns:
        numpy array of shape (N,) as float32
    """
    features = data.get("features", [])
    return np.array(features, dtype=np.float32)


def clear_flow_table() -> None:
    """
    Clear the in-memory flow tracking table.
    Call this periodically to prevent memory bloat
    during long-running capture sessions.
    """
    global _flow_table
    _flow_table = {}
    

if __name__ == "__main__":
    # ── Standalone test without real packets ────────────────
    print("Testing extract_features_from_dict...")

    test_data = {
        "features": [
            80.0, 443.0, 6.0, 0.5,
            10.0, 5.0, 1500.0, 800.0,
            150.0, 20.0, 200.0, 100.0,
            0.05, 0.01, 0.1, 0.001,
            1.0, 0.0, 0.0, 1.0
        ]
    }

    vec = extract_features_from_dict(test_data)
    print(f"Vector shape  : {vec.shape}")
    print(f"Vector dtype  : {vec.dtype}")
    print(f"First 5 values: {vec[:5]}")
    print(f"Last 5 values : {vec[-5:]}")
    print("features.py OK")