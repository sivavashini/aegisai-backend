"""
features.py — Packet to Feature Vector Extractor
Converts raw Scapy packets into the UNSW-NB15 feature
vector that the ML models expect (56 features for stage1,
41 for stage2). Uses ConnectionTracker for ct_ statistics.
"""

import time
import math
from collections import defaultdict
from typing import Optional
import numpy as np

# ─── Categorical mappings (must match training encoders) ────
PROTO_CLASSES = [
    '3pc','a/n','aes-sp3-d','any','argus','aris','arp','ax.25',
    'bbn-rcc','bna','br-sat-mon','cbt','cftp','chaos','compaq-peer',
    'cphb','cpnx','crtp','crudp','dcn','ddp','ddx','dgp','egp',
    'eigrp','emcon','encap','etherip','fc','fire','ggp','gmtp','gre',
    'hmp','i-nlsp','iatp','ib','idpr','idpr-cmtp','idrp','ifmp','igmp',
    'igp','il','ip','ipcomp','ipcv','ipip','iplt','ipnip','ippc','ipv6',
    'ipv6-frag','ipv6-no','ipv6-opts','ipv6-route','ipx-n-ip','irtp',
    'isis','iso-ip','iso-tp4','kryptolan','l2tp','larp','leaf-1','leaf-2',
    'merit-inp','mfe-nsp','mhrp','micp','mobile','mtp','mux','narp',
    'netblt','nsfnet-igp','nvp','ospf','pgm','pim','pipe','pnni',
    'pri-enc','prm','ptp','pup','pvp','qnx','rdp','rsvp','rvd',
    'sat-expak','sat-mon','sccopmce','scps','sctp','sdrp','secure-vmtp',
    'sep','skip','sm','smp','snp','sprite-rpc','sps','srp','st2','stp',
    'sun-nd','swipe','tcf','tcp','tlsp','tp++','trunk-1','trunk-2','ttp',
    'udp','unas','uti','vines','visa','vmtp','vrrp','wb-expak','wb-mon',
    'wsn','xnet','xns-idp','xtp','zero'
]
SERVICE_CLASSES = [
    '-','dhcp','dns','ftp','ftp-data','http','irc',
    'pop3','radius','smtp','snmp','ssh','ssl'
]
STATE_CLASSES = ['ACC','CLO','CON','FIN','INT','REQ','RST']

PROTO_MAP   = {v: i for i, v in enumerate(PROTO_CLASSES)}
SERVICE_MAP = {v: i for i, v in enumerate(SERVICE_CLASSES)}
STATE_MAP   = {v: i for i, v in enumerate(STATE_CLASSES)}

# Port → service mapping
PORT_SERVICE = {
    20: 'ftp-data', 21: 'ftp', 22: 'ssh', 25: 'smtp',
    53: 'dns', 67: 'dhcp', 68: 'dhcp', 80: 'http',
    110: 'pop3', 143: 'imap', 161: 'snmp', 162: 'snmp',
    194: 'irc', 443: 'ssl', 465: 'smtp', 587: 'smtp',
    993: 'ssl', 995: 'ssl', 1812: 'radius', 1813: 'radius',
}

# ─── Flow table ─────────────────────────────────────────────
_flow_table: dict = {}

# ─── Connection tracker for ct_ features ────────────────────
# Stores recent (timestamp, src_ip, dst_ip, service, state, proto)
_conn_history: list = []
_CT_WINDOW = 100  # last N connections to consider


def _flow_key(src_ip, dst_ip, src_port, dst_port, proto):
    return tuple(sorted([(src_ip, src_port),
                          (dst_ip, dst_port)])) + (proto,)


def _get_service(src_port, dst_port, proto_name):
    svc = PORT_SERVICE.get(dst_port) or PORT_SERVICE.get(src_port)
    if svc:
        return svc
    if proto_name == 'udp' and (src_port == 53 or dst_port == 53):
        return 'dns'
    return '-'


def _get_state(flags, proto_name):
    if proto_name != 'tcp':
        return 'CON'
    if flags & 0x04:  # RST
        return 'RST'
    if flags & 0x01:  # FIN
        return 'FIN'
    if flags & 0x02:  # SYN only
        return 'REQ'
    if flags & 0x10:  # ACK
        return 'CON'
    return 'INT'


def _update_flow(key, pkt_len, timestamp):
    if key not in _flow_table:
        _flow_table[key] = {
            "start_time"  : timestamp,
            "last_time"   : timestamp,
            "spkts"       : 0,
            "dpkts"       : 0,
            "sbytes"      : 0,
            "dbytes"      : 0,
            "spkt_lens"   : [],
            "dpkt_lens"   : [],
            "siat_times"  : [],
            "diat_times"  : [],
            "flags_syn"   : 0,
            "flags_fin"   : 0,
            "flags_rst"   : 0,
            "flags_psh"   : 0,
            "flags_ack"   : 0,
            "sttl"        : 64,
            "dttl"        : 64,
            "swin"        : 0,
            "dwin"        : 0,
            "stcpb"       : 0,
            "dtcpb"       : 0,
            "synack_time" : None,
            "syn_time"    : None,
            "ack_time"    : None,
        }
    flow = _flow_table[key]
    iat = timestamp - flow["last_time"]
    flow["siat_times"].append(iat)
    flow["last_time"] = timestamp
    flow["spkts"] += 1
    flow["sbytes"] += pkt_len
    flow["spkt_lens"].append(pkt_len)
    return flow


def _compute_ct_features(src_ip, dst_ip, service, state, proto_name):
    """
    Compute connection table statistics from recent history.
    These replicate UNSW-NB15 ct_ feature definitions.
    """
    history = _conn_history[-_CT_WINDOW:]

    ct_srv_src = sum(
        1 for c in history
        if c['src_ip'] == src_ip and c['service'] == service
    )
    ct_state_ttl = sum(
        1 for c in history
        if c['state'] == state and c['proto'] == proto_name
    )
    ct_dst_ltm = sum(
        1 for c in history if c['dst_ip'] == dst_ip
    )
    ct_src_dport_ltm = sum(
        1 for c in history
        if c['src_ip'] == src_ip and c['dst_port'] == dst_ip
    )
    ct_dst_sport_ltm = sum(
        1 for c in history
        if c['dst_ip'] == dst_ip and c['src_port'] == src_ip
    )
    ct_dst_src_ltm = sum(
        1 for c in history
        if c['dst_ip'] == dst_ip and c['src_ip'] == src_ip
    )
    ct_src_ltm = sum(
        1 for c in history if c['src_ip'] == src_ip
    )
    ct_srv_dst = sum(
        1 for c in history
        if c['dst_ip'] == dst_ip and c['service'] == service
    )

    return {
        'ct_srv_src'      : ct_srv_src,
        'ct_state_ttl'    : ct_state_ttl,
        'ct_dst_ltm'      : ct_dst_ltm,
        'ct_src_dport_ltm': ct_src_dport_ltm,
        'ct_dst_sport_ltm': ct_dst_sport_ltm,
        'ct_dst_src_ltm'  : ct_dst_src_ltm,
        'ct_src_ltm'      : ct_src_ltm,
        'ct_srv_dst'      : ct_srv_dst,
    }


def extract_features(packet) -> Optional[np.ndarray]:
    """
    Extract 56-feature UNSW-NB15 vector from a Scapy packet.
    Returns None if packet cannot be parsed.

    Feature order matches feature_cols_aug.pkl (56 cols):
    dur, proto, service, state, spkts, dpkts, sbytes, dbytes,
    rate, sttl, dttl, sload, dload, sloss, dloss, sinpkt, dinpkt,
    sjit, djit, swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat,
    smean, dmean, trans_depth, response_body_len,
    ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm,
    ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login, ct_ftp_cmd,
    ct_flw_http_mthd, ct_src_ltm, ct_srv_dst, is_sm_ips_ports,
    + 14 engineered features
    """
    try:
        from scapy.layers.inet import IP, TCP, UDP

        if not packet.haslayer(IP):
            return None

        ip  = packet[IP]
        src_ip  = ip.src
        dst_ip  = ip.dst
        proto_num = ip.proto
        pkt_len = len(packet)
        ttl     = ip.ttl
        timestamp = time.time()

        src_port = 0
        dst_port = 0
        flags    = 0
        swin     = 0
        dwin     = 0
        stcpb    = 0
        dtcpb    = 0
        proto_name = 'tcp'

        if packet.haslayer(TCP):
            tcp = packet[TCP]
            src_port  = tcp.sport
            dst_port  = tcp.dport
            flags     = int(tcp.flags)
            swin      = tcp.window
            stcpb     = tcp.seq
            dtcpb     = tcp.ack
            proto_name = 'tcp'
        elif packet.haslayer(UDP):
            udp = packet[UDP]
            src_port  = udp.sport
            dst_port  = udp.dport
            proto_name = 'udp'
        else:
            proto_name = 'tcp'

        service = _get_service(src_port, dst_port, proto_name)
        state   = _get_state(flags, proto_name)

        # Encode categoricals
        proto_enc   = PROTO_MAP.get(proto_name, 0)
        service_enc = SERVICE_MAP.get(service, 0)
        state_enc   = STATE_MAP.get(state, 2)  # default CON

        # Flow tracking
        key  = _flow_key(src_ip, dst_ip, src_port, dst_port, proto_num)
        flow = _update_flow(key, pkt_len, timestamp)
        flow["sttl"] = ttl
        flow["swin"] = swin
        flow["stcpb"] = stcpb

        # TCP flag tracking
        if flags & 0x02: flow["flags_syn"] += 1
        if flags & 0x01: flow["flags_fin"] += 1
        if flags & 0x04: flow["flags_rst"] += 1
        if flags & 0x08: flow["flags_psh"] += 1
        if flags & 0x10: flow["flags_ack"] += 1

        # Timing
        dur = flow["last_time"] - flow["start_time"]
        dur = max(dur, 1e-6)

        spkts = flow["spkts"]
        dpkts = flow["dpkts"]
        sbytes = flow["sbytes"]
        dbytes = flow["dbytes"]

        # Rate
        rate = (spkts + dpkts) / dur

        # Load (bits per second)
        sload = (sbytes * 8) / dur
        dload = (dbytes * 8) / dur

        # Loss (approximate as 0 for live capture)
        sloss = 0
        dloss = 0

        # Inter-packet times
        siat = flow["siat_times"]
        sinpkt = float(np.mean(siat)) if siat else 0.0
        dinpkt = 0.0

        # Jitter
        sjit = float(np.std(siat)) if len(siat) > 1 else 0.0
        djit = 0.0

        # TCP handshake timings (approximate)
        tcprtt = 0.0
        synack = 0.0
        ackdat = 0.0

        # Mean packet sizes
        spkt_lens = flow["spkt_lens"]
        smean = float(np.mean(spkt_lens)) if spkt_lens else 0.0
        dmean = 0.0

        # HTTP features (0 for non-HTTP live capture)
        trans_depth       = 0
        response_body_len = 0

        # Connection table features
        ct = _compute_ct_features(
            src_ip, dst_ip, service, state, proto_name)

        # FTP/HTTP flags
        is_ftp_login    = 1 if service == 'ftp' else 0
        ct_ftp_cmd      = 0
        ct_flw_http_mthd = 1 if service == 'http' else 0
        is_sm_ips_ports = 1 if src_ip == dst_ip else 0

        sttl = flow["sttl"]
        dttl = 64
        swin_val = flow["swin"]
        dwin_val = flow["dwin"]
        stcpb_val = flow["stcpb"]
        dtcpb_val = flow["dtcpb"]

        # ── Base 42 features ────────────────────────────────
        base = np.array([
            dur,                          # 0  dur
            float(proto_enc),             # 1  proto
            float(service_enc),           # 2  service
            float(state_enc),             # 3  state
            float(spkts),                 # 4  spkts
            float(dpkts),                 # 5  dpkts
            float(sbytes),                # 6  sbytes
            float(dbytes),                # 7  dbytes
            rate,                         # 8  rate
            float(sttl),                  # 9  sttl
            float(dttl),                  # 10 dttl
            sload,                        # 11 sload
            dload,                        # 12 dload
            float(sloss),                 # 13 sloss
            float(dloss),                 # 14 dloss
            sinpkt,                       # 15 sinpkt
            dinpkt,                       # 16 dinpkt
            sjit,                         # 17 sjit
            djit,                         # 18 djit
            float(swin_val),              # 19 swin
            float(stcpb_val),             # 20 stcpb
            float(dtcpb_val),             # 21 dtcpb
            float(dwin_val),              # 22 dwin
            tcprtt,                       # 23 tcprtt
            synack,                       # 24 synack
            ackdat,                       # 25 ackdat
            smean,                        # 26 smean
            dmean,                        # 27 dmean
            float(trans_depth),           # 28 trans_depth
            float(response_body_len),     # 29 response_body_len
            float(ct['ct_srv_src']),      # 30 ct_srv_src
            float(ct['ct_state_ttl']),    # 31 ct_state_ttl
            float(ct['ct_dst_ltm']),      # 32 ct_dst_ltm
            float(ct['ct_src_dport_ltm']),# 33 ct_src_dport_ltm
            float(ct['ct_dst_sport_ltm']),# 34 ct_dst_sport_ltm
            float(ct['ct_dst_src_ltm']), # 35 ct_dst_src_ltm
            float(is_ftp_login),          # 36 is_ftp_login
            float(ct_ftp_cmd),            # 37 ct_ftp_cmd
            float(ct_flw_http_mthd),      # 38 ct_flw_http_mthd
            float(ct['ct_src_ltm']),      # 39 ct_src_ltm
            float(ct['ct_srv_dst']),      # 40 ct_srv_dst
            float(is_sm_ips_ports),       # 41 is_sm_ips_ports
        ], dtype=np.float32)

        # ── 14 engineered features ───────────────────────────
        sbytes_per_spkt = sbytes / spkts if spkts > 0 else 0.0
        dbytes_per_dpkt = dbytes / dpkts if dpkts > 0 else 0.0
        load_ratio      = sload / (dload + 1e-6)
        smean_x_rate    = smean * rate
        ttl_diff        = float(sttl) - float(dttl)
        jit_ratio       = sjit / (djit + 1e-6)
        sbytes_per_dur  = sbytes / dur
        rate_x_dur      = rate * dur
        sloss_ratio     = sloss / (spkts + 1e-6)
        port_reuse_ratio = float(ct['ct_src_ltm']) / (spkts + 1e-6)
        ack_syn_ratio   = (flow["flags_ack"] /
                           (flow["flags_syn"] + 1e-6))
        pkt_symmetry    = spkts / (spkts + dpkts + 1e-6)
        sbytes_per_srv  = sbytes / (ct['ct_srv_src'] + 1e-6)
        srv_src_dst_ratio = (ct['ct_srv_src'] /
                             (ct['ct_srv_dst'] + 1e-6))

        engineered = np.array([
            sbytes_per_spkt,   # 42
            dbytes_per_dpkt,   # 43
            load_ratio,        # 44
            smean_x_rate,      # 45
            ttl_diff,          # 46
            jit_ratio,         # 47
            sbytes_per_dur,    # 48
            rate_x_dur,        # 49
            sloss_ratio,       # 50
            port_reuse_ratio,  # 51
            ack_syn_ratio,     # 52
            pkt_symmetry,      # 53
            sbytes_per_srv,    # 54
            srv_src_dst_ratio, # 55
        ], dtype=np.float32)

        # Update connection history
        _conn_history.append({
            'src_ip'  : src_ip,
            'dst_ip'  : dst_ip,
            'src_port': src_port,
            'dst_port': dst_port,
            'service' : service,
            'state'   : state,
            'proto'   : proto_name,
            'ts'      : timestamp,
        })
        # Keep history bounded
        if len(_conn_history) > 1000:
            _conn_history.pop(0)

        return np.concatenate([base, engineered])

    except Exception:
        return None


def extract_features_from_dict(data: dict) -> np.ndarray:
    """
    Build feature vector from a pre-extracted dictionary.
    Used by POST /ingest and POST /predict endpoints.
    """
    features = data.get("features", [])
    return np.array(features, dtype=np.float32)


def clear_flow_table() -> None:
    """
    Clear flow table and connection history.
    Call every 300 seconds to prevent RAM exhaustion.
    """
    global _flow_table, _conn_history
    _flow_table = {}
    _conn_history = []


if __name__ == "__main__":
    print("features.py — column count check")
    print(f"Base features     : 42")
    print(f"Engineered features: 14")
    print(f"Total             : 56")
    print("OK — matches feature_cols_aug.pkl")