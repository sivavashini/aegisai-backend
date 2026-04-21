"""
mitre_map.py — MITRE ATT&CK Mapping Dictionary
Maps attack labels from ML models to MITRE ATT&CK
technique IDs, tactic names, and descriptions.
"""

MITRE_MAP = {
    "DoS": {
        "id": "T1499",
        "tactic": "Impact",
        "name": "Endpoint Denial of Service",
        "description": "Adversary disrupts availability by exhausting resources."
    },
    "DDoS": {
        "id": "T1498",
        "tactic": "Impact",
        "name": "Network Denial of Service",
        "description": "Adversary floods network to degrade or block availability."
    },
    "PortScan": {
        "id": "T1046",
        "tactic": "Discovery",
        "name": "Network Service Discovery",
        "description": "Adversary scans to enumerate open ports and services."
    },
    "Brute Force": {
        "id": "T1110",
        "tactic": "Credential Access",
        "name": "Brute Force",
        "description": "Adversary attempts many passwords to gain access."
    },
    "BruteForce": {
        "id": "T1110",
        "tactic": "Credential Access",
        "name": "Brute Force",
        "description": "Adversary attempts many passwords to gain access."
    },
    "Infiltration": {
        "id": "T1078",
        "tactic": "Defense Evasion",
        "name": "Valid Accounts",
        "description": "Adversary uses legitimate credentials to evade detection."
    },
    "Web Attack": {
        "id": "T1190",
        "tactic": "Initial Access",
        "name": "Exploit Public-Facing Application",
        "description": "Adversary exploits weakness in internet-facing application."
    },
    "WebAttack": {
        "id": "T1190",
        "tactic": "Initial Access",
        "name": "Exploit Public-Facing Application",
        "description": "Adversary exploits weakness in internet-facing application."
    },
    "SQL Injection": {
        "id": "T1190",
        "tactic": "Initial Access",
        "name": "Exploit Public-Facing Application",
        "description": "Adversary injects malicious SQL to manipulate database."
    },
    "XSS": {
        "id": "T1059.007",
        "tactic": "Execution",
        "name": "JavaScript Execution",
        "description": "Adversary executes malicious scripts in victim browser."
    },
    "FTP-Patator": {
        "id": "T1110.001",
        "tactic": "Credential Access",
        "name": "Password Guessing",
        "description": "Adversary brute forces FTP credentials."
    },
    "SSH-Patator": {
        "id": "T1110.001",
        "tactic": "Credential Access",
        "name": "Password Guessing",
        "description": "Adversary brute forces SSH credentials."
    },
    "Botnet": {
        "id": "T1583.005",
        "tactic": "Resource Development",
        "name": "Botnet",
        "description": "Adversary leverages network of compromised systems."
    },
    "Heartbleed": {
        "id": "T1203",
        "tactic": "Execution",
        "name": "Exploitation for Client Execution",
        "description": "Adversary exploits OpenSSL Heartbleed vulnerability."
    },
    "BENIGN": {
        "id": "NONE",
        "tactic": "None",
        "name": "No Threat Detected",
        "description": "Traffic classified as normal benign activity."
    },
    "Normal": {
        "id": "NONE",
        "tactic": "None",
        "name": "No Threat Detected",
        "description": "Traffic classified as normal benign activity."
    },
    "UNKNOWN": {
        "id": "T1205",
        "tactic": "Defense Evasion",
        "name": "Traffic Signaling",
        "description": "Unrecognized traffic pattern flagged for review."
    }
}


def get_mitre(label: str) -> dict:
    """
    Look up MITRE ATT&CK entry for a given attack label.
    Returns UNKNOWN entry if label not found in map.

    Args:
        label: Attack label string from ML model output

    Returns:
        Dictionary with id, tactic, name, description
    """
    return MITRE_MAP.get(label, MITRE_MAP["UNKNOWN"])