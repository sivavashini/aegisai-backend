"""
veritascore.py — VeritasCore Layer 3 Decision Logic
Standalone module for VeritasCore final verdict.
Wraps the predictor._run_veritascore logic with
additional context and logging for main.py usage.
"""

import logging
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def get_final_verdict(
    feature_dict: dict,
    label       : str,
    score       : float,
    phantomnet  : dict
) -> dict:
    """
    Run VeritasCore Layer 3 and return final verdict.
    Called by main.py when PhantomNet routes to
    VeritasCore for final decision.

    Decision logic:
      Both SVM and GNB agree attack + confidence > 0.60
        → ISOLATE
      Both agree Normal
        → ALLOW
      Any disagreement
        → SUSPICIOUS — log only

    Args:
        feature_dict: Encoded feature dictionary
        label       : Attack label from Stage 2
        score       : Threat score from AWT
        phantomnet  : PhantomNet result dictionary

    Returns:
        Complete verdict dictionary with decision,
        confidence, svm_pred, gnb_pred, reasoning
    """
    try:
        from predictor import _run_veritascore

        vt_result = _run_veritascore(
            feature_dict, label)

        decision   = vt_result.get(
            "veritascore_decision", "SUSPICIOUS")
        confidence = vt_result.get("confidence", 0.0)
        svm_pred   = vt_result.get(
            "svm_pred", "UNKNOWN")
        gnb_pred   = vt_result.get(
            "gnb_pred", "UNKNOWN")

        # Build reasoning string for dashboard
        if decision == "ISOLATE":
            reasoning = (
                f"SVM={svm_pred} GNB={gnb_pred} "
                f"both agree attack with "
                f"confidence={confidence:.2f}"
            )
        elif decision == "ALLOW":
            reasoning = (
                f"SVM={svm_pred} GNB={gnb_pred} "
                f"both agree Normal traffic"
            )
        else:
            reasoning = (
                f"SVM={svm_pred} GNB={gnb_pred} "
                f"disagreement — flagged suspicious"
            )

        verdict = {
            "decision"   : decision,
            "confidence" : confidence,
            "svm_pred"   : svm_pred,
            "gnb_pred"   : gnb_pred,
            "reasoning"  : reasoning,
            "label"      : label,
            "score"      : score,
            "phantomnet" : phantomnet,
            "timestamp"  : datetime.now(
                timezone.utc).isoformat(),
            "layer"      : "VeritasCore"
        }

        logger.info(
            f"[VERITASCORE] Verdict — "
            f"decision={decision} "
            f"svm={svm_pred} gnb={gnb_pred} "
            f"confidence={confidence:.2f}"
        )

        return verdict

    except Exception as e:
        logger.error(
            f"[VERITASCORE] Error: {e}")
        return {
            "decision"  : "SUSPICIOUS",
            "confidence": 0.0,
            "svm_pred"  : "UNKNOWN",
            "gnb_pred"  : "UNKNOWN",
            "reasoning" : f"Error: {e}",
            "label"     : label,
            "score"     : score,
            "phantomnet": phantomnet,
            "timestamp" : datetime.now(
                timezone.utc).isoformat(),
            "layer"     : "VeritasCore"
        }


def explain_verdict(verdict: dict) -> str:
    """
    Generate human-readable explanation of verdict.
    Used for dashboard display and judge demo.

    Args:
        verdict: Verdict dictionary from get_final_verdict

    Returns:
        Human readable explanation string
    """
    decision   = verdict.get("decision",   "UNKNOWN")
    confidence = verdict.get("confidence", 0.0)
    svm_pred   = verdict.get("svm_pred",   "UNKNOWN")
    gnb_pred   = verdict.get("gnb_pred",   "UNKNOWN")
    label      = verdict.get("label",      "UNKNOWN")
    score      = verdict.get("score",      0.0)

    lines = [
        f"VeritasCore Analysis",
        f"{'─' * 30}",
        f"Threat Label     : {label}",
        f"Risk Score       : {score:.2f}/100",
        f"SVM Prediction   : {svm_pred}",
        f"GNB Prediction   : {gnb_pred}",
        f"Confidence       : {confidence:.2%}",
        f"Final Decision   : {decision}",
        f"{'─' * 30}",
        f"Reasoning: {verdict.get('reasoning', '')}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    print("Testing VeritasCore...")

    # Load models first
    print("Loading models...")
    from predictor import load_all_models
    load_all_models()
    print()

    # Test feature dict
    test_features = {
        "dur": 0.5, "proto": "tcp",
        "service": "http", "state": "FIN",
        "spkts": 10, "dpkts": 8,
        "sbytes": 1500, "dbytes": 800,
        "rate": 36.0, "sttl": 64, "dttl": 64,
        "sload": 24000.0, "dload": 12800.0,
        "sloss": 0, "dloss": 0,
        "sinpkt": 0.05, "dinpkt": 0.06,
        "sjit": 0.001, "djit": 0.001,
        "swin": 255, "stcpb": 0, "dtcpb": 0,
        "dwin": 255, "tcprtt": 0.01,
        "synack": 0.005, "ackdat": 0.005,
        "smean": 150, "dmean": 100,
        "trans_depth": 1, "response_body_len": 500,
        "ct_srv_src": 5, "ct_state_ttl": 2,
        "ct_dst_ltm": 3, "ct_src_dport_ltm": 2,
        "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 4,
        "is_ftp_login": 0, "ct_ftp_cmd": 0,
        "ct_flw_http_mthd": 0, "ct_src_ltm": 3,
        "ct_srv_dst": 4, "is_sm_ips_ports": 0,
    }

    # Test 1 — get verdict
    print("Test 1 — get_final_verdict()")
    verdict = get_final_verdict(
        feature_dict = test_features,
        label        = "Infiltration",
        score        = 61.7,
        phantomnet   = {
            "ocsvm_anomaly"    : True,
            "lstm_anomaly"     : False,
            "phantomnet_result": "VERITASCORE"
        }
    )
    print(f"Decision   : {verdict['decision']}")
    print(f"Confidence : {verdict['confidence']:.2f}")
    print(f"SVM pred   : {verdict['svm_pred']}")
    print(f"GNB pred   : {verdict['gnb_pred']}")
    print()

    # Test 2 — explain verdict
    print("Test 2 — explain_verdict()")
    explanation = explain_verdict(verdict)
    print(explanation)

    print("\nveritascore.py OK")