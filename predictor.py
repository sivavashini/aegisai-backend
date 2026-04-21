"""
predictor.py — AegisAI Trinity Inference Engine
Loads all ML models and runs the complete three-layer
inference pipeline on a feature vector.

Pipeline:
  Raw features → categorical encode → scaler → interaction
  features → Stage1 (binary) → Stage2 (multiclass) →
  AWT score → PhantomNet → VeritasCore → final decision
"""

import os
import json
import joblib
import numpy as np
import onnxruntime as ort
from typing import Optional
import config
from mitre_map import get_mitre


# ─── Model file paths ───────────────────────────────────────
M = config.MODELS_DIR

PATHS = {
    "cat_encoders"   : os.path.join(M, "categorical_encoders.pkl"),
    "scaler"         : os.path.join(M, "scaler.pkl"),
    "feature_cols1"  : os.path.join(M, "feature_cols_aug.pkl"),
    "feature_cols2"  : os.path.join(M, "feature_cols_stage2.pkl"),
    "stage1"         : os.path.join(M, "sentinelx_lgbm_calibrated_stage1.pkl"),
    "stage2"         : os.path.join(M, "sentinelx_stage2.pkl"),
    "if_model"       : os.path.join(M, "sentinelx_if.pkl"),
    "awt"            : os.path.join(M, "sentinelx_awt.pkl"),
    "norm_json"      : os.path.join(M, "sentinelx_normalization.json"),
    "label_enc"      : os.path.join(M, "label_encoder_merged.pkl"),
    "train_medians"  : os.path.join(M, "train_medians.pkl"),
    "ocsvm"          : os.path.join(M, "phantomnet_ocsvm.pkl"),
    "ocsvm_scaler"   : os.path.join(M, "phantomnet_ocsvm_scaler.pkl"),
    "lstm_onnx"      : os.path.join(M, "phantomnet_lstm.onnx"),
    "lstm_norm"      : os.path.join(M, "lstm_norm_params.json"),
    "lstm_threshold" : os.path.join(M, "lstm_threshold.json"),
    "phantom_feats"  : os.path.join(M, "phantomnet_feature_names.pkl"),
    "vt_svm"         : os.path.join(M, "veritascore_svm.pkl"),
    "vt_gnb"         : os.path.join(M, "veritascore_gnb.pkl"),
    "vt_config"      : os.path.join(M, "veritascore_config.json"),
    "mitre_merged"   : os.path.join(M, "mitre_map_merged.json"),
}


# ─── Global model registry ──────────────────────────────────
_models: dict = {}
_loaded: bool = False

# ─── LSTM rolling window buffer ─────────────────────────────
# Stores last 30 behavioral samples for LSTM sequence input
# Each sample is a list of 5 floats
_lstm_buffer: list = []
_LSTM_WINDOW  = 30
_LSTM_FEATURES = 5


def collect_behavioral_sample() -> list:
    """
    Collect one behavioral sample from the host system.
    Returns list of 5 floats matching LSTM feature order:
      [syscall_rate_per_sec, file_access_per_min,
       cpu_percent, memory_mb, network_conns_per_min]

    On Windows returns psutil values where available.
    syscall_rate uses context switches as proxy.
    """
    try:
        import psutil
        proc   = psutil.Process()
        ctx    = proc.num_ctx_switches()
        # voluntary + involuntary context switches as syscall proxy
        syscall_proxy    = float(ctx.voluntary +
                                 ctx.involuntary)
        file_access      = float(len(proc.open_files()))
        cpu_pct          = float(psutil.cpu_percent())
        memory_mb        = float(
            psutil.virtual_memory().used / 1024 / 1024)
        net_conns        = float(
            len(psutil.net_connections()))
        return [syscall_proxy, file_access,
                cpu_pct, memory_mb, net_conns]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0, 0.0]


def update_lstm_buffer(sample: list = None) -> None:
    """
    Add one behavioral sample to the rolling window buffer.
    Maintains a maximum of 30 samples (FIFO).
    If sample is None, collects from system automatically.

    Args:
        sample: Optional list of 5 floats. If None,
                collect_behavioral_sample() is called.
    """
    global _lstm_buffer
    if sample is None:
        sample = collect_behavioral_sample()
    _lstm_buffer.append(sample)
    if len(_lstm_buffer) > _LSTM_WINDOW:
        _lstm_buffer.pop(0)


def get_lstm_sequence() -> np.ndarray:
    """
    Build the (1, 30, 5) input tensor for LSTM inference.
    Pads with zeros if buffer has fewer than 30 samples.

    Returns:
        numpy array of shape (1, 30, 5) as float32
    """
    global _lstm_buffer
    buf = list(_lstm_buffer)
    # Pad with zeros if not enough samples yet
    while len(buf) < _LSTM_WINDOW:
        buf.insert(0, [0.0] * _LSTM_FEATURES)
    arr = np.array(buf[-_LSTM_WINDOW:],
                   dtype=np.float32)
    return arr.reshape(1, _LSTM_WINDOW, _LSTM_FEATURES)


def load_all_models() -> dict:
    """
    Load all ML models into memory at startup.
    Called once when FastAPI starts.
    Returns status dict showing which models loaded.

    Returns:
        Dictionary with model names as keys and
        True/False as load status values
    """
    global _models, _loaded
    status = {}

    # sklearn and joblib models
    joblib_keys = [
        "cat_encoders", "scaler", "feature_cols1",
        "feature_cols2", "stage1", "stage2",
        "if_model", "awt", "label_enc",
        "train_medians", "ocsvm", "ocsvm_scaler",
        "phantom_feats", "vt_svm", "vt_gnb",
    ]

    for key in joblib_keys:
        try:
            _models[key] = joblib.load(PATHS[key])
            status[key] = True
        except Exception as e:
            print(f"[PREDICTOR] WARNING — could not load {key}: {e}")
            _models[key] = None
            status[key] = False

    # JSON config files
    json_keys = [
        "norm_json", "lstm_norm",
        "lstm_threshold", "vt_config", "mitre_merged"
    ]

    for key in json_keys:
        try:
            with open(PATHS[key], "r") as f:
                _models[key] = json.load(f)
            status[key] = True
        except Exception as e:
            print(f"[PREDICTOR] WARNING — could not load {key}: {e}")
            _models[key] = None
            status[key] = False

    # ONNX runtime session for LSTM
    try:
        _models["lstm_session"] = ort.InferenceSession(
            PATHS["lstm_onnx"],
            providers=["CPUExecutionProvider"]
        )
        status["lstm_onnx"] = True
    except Exception as e:
        print(f"[PREDICTOR] WARNING — could not load LSTM ONNX: {e}")
        _models["lstm_session"] = None
        status["lstm_onnx"] = False

    _loaded = True
    loaded_count = sum(1 for v in status.values() if v)
    total_count = len(status)
    print(f"[PREDICTOR] Models loaded: {loaded_count}/{total_count}")
    return status


def _encode_categoricals(feature_dict: dict) -> dict:
    """
    Encode proto, service, state fields using fitted
    LabelEncoders. Unknown values map to 0.

    Args:
        feature_dict: Raw feature dictionary with string
                      values for proto/service/state

    Returns:
        Feature dictionary with encoded integer values
    """
    cat_encoders = _models.get("cat_encoders")
    if cat_encoders is None:
        return feature_dict

    result = dict(feature_dict)
    for col in ["proto", "service", "state"]:
        if col in result:
            val = str(result[col])
            known = set(cat_encoders[col].classes_)
            if val in known:
                result[col] = int(
                    cat_encoders[col].transform([val])[0]
                )
            else:
                result[col] = 0
    return result


def _build_stage1_vector(feature_dict: dict) -> np.ndarray:
    """
    Build the 56-feature vector for Stage 1 inference.

    Pipeline:
      42 raw features → scaler.pkl → 42 scaled →
      add_interaction_features() → 56 features

    Args:
        feature_dict: Encoded feature dictionary

    Returns:
        numpy array of shape (1, 56) as float32
    """
    scaler        = _models.get("scaler")
    train_medians = _models.get("train_medians", {})

    # These are the exact 42 base columns scaler expects
    # feature_cols_aug has 56 — first 42 are base columns
    # last 14 are interaction columns added after scaling
    interaction_cols = set(_get_interaction_cols())
    feature_cols1    = _models.get("feature_cols1", [])
    base_cols = [c for c in feature_cols1
                 if c not in interaction_cols]

    row = []
    for col in base_cols:
        val = feature_dict.get(col, 0.0)
        if val is None:
            val = 0.0
        if isinstance(val, float) and np.isnan(val):
            val = train_medians.get(col, 0.0) \
                if isinstance(train_medians, dict) else 0.0
        row.append(float(val))

    arr = np.array(row, dtype=np.float32).reshape(1, -1)

    # Apply StandardScaler on 42 base features
    if scaler is not None:
        try:
            import pandas as pd
            df_input = pd.DataFrame(arr, columns=base_cols)
            arr = scaler.transform(df_input)
        except Exception as e:
            print(f"[PREDICTOR] Scaler error: {e}")

    # Append 14 interaction features → 56 total
    arr = _add_interaction_features(arr, base_cols)
    return arr.astype(np.float32)


def _get_interaction_cols() -> list:
    """
    Return list of the 14 interaction feature column names.
    These are derived features added after scaling.
    42 scaled features + 14 interaction = 56 total.

    Returns:
        List of 14 interaction column name strings
    """
    return [
        "sbytes_per_spkt",
        "dbytes_per_dpkt",
        "load_ratio",
        "smean_x_rate",
        "ttl_diff",
        "jit_ratio",
        "sbytes_per_dur",
        "rate_x_dur",
        "sloss_ratio",
        "port_reuse_ratio",
        "ack_syn_ratio",
        "pkt_symmetry",
        "sbytes_per_srv",
        "srv_src_dst_ratio",
    ]


def _add_interaction_features(arr: np.ndarray,
                               base_cols: list) -> np.ndarray:
    """
    Compute the exact 14 interaction features used during
    training. Appends them to the 42 scaled features
    to produce the 56-feature vector Stage 1 expects.

    Args:
        arr       : Scaled base feature array shape (1, 42)
        base_cols : List of 42 column names matching arr

    Returns:
        Array of shape (1, 56) with interactions appended
    """
    def _get(col):
        try:
            idx = base_cols.index(col)
            return float(arr[0, idx])
        except (ValueError, IndexError):
            return 0.0

    sbytes          = _get("sbytes")
    dbytes          = _get("dbytes")
    spkts           = _get("spkts")
    dpkts           = _get("dpkts")
    sload           = _get("sload")
    dload           = _get("dload")
    smean           = _get("smean")
    rate            = _get("rate")
    sttl            = _get("sttl")
    dttl            = _get("dttl")
    sjit            = _get("sjit")
    djit            = _get("djit")
    dur             = _get("dur")
    sloss           = _get("sloss")
    ct_dst_src_ltm  = _get("ct_dst_src_ltm")
    ct_src_dport_ltm= _get("ct_src_dport_ltm")
    ackdat          = _get("ackdat")
    synack          = _get("synack")
    ct_srv_src      = _get("ct_srv_src")
    ct_srv_dst      = _get("ct_srv_dst")

    interactions = [
        sbytes / (spkts + 1),                        # sbytes_per_spkt
        dbytes / (dpkts + 1),                        # dbytes_per_dpkt
        sload  / (dload + 1e-6),                     # load_ratio
        smean  * rate,                               # smean_x_rate
        sttl   - dttl,                               # ttl_diff
        sjit   / (djit + 1e-6),                      # jit_ratio
        sbytes / (dur + 1e-6),                       # sbytes_per_dur
        rate   * (dur + 1e-6),                       # rate_x_dur
        sloss  / (spkts + 1),                        # sloss_ratio
        ct_dst_src_ltm / (ct_src_dport_ltm + 1),    # port_reuse_ratio
        ackdat / (synack + 1e-6),                    # ack_syn_ratio
        spkts  / (dpkts + 1),                        # pkt_symmetry
        sbytes / (ct_srv_src + 1),                   # sbytes_per_srv
        ct_srv_src / (ct_srv_dst + 1),               # srv_src_dst_ratio
    ]

    inter_arr = np.array(interactions,
                         dtype=np.float32).reshape(1, -1)
    return np.hstack([arr, inter_arr])


def _build_stage2_vector(feature_dict: dict) -> np.ndarray:
    """
    Build the 41-feature vector for Stage 2 inference.
    Uses sentinelx_normalization.json instead of scaler.pkl.

    Args:
        feature_dict: Encoded feature dictionary

    Returns:
        numpy array of shape (1, 41) as float32
    """
    feature_cols2 = _models.get("feature_cols2", [])
    norm = _models.get("norm_json")

    row = []
    for col in feature_cols2:
        val = feature_dict.get(col, 0.0)
        if val is None:
            val = 0.0
        row.append(float(val))

    arr = np.array(row, dtype=np.float32)

    # Apply global normalization from JSON
    if norm is not None:
        try:
            mean = np.array(norm["global_mean"],
                            dtype=np.float32)
            std  = np.array(norm["global_std"],
                            dtype=np.float32)
            std  = np.where(std < 1e-8, 1.0, std)
            arr  = (arr - mean) / std
        except Exception:
            pass

    return arr.reshape(1, -1).astype(np.float32)


def _awt_score(rf_attack_prob: float,
               if_score: float) -> float:
    """
    Compute Adaptive Weighted Threat score 0-100.
    Combines LightGBM attack probability with
    Isolation Forest anomaly score.

    Args:
        rf_attack_prob : P(attack) from Stage 1 LightGBM
        if_score       : Raw Isolation Forest score

    Returns:
        Float threat score between 0 and 100
    """
    awt_params = _models.get("awt")

    if awt_params is not None:
        try:
            rf_weight = float(awt_params.get(
                "rf_weight", config.RF_WEIGHT))
            if_weight = float(awt_params.get(
                "if_weight", config.IF_WEIGHT))
        except Exception:
            rf_weight = config.RF_WEIGHT
            if_weight = config.IF_WEIGHT
    else:
        rf_weight = config.RF_WEIGHT
        if_weight = config.IF_WEIGHT

    # Sigmoid normalize IF score
    if_normalized = 1.0 / (1.0 + np.exp(-if_score))

    score = (rf_attack_prob * rf_weight +
             float(if_normalized) * if_weight) * 100.0
    return float(np.clip(score, 0.0, 100.0))


def _run_phantomnet(feature_dict: dict) -> dict:
    """
    Run PhantomNet Layer 2 inference.
    Uses OC-SVM for behavioral profiling and
    LSTM autoencoder for sequence anomaly detection.

    Args:
        feature_dict: Encoded feature dictionary

    Returns:
        Dictionary with ocsvm_anomaly, lstm_anomaly,
        and phantomnet_result keys
    """
    result = {
        "ocsvm_anomaly" : False,
        "lstm_anomaly"  : False,
        "phantomnet_result": "NORMAL"
    }

    # OC-SVM inference
    try:
        ocsvm        = _models.get("ocsvm")
        ocsvm_scaler = _models.get("ocsvm_scaler")
        phantom_feats= _models.get("phantom_feats", [])

        if ocsvm is not None and ocsvm_scaler is not None:
            row = [float(feature_dict.get(f, 0.0))
                   for f in phantom_feats]
            arr = np.array(row,
                           dtype=np.float32).reshape(1, -1)
            arr_scaled = ocsvm_scaler.transform(arr)
            pred = ocsvm.predict(arr_scaled)
            # OC-SVM returns -1 for anomaly, 1 for normal
            result["ocsvm_anomaly"] = bool(pred[0] == -1)
    except Exception as e:
        print(f"[PREDICTOR] PhantomNet OC-SVM error: {e}")

    # LSTM autoencoder inference using rolling window buffer
    try:
        session  = _models.get("lstm_session")
        lstm_thr = _models.get("lstm_threshold")

        if session is not None and lstm_thr is not None:
            # Collect current behavioral sample into buffer
            update_lstm_buffer()

            # Get (1, 30, 5) sequence
            seq        = get_lstm_sequence()
            input_name = session.get_inputs()[0].name
            output     = session.run(None,
                                     {input_name: seq})
            reconstruction = output[0]

            # Mean squared reconstruction error
            mse       = float(np.mean(
                (seq.flatten() -
                 reconstruction.flatten()) ** 2))
            threshold = float(lstm_thr.get(
                "threshold", 0.004470))
            result["lstm_anomaly"] = mse > threshold

    except Exception as e:
        print(f"[PREDICTOR] PhantomNet LSTM error: {e}")

    # Combine results
    ocsvm_a = result["ocsvm_anomaly"]
    lstm_a  = result["lstm_anomaly"]

    if ocsvm_a and lstm_a:
        result["phantomnet_result"] = "ISOLATE"
    elif ocsvm_a or lstm_a:
        result["phantomnet_result"] = "VERITASCORE"
    else:
        result["phantomnet_result"] = "NORMAL"

    return result


def _run_veritascore(feature_dict: dict,
                     label: str) -> dict:
    """
    Run VeritasCore Layer 3 final verdict.
    Uses SVM RBF and Gaussian Naive Bayes ensemble.

    Args:
        feature_dict: Encoded feature dictionary
        label       : Attack label from Stage 2

    Returns:
        Dictionary with svm_pred, gnb_pred,
        confidence, and veritascore_decision keys
    """
    result = {
        "svm_pred"           : "UNKNOWN",
        "gnb_pred"           : "UNKNOWN",
        "confidence"         : 0.0,
        "veritascore_decision": "SUSPICIOUS"
    }

    try:
        vt_svm    = _models.get("vt_svm")
        vt_gnb    = _models.get("vt_gnb")
        vt_config = _models.get("vt_config", {})

        # VeritasCore uses 42 scaled features
        # Same as scaler.pkl output — includes ct_flw_http_mthd
        # Do NOT drop ct_flw_http_mthd here
        VT_COLS = [
            "dur", "proto", "service", "state", "spkts",
            "dpkts", "sbytes", "dbytes", "rate", "sttl",
            "dttl", "sload", "dload", "sloss", "dloss",
            "sinpkt", "dinpkt", "sjit", "djit", "swin",
            "stcpb", "dtcpb", "dwin", "tcprtt", "synack",
            "ackdat", "smean", "dmean", "trans_depth",
            "response_body_len", "ct_srv_src", "ct_state_ttl",
            "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
            "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
            "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
            "is_sm_ips_ports"
        ]
        # Encode categoricals before building array
        encoded_dict = _encode_categoricals(
            feature_dict)
        row = [float(encoded_dict.get(f, 0.0))
               for f in VT_COLS]
        arr = np.array(row,
                       dtype=np.float32).reshape(1, -1)

        # Apply scaler to get normalized 42 features
        scaler = _models.get("scaler")
        if scaler is not None:
            try:
                import pandas as pd
                df_input = pd.DataFrame(arr, columns=VT_COLS)
                arr = scaler.transform(df_input).astype(np.float32)
            except Exception as e:
                print(f"[PREDICTOR] VeritasCore scaler error: {e}")

        conf_threshold = float(vt_config.get(
            "confidence_threshold",
            config.PHANTOMNET_CONFIDENCE_THRESHOLD))

        svm_pred = "UNKNOWN"
        gnb_pred = "UNKNOWN"
        svm_conf = 0.0
        gnb_conf = 0.0

        if vt_svm is not None:
            svm_p = vt_svm.predict(arr)[0]
            svm_pred = str(svm_p)
            if hasattr(vt_svm, "predict_proba"):
                proba    = vt_svm.predict_proba(arr)[0]
                svm_conf = float(np.max(proba))

        if vt_gnb is not None:
            gnb_p = vt_gnb.predict(arr)[0]
            gnb_pred = str(gnb_p)
            if hasattr(vt_gnb, "predict_proba"):
                proba    = vt_gnb.predict_proba(arr)[0]
                gnb_conf = float(np.max(proba))

        avg_conf = (svm_conf + gnb_conf) / 2.0
        result["svm_pred"]   = svm_pred
        result["gnb_pred"]   = gnb_pred
        result["confidence"] = avg_conf

        # Decision logic per spec
        if (svm_pred == gnb_pred and
                svm_pred != "Normal" and
                avg_conf > conf_threshold):
            result["veritascore_decision"] = "ISOLATE"
        elif svm_pred == gnb_pred == "Normal":
            result["veritascore_decision"] = "ALLOW"
        else:
            result["veritascore_decision"] = "SUSPICIOUS"

    except Exception as e:
        print(f"[PREDICTOR] VeritasCore error: {e}")

    return result


def predict(feature_dict: dict) -> dict:
    """
    Run the complete AegisAI Trinity inference pipeline.
    Entry point called by main.py for every event.

    Args:
        feature_dict: Dictionary mapping feature names
                      to their values. Must include all
                      columns from feature_cols_aug.pkl

    Returns:
        Complete result dictionary with keys:
          label, score, decision, mitre,
          stage1_pred, stage2_pred,
          phantomnet, veritascore,
          confidence, layer
    """
    if not _loaded:
        load_all_models()

    # Default result in case of total failure
    result = {
        "label"      : "UNKNOWN",
        "score"      : 0.0,
        "decision"   : "ALLOW",
        "mitre"      : get_mitre("UNKNOWN"),
        "stage1_pred": 0,
        "stage2_pred": "Normal",
        "phantomnet" : {},
        "veritascore": {},
        "confidence" : 0.0,
        "layer"      : "SentinelX"
    }

    try:
        # Step 1 — encode categoricals
        encoded = _encode_categoricals(feature_dict)

        # Step 2 — build Stage 1 vector and predict
        x1 = _build_stage1_vector(encoded)
        stage1_model = _models.get("stage1")
        if_model     = _models.get("if_model")

        stage1_pred  = 0
        attack_prob  = 0.0
        if_score     = 0.0

        if stage1_model is not None:
            stage1_pred = int(
                stage1_model.predict(x1)[0])
            proba = stage1_model.predict_proba(x1)[0]
            # Index 1 = Attack probability
            attack_prob = float(proba[1]) \
                if len(proba) > 1 else float(proba[0])

        if if_model is not None:
            # IF was trained on 42 base features only
            # x1 has 56 features — slice first 42
            x1_base = x1[:, :42]
            if_score = float(
                if_model.score_samples(x1_base)[0])

        result["stage1_pred"] = stage1_pred

        # Step 3 — AWT threat score
        score = _awt_score(attack_prob, if_score)
        result["score"] = score

        # Step 4 — if Normal, stop here
        if stage1_pred == 0:
            result["label"]    = "Normal"
            result["decision"] = "ALLOW"
            result["mitre"]    = get_mitre("Normal")
            return result

        # Step 5 — Stage 2 multiclass classification
        x2 = _build_stage2_vector(encoded)
        stage2_model  = _models.get("stage2")
        label_encoder = _models.get("label_enc")
        stage2_label  = "UNKNOWN"

        if stage2_model is not None:
            stage2_pred_int = int(
                stage2_model.predict(x2)[0])
            if label_encoder is not None:
                stage2_label = str(
                    label_encoder.classes_[
                        stage2_pred_int])
            else:
                stage2_label = str(stage2_pred_int)

        result["stage2_pred"] = stage2_label
        result["label"]       = stage2_label
        result["mitre"]       = get_mitre(stage2_label)

        # Step 6 — threshold decision
        # Only apply thresholds for confirmed attacks
        if label == "Normal":
            result["decision"] = "ALLOW"
            return result
        if score >= config.THREAT_BLOCK_THRESHOLD:
            result["decision"] = "BLOCK"
            result["layer"]    = "SentinelX"
            return result

        if score >= config.THREAT_SANDBOX_THRESHOLD:
            # Step 7 — PhantomNet sandbox layer
            phantom = _run_phantomnet(encoded)
            result["phantomnet"] = phantom
            result["layer"]      = "PhantomNet"

            pn_result = phantom.get(
                "phantomnet_result", "NORMAL")

            if pn_result == "ISOLATE":
                result["decision"] = "ISOLATE"
                return result

            if pn_result == "VERITASCORE":
                # Step 8 — VeritasCore final verdict
                vt = _run_veritascore(encoded,
                                      stage2_label)
                result["veritascore"] = vt
                result["confidence"]  = vt["confidence"]
                result["layer"]       = "VeritasCore"
                result["decision"]    = vt[
                    "veritascore_decision"]
                return result

            # Both PhantomNet models say normal
            result["decision"] = "ALLOW"
            return result

        # Score below sandbox threshold
        result["decision"] = "ALLOW"
        return result

    except Exception as e:
        print(f"[PREDICTOR] Pipeline error: {e}")
        result["decision"] = "ALLOW"
        return result


if __name__ == "__main__":
    print("Loading models...")
    status = load_all_models()
    print("\nModel load status:")
    for k, v in status.items():
        print(f"  {'OK' if v else 'FAIL':4} — {k}")

    print("\nRunning test prediction...")
    test_input = {
        "dur": 0.5, "proto": "tcp", "service": "http",
        "state": "FIN", "spkts": 10, "dpkts": 8,
        "sbytes": 1500, "dbytes": 800, "sttl": 64,
        "dttl": 64, "sloss": 0, "dloss": 0,
        "sload": 24000.0, "dload": 12800.0,
        "sinpkt": 0.05, "dinpkt": 0.06,
        "sjit": 0.001, "djit": 0.001,
        "swin": 255, "stcpb": 0, "dtcpb": 0,
        "dwin": 255, "tcprtt": 0.01, "synack": 0.005,
        "ackdat": 0.005, "smean": 150, "dmean": 100,
        "trans_depth": 1, "response_body_len": 500,
        "ct_srv_src": 5, "ct_state_ttl": 2,
        "ct_dst_ltm": 3, "ct_src_dport_ltm": 2,
        "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 4,
        "is_ftp_login": 0, "ct_ftp_cmd": 0,
        "ct_flw_http_mthd": 0, "ct_src_ltm": 3,
        "ct_srv_dst": 4, "is_sm_ips_ports": 0,
        "sport": 12345, "dport": 80,
        "attack_cat": "", "label": 0
    }

    output = predict(test_input)
    print(f"\nPrediction result:")
    print(f"  Label    : {output['label']}")
    print(f"  Score    : {output['score']:.2f}")
    print(f"  Decision : {output['decision']}")
    print(f"  Layer    : {output['layer']}")
    print(f"  MITRE    : {output['mitre']['id']} "
          f"— {output['mitre']['name']}")
    print("\npredictor.py OK")