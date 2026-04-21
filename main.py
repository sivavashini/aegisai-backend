"""
main.py — AegisAI Trinity FastAPI Application
Wires all modules together into a single API server.
Handles all 8 endpoints plus WebSocket live stream.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional, List
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import predictor
import responder
import gpio_controller
import file_isolation
import otx_check
import trainiq
import capture
import cef_writer
import logger as incident_logger
from mitre_map import get_mitre

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─── WebSocket connection manager ───────────────────────────
class ConnectionManager:
    """Manages all active WebSocket connections."""

    def __init__(self):
        self.active: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self.active.append(ws)
        log.info(
            f"[WS] Client connected — "
            f"total={len(self.active)}")

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self.active:
                self.active.remove(ws)
        log.info(
            f"[WS] Client disconnected — "
            f"total={len(self.active)}")

    async def broadcast(self, message: dict) -> None:
        """
        Broadcast message to all connected clients.
        Removes dead connections silently.
        Never blocks the main thread.
        """
        if not self.active:
            return
        dead = []
        async with self._lock:
            clients = list(self.active)
        for ws in clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)


manager = ConnectionManager()


def sync_broadcast(message: dict) -> None:
    """
    Synchronous wrapper for broadcast.
    Called from background threads (capture, isolation).
    Schedules broadcast on the event loop safely.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(message), loop)
    except Exception as e:
        log.debug(f"[WS] Broadcast error: {e}")


# ─── Application lifespan ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic.
    Loads models, starts background threads,
    initializes GPIO on startup.
    Cleans up on shutdown.
    """
    log.info("[MAIN] AegisAI Trinity starting up...")

    # Load all ML models
    model_status = predictor.load_all_models()
    loaded = sum(1 for v in model_status.values() if v)
    total  = len(model_status)
    log.info(f"[MAIN] Models: {loaded}/{total} loaded")

    # Initialize GPIO
    gpio_controller.setup_gpio()
    gpio_controller.system_safe()

    # Initialize OTX database and start sync
    otx_check.init_db()
    otx_check.start_background_sync()
    log.info("[MAIN] OTX sync started")

    # Build file isolation baseline
    file_isolation.load_baseline()
    file_isolation.set_websocket_callback(
        sync_broadcast)

    # Start TrainIQ decay thread
    trainiq.start_decay_thread()
    log.info("[MAIN] TrainIQ decay thread started")

    # Start packet capture
    capture.set_event_callback(sync_broadcast)
    capture.start_capture(
        simulate=not capture.IS_LINUX)
    log.info("[MAIN] Capture daemon started")

    log.info("[MAIN] All systems online")
    yield

    # Shutdown
    log.info("[MAIN] Shutting down...")
    capture.stop_capture()
    otx_check.stop_background_sync()
    trainiq.stop_decay_thread()
    gpio_controller.system_safe()
    gpio_controller.cleanup_gpio()
    log.info("[MAIN] Shutdown complete")


# ─── FastAPI app ────────────────────────────────────────────
app = FastAPI(
    title      = "AegisAI Trinity",
    description= "Three-layer AI cybersecurity system",
    version    = "1.0.0",
    lifespan   = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─── Pydantic request models ────────────────────────────────
class IngestRequest(BaseModel):
    features : List[float]
    src_ip   : str = "0.0.0.0"
    dst_ip   : str = "0.0.0.0"


class PredictRequest(BaseModel):
    features: List[float]


class RespondRequest(BaseModel):
    action  : str
    src_ip  : str
    dst_ip  : str = "0.0.0.0"
    label   : str = "UNKNOWN"
    score   : float = 0.0


class CompleteModuleRequest(BaseModel):
    ip    : str
    module: str


class TrainIQSignalRequest(BaseModel):
    ip    : str
    signal: str


# ─── Endpoints ──────────────────────────────────────────────

@app.post("/ingest")
async def ingest(request: IngestRequest):
    """
    Receive pre-extracted feature vector for scoring.
    Runs full pipeline and triggers response actions.
    """
    try:
        # Check OTX first
        if otx_check.check_ip(request.src_ip):
            result = {
                "label"    : "OTX_BLACKLIST",
                "score"    : 100.0,
                "decision" : "BLOCK",
                "mitre"    : get_mitre("UNKNOWN"),
                "layer"    : "OTX",
                "src_ip"   : request.src_ip,
                "dst_ip"   : request.dst_ip,
            }
            responder.execute_response(
                decision = "BLOCK",
                src_ip   = request.src_ip,
                dst_ip   = request.dst_ip,
                label    = "OTX_BLACKLIST",
                score    = 100.0,
                mitre    = get_mitre("UNKNOWN"),
                layer    = "OTX"
            )
            await manager.broadcast({
                "type"  : "ingest_event",
                **result
            })
            return result

        # Build feature dict
        feature_cols = predictor._models.get(
            "feature_cols1", [])
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
            if i < len(request.features):
                feature_dict[col] = request.features[i]

        feature_dict["src_ip"] = request.src_ip
        feature_dict["dst_ip"] = request.dst_ip

        # Run prediction
        prediction = predictor.predict(feature_dict)

        decision = prediction.get("decision", "ALLOW")
        label    = prediction.get("label",    "Normal")
        score    = prediction.get("score",    0.0)
        mitre    = prediction.get("mitre",    {})
        layer    = prediction.get("layer",
                                  "SentinelX")

        # Execute response
        responder.execute_response(
            decision = decision,
            src_ip   = request.src_ip,
            dst_ip   = request.dst_ip,
            label    = label,
            score    = score,
            mitre    = mitre,
            layer    = layer
        )

        # Trigger file isolation on block/isolate
        if decision in ("BLOCK", "ISOLATE"):
            file_isolation.trigger_isolation(
                blocking=False)

        # Log to MongoDB
        asyncio.create_task(
            incident_logger.log_incident(
                src_ip   = request.src_ip,
                dst_ip   = request.dst_ip,
                label    = label,
                score    = score,
                decision = decision,
                mitre    = mitre,
                layer    = layer
            )
        )

        result = {
            "label"    : label,
            "score"    : score,
            "decision" : decision,
            "mitre"    : mitre,
            "layer"    : layer,
            "src_ip"   : request.src_ip,
            "dst_ip"   : request.dst_ip,
            "phantomnet" : prediction.get(
                "phantomnet", {}),
            "veritascore": prediction.get(
                "veritascore", {}),
        }

        await manager.broadcast({
            "type": "ingest_event", **result})

        return result

    except Exception as e:
        log.error(f"[MAIN] /ingest error: {e}")
        raise HTTPException(
            status_code=500, detail=str(e))


@app.post("/predict")
async def predict_only(request: PredictRequest):
    """
    Return prediction without triggering response.
    Safe for testing and dashboard preview.
    """
    try:
        feature_cols = predictor._models.get(
            "feature_cols1", [])
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
            if i < len(request.features):
                feature_dict[col] = request.features[i]

        prediction = predictor.predict(feature_dict)

        return {
            "label"     : prediction.get(
                "label",    "Normal"),
            "score"     : prediction.get(
                "score",    0.0),
            "decision"  : prediction.get(
                "decision", "ALLOW"),
            "mitre"     : prediction.get(
                "mitre",    {}),
            "layer"     : prediction.get(
                "layer",    "SentinelX"),
            "confidence": prediction.get(
                "confidence", 0.0),
            "phantomnet" : prediction.get(
                "phantomnet", {}),
            "veritascore": prediction.get(
                "veritascore", {}),
        }

    except Exception as e:
        log.error(f"[MAIN] /predict error: {e}")
        raise HTTPException(
            status_code=500, detail=str(e))


@app.post("/respond")
async def respond(request: RespondRequest):
    """
    Manually trigger a response action.
    Used by dashboard manual override controls.
    """
    try:
        mitre  = get_mitre(request.label)
        result = responder.execute_response(
            decision = request.action,
            src_ip   = request.src_ip,
            dst_ip   = request.dst_ip,
            label    = request.label,
            score    = request.score,
            mitre    = mitre,
            layer    = "Manual"
        )

        await manager.broadcast({
            "type"  : "manual_response",
            "action": request.action,
            "src_ip": request.src_ip,
            "result": result
        })

        return result

    except Exception as e:
        log.error(f"[MAIN] /respond error: {e}")
        raise HTTPException(
            status_code=500, detail=str(e))


@app.get("/status")
async def status():
    """
    System health check endpoint.
    Returns status of all subsystems.
    """
    try:
        mongo_ok = await incident_logger.ping_mongodb()
        gpio_state = gpio_controller.get_state()
        capture_stats = capture.get_capture_stats()
        intel_counts  = otx_check.get_intel_counts()

        # Determine threat level from GPIO state
        if gpio_state.get("red_led"):
            threat_level = "HIGH"
        elif gpio_state.get("orange_led"):
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"

        return {
            "status"        : "online",
            "timestamp"     : datetime.now(
                timezone.utc).isoformat(),
            "models_loaded" : True,
            "mongodb"       : mongo_ok,
            "gpio_available": gpio_controller.IS_PI,
            "gpio_state"    : gpio_state,
            "threat_level"  : threat_level,
            "relay_state"   : gpio_state.get(
                "relay", False),
            "otx_last_sync" : otx_check.get_last_sync_time(),
            "otx_intel"     : intel_counts,
            "capture"       : {
                "active"  : capture.is_capturing(),
                "stats"   : capture_stats,
            },
            "blocked_ips"   : responder.get_blocked_ips(),
        }

    except Exception as e:
        log.error(f"[MAIN] /status error: {e}")
        return {
            "status"   : "degraded",
            "error"    : str(e),
            "timestamp": datetime.now(
                timezone.utc).isoformat()
        }


@app.get("/incidents")
async def get_incidents(
    limit    : int   = Query(50,  ge=1, le=500),
    min_score: float = Query(0.0, ge=0, le=100),
    label    : Optional[str] = Query(None)
):
    """
    Get incident history from MongoDB.
    """
    try:
        incidents = await incident_logger.get_incidents(
            limit     = limit,
            min_score = min_score,
            label     = label
        )
        return {
            "count"    : len(incidents),
            "incidents": incidents
        }
    except Exception as e:
        log.error(f"[MAIN] /incidents error: {e}")
        raise HTTPException(
            status_code=500, detail=str(e))


@app.get("/file-isolation/report")
async def file_isolation_report():
    """
    Get latest file isolation scan results.
    """
    try:
        report = file_isolation.get_scan_report()
        db_report = await \
            incident_logger.get_file_isolation_report()
        return {
            **report,
            "db_stats": db_report
        }
    except Exception as e:
        log.error(
            f"[MAIN] /file-isolation error: {e}")
        raise HTTPException(
            status_code=500, detail=str(e))


@app.get("/trainiq/risk-score")
async def trainiq_risk_score(
    ip: str = Query(..., description="User IP address")
):
    """
    Get current risk score for a user by IP.
    """
    try:
        data = trainiq.get_risk_score(ip)
        return data
    except Exception as e:
        log.error(
            f"[MAIN] /trainiq/risk-score error: {e}")
        raise HTTPException(
            status_code=500, detail=str(e))


@app.post("/trainiq/complete-module")
async def complete_module(
        request: CompleteModuleRequest):
    """
    Mark a training module as completed for a user.
    """
    try:
        result = trainiq.complete_module(
            request.ip, request.module)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Module {request.module} "
                       f"not assigned to {request.ip}"
            )
        await manager.broadcast({
            "type"  : "trainiq_update",
            "ip"    : request.ip,
            "module": request.module,
            "result": result
        })
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.error(
            f"[MAIN] /trainiq/complete error: {e}")
        raise HTTPException(
            status_code=500, detail=str(e))


@app.get("/trainiq/all-users")
async def trainiq_all_users():
    """
    Get all user risk scores for admin dashboard.
    """
    try:
        users = trainiq.get_all_users()
        return {
            "count": len(users),
            "users": users
        }
    except Exception as e:
        log.error(
            f"[MAIN] /trainiq/all-users error: {e}")
        raise HTTPException(
            status_code=500, detail=str(e))


@app.post("/trainiq/signal")
async def record_signal(
        request: TrainIQSignalRequest):
    """
    Record a behavioral signal for a user IP.
    """
    try:
        result = trainiq.record_signal(
            request.ip, request.signal)
        await manager.broadcast({
            "type"  : "trainiq_signal",
            "ip"    : request.ip,
            "signal": request.signal,
            "result": result
        })
        return result
    except Exception as e:
        log.error(
            f"[MAIN] /trainiq/signal error: {e}")
        raise HTTPException(
            status_code=500, detail=str(e))


@app.websocket("/live")
async def websocket_live(ws: WebSocket):
    """
    WebSocket endpoint for real-time event stream.
    Pushes every event from all layers in real time.
    """
    await manager.connect(ws)
    try:
        await ws.send_json({
            "type"     : "connected",
            "message"  : "AegisAI Trinity live feed",
            "timestamp": datetime.now(
                timezone.utc).isoformat()
        })
        # Keep connection alive
        while True:
            try:
                await asyncio.wait_for(
                    ws.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send keepalive ping
                await ws.send_json({
                    "type"     : "ping",
                    "timestamp": datetime.now(
                        timezone.utc).isoformat()
                })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.debug(f"[WS] Connection error: {e}")
    finally:
        await manager.disconnect(ws)


@app.get("/")
async def root():
    """Root endpoint — confirms API is running."""
    return {
        "name"   : "AegisAI Trinity",
        "version": "1.0.0",
        "status" : "online",
        "docs"   : "/docs"
    }