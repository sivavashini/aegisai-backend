"""
logger.py — MongoDB Incident Logger
Logs every security event to MongoDB Atlas.
Provides query functions for the /incidents endpoint.
Falls back gracefully if MongoDB is unavailable —
system keeps running without logging.
"""

import logging
from datetime import datetime, timezone
from typing import Optional
import config

logger = logging.getLogger(__name__)

# ─── Lazy imports — motor is async MongoDB driver ───────────
_client     = None
_db         = None
_connected  = False


def _get_db():
    """
    Return MongoDB database handle.
    Creates connection on first call — lazy init.
    Returns None if connection fails.
    """
    global _client, _db, _connected

    if _db is not None:
        return _db

    try:
        from motor.motor_asyncio import (
            AsyncIOMotorClient
        )
        import os
        import certifi
        tls_insecure = os.getenv(
            "MONGO_TLS_INSECURE", "false"
        ).lower() == "true"
        _client    = AsyncIOMotorClient(
            config.MONGO_URI,
            serverSelectionTimeoutMS=10000,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidCertificates=tls_insecure,
            tlsAllowInvalidHostnames=tls_insecure
        )
        _db        = _client[config.MONGO_DB_NAME]
        _connected = True
        logger.info("[LOGGER] MongoDB connection established")
        return _db
    except Exception as e:
        logger.error(
            f"[LOGGER] MongoDB connection failed: {e}")
        _connected = False
        return None


def is_connected() -> bool:
    """
    Return current MongoDB connection status.

    Returns:
        True if connected, False otherwise
    """
    return _connected


async def log_incident(
    src_ip      : str,
    dst_ip      : str,
    label       : str,
    score       : float,
    decision    : str,
    mitre       : dict,
    layer       : str,
    shap        : Optional[dict] = None,
    phantomnet  : Optional[dict] = None,
    veritascore : Optional[dict] = None,
    extra       : Optional[dict] = None
) -> Optional[str]:
    """
    Log one security incident to MongoDB.
    Non-blocking — failure does not crash the system.

    Args:
        src_ip     : Source IP address
        dst_ip     : Destination IP address
        label      : Attack label from ML
        score      : Threat score 0-100
        decision   : BLOCK/ISOLATE/SANDBOX/SUSPICIOUS/ALLOW
        mitre      : MITRE ATT&CK dict
        layer      : Detection layer name
        shap       : Optional SHAP explanation dict
        phantomnet : Optional PhantomNet result dict
        veritascore: Optional VeritasCore result dict
        extra      : Optional additional metadata

    Returns:
        Inserted document ID string or None on failure
    """
    db = _get_db()
    if db is None:
        return None

    try:
        doc = {
            "timestamp"  : datetime.now(
                timezone.utc).isoformat(),
            "src_ip"     : src_ip,
            "dst_ip"     : dst_ip,
            "label"      : label,
            "score"      : round(float(score), 2),
            "decision"   : decision,
            "mitre"      : mitre,
            "layer"      : layer,
            "shap"       : shap       or {},
            "phantomnet" : phantomnet or {},
            "veritascore": veritascore or {},
            "extra"      : extra      or {},
        }

        collection = db[
            config.MONGO_COLLECTION_INCIDENTS]
        result = await collection.insert_one(doc)
        inserted_id = str(result.inserted_id)
        logger.debug(
            f"[LOGGER] Incident logged — "
            f"id={inserted_id} "
            f"label={label} "
            f"score={score:.2f} "
            f"decision={decision}"
        )
        return inserted_id

    except Exception as e:
        logger.error(f"[LOGGER] Log incident error: {e}")
        return None


async def get_incidents(
    limit    : int = 50,
    min_score: float = 0.0,
    label    : Optional[str] = None
) -> list:
    """
    Query incident history from MongoDB.
    Used by the GET /incidents endpoint.

    Args:
        limit    : Maximum number of incidents to return
        min_score: Minimum threat score filter
        label    : Optional label filter string

    Returns:
        List of incident documents as dicts
    """
    db = _get_db()
    if db is None:
        return []

    try:
        query = {"score": {"$gte": min_score}}
        if label:
            query["label"] = label

        collection = db[
            config.MONGO_COLLECTION_INCIDENTS]
        cursor = collection.find(
            query,
            {"_id": 0}
        ).sort(
            "timestamp", -1
        ).limit(limit)

        results = []
        async for doc in cursor:
            results.append(doc)
        return results

    except Exception as e:
        logger.error(
            f"[LOGGER] Get incidents error: {e}")
        return []


async def log_file_isolation(
    event_type : str,
    file_path  : str,
    file_hash  : Optional[str] = None,
    status     : str = "quarantined",
    extra      : Optional[dict] = None
) -> Optional[str]:
    """
    Log a file isolation event to MongoDB.

    Args:
        event_type: corrupted/suspicious/restored/clean
        file_path : Full path of the affected file
        file_hash : SHA-256 hash of the file
        status    : quarantined/restored/clean
        extra     : Optional additional metadata

    Returns:
        Inserted document ID string or None on failure
    """
    db = _get_db()
    if db is None:
        return None

    try:
        doc = {
            "timestamp" : datetime.now(
                timezone.utc).isoformat(),
            "event_type": event_type,
            "file_path" : file_path,
            "file_hash" : file_hash or "",
            "status"    : status,
            "extra"     : extra or {},
        }

        collection = db[
            config.MONGO_COLLECTION_FILES]
        result = await collection.insert_one(doc)
        return str(result.inserted_id)

    except Exception as e:
        logger.error(
            f"[LOGGER] Log file isolation error: {e}")
        return None


async def get_file_isolation_report() -> dict:
    """
    Get summary statistics for file isolation events.
    Used by GET /file-isolation/report endpoint.

    Returns:
        Dictionary with monitored, scanned,
        quarantined, restored counts
    """
    db = _get_db()
    if db is None:
        return {
            "monitored"  : 0,
            "scanned"    : 0,
            "quarantined": 0,
            "restored"   : 0
        }

    try:
        collection = db[
            config.MONGO_COLLECTION_FILES]

        quarantined = await collection.count_documents(
            {"status": "quarantined"})
        restored    = await collection.count_documents(
            {"status": "restored"})
        scanned     = await collection.count_documents({})

        return {
            "monitored"  : scanned,
            "scanned"    : scanned,
            "quarantined": quarantined,
            "restored"   : restored
        }

    except Exception as e:
        logger.error(
            f"[LOGGER] File report error: {e}")
        return {
            "monitored"  : 0,
            "scanned"    : 0,
            "quarantined": 0,
            "restored"   : 0
        }


async def ping_mongodb() -> bool:
    """
    Test MongoDB connectivity with a ping command.
    Used by GET /status endpoint health check.

    Returns:
        True if MongoDB responds, False otherwise
    """
    global _connected
    db = _get_db()
    if db is None:
        return False

    try:
        await db.command("ping")
        _connected = True
        return True
    except Exception as e:
        logger.error(f"[LOGGER] Ping failed: {e}")
        _connected = False
        return False


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def run_tests():
        print("Testing MongoDB logger...")

        # Test 1 — ping
        print("\nTest 1 — MongoDB ping...")
        ok = await ping_mongodb()
        print(f"MongoDB ping: {'OK' if ok else 'FAIL — check MONGO_URI in .env'}")

        if not ok:
            print("Cannot proceed without MongoDB.")
            print("Check your MONGO_URI and Atlas network access.")
            return

        # Test 2 — log an incident
        print("\nTest 2 — Log test incident...")
        doc_id = await log_incident(
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
        print(f"Incident logged: "
              f"{'OK id=' + doc_id if doc_id else 'FAIL'}")

        # Test 3 — query incidents
        print("\nTest 3 — Query incidents...")
        incidents = await get_incidents(limit=5)
        print(f"Incidents returned: {len(incidents)}")
        if incidents:
            first = incidents[0]
            print(f"  Latest — label={first.get('label')}"
                  f" score={first.get('score')}"
                  f" decision={first.get('decision')}")

        # Test 4 — log file isolation
        print("\nTest 4 — Log file isolation event...")
        fid = await log_file_isolation(
            event_type = "corrupted",
            file_path  = "/home/pi/protected/test.txt",
            file_hash  = "abc123def456",
            status     = "quarantined"
        )
        print(f"File event logged: "
              f"{'OK id=' + fid if fid else 'FAIL'}")

        # Test 5 — file isolation report
        print("\nTest 5 — File isolation report...")
        report = await get_file_isolation_report()
        print(f"Report: {report}")

        print("\nlogger.py OK")

    asyncio.run(run_tests())