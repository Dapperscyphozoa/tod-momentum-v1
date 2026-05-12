"""GitHub-backed SQLite DB backup.

Restores DB file from GitHub on boot. Schedules a debounced upload after
each transaction so paper PnL survives Render restarts.

Free, uses existing GITHUB_TOKEN. ~30s upload latency after a close trade.
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
import threading
import time
import urllib.request
import urllib.error
from typing import Optional


_GITHUB_OWNER = "Dapperscyphozoa"
_GITHUB_REPO = "multica"
_ENGINE_NAME = os.environ.get("ENGINE_NAME", "unknown")
_BRANCH = "main"
_DEBOUNCE_SECONDS = 30   # upload at most once every 30s
_MAX_DB_BYTES = 50 * 1024 * 1024  # 50 MB safety cap

_state_lock = threading.RLock()
_pending_upload = False
_last_upload_ts = 0.0
_sha: Optional[str] = None
_db_path_cached: Optional[str] = None
_last_uploaded_hash: Optional[str] = None


def _repo_path() -> str:
    return f"engine_state/{_ENGINE_NAME}/state.db"


def _api_url() -> str:
    return f"https://api.github.com/repos/{_GITHUB_OWNER}/{_GITHUB_REPO}/contents/{_repo_path()}"


def _hdrs() -> dict:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "multica-dbbackup/1.0"}
    tok = os.environ.get("GITHUB_TOKEN", "")
    if tok: h["Authorization"] = f"Bearer {tok}"
    return h


def restore_on_boot(db_path: str) -> bool:
    """Download DB from GitHub if local doesn't exist or is empty.
    Returns True if a restore happened."""
    global _sha, _db_path_cached
    _db_path_cached = db_path

    # Skip if local DB already has data
    if os.path.exists(db_path) and os.path.getsize(db_path) > 1024:
        # Try to read sha for future updates
        try:
            req = urllib.request.Request(f"{_api_url()}?ref={_BRANCH}", headers=_hdrs())
            with urllib.request.urlopen(req, timeout=15) as r:
                body = json.loads(r.read())
            _sha = body.get("sha")
        except Exception:
            pass
        return False

    # Download from GitHub
    try:
        req = urllib.request.Request(f"{_api_url()}?ref={_BRANCH}", headers=_hdrs())
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
        _sha = body.get("sha")
        # large files (>1MB) need separate download via download_url
        if body.get("encoding") == "base64" and body.get("content"):
            content = body["content"].replace("\n", "")
            data = base64.b64decode(content)
        elif body.get("download_url"):
            dl_req = urllib.request.Request(body["download_url"], headers=_hdrs())
            with urllib.request.urlopen(dl_req, timeout=60) as r2:
                data = r2.read()
        else:
            return False

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with open(db_path, "wb") as f:
            f.write(data)
        print(f"[db_backup] restored {len(data)} bytes from GitHub to {db_path}", flush=True)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"[db_backup] no prior backup at {_repo_path()} — starting fresh", flush=True)
        else:
            print(f"[db_backup] restore HTTP {e.code}: {e.read().decode()[:200]}", flush=True)
        return False
    except Exception as e:
        print(f"[db_backup] restore err: {e}", flush=True)
        return False


def _upload_now() -> bool:
    """Upload current DB file to GitHub. Internal use only."""
    global _sha, _last_uploaded_hash
    if not _db_path_cached or not os.path.exists(_db_path_cached):
        return False
    try:
        size = os.path.getsize(_db_path_cached)
        if size > _MAX_DB_BYTES:
            print(f"[db_backup] DB size {size} > {_MAX_DB_BYTES} — skipping", flush=True)
            return False
        if size == 0:
            return False
        with open(_db_path_cached, "rb") as f:
            data = f.read()

        # Skip if no change since last upload
        h = hashlib.sha256(data).hexdigest()
        if h == _last_uploaded_hash:
            return True

        payload = {
            "message": f"{_ENGINE_NAME}: state @ {int(time.time())}",
            "content": base64.b64encode(data).decode("ascii"),
            "branch": _BRANCH,
        }
        if _sha:
            payload["sha"] = _sha
        req = urllib.request.Request(_api_url(), data=json.dumps(payload).encode(),
                                      method="PUT", headers=_hdrs())
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read())
        _sha = body.get("content", {}).get("sha")
        _last_uploaded_hash = h
        return True
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        if e.code == 409:
            # sha conflict — reload sha and retry once
            try:
                req = urllib.request.Request(f"{_api_url()}?ref={_BRANCH}", headers=_hdrs())
                with urllib.request.urlopen(req, timeout=15) as r:
                    info = json.loads(r.read())
                _sha = info.get("sha")
                # retry once
                with open(_db_path_cached, "rb") as f: data = f.read()
                payload["sha"] = _sha
                payload["content"] = base64.b64encode(data).decode("ascii")
                req = urllib.request.Request(_api_url(), data=json.dumps(payload).encode(),
                                              method="PUT", headers=_hdrs())
                with urllib.request.urlopen(req, timeout=60) as r:
                    body2 = json.loads(r.read())
                _sha = body2.get("content", {}).get("sha")
                return True
            except Exception as e2:
                print(f"[db_backup] retry err: {e2}", flush=True)
                return False
        print(f"[db_backup] upload HTTP {e.code}: {body}", flush=True)
        return False
    except Exception as e:
        print(f"[db_backup] upload err: {e}", flush=True)
        return False


def schedule_backup():
    """Mark DB as dirty. Background thread will upload after debounce window.
    Call this after every close_trade / record_closure."""
    global _pending_upload
    with _state_lock:
        _pending_upload = True


def _backup_loop():
    global _pending_upload, _last_upload_ts
    while True:
        try:
            time.sleep(_DEBOUNCE_SECONDS)
            now = time.time()
            with _state_lock:
                should = _pending_upload and (now - _last_upload_ts >= _DEBOUNCE_SECONDS)
                if should:
                    _pending_upload = False
                    _last_upload_ts = now
            if should:
                _upload_now()
        except Exception as e:
            print(f"[db_backup] loop err: {e}", flush=True)
            time.sleep(60)


_thread_started = False


def start_background_thread():
    global _thread_started
    if _thread_started: return
    _thread_started = True
    t = threading.Thread(target=_backup_loop, daemon=True, name="db_backup")
    t.start()
    print(f"[db_backup] background thread started (debounce={_DEBOUNCE_SECONDS}s)", flush=True)
