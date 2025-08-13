import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import datetime
from utils.db import init_db, SessionLocal, User
from utils.crypto import get_or_create_key, encrypt_embedding, decrypt_embedding

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

init_db()
KEY = get_or_create_key()


# -------------------------
# Paths and basic config
# -------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
ENROLL_DIR = PROJECT_ROOT / "enrollments"
CONFIG_JSON = PROJECT_ROOT / "config.json"


def read_seq_len(default: int = 50) -> int:
    try:
        with open(CONFIG_JSON, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg["data"]["keystroke_sequence_len"])  # type: ignore
    except Exception:
        return default


SEQ_LEN = read_seq_len(50)


# -------------------------
# Model utilities
# -------------------------
def pick_best_model(models_dir: Path) -> Optional[Path]:
    import re
    best_path = None
    best_eer = float("inf")
    pattern = re.compile(r"best_model_eer_([0-9]+\.[0-9]+)\.pt$")
    for p in models_dir.glob("best_model_eer_*.pt"):
        m = pattern.search(p.name)
        if not m:
            continue
        try:
            eer_val = float(m.group(1))
        except ValueError:
            continue
        if eer_val < best_eer:
            best_eer = eer_val
            best_path = p
    return best_path


def load_threshold_from_results(results_dir: Path) -> Optional[float]:
    import csv
    eval_csv = results_dir / "evaluation_results.csv"
    if not eval_csv.exists():
        return None
    try:
        with open(eval_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Metric", "").strip().lower() == "threshold(distance,eer)":
                    v = row.get("Value", None)
                    if v is not None and str(v).strip() != "":
                        return float(v)
    except Exception:
        return None
    return None


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    sim = float((a * b).sum(dim=-1))
    return 1.0 - sim


def compute_distance(a: torch.Tensor, b: torch.Tensor, metric: str = "l2") -> float:
    if metric == "cosine":
        return cosine_distance(a, b)
    diff = a - b
    return float(torch.sqrt(torch.clamp((diff * diff).sum(), min=1e-12)))


class ModelWrapper:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if (torch.cuda.is_available() and os.getenv("WEB_USE_GPU") == "1") else "cpu")
        # Fixed model path as requested
        fixed = MODELS_DIR / "best_model_eer_4.6800.pt"
        model_path = fixed if fixed.exists() else pick_best_model(MODELS_DIR)
        if model_path is None or not Path(model_path).exists():
            raise RuntimeError("Required model not found: models/best_model_eer_4.6800.pt")
        self.model = torch.load(str(model_path), map_location=self.device, weights_only=False)
        self.model.eval()

    @torch.no_grad()
    def embed(self, sequence_3x: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(sequence_3x).unsqueeze(0).to(self.device)
        emb = self.model(x)
        return emb.squeeze(0).detach().cpu()


def pad_or_truncate(sequence: np.ndarray, seq_len: int) -> np.ndarray:
    if sequence.shape[0] == seq_len:
        return sequence
    if sequence.shape[0] < seq_len:
        pad = np.zeros((seq_len - sequence.shape[0], 3), dtype=np.float32)
        return np.vstack([sequence, pad]).astype(np.float32)
    return sequence[:seq_len].astype(np.float32)


def build_features_from_events(events: List[Dict[str, Any]]) -> np.ndarray:
    """events: [{"press": float, "release": float, "key": str}] in seconds
    Return Nx3 features (dwell, flight, press_interval) in seconds
    """
    if not events:
        return np.zeros((0, 3), dtype=np.float32)
    # Ensure sorted by press time
    # Remove invalid/duplicate events and ensure press <= release
    filtered: List[Dict[str, Any]] = []
    seen = set()
    for e in events:
        try:
            p = float(e.get("press", 0.0))
            r = float(e.get("release", 0.0))
            k = str(e.get("key", ""))
        except Exception:
            continue
        if not np.isfinite(p) or not np.isfinite(r):
            continue
        if r < p:
            continue
        key_id = (k, p, r)
        if key_id in seen:
            continue
        seen.add(key_id)
        filtered.append({"press": p, "release": r, "key": k})
    events_sorted = sorted(filtered, key=lambda e: float(e.get("press", 0.0)))
    press_times = np.array([float(e["press"]) for e in events_sorted], dtype=np.float64)
    release_times = np.array([float(e["release"]) for e in events_sorted], dtype=np.float64)
    dwell = np.maximum(0.0, release_times - press_times)
    n = len(events_sorted)
    flight = np.zeros(n, dtype=np.float64)
    press_interval = np.zeros(n, dtype=np.float64)
    if n > 1:
        next_press = press_times[1:]
        flight[:-1] = np.maximum(0.0, next_press - release_times[:-1])
        press_interval[:-1] = np.maximum(0.0, next_press - press_times[:-1])
    feats = np.stack([dwell, flight, press_interval], axis=1).astype(np.float32)
    return feats


def save_template(user: str, template: torch.Tensor) -> Path:
    ENROLL_DIR.mkdir(parents=True, exist_ok=True)
    path = ENROLL_DIR / f"{user}.npy"
    np.save(str(path), template.detach().cpu().numpy().astype(np.float32))
    return path


def load_template(user: str) -> Optional[torch.Tensor]:
    path = ENROLL_DIR / f"{user}.npy"
    if not path.exists():
        return None
    return torch.from_numpy(np.load(str(path)))


def list_users() -> List[str]:
    if not ENROLL_DIR.exists():
        return []
    return sorted([p.stem for p in ENROLL_DIR.glob("*.npy")])


def delete_all_users() -> int:
    """Delete all enrolled user templates (.npy files) robustly on Windows.

    Attempts file-by-file deletion first to avoid rmtree failures due to
    transient file locks. Ensures the enrollments directory exists at the end.
    """
    if not ENROLL_DIR.exists():
        ENROLL_DIR.mkdir(parents=True, exist_ok=True)
        return 0

    removed_count = 0
    # First, try to unlink all .npy files individually
    for npy_path in ENROLL_DIR.rglob("*.npy"):
        try:
            npy_path.unlink()
            removed_count += 1
        except Exception:
            # Best-effort: continue deleting others
            pass

    # Then, try to clean up any leftover empty subdirectories
    try:
        for entry in ENROLL_DIR.iterdir():
            try:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
            except Exception:
                pass
    finally:
        ENROLL_DIR.mkdir(parents=True, exist_ok=True)

    return removed_count


def delete_user(user: str) -> bool:
    if not user:
        return False
    path = ENROLL_DIR / f"{user}.npy"
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except Exception:
        return False


# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Keystroke Web Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/static", StaticFiles(directory=str(CURRENT_DIR / "static"), html=False), name="static")


@app.get("/")
def root_redirect_to_enroll():
    enroll_path = CURRENT_DIR / "static" / "enroll.html"
    if not enroll_path.exists():
        raise HTTPException(status_code=404, detail="enroll.html not found")
    return FileResponse(str(enroll_path))


@app.get("/enroll")
def enroll_page():
    enroll_path = CURRENT_DIR / "static" / "enroll.html"
    if not enroll_path.exists():
        raise HTTPException(status_code=404, detail="enroll.html not found")
    return FileResponse(str(enroll_path))


@app.get("/verify")
def verify_page():
    verify_path = CURRENT_DIR / "static" / "verify.html"
    if not verify_path.exists():
        raise HTTPException(status_code=404, detail="verify.html not found")
    return FileResponse(str(verify_path))


class EnrollBody(BaseModel):
    user: str
    sessions: List[List[Dict[str, Any]]]  # list of event arrays
    metric: Optional[str] = "l2"


class VerifyBody(BaseModel):
    user: str
    events: List[Dict[str, Any]]
    metric: Optional[str] = "l2"


class IdentifyBody(BaseModel):
    events: List[Dict[str, Any]]
    metric: Optional[str] = "l2"
    topk: Optional[int] = 3


# Lazy model init (loaded on first request)
_MODEL: Optional[ModelWrapper] = None
_THRESHOLD: Optional[float] = None


def get_model() -> ModelWrapper:
    global _MODEL
    if _MODEL is None:
        _MODEL = ModelWrapper()
    return _MODEL


def get_threshold() -> Optional[float]:
    global _THRESHOLD
    if _THRESHOLD is None:
        _THRESHOLD = load_threshold_from_results(RESULTS_DIR)
    return _THRESHOLD


@app.get("/api/config")
def api_config():
    return {
        "seq_len": SEQ_LEN,
        "threshold": get_threshold(),
        "users": list_users(),
    }


@app.get("/api/users")
def api_users():
    return {"users": list_users()}


@app.delete("/api/users")
def api_delete_users():
    deleted = delete_all_users()
    return {"deleted": int(deleted), "users": list_users()}


@app.delete("/api/users/{user}")
def api_delete_user(user: str):
    ok = delete_user(user)
    if not ok:
        raise HTTPException(status_code=404, detail="User template not found or failed to delete")
    return {"deleted": user, "users": list_users()}


@app.post("/api/enroll")
def api_enroll_db(body: EnrollBody):
    if not body.user or len(body.user.strip()) == 0:
        raise HTTPException(status_code=400, detail="user is required")
    if not body.sessions or len(body.sessions) == 0:
        raise HTTPException(status_code=400, detail="sessions are required")
    db = SessionLocal()
    try:
        # 임베딩 생성
        model = get_model()
        embeddings = []
        for events in body.sessions:
            feats = build_features_from_events(events)
            feats = pad_or_truncate(feats, SEQ_LEN)
            emb = model.embed(feats)
            embeddings.append(emb.detach().cpu().numpy())
        template = np.stack(embeddings, axis=0).mean(axis=0)
        # 암호화
        enc = encrypt_embedding(template, KEY)
        # DB 저장(업데이트 or 신규)
        user = db.query(User).filter_by(username=body.user).first()
        if user:
            user.embedding = enc
            user.last_verified_at = None
        else:
            user = User(username=body.user, embedding=enc)
            db.add(user)
        db.commit()
        return {"ok": True, "user": body.user}
    finally:
        db.close()


@app.post("/api/verify")
def api_verify_db(body: VerifyBody):
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(username=body.user).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        # 임베딩 복호화
        template = decrypt_embedding(user.embedding, KEY)
        # 입력 세션 임베딩 생성
        model = get_model()
        feats = build_features_from_events(body.events)
        feats = pad_or_truncate(feats, SEQ_LEN)
        emb = model.embed(feats).detach().cpu().numpy()
        # 거리 계산 (L2)
        dist = float(np.linalg.norm(emb - template))
        thr = get_threshold()
        decision = None
        if thr is not None:
            decision = "ACCEPT" if dist <= thr else "REJECT"
        # 인증 성공 시 last_verified_at 갱신
        if decision == "ACCEPT":
            user.last_verified_at = datetime.datetime.utcnow()
            db.commit()
        return {"distance": dist, "threshold": thr, "decision": decision}
    finally:
        db.close()


@app.post("/api/identify")
def api_identify(body: IdentifyBody):
    users = list_users()
    if not users:
        raise HTTPException(status_code=404, detail="No enrolled users")
    model = get_model()
    feats = build_features_from_events(body.events)
    feats = pad_or_truncate(feats, SEQ_LEN)
    emb = model.embed(feats)
    if emb.numel() > 0:
        emb = emb + 1e-9 * torch.randn_like(emb)
    results: List[Tuple[str, float]] = []
    for user in users:
        t = load_template(user)
        if t is None:
            continue
        d = compute_distance(emb, t, body.metric or "l2")
        results.append((user, d))
    results.sort(key=lambda x: x[1])
    topk = max(1, int(body.topk or 3))
    top = [{"user": u, "distance": float(d)} for (u, d) in results[:topk]]
    thr = get_threshold()
    decision = None
    if thr is not None and top:
        decision = "ACCEPT" if top[0]["distance"] <= thr else "REVERIFY/REJECT"
    return {"top": top, "threshold": thr, "decision": decision}


