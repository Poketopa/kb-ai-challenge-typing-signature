import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def ensure_pynput_installed():
    try:
        from pynput import keyboard  # noqa: F401
        return True
    except Exception:
        return False


def pick_best_model(models_dir: Path) -> Optional[Path]:
    """Pick the model with the smallest EER from filenames like best_model_eer_12.3400.pt"""
    if not models_dir.exists():
        return None
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


def load_threshold_from_results(results_dir: Path, mode: str = "eer") -> Optional[float]:
    """Load a distance threshold from results files.

    mode:
      - "eer" -> from evaluation_results.csv row "Threshold(distance,EER)"
      - "far01" -> from operational_thresholds.csv row "FAR@0.1%"
      - "far001" -> from operational_thresholds.csv row "FAR@0.01%"
      - "f1max" -> from operational_thresholds.csv row "F1-max"
    """
    eval_csv = results_dir / "evaluation_results.csv"
    ops_csv = results_dir / "operational_thresholds.csv"

    try:
        if mode.lower() == "eer" and eval_csv.exists():
            with open(eval_csv, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Metric", "").strip().lower() == "threshold(distance,eer)":
                        val = row.get("Value", None)
                        if val is not None and str(val).strip() != "":
                            return float(val)
        elif ops_csv.exists():
            wanted = {
                "far01": "FAR@0.1%",
                "far001": "FAR@0.01%",
                "f1max": "F1-max",
            }.get(mode.lower())
            if wanted is None:
                return None
            with open(ops_csv, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Mode", "").strip() == wanted:
                        thr = row.get("Threshold(distance)", None)
                        if thr is not None and str(thr).strip() != "":
                            return float(thr)
    except Exception:
        return None
    return None


class KeystrokeRecorder:
    """Capture key press/release timings until Enter is pressed."""

    def __init__(self, exclude_non_printable: bool = True):
        self.exclude_non_printable = exclude_non_printable
        self._press_times: Dict[str, List[float]] = {}
        self._events: List[Tuple[float, float, str]] = []  # (press_time, release_time, key_str)
        self._finished = False

    def _key_to_str(self, key) -> Optional[str]:
        try:
            from pynput import keyboard
        except Exception:
            return None
        if isinstance(key, keyboard.KeyCode):
            # Printable char
            if key.char is None:
                return None
            return key.char
        else:
            # Special keys
            if self.exclude_non_printable:
                return None
            return str(key)

    def _is_finish_key(self, key) -> bool:
        try:
            from pynput import keyboard
        except Exception:
            return False
        return key == keyboard.Key.enter

    def _on_press(self, key):
        if self._is_finish_key(key):
            # Stop immediately on Enter press
            self._finished = True
            return False
        key_str = self._key_to_str(key)
        if key_str is None:
            return True
        now = time.perf_counter()
        self._press_times.setdefault(key_str, []).append(now)
        return True

    def _on_release(self, key):
        key_str = self._key_to_str(key)
        if key_str is None:
            return True
        now = time.perf_counter()
        if key_str in self._press_times and len(self._press_times[key_str]) > 0:
            press_time = self._press_times[key_str].pop(0)
            # Save event (press, release)
            if now >= press_time:
                self._events.append((press_time, now, key_str))
        return True

    def record_once(self, prompt: str = "Type your phrase then press Enter") -> List[Tuple[float, float, str]]:
        if not ensure_pynput_installed():
            print("Missing dependency: pynput. Install with: python -m pip install pynput")
            sys.exit(1)
        from pynput import keyboard
        self._press_times.clear()
        self._events.clear()
        self._finished = False

        print(f"\n{prompt}\n(Recording... Press Enter to finish)\n")
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()
        # Sort by press time
        self._events.sort(key=lambda t: t[0])
        return self._events


def build_features_from_events(events: List[Tuple[float, float, str]]) -> np.ndarray:
    """Convert (press, release, key) events into Nx3 features in seconds.

    Features per key i:
      - dwell_time = release_i - press_i
      - flight_time = press_{i+1} - release_i (0 for last)
      - press_interval = press_{i+1} - press_i (0 for last)
    """
    if len(events) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    press_times = [p for (p, r, k) in events]
    release_times = [r for (p, r, k) in events]

    dwell = np.maximum(0.0, np.array(release_times) - np.array(press_times))

    flight = np.zeros(len(events), dtype=np.float64)
    press_interval = np.zeros(len(events), dtype=np.float64)
    for i in range(len(events) - 1):
        next_press = press_times[i + 1]
        flight[i] = max(0.0, next_press - release_times[i])
        press_interval[i] = max(0.0, next_press - press_times[i])

    feats = np.stack([dwell, flight, press_interval], axis=1).astype(np.float32)
    return feats


def pad_or_truncate(sequence: np.ndarray, seq_len: int) -> np.ndarray:
    if sequence.shape[0] == seq_len:
        return sequence
    if sequence.shape[0] < seq_len:
        pad = np.zeros((seq_len - sequence.shape[0], 3), dtype=np.float32)
        return np.vstack([sequence, pad]).astype(np.float32)
    return sequence[:seq_len].astype(np.float32)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    sim = float((a * b).sum(dim=-1))
    return 1.0 - sim


class ModelWrapper:
    def __init__(self, model_path: Path, device: torch.device):
        self.device = device
        # Load entire model object as saved
        self.model = torch.load(str(model_path), map_location=device, weights_only=False)
        self.model.eval()

    @torch.no_grad()
    def embed(self, sequence_3x: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(sequence_3x).unsqueeze(0).to(self.device)
        emb = self.model(x)  # Already normalized in model
        return emb.squeeze(0).detach().to("cpu")


def compute_distance(a: torch.Tensor, b: torch.Tensor, metric: str = "l2") -> float:
    if metric == "cosine":
        return cosine_distance(a, b)
    # L2 distance
    diff = a - b
    return float(torch.sqrt(torch.clamp((diff * diff).sum(), min=1e-12)))


def save_template(user: str, template: torch.Tensor, enroll_dir: Path) -> Path:
    enroll_dir.mkdir(parents=True, exist_ok=True)
    path = enroll_dir / f"{user}.npy"
    np.save(str(path), template.detach().cpu().numpy().astype(np.float32))
    return path


def load_template(user: str, enroll_dir: Path) -> Optional[torch.Tensor]:
    path = enroll_dir / f"{user}.npy"
    if not path.exists():
        return None
    arr = np.load(str(path))
    return torch.from_numpy(arr)


def list_templates(enroll_dir: Path) -> List[Path]:
    if not enroll_dir.exists():
        return []
    return sorted(enroll_dir.glob("*.npy"))


def main():
    parser = argparse.ArgumentParser(description="Interactive keystroke test (enroll/verify/identify)")
    parser.add_argument("--mode", choices=["enroll", "verify", "identify"], required=True)
    parser.add_argument("--user", type=str, default=None, help="Username for enroll/verify")
    parser.add_argument("--sessions", type=int, default=5, help="Enrollment sessions to capture")
    parser.add_argument("--phrase", type=str, default="secure typing", help="Prompt phrase to type")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model (.pt). If not set, pick best in models/")
    parser.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument("--threshold_mode", choices=["eer", "far01", "far001", "f1max", "none"], default="eer")
    parser.add_argument("--threshold", type=float, default=None, help="Override distance threshold")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    enroll_dir = project_root / "enrollments"

    # Device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model path
    model_path = Path(args.model_path) if args.model_path else pick_best_model(models_dir)
    if model_path is None or not model_path.exists():
        print("Could not find a model. Pass --model_path or place best_model_eer_*.pt under models/.")
        sys.exit(1)
    print(f"Loading model: {model_path}")

    # Config: read seq_len from config.json via utils.Config
    try:
        sys.path.append(str((project_root / "utils").resolve()))
        from config import Config  # type: ignore
        cfg = Config(path="../../config.json")
        seq_len = int(cfg.get_config_dict()["data"]["keystroke_sequence_len"])  # type: ignore
    except Exception:
        seq_len = 50
        print("Warning: Failed to read config.json. Defaulting seq_len=50")

    # Threshold
    threshold: Optional[float] = args.threshold
    if threshold is None and args.threshold_mode.lower() != "none":
        threshold = load_threshold_from_results(results_dir, args.threshold_mode)
        if threshold is not None:
            print(f"Loaded threshold ({args.threshold_mode}) = {threshold:.6f} (distance)")
        else:
            print("Threshold not found. Proceeding without a threshold.")

    # Model
    model = ModelWrapper(model_path, device)

    recorder = KeystrokeRecorder(exclude_non_printable=True)

    if args.mode == "enroll":
        if not args.user:
            print("--user is required for enroll mode")
            sys.exit(1)
        embeddings: List[torch.Tensor] = []
        for i in range(args.sessions):
            prompt = f"[{i+1}/{args.sessions}] Type: '{args.phrase}' then press Enter"
            events = recorder.record_once(prompt)
            feats = build_features_from_events(events)
            feats = pad_or_truncate(feats, seq_len)
            emb = model.embed(feats)
            embeddings.append(emb)
            print(f"Captured keys: {len(events)} | Non-zero frames: {int((feats.sum(axis=1)>0).sum())}")
        # Template = mean
        template = torch.stack(embeddings, dim=0).mean(dim=0)
        path = save_template(args.user, template, enroll_dir)
        print(f"Enrollment saved: {path}")
        return

    if args.mode == "verify":
        if not args.user:
            print("--user is required for verify mode")
            sys.exit(1)
        template = load_template(args.user, enroll_dir)
        if template is None:
            print(f"No enrollment found for user '{args.user}'. Run enroll first.")
            sys.exit(1)
        events = recorder.record_once(f"Verify for '{args.user}'. Type: '{args.phrase}' then press Enter")
        feats = build_features_from_events(events)
        feats = pad_or_truncate(feats, seq_len)
        emb = model.embed(feats)
        dist = compute_distance(emb, template, args.metric)
        print(f"Distance ({args.metric}): {dist:.6f}")
        if threshold is not None:
            decision = "ACCEPT" if dist <= threshold else "REJECT"
            print(f"Decision @ thr={threshold:.6f}: {decision}")
        else:
            print("No threshold provided. Use --threshold or --threshold_mode to get a decision.")
        return

    if args.mode == "identify":
        templates = list_templates(enroll_dir)
        if not templates:
            print("No enrollments found. Enroll at least one user first.")
            sys.exit(1)
        events = recorder.record_once(f"Identify. Type: '{args.phrase}' then press Enter")
        feats = build_features_from_events(events)
        feats = pad_or_truncate(feats, seq_len)
        emb = model.embed(feats)
        scores: List[Tuple[str, float]] = []
        for p in templates:
            user = p.stem
            t = torch.from_numpy(np.load(str(p)))
            d = compute_distance(emb, t, args.metric)
            scores.append((user, d))
        scores.sort(key=lambda x: x[1])
        print("Top matches (lower is better):")
        for user, d in scores[:5]:
            print(f"  {user}: {d:.6f}")
        best_user, best_dist = scores[0]
        if threshold is not None:
            decision = "ACCEPT" if best_dist <= threshold else "REVERIFY/REJECT"
            print(f"Best: {best_user} | dist={best_dist:.6f} | decision @ thr {threshold:.6f}: {decision}")
        else:
            print(f"Best: {best_user} | dist={best_dist:.6f} (no threshold)")
        return


if __name__ == "__main__":
    main()


