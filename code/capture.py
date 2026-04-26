"""
capture.py — Screenshot capture + analysis loop.

Captures a screenshot every N seconds, sends to local Ollama VLM for
classification, appends result to log.csv. Designed to run continuously.

Run: python capture.py
Stop: Ctrl+C
"""

import csv
import json
import logging
import re
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import mss
import ollama
import yaml
from PIL import Image

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("timetracker")


# ---------- Config loading ----------
def load_config(path: Path) -> dict:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against the config file's directory (code/)
    base = path.parent.resolve()
    cfg["capture"]["screenshot_dir"] = (base / cfg["capture"]["screenshot_dir"]).resolve()
    cfg["storage"]["log_csv"] = (base / cfg["storage"]["log_csv"]).resolve()
    cfg["storage"]["reports_dir"] = (base / cfg["storage"]["reports_dir"]).resolve()
    return cfg


# ---------- Screenshot capture ----------
SCT = None

def get_sct():
    global SCT
    if SCT is None:
        SCT = mss.MSS()
    return SCT

def capture_screenshot(screenshot_dir: Path, max_width: int) -> Path:
    """Capture full screen, resize, save as PNG. Returns path."""
    now = datetime.now()
    day_dir = screenshot_dir / f"{now.year:04d}" / f"{now.month:02d}" / f"{now.day:02d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    out_path = day_dir / f"{now.strftime('%H%M%S')}.png"

    sct = get_sct()
    # monitors[0] = all monitors combined, monitors[1] = primary
    monitor = sct.monitors[1]
    raw = sct.grab(monitor)
    img = Image.frombytes("RGB", raw.size, raw.rgb)

    # Resize if wider than max_width (preserves aspect ratio)
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    img.save(out_path, "PNG")
    return out_path


# ---------- Prompt construction ----------
def build_prompt(categories: list[dict]) -> str:
    """Build the classification prompt from the taxonomy in config."""
    cat_lines = "\n".join(f"- {c['id']}: {c['description']}" for c in categories)
    cat_ids = ", ".join(c["id"] for c in categories)
    return f"""You are classifying a screenshot of a Mac desktop into exactly ONE category.

Available categories:
{cat_lines}

Look at the screenshot. Identify the active app, visible content, and any URLs or window titles. Pick the single best-fitting category.

Respond ONLY with valid JSON, no other text, no markdown fences:
{{"category": "<one of: {cat_ids}>", "app_name": "<Name of the active application, e.g., VS Code, Chrome>", "window_title": "<Specific document name, URL, or window title>", "description": "<one short sentence describing what is visible>"}}
"""


# ---------- Robust JSON extraction ----------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

def parse_model_output(raw: str, valid_categories: set[str]) -> tuple[str, str, str, str]:
    """
    Extract (category, description, app_name, window_title) from model output. Falls back gracefully.
    Returns ("ERROR", reason, "Unknown", "Unknown") if parsing fails.
    """
    if not raw or not raw.strip():
        return "ERROR", "empty model output", "Unknown", "Unknown"

    # Strip markdown fences if present
    candidate = raw.strip()
    fence = _JSON_FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    # Find first {...} block (in case model added preamble)
    brace_start = candidate.find("{")
    brace_end = candidate.rfind("}")
    if brace_start == -1 or brace_end == -1 or brace_end <= brace_start:
        return "ERROR", f"no JSON object found in: {raw[:120]!r}", "Unknown", "Unknown"

    json_str = candidate[brace_start : brace_end + 1]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return "ERROR", f"JSON parse failed: {e}; raw: {raw[:120]!r}", "Unknown", "Unknown"

    category = str(parsed.get("category", "")).strip()
    description = str(parsed.get("description", "")).strip()
    app_name = str(parsed.get("app_name", "Unknown")).strip()
    window_title = str(parsed.get("window_title", "Unknown")).strip()

    if category not in valid_categories:
        return "ERROR", f"invalid category {category!r}; raw: {raw[:120]!r}", "Unknown", "Unknown"

    return category, description, app_name, window_title


# ---------- Ollama call ----------
CLIENT = None

def get_client(host: str, timeout: int):
    global CLIENT
    if CLIENT is None:
        CLIENT = ollama.Client(host=host, timeout=timeout)
    return CLIENT


def classify(model: str, host: str, timeout: int, image_path: Path, prompt: str) -> tuple[str, int]:
    """Call Ollama, return (raw_text_response, latency_ms)."""
    client = get_client(host, timeout)
    t0 = time.monotonic()
    resp = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }
        ],
        options={
            "temperature": 0.2,  # low for classification consistency
        },
        keep_alive="15m",  # keep model loaded between captures
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    return resp["message"]["content"], latency_ms


# ---------- CSV logging ----------
CSV_HEADER = ["timestamp", "screenshot_path", "category", "app_name", "window_title", "description", "latency_ms", "raw_output", "model"]

def ensure_csv_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

def append_row(csv_path: Path, row: dict) -> None:
    """Append-only, never rewrite."""
    with csv_path.open("a", newline="") as f:
        csv.writer(f).writerow([row[k] for k in CSV_HEADER])


# ---------- Main loop ----------
def main():
    config_path = Path(__file__).parent / "config.yaml"
    cfg = load_config(config_path)

    interval = int(cfg["capture"]["interval_seconds"])
    max_width = int(cfg["capture"]["max_width"])
    screenshot_dir = Path(cfg["capture"]["screenshot_dir"])
    csv_path = Path(cfg["storage"]["log_csv"])
    data_root = csv_path.parent  # for computing relative screenshot paths

    model = cfg["model"]["name"]
    host = cfg["model"]["ollama_host"]
    timeout = int(cfg["model"]["timeout_seconds"])

    categories = cfg["categories"]
    valid_cat_ids = {c["id"] for c in categories}
    prompt = build_prompt(categories)
    print(prompt)

    screenshot_dir.mkdir(parents=True, exist_ok=True)
    ensure_csv_header(csv_path)

    log.info("starting timetracker | interval=%ds | model=%s", interval, model)
    log.info("screenshots → %s", screenshot_dir)
    log.info("log csv     → %s", csv_path)

    # Graceful Ctrl+C
    stop_event = threading.Event()
    def _stop(signum, frame):
        log.info("stop requested (signal %d) — exiting after current iteration", signum)
        stop_event.set()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while not stop_event.is_set():
        loop_start = time.monotonic()
        timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

        try:
            shot_path = capture_screenshot(screenshot_dir, max_width)
        except Exception as e:
            log.exception("screenshot capture failed: %s", e)
            append_row(csv_path, {
                "timestamp": timestamp,
                "screenshot_path": "",
                "category": "ERROR",
                "app_name": "Unknown",
                "window_title": "Unknown",
                "description": f"capture_failed: {e}",
                "latency_ms": 0,
                "raw_output": "",
                "model": model,
            })
            _sleep_remaining(loop_start, interval, stop_event)
            continue

        # Make screenshot path relative to data/ for portability
        try:
            rel_path = str(shot_path.relative_to(data_root))
        except ValueError:
            rel_path = str(shot_path)

        try:
            raw_output, latency_ms = classify(model, host, timeout, shot_path, prompt)
            category, description, app_name, window_title = parse_model_output(raw_output, valid_cat_ids)
        except Exception as e:
            log.exception("classification failed: %s", e)
            raw_output = ""
            latency_ms = 0
            category = "ERROR"
            description = f"classify_failed: {e}"
            app_name = "Unknown"
            window_title = "Unknown"

        append_row(csv_path, {
            "timestamp": timestamp,
            "screenshot_path": rel_path,
            "category": category,
            "app_name": app_name,
            "window_title": window_title,
            "description": description,
            "latency_ms": latency_ms,
            "raw_output": raw_output,
            "model": model,
        })

        log.info("[%s] %s | %dms | %s", timestamp, category, latency_ms, description[:80])

        _sleep_remaining(loop_start, interval, stop_event)

    log.info("stopped cleanly")


def _sleep_remaining(loop_start: float, interval: int, stop_event: threading.Event) -> None:
    """Sleep until the next interval tick, but wake early if stopping."""
    elapsed = time.monotonic() - loop_start
    remaining = max(0.0, interval - elapsed)
    if remaining > 0.0:
        stop_event.wait(timeout=remaining)


if __name__ == "__main__":
    main()