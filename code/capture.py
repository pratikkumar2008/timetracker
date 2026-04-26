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
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import mss
import ollama
import yaml
from PIL import Image

# ---------- Logging ----------
log_dir = Path(__file__).parent.resolve().parent / "data" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_dir / "capture.log", mode="a"),
        logging.StreamHandler()
    ]
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
    return f"""You are classifying a Mac desktop screenshot to track HOW THE USER IS SPENDING THEIR TIME. Classify by **behavior**, not by app or website — the same site can fall into different categories depending on what the user is doing.

Available categories:
{cat_lines}

## Signal priority
1. **Active window first** — the frontmost/focused window (visible title bar, menu bar app name). Ignore background windows.
2. Use URL, window title, and visible content together to infer the behavior.

## Disambiguation rules
- **LinkedIn**: job search / applications / recruiter messages → `JobHunt`; feed scrolling → `Doomscroll`
- **YouTube**: lecture / tutorial / technical talk → `Learning`; music video / vlog / entertainment → `Entertainment`; Shorts → `Doomscroll`
- **Email**: job-related (application, recruiter, company research) → `JobHunt`; other professional email → `Communication`
- **ML content**: interview-prep focus (ML system design, model evaluation questions) → `InterviewPrep-ML`; general AI research, papers, news → `Learning`
- **LLM chats** (Claude, ChatGPT, Gemini): classify by topic — DSA problems → `InterviewPrep-DSA`; ML concepts → `InterviewPrep-ML`; general coding/debugging → `Learning`; life admin → `Personal`
- **Messaging**: WhatsApp / iMessage / personal chats → `Social`; Slack / email with colleagues → `Communication`
- **Rest vs Idle**: `Rest` is intentional downtime (music app in focus, screen idle on a break); `Idle` is a screen you genuinely cannot interpret

## Examples

Screen: Chrome, LinkedIn social feed, scrolling posts from connections, no job listing open.
{{"reasoning": "LinkedIn is open but showing the social feed, not job listings or recruiter messages. Feed browsing → Doomscroll.", "category": "Doomscroll", "app_name": "Chrome", "window_title": "LinkedIn Feed", "description": "Passively scrolling LinkedIn social feed reading posts from connections with no job search activity visible."}}

Screen: Chrome, YouTube video titled 'Stanford CS229: Machine Learning Lecture 4', progress bar at 22 minutes.
{{"reasoning": "YouTube is open but it is a university ML lecture, clearly educational. Educational YouTube → Learning.", "category": "Learning", "app_name": "Chrome", "window_title": "Stanford CS229: Machine Learning Lecture 4 | YouTube", "description": "Watching Stanford CS229 Machine Learning lecture on YouTube, 22 minutes into the video."}}

Screen: Chrome, ChatGPT conversation, visible message reads 'Can you explain how to solve Longest Common Subsequence using DP?'
{{"reasoning": "ChatGPT is open; the conversation is about a DP algorithm problem. LLM chat classified by topic → InterviewPrep-DSA.", "category": "InterviewPrep-DSA", "app_name": "Chrome", "window_title": "ChatGPT", "description": "Asking ChatGPT to explain the dynamic programming solution for the Longest Common Subsequence problem."}}

Screen: Chrome, Gmail, email subject line reads 'Interview Invitation — Senior ML Engineer at Cohere'.
{{"reasoning": "Gmail is open but the email is an interview invitation from a company. Job-related email → JobHunt.", "category": "JobHunt", "app_name": "Chrome", "window_title": "Interview Invitation — Senior ML Engineer at Cohere | Gmail", "description": "Reading an interview invitation email from Cohere for a Senior ML Engineer position."}}

Screen: Spotify desktop app in foreground playing a playlist, no other windows visible.
{{"reasoning": "Spotify is the frontmost app with music playing. Intentional downtime with a music app in focus → Rest, not Idle.", "category": "Rest", "app_name": "Spotify", "window_title": "Daily Mix 1 — Spotify", "description": "Listening to music on Spotify with the app in focus, taking an intentional break from screen work."}}

## When uncertain
Prefer `Idle` over a wrong guess — use it when the screen is blank, locked, or when two categories seem equally plausible.

## Output
Respond with ONLY a single JSON object, starting with {{ and ending with }}. No markdown, no preamble, no explanation outside the JSON. Fields MUST appear in this exact order.
For `description`: write 15–25 words on the specific activity, not just the app. Good: "Reviewing a LeetCode submission for the Two Sum problem in Java." Bad: "Code editor open."

{{"reasoning": "<2-3 sentences: what you see, which behavior, which rule applies>", "category": "<one of: {cat_ids}>", "app_name": "<frontmost app, e.g. VS Code, Chrome; 'Unknown' if unclear>", "window_title": "<specific document, URL, or window title; 'Unknown' if unclear>", "description": "<15-25 word sentence describing the specific activity>"}}
"""


# ---------- Robust JSON extraction ----------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

def parse_model_output(raw: str, valid_categories: set[str]) -> tuple[str, str, str, str, str]:
    """
    Extract (category, description, app_name, window_title, reasoning) from model output. Falls back gracefully.
    Returns ("ERROR", reason, "Unknown", "Unknown", "") if parsing fails.
    """
    if not raw or not raw.strip():
        return "ERROR", "empty model output", "Unknown", "Unknown", ""

    # Strip markdown fences if present
    candidate = raw.strip()
    fence = _JSON_FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    # Find first {...} block (in case model added preamble)
    brace_start = candidate.find("{")
    brace_end = candidate.rfind("}")
    if brace_start == -1 or brace_end == -1 or brace_end <= brace_start:
        return "ERROR", f"no JSON object found in: {raw[:120]!r}", "Unknown", "Unknown", ""

    json_str = candidate[brace_start : brace_end + 1]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return "ERROR", f"JSON parse failed: {e}; raw: {raw[:120]!r}", "Unknown", "Unknown", ""

    category = str(parsed.get("category", "")).strip()
    description = str(parsed.get("description", "")).strip()
    app_name = str(parsed.get("app_name", "Unknown")).strip()
    window_title = str(parsed.get("window_title", "Unknown")).strip()
    reasoning = str(parsed.get("reasoning", "")).strip()

    if category not in valid_categories:
        return "ERROR", f"invalid category {category!r}; raw: {raw[:120]!r}", "Unknown", "Unknown", ""

    return category, description, app_name, window_title, reasoning


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
CSV_HEADER = ["timestamp", "screenshot_path", "category", "app_name", "window_title", "description", "reasoning", "latency_ms", "raw_output", "model"]

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

    screenshot_dir.mkdir(parents=True, exist_ok=True)
    ensure_csv_header(csv_path)

    log.info("starting timetracker | interval=%ds | model=%s", interval, model)
    log.info("screenshots → %s", screenshot_dir)
    log.info("log csv     → %s", csv_path)

    # Graceful Ctrl+C
    stop_event = threading.Event()
    def _stop(signum, _frame):
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
                "reasoning": "",
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
            category, description, app_name, window_title, reasoning = parse_model_output(raw_output, valid_cat_ids)
        except Exception as e:
            log.exception("classification failed: %s", e)
            raw_output = ""
            latency_ms = 0
            category = "ERROR"
            description = f"classify_failed: {e}"
            app_name = "Unknown"
            window_title = "Unknown"
            reasoning = ""

        append_row(csv_path, {
            "timestamp": timestamp,
            "screenshot_path": rel_path,
            "category": category,
            "app_name": app_name,
            "window_title": window_title,
            "description": description,
            "reasoning": reasoning,
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