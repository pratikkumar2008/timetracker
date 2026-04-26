"""
menubar.py — macOS menu bar app showing live time tracker status.

Reads log.csv every 60s, updates the menu bar title with the current category,
and shows today's top 5 categories in the dropdown.

This process does NOT capture screenshots — it only reads what capture.py writes.
Run capture.py separately.

Usage:
    nohup python menubar.py > /tmp/timetracker_menubar.log 2>&1 &
    (or just `python menubar.py` while you watch it)

Stop:
    Click the menu bar icon -> Quit
"""

import csv
import logging
import subprocess
import sys
import threading
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import rumps
import yaml

log_dir = Path(__file__).parent.resolve().parent / "data" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=str(log_dir / "menubar.log"),
    filemode="a",
)
log = logging.getLogger("menubar")

# ---------- Config loading ----------
def load_config(path: Path) -> dict:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    base = path.parent.resolve()
    cfg["storage"]["log_csv"] = (base / cfg["storage"]["log_csv"]).resolve()
    cfg["storage"]["reports_dir"] = (base / cfg["storage"]["reports_dir"]).resolve()
    cfg["capture"]["interval_seconds"] = int(cfg["capture"]["interval_seconds"])
    return cfg


# ---------- CSV reading ----------
def load_today_rows(csv_path: Path) -> list[dict]:
    """Read all of today's rows. Tolerant of missing/old columns."""
    if not csv_path.exists():
        return []

    today = date.today()
    rows: list[dict] = []
    try:
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                ts_str = (r.get("timestamp") or "").strip()
                if not ts_str:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str)
                except ValueError:
                    continue
                if ts.date() != today:
                    continue
                r["_ts"] = ts
                rows.append(r)
    except Exception as e:
        log.exception("failed reading CSV: %s", e)
        return []

    rows.sort(key=lambda x: x["_ts"])
    return rows


# ---------- Aggregations ----------
def fmt_duration(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def time_per_category(rows: list[dict], interval: int) -> list[tuple[str, int]]:
    """List of (category, seconds) sorted desc, excluding ERROR."""
    counts = Counter(r["category"] for r in rows if r["category"] != "ERROR")
    result = [(cat, n * interval) for cat, n in counts.items()]
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def total_tracked_seconds(cat_breakdown: list[tuple[str, int]]) -> int:
    return sum(s for _, s in cat_breakdown)


def truncate_title(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "\u2026"


# ---------- Menu bar app ----------
class TimeTrackerMenu(rumps.App):
    def __init__(self, cfg: dict):
        # Initial title is set before we have data
        super().__init__("\u23f1", quit_button=None)

        self.cfg = cfg
        self.csv_path: Path = cfg["storage"]["log_csv"]
        self.reports_dir: Path = cfg["storage"]["reports_dir"]
        self.code_dir: Path = Path(__file__).parent.resolve()
        self.capture_interval: int = cfg["capture"]["interval_seconds"]

        alert_cfg = cfg.get("alerts", {})
        self.username: str = alert_cfg.get("username", "User")
        self.distraction_cats: set[str] = set(alert_cfg.get("distraction_categories", []))
        self.alert_threshold: int = int(alert_cfg.get("threshold_minutes", 15)) * 60
        self.repeat_interval_mins: int = int(alert_cfg.get("repeat_interval_minutes", 5))
        self.snooze_duration: int = int(alert_cfg.get("snooze_minutes", 30)) * 60
        self._snoozed_until: datetime | None = None
        self._last_notified_mins: int = 0

        mb = cfg.get("menubar", {})
        self.refresh_interval: int = int(mb.get("refresh_interval_seconds", 60))
        self.stale_threshold: int = int(mb.get("stale_threshold_seconds", 300))
        self.top_n: int = int(mb.get("top_n_categories", 5))
        self.max_title_chars: int = int(mb.get("max_title_chars", 32))

        # ----- Build static menu structure -----
        # Header: today total
        self.today_total_item = rumps.MenuItem("Today (so far): \u2014")

        # Category breakdown placeholders (we update them on refresh)
        self.category_items: list[rumps.MenuItem] = [
            rumps.MenuItem("\u2014") for _ in range(self.top_n)
        ]

        # Status lines
        self.last_capture_item = rumps.MenuItem("Last capture: \u2014")
        self.capture_status_item = rumps.MenuItem("Capture: \u2014")

        # Action items
        self.action_report = rumps.MenuItem("Generate today's report",
                                            callback=self._action_generate_report)
        self.action_refresh = rumps.MenuItem("Refresh now",
                                             callback=self._action_refresh_now)
        self.action_snooze = rumps.MenuItem(
            f"Snooze alerts ({alert_cfg.get('snooze_minutes', 30)}m)",
            callback=self._action_snooze,
        )
        self.action_quit = rumps.MenuItem("Quit", callback=self._action_quit)

        self.menu = [
            self.today_total_item,
            None,  # separator
            *self.category_items,
            None,
            self.last_capture_item,
            self.capture_status_item,
            None,
            self.action_report,
            self.action_refresh,
            self.action_snooze,
            None,
            self.action_quit,
        ]

        # Disable the today_total_item & status lines (display-only)
        for item in (self.today_total_item, self.last_capture_item,
                     self.capture_status_item):
            item.set_callback(None)
        for ci in self.category_items:
            ci.set_callback(None)

        # Lock to avoid concurrent refreshes (timer + menu click)
        self._refresh_lock = threading.Lock()

        # First refresh immediately, then on a timer
        self.timer = rumps.Timer(self._on_timer, self.refresh_interval)
        self.timer.start()
        self._refresh()

    # ---------- Refresh logic ----------
    def _on_timer(self, _sender):
        try:
            self._refresh()
        except Exception:
            log.exception("refresh failed")

    def _refresh(self):
        with self._refresh_lock:
            rows = load_today_rows(self.csv_path)
            now = datetime.now().astimezone()
            cat_breakdown = time_per_category(rows, self.capture_interval)
            total_secs = total_tracked_seconds(cat_breakdown)

            # ----- Title -----
            self.title = self._compute_title(rows, now, cat_breakdown, total_secs)

            # ----- Today total -----
            self.today_total_item.title = f"Today (so far): {fmt_duration(total_secs)}"

            # ----- Top N categories -----
            for i, item in enumerate(self.category_items):
                if i < len(cat_breakdown):
                    cat, secs = cat_breakdown[i]
                    pct = (secs / total_secs * 100) if total_secs else 0
                    item.title = f"  {cat:<22} {fmt_duration(secs):>7}  {pct:4.1f}%"
                else:
                    item.title = "  \u2014"

            # ----- Distraction alert -----
            self._check_distraction_alert(cat_breakdown)

            # ----- Status lines -----
            if rows:
                last_ts = rows[-1]["_ts"]
                age_sec = max(0, int((now - last_ts).total_seconds()))
                self.last_capture_item.title = (
                    f"Last capture: {last_ts.strftime('%H:%M')} "
                    f"({self._human_age(age_sec)} ago)"
                )
                if age_sec > self.stale_threshold:
                    self.capture_status_item.title = "Capture: \u23f8 paused / stale"
                else:
                    self.capture_status_item.title = "Capture: \u25cf running"
            else:
                self.last_capture_item.title = "Last capture: \u2014"
                self.capture_status_item.title = "Capture: no data yet"

    def _check_distraction_alert(self, cat_breakdown: list[tuple[str, int]]):
        now = datetime.now().astimezone()

        if self._snoozed_until:
            if now < self._snoozed_until:
                remaining = int((self._snoozed_until - now).total_seconds() // 60)
                self.action_snooze.title = f"Snoozed ({remaining}m left)"
                return
            else:
                self._snoozed_until = None
                self._last_notified_mins = 0
                self.action_snooze.title = f"Snooze alerts ({self.snooze_duration // 60}m)"

        distraction_secs = sum(s for c, s in cat_breakdown if c in self.distraction_cats)
        mins = distraction_secs // 60

        # Reset if time somehow goes backward (e.g., new day starts)
        if mins < self._last_notified_mins:
            self._last_notified_mins = 0

        if distraction_secs < self.alert_threshold:
            self._last_notified_mins = 0
            return

        # Fire if it's the first time crossing threshold OR if N more mins have passed
        if self._last_notified_mins == 0 or (mins - self._last_notified_mins) >= self.repeat_interval_mins:
            self._last_notified_mins = mins
            
            messages = {
                15: "Distraction alert {name}. You've hit your {mins} mins limit.",
                20: "Hey {name}, you've been slacking for {mins} mins. Get back to work!",
                25: "{name}, {mins} minutes. Come on, let's refocus.",
                30: "Seriously? {name}! {mins} mins wasted. Close that tab NOW.",
                35: "{mins} minutes {name} !!! You are literally stealing from your own future.",
                40: "Bro. {mins} minutes. {name}, your future self is disappointed.",
                60: "AN HOUR OF SLACKING {name}!!! {mins} mins gone. What are you doing with your life 💀",
            }
            # Find the largest milestone you've crossed
            best_key = max([k for k in messages.keys() if k <= mins], default=None)
            if best_key:
                msg = messages[best_key].format(mins=mins, name=self.username)
            else:
                msg = f"Still slacking {self.username} ... {mins} mins gone. Fix it."
            
            log.info("triggering distraction notification (%dm >= threshold)", mins)
            try:
                # Use osascript for 100% reliable notifications
                title = "Study Agent 🚨"
                subtitle = "Distraction alert"
                
                # Add the 'sound name' parameter for a system ping
                script = f'display notification "{msg}" with title "{title}" subtitle "{subtitle}" sound name "Basso"'
                subprocess.run(["osascript", "-e", script], check=True)
                
                # Play a spoken audio alert in the background
                spoken_msg = msg.replace("💀", "").replace("🚨", "")
                subprocess.Popen(["say", spoken_msg])
                
                log.info("notification and spoken sound sent")
            except Exception as e:
                log.error("failed to send notification: %s", e)

    def _action_snooze(self, _sender):
        self._snoozed_until = datetime.now().astimezone() + timedelta(seconds=self.snooze_duration)
        self._last_notified_mins = 0
        log.info("alerts snoozed until %s", self._snoozed_until)

    def _compute_title(
        self,
        rows: list[dict],
        now: datetime,
        cat_breakdown: list[tuple[str, int]],
        total_secs: int,
    ) -> str:
        if not rows:
            return "\u23f1 no data"

        last_ts = rows[-1]["_ts"]
        age_sec = (now - last_ts).total_seconds()

        if age_sec > self.stale_threshold:
            mins = int(age_sec // 60)
            return truncate_title(f"\u23f8 Paused ({mins}m ago)", self.max_title_chars)

        last_valid = next((r for r in reversed(rows) if r["category"] != "ERROR"), None)
        if last_valid is None:
            return "\u23f1 ERROR"

        current_cat = last_valid["category"]
        cat_secs = next((s for c, s in cat_breakdown if c == current_cat), 0)
        pct = (cat_secs / total_secs * 100) if total_secs else 0

        return truncate_title(
            f"\u23f1 {current_cat} \u00b7 {fmt_duration(cat_secs)} \u00b7 {pct:.0f}%",
            self.max_title_chars,
        )

    @staticmethod
    def _human_age(seconds: int) -> str:
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m"
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"

    # ---------- Actions ----------
    def _action_generate_report(self, _sender):
        """Run report.py for today, then open the resulting markdown file."""
        log.info("generate report clicked")
        try:
            # Run report.py in our code dir using same Python we're running
            result = subprocess.run(
                [sys.executable, str(self.code_dir / "report.py"), "today"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                log.error("report.py failed: %s\n%s", result.stdout, result.stderr)
                rumps.alert("Report failed",
                            "Could not generate report. See /tmp/timetracker_menubar.log.")
                return

            today_md = self.reports_dir / f"{date.today().isoformat()}.md"
            if today_md.exists():
                # Open via macOS `open` so user's default .md viewer (or VS Code) opens it
                subprocess.run(["open", str(today_md)], check=False)
            else:
                rumps.alert("Report generated", f"Report file not found at {today_md}")
        except Exception as e:
            log.exception("error generating report")
            rumps.alert("Report error", str(e))

    def _action_refresh_now(self, _sender):
        log.info("refresh-now clicked")
        try:
            self._refresh()
        except Exception:
            log.exception("manual refresh failed")

    def _action_quit(self, _sender):
        log.info("quit clicked")
        rumps.quit_application()


# ---------- Main ----------
def main():
    config_path = Path(__file__).parent / "config.yaml"
    cfg = load_config(config_path)
    log.info("starting menubar | csv=%s", cfg["storage"]["log_csv"])
    TimeTrackerMenu(cfg).run()


if __name__ == "__main__":
    main()