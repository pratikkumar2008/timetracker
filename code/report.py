"""
report.py — Daily markdown report generator.

Reads log.csv for a given date, aggregates time per category and per app,
groups consecutive captures into activity blocks, samples window titles,
and writes a markdown summary to data/reports/YYYY-MM-DD.md.

Usage:
    python report.py              # today
    python report.py yesterday    # yesterday
    python report.py 2026-04-25   # specific date (ISO format)
"""

import csv
import logging
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("report")


# ---------- Config ----------
def load_config(path: Path) -> dict:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    base = path.parent.resolve()
    cfg["storage"]["log_csv"] = (base / cfg["storage"]["log_csv"]).resolve()
    cfg["storage"]["reports_dir"] = (base / cfg["storage"]["reports_dir"]).resolve()
    cfg["capture"]["interval_seconds"] = int(cfg["capture"]["interval_seconds"])
    return cfg


# ---------- Date parsing ----------
def parse_target_date(arg: str | None) -> date:
    """today / yesterday / YYYY-MM-DD -> date."""
    if arg is None or arg.lower() == "today":
        return date.today()
    if arg.lower() == "yesterday":
        return date.today() - timedelta(days=1)
    try:
        return datetime.strptime(arg, "%Y-%m-%d").date()
    except ValueError as e:
        raise SystemExit(f"Invalid date {arg!r}. Use 'today', 'yesterday', or YYYY-MM-DD.") from e


# ---------- CSV loading ----------
def load_rows_for_date(csv_path: Path, target: date) -> list[dict]:
    """Return rows whose timestamp date matches target, in chronological order.
    Tolerant of older rows missing newer columns."""
    if not csv_path.exists():
        return []

    rows = []
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
            if ts.date() != target:
                continue

            # Backfill missing columns with sensible defaults
            r.setdefault("app_name", "Unknown")
            r.setdefault("window_title", "Unknown")
            r.setdefault("model", "")
            r["app_name"] = (r["app_name"] or "Unknown").strip()
            r["window_title"] = (r["window_title"] or "Unknown").strip()

            r["_ts"] = ts
            rows.append(r)

    rows.sort(key=lambda x: x["_ts"])
    return rows


# ---------- Aggregation helpers ----------
def fmt_duration(seconds: int) -> str:
    """3725s -> '1h 02m'. Drops hours if zero."""
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def time_by_key(rows: list[dict], key: str, interval: int) -> list[tuple[str, int, int]]:
    """Generic: bucket rows by `key`, return list of (value, seconds, count) sorted by seconds desc."""
    counts = Counter((r.get(key) or "Unknown") for r in rows)
    result = [(k, n * interval, n) for k, n in counts.items()]
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def build_timeline_blocks(rows: list[dict], interval: int) -> list[dict]:
    """
    Group consecutive same-(category, app) captures into blocks.
    A gap larger than 2x interval breaks the block (e.g. laptop sleep).
    """
    if not rows:
        return []

    def block_from(r: dict) -> dict:
        return {
            "category": r["category"],
            "app_name": r.get("app_name", "Unknown"),
            "start": r["_ts"],
            "end": r["_ts"],
            "count": 1,
            "window_titles": [r.get("window_title", "")],
            "descriptions": [r.get("description", "")],
        }

    blocks = []
    cur = block_from(rows[0])

    gap_threshold = timedelta(seconds=interval * 2)

    for r in rows[1:]:
        same_cat = r["category"] == cur["category"]
        same_app = r.get("app_name", "Unknown") == cur["app_name"]
        gap = r["_ts"] - cur["end"]
        within_window = gap <= gap_threshold

        if same_cat and same_app and within_window:
            cur["end"] = r["_ts"]
            cur["count"] += 1
            cur["window_titles"].append(r.get("window_title", ""))
            cur["descriptions"].append(r.get("description", ""))
        else:
            blocks.append(cur)
            cur = block_from(r)

    blocks.append(cur)

    for b in blocks:
        b["seconds"] = b["count"] * interval
    return blocks


def top_titles_per_category(rows: list[dict], top_n: int = 5) -> dict[str, list[tuple[str, int]]]:
    """For each category, return up to top_n most common window_titles with counts."""
    by_cat: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        title = (r.get("window_title") or "").strip()
        if title and title.lower() != "unknown":
            by_cat[r["category"]][title] += 1
    return {cat: c.most_common(top_n) for cat, c in by_cat.items()}


# ---------- Markdown rendering ----------
def render_markdown(target: date, rows: list[dict], interval: int) -> str:
    if not rows:
        return f"# Time tracker - {target.isoformat()}\n\n_No captures recorded for this date._\n"

    total_captures = len(rows)
    error_captures = sum(1 for r in rows if r["category"] == "ERROR")
    valid_rows = [r for r in rows if r["category"] != "ERROR"]
    valid_captures = len(valid_rows)
    total_seconds = valid_captures * interval

    cat_breakdown = time_by_key(valid_rows, "category", interval)
    app_breakdown = time_by_key(valid_rows, "app_name", interval)
    blocks = build_timeline_blocks(rows, interval)
    top_titles = top_titles_per_category(valid_rows)
    models_used = sorted({(r.get("model") or "").strip() for r in rows if (r.get("model") or "").strip()})

    first_ts = rows[0]["_ts"].strftime("%H:%M")
    last_ts = rows[-1]["_ts"].strftime("%H:%M")

    out = []
    out.append(f"# Time tracker - {target.isoformat()}")
    out.append("")
    out.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    out.append("")

    # ----- Summary -----
    out.append("## Summary")
    out.append("")
    out.append(f"- **Tracked window**: {first_ts} - {last_ts}")
    out.append(f"- **Total tracked time**: {fmt_duration(total_seconds)} ({valid_captures} captures x {interval}s)")
    out.append(f"- **Errors**: {error_captures} capture(s)")
    out.append(f"- **Categories observed**: {len(cat_breakdown)}")
    out.append(f"- **Apps observed**: {len(app_breakdown)}")
    if models_used:
        out.append(f"- **Model(s)**: {', '.join(models_used)}")
    out.append("")

    # ----- Time per category -----
    out.append("## Time per category")
    out.append("")
    if cat_breakdown:
        out.append("| Category | Time | % | Captures |")
        out.append("|---|---:|---:|---:|")
        for cat, secs, n in cat_breakdown:
            pct = (secs / total_seconds * 100) if total_seconds else 0
            out.append(f"| {cat} | {fmt_duration(secs)} | {pct:.1f}% | {n} |")
    else:
        out.append("_No valid categories recorded._")
    out.append("")

    # ----- Time per app -----
    out.append("## Time per app")
    out.append("")
    if app_breakdown:
        out.append("| App | Time | % | Captures |")
        out.append("|---|---:|---:|---:|")
        for app, secs, n in app_breakdown:
            pct = (secs / total_seconds * 100) if total_seconds else 0
            out.append(f"| {app} | {fmt_duration(secs)} | {pct:.1f}% | {n} |")
    else:
        out.append("_No app data available._")
    out.append("")

    # ----- Timeline -----
    out.append("## Timeline")
    out.append("")
    if blocks:
        for b in blocks:
            start = b["start"].strftime("%H:%M")
            end = b["end"].strftime("%H:%M")
            dur = fmt_duration(b["seconds"])
            title_counter = Counter(t for t in b["window_titles"] if t and t.lower() != "unknown")
            sample_title = title_counter.most_common(1)[0][0] if title_counter else ""
            label = f"`{b['category']}` * {b['app_name']}"
            line = f"- **{start} - {end}** ({dur}) * {label}"
            if sample_title:
                line += f" - _{sample_title}_"
            out.append(line)
    else:
        out.append("_No timeline blocks._")
    out.append("")

    # ----- Top window titles per category -----
    out.append("## Top window titles per category")
    out.append("")
    has_any_titles = any(top_titles.get(cat) for cat, _, _ in cat_breakdown)
    if has_any_titles:
        for cat, _, _ in cat_breakdown:
            titles = top_titles.get(cat, [])
            if not titles:
                continue
            out.append(f"### {cat}")
            for title, count in titles:
                out.append(f"- ({count}x) {title}")
            out.append("")
    else:
        out.append("_No window titles available._")
        out.append("")

    return "\n".join(out)


# ---------- Main ----------
def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    target = parse_target_date(arg)

    config_path = Path(__file__).parent / "config.yaml"
    cfg = load_config(config_path)

    csv_path = Path(cfg["storage"]["log_csv"])
    reports_dir = Path(cfg["storage"]["reports_dir"])
    interval = cfg["capture"]["interval_seconds"]

    log.info("generating report for %s", target.isoformat())
    log.info("reading %s", csv_path)

    rows = load_rows_for_date(csv_path, target)
    log.info("found %d captures for %s", len(rows), target.isoformat())

    md = render_markdown(target, rows, interval)

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"{target.isoformat()}.md"
    out_path.write_text(md)

    log.info("wrote %s (%d bytes)", out_path, out_path.stat().st_size)
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
