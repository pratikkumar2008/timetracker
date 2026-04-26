"""
report.py — Daily markdown report generator.

Reads log.csv for a given date, aggregates time per category,
groups consecutive captures into activity blocks, samples descriptions,
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
    """today / yesterday / YYYY-MM-DD → date."""
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
    """Return rows whose timestamp date matches target, in chronological order."""
    if not csv_path.exists():
        return []

    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ts_str = r.get("timestamp", "").strip()
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue
            if ts.date() != target:
                continue
            r["_ts"] = ts
            rows.append(r)

    rows.sort(key=lambda x: x["_ts"])
    return rows


# ---------- Aggregation ----------
def fmt_duration(seconds: int) -> str:
    """3725s -> '1h 02m'. Drops hours if zero."""
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def time_per_category(rows: list[dict], interval: int) -> list[tuple[str, int, int]]:
    """
    Each capture represents `interval` seconds of activity.
    Returns list of (category, seconds, count) sorted by seconds desc.
    """
    counts = Counter(r["category"] for r in rows)
    result = [(cat, n * interval, n) for cat, n in counts.items()]
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def build_timeline_blocks(rows: list[dict], interval: int) -> list[dict]:
    """
    Group consecutive same-category captures into blocks.
    A gap larger than 2× interval breaks the block (e.g. laptop sleep).
    """
    if not rows:
        return []

    blocks = []
    cur = {
        "category": rows[0]["category"],
        "start": rows[0]["_ts"],
        "end": rows[0]["_ts"],
        "count": 1,
        "descriptions": [rows[0].get("description", "")],
    }

    gap_threshold = timedelta(seconds=interval * 2)

    for r in rows[1:]:
        same_cat = r["category"] == cur["category"]
        gap = r["_ts"] - cur["end"]
        within_window = gap <= gap_threshold

        if same_cat and within_window:
            cur["end"] = r["_ts"]
            cur["count"] += 1
            cur["descriptions"].append(r.get("description", ""))
        else:
            blocks.append(cur)
            cur = {
                "category": r["category"],
                "start": r["_ts"],
                "end": r["_ts"],
                "count": 1,
                "descriptions": [r.get("description", "")],
            }

    blocks.append(cur)

    # Compute duration. A block of N captures spans (N * interval) seconds of represented activity.
    for b in blocks:
        b["seconds"] = b["count"] * interval

    return blocks


def top_descriptions_per_category(rows: list[dict], top_n: int = 3) -> dict[str, list[tuple[str, int]]]:
    """For each category, return up to top_n most common descriptions with counts."""
    by_cat: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        desc = (r.get("description") or "").strip()
        if desc:
            by_cat[r["category"]][desc] += 1
    return {cat: c.most_common(top_n) for cat, c in by_cat.items()}


# ---------- Markdown rendering ----------
def render_markdown(target: date, rows: list[dict], interval: int) -> str:
    if not rows:
        return f"# Time tracker — {target.isoformat()}\n\n_No captures recorded for this date._\n"

    total_captures = len(rows)
    error_captures = sum(1 for r in rows if r["category"] == "ERROR")
    valid_captures = total_captures - error_captures
    total_seconds = valid_captures * interval

    cat_breakdown = time_per_category([r for r in rows if r["category"] != "ERROR"], interval)
    blocks = build_timeline_blocks(rows, interval)
    top_desc = top_descriptions_per_category([r for r in rows if r["category"] != "ERROR"])

    first_ts = rows[0]["_ts"].strftime("%H:%M")
    last_ts = rows[-1]["_ts"].strftime("%H:%M")

    out = []
    out.append(f"# Time tracker — {target.isoformat()}")
    out.append("")
    out.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    out.append("")

    # ----- Summary -----
    out.append("## Summary")
    out.append("")
    out.append(f"- **Tracked window**: {first_ts} – {last_ts}")
    out.append(f"- **Total tracked time**: {fmt_duration(total_seconds)} ({valid_captures} captures × {interval}s)")
    out.append(f"- **Errors**: {error_captures} capture(s)")
    out.append(f"- **Categories observed**: {len(cat_breakdown)}")
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

    # ----- Timeline -----
    out.append("## Timeline")
    out.append("")
    if blocks:
        for b in blocks:
            start = b["start"].strftime("%H:%M")
            end = b["end"].strftime("%H:%M")
            dur = fmt_duration(b["seconds"])
            # Use the most common description in the block as a subtitle
            desc_counter = Counter(d for d in b["descriptions"] if d)
            sample_desc = desc_counter.most_common(1)[0][0] if desc_counter else ""
            line = f"- **{start} – {end}** ({dur}) · `{b['category']}`"
            if sample_desc:
                line += f" — {sample_desc}"
            out.append(line)
    else:
        out.append("_No timeline blocks._")
    out.append("")

    # ----- Top descriptions per category -----
    out.append("## Top descriptions per category")
    out.append("")
    if top_desc:
        for cat, _, _ in cat_breakdown:
            descs = top_desc.get(cat, [])
            if not descs:
                continue
            out.append(f"### {cat}")
            for desc, count in descs:
                out.append(f"- ({count}×) {desc}")
            out.append("")
    else:
        out.append("_No descriptions available._")
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
    print(f"\n✓ Report: {out_path}")


if __name__ == "__main__":
    main()