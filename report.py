"""
report.py — Daily markdown report generator.

Reads log.csv for a given date, aggregates time per category,
writes a markdown summary to data/reports/YYYY-MM-DD.md.

Run: python report.py             # today
     python report.py 2026-04-25  # specific date
"""
