# Time Tracker ⏳

A fully local, privacy-first time tracking and distraction management system for macOS. It uses local Vision Language Models (like `gemma4:26b` via Ollama) to automatically classify screenshots of your desktop into defined behavior categories, giving you a detailed breakdown of your day without sending any data to the cloud.

## System Architecture

The project is broken into three separate, decoupled Python scripts to ensure performance, stability, and modularity:

### 1. `capture.py` (The Engine)
This is the background worker. It runs in an infinite loop and does all the heavy lifting:
- Takes a screenshot of your primary monitor at intervals defined in `config.yaml`.
- Passes the screenshot to a local Ollama VLM with a strict system prompt to determine exactly what you are doing.
- Gracefully handles edge cases (locked screen, blank desktop, timeouts).
- Appends the resulting JSON (category, active app, window title, reasoning) as a new row in `data/log.csv`.

**Usage:** `python code/capture.py`

### 2. `menubar.py` (The Interface & Alerts)
A lightweight macOS menu bar app (built with `rumps`) that gives you a live dashboard of your day.
- **Read-only:** It does *not* capture screenshots or call the LLM. It simply reads `data/log.csv` every 60 seconds.
- **Dashboard:** Clicking the menu bar icon shows your total tracked time and a live breakdown of your top 5 categories for the day.
- **Distraction Alerts:** If you exceed the configured threshold (e.g., 5 minutes) on a "distraction" category (like `Doomscroll` or `Entertainment`), it will immediately trigger a native macOS banner notification and a spoken audio alert to snap you out of it.
- **Actions:** Allows you to quickly generate daily reports or snooze alerts directly from the menu bar.

**Usage:** `python code/menubar.py` (Can be run alongside `capture.py`)

### 3. `report.py` (The Analyst)
A data aggregator that reads the raw `log.csv` and compiles a beautiful Markdown summary of your day.
- **Timeline:** Groups consecutive rows of the same activity into continuous "time blocks" (e.g., "14:00 - 15:30: JobHunt").
- **App Breakdown:** Calculates exactly how much time you spent in specific applications (like VS Code vs Chrome).
- **Deep Insights:** Extracts the top window titles for each category, letting you see exactly *which* YouTube videos or *which* websites consumed your time.

**Usage:** `python code/report.py` (defaults to today) or `python code/report.py 2026-04-25`

## Getting Started
1. Install Ollama and pull your desired model (e.g., `ollama pull gemma4:26b`).
2. Update the `code/config.yaml` with your preferred model and capture intervals.
3. Run `capture.py` in the background.
4. Run `menubar.py` to keep track of your progress!
