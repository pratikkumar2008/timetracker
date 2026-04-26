# timetracker

Local-only time tracker for macOS using a vision LLM (Gemma 4 26B via Ollama).

Captures a screenshot every 60s, classifies it into one of 12 fixed categories,
and surfaces the data three ways: append-only CSV log, daily markdown reports,
and a live menu bar app.

All processing is on-device. Nothing leaves the Mac.

## Layout

    timetracker/
    ├── code/   ← this repo (Python source)
    └── data/   ← screenshots, log.csv, daily reports (git-ignored)

## Setup

    conda activate myenv
    pip install -r requirements.txt
    # Make sure ollama is running and the configured model is pulled:
    ollama pull gemma4:26b   # or whatever tag your config.yaml points to

## Run

    python capture.py    # main loop
    python report.py     # generate today's markdown report
    python menubar.py    # menu bar surface (run separately)

## Config

Edit `config.yaml` to change capture interval, model tag, or taxonomy.
