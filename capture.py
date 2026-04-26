"""
capture.py — Screenshot capture + analysis loop.

Captures a screenshot every N seconds, sends to local Ollama VLM for
classification, appends result to log.csv. Designed to run continuously.

Run: python capture.py
"""
