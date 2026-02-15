# AIOps Sentinel — External Log Ingestion

This repo supports ingesting external logs (uploaded files or pasted lines), runs preprocessing on every dataset, and appends structured rows for training and provenance.

## Features
- Clean and robust preprocessing pipeline using Drain3 and sanitization
- Masking of sensitive data (IP, email, tokens, usernames)
- Saves structured provenance CSV to `data/logs/<source>_structured_<timestamp>.csv`
- Appends structured rows to `structured_logs.csv` so training uses all available data
- Streamlit UI: upload/paste external logs from the web dashboard (sidebar). The UI now collects optional `Uploader` metadata and offers CSV preview, download, and an ingestion history with per-ingest metadata.
- CLI: `python scripts/ingest_external.py path/to/external.log` (optional `--uploader NAME` to record uploader metadata)

## How preprocessing handles issues
- Removes noise lines (heartbeat, cron, etc.)
- Ensures `Raw_Log`, `Event_ID`, `Event_Template`, and `Timestamp` exist
- Fills missing `Event_ID`/`Event_Template` with `unknown`
- Parses or fills missing timestamps with current UTC time
- Removes duplicate entries and fills feature matrix NaNs with zeros (integers)

## Usage
- From CLI:
  - `python scripts/ingest_external.py path/to/external.log`
- From dashboard:
  - Open the app with `streamlit run app.py` and use the **External Log Ingest** in the sidebar

## Implementation notes
- `ingestion/external_ingest.py` is the new helper for external sources.
- `preprocessing/log_processor.py` adds `sanitize_structured_df` to avoid missing-value issues.
