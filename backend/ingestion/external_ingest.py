"""
External ingestion helper
- Accepts external logs (list of strings), runs masking/normalization
- Runs preprocessing pipeline (LogProcessor)
- Saves structured rows to `structured_logs.csv` and also to `data/logs` for provenance
- Returns processed feature matrix + metadata
"""
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any
import pandas as pd
import os

from ingestion.log_ingester import LogIngester
from preprocessing.log_processor import LogProcessor
from config.settings import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger('external_ingest')

STRUCTURED_CSV = Path("structured_logs.csv")
PROVENANCE_DIR = Path(DATA_DIR) / "logs"
PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)


def process_external_logs(lines: List[str], source: str = "external", uploader: str = None) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Process a list of raw log lines coming from an external source.

    Steps:
    - Mask and normalize using LogIngester
    - Preprocess with LogProcessor (noise filter, template extraction, windows, features)
    - Save structured rows to `structured_logs.csv` (append)
    - Save a provenance CSV to `data/logs/<source>_structured_<timestamp>.csv`

    Returns: (feature_matrix, feature_columns, metadata)
    """
    try:
        ingester = LogIngester()
        processor = LogProcessor()

        # 1. Normalize + mask
        processed = []
        for line in lines:
            if not isinstance(line, str):
                line = str(line)
            line = line.strip()
            if not line:
                continue
            line = ingester.normalize_timestamp(line)
            line = ingester.mask_sensitive_data(line)
            processed.append(line)

        if not processed:
            logger.warning("No valid lines provided to external ingest")
            return pd.DataFrame(), [], {"processed_lines": 0}

        # 2. Preprocess
        feature_matrix, feature_columns = processor.process_logs(processed)

        # 3. Save structured outputs for provenance and training
        # Re-run extraction to get structured rows (we want the structured_df)
        structured_df = processor.extract_templates(processed)
        structured_df = processor.sanitize_structured_df(structured_df)

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        provenance_name = f"{source}_structured_{timestamp}.csv"
        provenance_path = PROVENANCE_DIR / provenance_name
        structured_df.to_csv(provenance_path, index=False)

        # Save companion metadata JSON
        meta = {
            'source': source,
            'uploader': uploader,
            'ingest_time': datetime.utcnow().isoformat() + 'Z',
            'rows': len(structured_df)
        }
        try:
            import json
            meta_path = provenance_path.with_suffix('.meta.json')
            with open(meta_path, 'w', encoding='utf-8') as mf:
                json.dump(meta, mf)
        except Exception as e:
            logger.warning(f"Failed to write metadata for {provenance_path}: {e}")

        # 4. Append to main structured CSV for training (if exists)
        try:
            def _read_structured_safe(path):
                # Robustly parse existing structured file that may have variable columns
                recs = []
                if not path.exists():
                    return None
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    hdr = f.readline()
                    for line in f:
                        line = line.rstrip('\n')
                        if not line.strip():
                            continue
                        parts = line.rsplit(',', 3)
                        if len(parts) == 4:
                            raw, event_id, event_template, ts = parts
                        elif len(parts) == 3:
                            raw, event_id, event_template = parts
                            ts = ''
                        else:
                            raw = parts[0]
                            event_id = parts[1] if len(parts) > 1 else ''
                            event_template = parts[2] if len(parts) > 2 else ''
                            ts = ''
                        recs.append({'Raw_Log': raw, 'Event_ID': event_id, 'Event_Template': event_template, 'Timestamp': ts})
                import pandas as _pd
                return _pd.DataFrame(recs)

            def _append_structured(df_to_append):
                # Ensure schema
                for c in ['Raw_Log', 'Event_ID', 'Event_Template', 'Timestamp']:
                    if c not in df_to_append.columns:
                        df_to_append[c] = ''
                df_to_append = df_to_append[['Raw_Log', 'Event_ID', 'Event_Template', 'Timestamp']]

                if STRUCTURED_CSV.exists():
                    existing = _read_structured_safe(STRUCTURED_CSV)
                    if existing is None:
                        # Fallback to writing only new
                        df_to_append.to_csv(STRUCTURED_CSV, mode='a', header=False, index=False)
                        return
                    combined = pd.concat([existing, df_to_append], ignore_index=True, sort=False)
                    combined = combined.drop_duplicates()
                    combined.to_csv(STRUCTURED_CSV, index=False)
                else:
                    df_to_append.to_csv(STRUCTURED_CSV, index=False)

            _append_structured(structured_df)
        except Exception as e:
            logger.error(f"Failed to append to {STRUCTURED_CSV}: {e}")

        metadata = {
            "processed_lines": len(processed),
            "saved_provenance": str(provenance_path) if provenance_path else None,
            "meta_path": str(meta_path) if 'meta_path' in locals() else None,
            "appended_to": str(STRUCTURED_CSV)
        }

        logger.info(f"External ingest complete: {metadata}")
        return feature_matrix, feature_columns, metadata

    except Exception as e:
        logger.error(f"External ingest failed: {e}")
        return pd.DataFrame(), [], {"error": str(e)}


def list_provenance_files(limit: int = 20):
    """Return a list of recent provenance CSV files with metadata"""
    files = list(PROVENANCE_DIR.glob("*_structured_*.csv"))
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for p in files:
        try:
            mtime = p.stat().st_mtime
            # Try to read a small sample of the file to get row count
            try:
                df = pd.read_csv(p)
                rows = len(df)
            except Exception:
                rows = None

            # Try to read companion metadata JSON
            meta = None
            try:
                import json
                meta_path = p.with_suffix('.meta.json')
                if meta_path.exists():
                    with open(meta_path, 'r', encoding='utf-8') as mf:
                        meta = json.load(mf)
            except Exception:
                meta = None

            out.append({
                "name": p.name,
                "path": str(p),
                "mtime": mtime,
                "rows": rows,
                "meta": meta
            })
        except Exception:
            continue
    return out
