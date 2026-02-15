from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

   # Fix path to allow imports from sibling folders
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.monitor_worker import get_monitor_worker
from ingestion.external_ingest import process_external_logs

app = FastAPI()

   # Allow React to talk to Python
app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )

worker = get_monitor_worker()
worker.start()

@app.get("/api/live-data")
async def get_live_data():
       latest = worker.latest()
       logs = worker.get_logs(limit=20)
       history = worker.get_history()
       
       # Convert history to list of dicts for React
       chart_data = []
       if hasattr(history, 'tolist'):
           history = history.tolist()
       
       for i, score in enumerate(history):
           chart_data.append({"time": i, "score": score})

       return {
           "server_status": "Online",
           "ai_diagnosis": "CRITICAL" if latest['prediction'] == -1 else "Healthy",
           "anomaly_score": float(latest['score']),
           "is_anomaly": bool(latest['prediction'] == -1),
           "log_volume": len(logs),
           "logs": logs,
           "history": chart_data
       }

@app.post("/api/control/inject-attack")
async def inject_attack():
       worker.inject_attack()
       return {"status": "Attack Injected"}

@app.post("/api/ingest/upload")
async def upload_logs(file: UploadFile = File(...)):
       content = (await file.read()).decode("utf-8")
       lines = content.splitlines()
       fm, cols, meta = process_external_logs(lines, source='react_upload')
       return {"lines_processed": meta.get('processed_lines', 0)}
