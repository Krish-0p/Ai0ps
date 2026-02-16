from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import sys
import time
import signal

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.monitor_worker import get_monitor_worker

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

# --- MODE CONTROL VARIABLES ---
current_process = None
current_mode = "stopped" 
LOG_FILE = "live_server.log" # Ensure this matches your other scripts

def stop_current_logger():
    """Aggressively kills the currently running log script"""
    global current_process
    if current_process:
        try:
            print(f"⚠️ Killing process {current_process.pid}...")
            if sys.platform == "win32":
                # Windows: Force kill the entire process tree (/T)
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(current_process.pid)], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # Linux/Mac
                os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
            
            current_process.wait(timeout=1) # Wait for it to actually die
        except Exception as e:
            print(f"Error stopping process: {e}")
        finally:
            current_process = None

@app.post("/api/control/switch-mode/{mode}")
async def switch_mode(mode: str):
    global current_process, current_mode
    
    print(f"🔄 Switching to mode: {mode}")
    
    # 1. STOP everything first
    stop_current_logger()
    
    # 2. CLEAR the log file so the graph resets visually
    # (Optional: remove this block if you want to keep old history)
    with open(LOG_FILE, 'w') as f:
        f.write("") 

    # 3. SELECT the new script
    script_name = ""
    if mode == "real":
        script_name = "real_log_shipper.py"
    elif mode == "fake":
        script_name = "log_generator.py" # Make sure this matches your file name!
    else:
        current_mode = "stopped"
        return {"status": "Stopped logging"}

    # 4. START the new script
    try:
        if sys.platform == "win32":
            current_process = subprocess.Popen(
                ["python", script_name], 
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            current_process = subprocess.Popen(
                ["python", script_name], 
                preexec_fn=os.setsid
            )
            
        current_mode = mode
        return {"status": f"Switched to {mode} mode"}
    except Exception as e:
        print(f"❌ Failed to start script: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/control/inject-attack")
async def inject_attack():
    """Manually writes an attack log to the file"""
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        attack_log = f"{timestamp} ERROR: CRITICAL SQL Injection attempt detected from IP 192.168.0.666\n"
        
        with open(LOG_FILE, "a") as f:
            f.write(attack_log)
            
        return {"status": "Attack Injected", "log": attack_log}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/live-data")
async def get_live_data():
    latest = worker.latest()
    logs = worker.get_logs(limit=20)
    history = worker.get_history()
    
    # Format for Recharts
    chart_data = [{"time": i, "score": score} for i, score in enumerate(history)]

    return {
        "server_status": "Online",
        "mode": current_mode,
        "ai_diagnosis": "CRITICAL" if latest['prediction'] == -1 else "Healthy",
        "anomaly_score": float(latest['score']),
        "is_anomaly": bool(latest['prediction'] == -1),
        "logs": logs,
        "history": chart_data
    }