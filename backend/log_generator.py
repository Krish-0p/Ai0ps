import time
import random
import os

# Define file to write to
LOG_FILE = "live_server.log"

# Define templates (Must match what we trained on roughly)
NORMAL_LOGS = [
    "INFO: User login successful from IP 192.168.1.{}", 
    "INFO: Data packet sent to block_id_{} size 500",
    "WARN: Connection delay detected at node_{}",
    "INFO: Health check passed for service_{}"
]

ATTACK_LOGS = [
    "ERROR: Buffer Overflow exception in memory block {}",
    "FATAL: Database connection refused from IP 0.0.0.0",
    "CRITICAL: Unidentified protocol packet received",
    "SECURITY: Multiple failed login attempts (Brute Force)"
]

def write_log(text):
    with open(LOG_FILE, "a") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} {text}\n")

print(f"--- SERVER STARTED ---")
print(f"Writing logs to: {LOG_FILE}")
print("Press Ctrl+C to stop.")

# Create file if not exists
if not os.path.exists(LOG_FILE):
    open(LOG_FILE, 'w').close()

try:
    while True:
        # 1. Decide: Normal or Attack?
        # 95% chance of normal, unless we force it
        if random.random() > 0.95:
            log_template = random.choice(ATTACK_LOGS)
            log_text = log_template.format(random.randint(100, 999))
            print(f"⚠️  Generating ERROR: {log_text}")
        else:
            log_template = random.choice(NORMAL_LOGS)
            log_text = log_template.format(random.randint(1, 255))
            print(f"✅ Generating Normal: {log_text}")
        
        write_log(log_text)
        
        # Speed of logs (Fast = 0.5s, Slow = 2s)
        time.sleep(1.0) 

except KeyboardInterrupt:
    print("\nServer Stopped.")