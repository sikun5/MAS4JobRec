import json,time
from pathlib import Path
from seal import seal
status=Path(__file__).resolve().parents[1]/'jobs/handoff/status.json'
while json.loads(status.read_text())['state']=='running':
    time.sleep(45)
seal()
