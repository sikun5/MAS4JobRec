import json
import subprocess
import sys
import time
from pathlib import Path

run=Path("runtime/runs/v2-phase3-20261121")
def wait(stage):
    receipt=run/"jobs"/stage/"status.json"
    while True:
        if receipt.exists():
            state=json.loads(receipt.read_text())["state"]
            if state!="running":return state
        time.sleep(3)

if wait("teacher-extra")!="done":raise RuntimeError("supplemental teacher failed")
demos=json.loads((run/"pilot/teacher_supplement_demos.json").read_text())
print("additional typical demonstrations",len(demos),flush=True)
if not demos:raise RuntimeError("no retained typical demonstrations")
for mode in ["lora","full"]:
    stage=mode+"-matched"
    subprocess.run([sys.executable,"code/launch_stage.py",str(run),stage,"v2","--",sys.executable,"-u","code/phase3_probe.py",mode],check=True)
    print(mode, wait(stage),flush=True)
