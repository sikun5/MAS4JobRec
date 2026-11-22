from experiment_paths import phase, lock_sources
import json,time
from pathlib import Path
from phase7_seal import seal
def main():
    status = phase(7) / 'jobs/handoff/status.json'
    while not status.exists() or json.loads(status.read_text())['state'] == 'running':
        time.sleep(45)
    seal()


if __name__ == "__main__":
    lock_sources(7)
    main()
