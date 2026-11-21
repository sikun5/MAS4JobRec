"""Launch a restartable named stage on the experiment host (Linux)."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def save(path, record):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("stage")
    parser.add_argument("revision")
    parser.add_argument("command", nargs="+")
    args = parser.parse_args()
    if not args.stage.replace("-", "").replace("_", "").isalnum():
        parser.error("invalid stage name")
    folder = args.run.resolve() / "jobs" / args.stage
    folder.mkdir(parents=True, exist_ok=True)
    state = folder / "status.json"
    if "IOTA_STAGE_LOCK_FD" in os.environ:
        lock_fd = int(os.environ.pop("IOTA_STAGE_LOCK_FD"))
        started = time.time()
        record = dict(pid=os.getpid(), started_epoch=started, revision=args.revision,
                      command=args.command, state="running")
        save(state, record)
        rc = 1
        try:
            rc = subprocess.run(args.command).returncode
        finally:
            record.update(exit_code=rc, ended_epoch=time.time(), state="done" if rc == 0 else "failed")
            save(state, record)
            os.close(lock_fd)
        return rc
    with (folder / ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("already running", folder)
            return 0
        if state.exists():
            previous = json.loads(state.read_text())
            if (previous["revision"], previous["command"]) != (args.revision, args.command):
                parser.error("changed configuration requires a new stage name or run ID")
            if previous["state"] == "done":
                print("reused completed stage", folder)
                return 0
        env = dict(os.environ, IOTA_STAGE_LOCK_FD=str(lock.fileno()))
        with (folder / "stage.log").open("ab") as log:
            proc = subprocess.Popen([sys.executable, __file__, *sys.argv[1:]],
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                    pass_fds=(lock.fileno(),), env=env, start_new_session=True)
        print("launched", proc.pid, folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
