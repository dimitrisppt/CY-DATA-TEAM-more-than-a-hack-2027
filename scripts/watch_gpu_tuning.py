"""Single-owner detached supervisor with a persistent wall-clock deadline."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from possession.storage import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, required=True)
    args = parser.parse_args()
    with (args.output / ".supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = [sys.executable, "-u", "-m", "experiments.gpu_tuning", "--repo", str(args.repo),
                   "--output", str(args.output), "--hours", str(args.hours)]
        started = time.time()
        with (args.output / "stdout.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        write_json(args.output / "supervisor.json", {"pid": os.getpid(), "child_pid": child.pid,
                   "started_unix": started, "command": command})
        deadline = started + args.hours * 3600
        stopped = False
        while child.poll() is None:
            clock = args.output / "clock.json"
            if clock.exists():
                deadline = json.loads(clock.read_text())["payload"]["deadline_unix"]
            if time.time() > deadline + 20 and not stopped:
                child.send_signal(signal.SIGTERM)
                stopped = True
            if time.time() > deadline + 140:
                child.kill()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
        write_json(args.output / "exit.json", {"child_pid": child.pid, "returncode": child.returncode,
                   "deadline_unix": deadline, "finished_unix": time.time(), "watchdog_stop": stopped})
        print(json.dumps({"returncode": child.returncode, "watchdog_stop": stopped}), flush=True)


if __name__ == "__main__":
    main()
