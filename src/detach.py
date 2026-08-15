"""Run a command in its own session so it survives the parent shell going away.

Long folds kept dying at ~11 minutes with no traceback, no OOM and plenty of
disk. The cause was the harness tearing down the process group when a session
ended: nohup blocks SIGHUP but not a group-directed SIGKILL. os.setsid() puts
the child in a fresh session with no controlling terminal, so a kill aimed at
the old group does not reach it.

Usage:
    python src/detach.py <logfile> <command> [args...]
"""
import os
import sys

log, cmd = sys.argv[1], sys.argv[2:]
if not cmd:
    raise SystemExit("usage: detach.py <logfile> <command> [args...]")
if os.fork() != 0:
    sys.exit(0)                      # parent returns immediately
os.setsid()                          # new session, detached from the old group
if os.fork() != 0:
    os._exit(0)                      # give up session leadership
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1)
os.dup2(fd, 2)
os.close(os.open(os.devnull, os.O_RDONLY))
os.execvp(cmd[0], cmd)
