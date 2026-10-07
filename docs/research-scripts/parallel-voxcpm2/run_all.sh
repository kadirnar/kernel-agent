#!/bin/sh
# Each GPU job under the shared lock, < 10 min each.
cd /tmp/claude-1000/-home-kadir-kadir-projects-kernel-agent/ecc94bbb-9c5b-4acc-bcfd-5c76518d4c95/scratchpad/136
PY=/home/kadir/kadir_projects/kernel-agent/.venv/bin/python
SRC=/home/kadir/kadir_projects/kernel-agent/.claude/worktrees/agent-a72b2ec6d48737a6f/src
for job in "$@"; do
  echo "== $job $(date +%T)"
  flock /home/kadir/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1 PYTHONPATH=$SRC timeout 590 $PY $job > out/$(echo $job | cut -d' ' -f1 | sed 's/.py$//').log 2>&1
  echo "exit $? $(date +%T)"
done
