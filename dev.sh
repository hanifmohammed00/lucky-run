#!/usr/bin/env bash
# Shortcuts for the .venv/bin/python commands in README.md's "Running it"
# section, so you don't have to retype the venv path and module flags.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "no .venv found - run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# no subcommand -> `run`: this is what a terminal profile configured to
# launch this script on open actually wants (a live dry-run session), not a
# usage message that exits immediately and looks like a crashed profile.
cmd="${1:-run}"
[ $# -gt 0 ] && shift

usage() {
  cat >&2 <<EOF
usage: ./dev.sh <command> [args...]

  run         live dry run (caffeinate, serves the dashboard) -- -m Lucky.runner  [default]
  replay      rebuild a missed session -- -m Lucky.replay 2026-08-17 2026-08-27
  dashboard   standalone dashboard, no runner attached -- -m Lucky.dashboard
  probe       latency probe only -- -m Lucky.probe_latency
  check       strategy self-check -- gap_vwap_strategy.py
  robustness  bootstrap / Monte-Carlo checks -- -m Lucky.robustness
  fetch       chunked 1-minute bar fetcher -- fetch_1m.py

any extra args are passed straight through, e.g.:
  ./dev.sh run --no-dashboard --until 15:00
  ./dev.sh replay 2026-09-01 2026-09-11 --dry-run
EOF
}

case "$cmd" in
  run)        exec caffeinate -i "$PY" -m Lucky.runner "$@" ;;
  replay)     exec "$PY" -m Lucky.replay "$@" ;;
  dashboard)  exec "$PY" -m Lucky.dashboard "$@" ;;
  probe)      exec "$PY" -m Lucky.probe_latency "$@" ;;
  check)      exec "$PY" gap_vwap_strategy.py "$@" ;;
  robustness) exec "$PY" -m Lucky.robustness "$@" ;;
  fetch)      exec "$PY" fetch_1m.py "$@" ;;
  help|-h|--help) usage; exit 0 ;;
  *)          usage; exit 1 ;;
esac
