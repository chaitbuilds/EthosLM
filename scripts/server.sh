#!/usr/bin/env bash
# start | stop | cmd <console command> | log Console stdin is held open by a FIFO so we
# can push commands (save-all, setbuildarea).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
SRV="$ROOT/run/server"
FIFO="$SRV/console.in"
PIDF="$SRV/server.pid"
LOG="$SRV/console.log"

case "${1:-}" in
  start)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "already running"; exit 0; fi
    rm -f "$FIFO"; mkfifo "$FIFO"
    cd "$SRV"
    # keep the fifo open for writing so the JVM never sees EOF on stdin
    sleep infinity > "$FIFO" &
    echo $! > "$SRV/fifo.pid"
    LOADER=(); [ -n "${ETHOSLM_JAVA_LOADER:-}" ] && LOADER=("$ETHOSLM_JAVA_LOADER")
    ( ${LOADER[@]+"${LOADER[@]}"} "$ETHOSLM_JAVA" \
        -Xms2G -Xmx8G -jar fabric-server-launch.jar nogui < "$FIFO" > "$LOG" 2>&1 & echo $! > "$PIDF" )
    echo "started pid $(cat "$PIDF"); tail $LOG"
    ;;
  cmd)
    shift
    echo "$*" > "$FIFO"
    ;;
  stop)
    [ -p "$FIFO" ] && echo stop > "$FIFO" || true
    for _ in $(seq 1 60); do
      [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null || break
      sleep 1
    done
    [ -f "$SRV/fifo.pid" ] && kill "$(cat "$SRV/fifo.pid")" 2>/dev/null || true
    rm -f "$PIDF" "$SRV/fifo.pid" "$FIFO"
    echo stopped
    ;;
  log) tail -n "${2:-40}" "$LOG" ;;
  *) echo "usage: $0 {start|stop|cmd <console cmd>|log [n]}"; exit 1 ;;
esac
