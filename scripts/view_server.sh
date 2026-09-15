#!/usr/bin/env bash
# view_server.sh <seconds> [username-to-op] Starts the server with GDMC-HTTP disabled so
# a vanilla client can connect. The mod registers registry entries the client must also
# have; nothing in the world depends on it, so for looking around it can simply be left
# out.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
SRV="$ROOT/run/server"
LOG="$SRV/console.log"
FIFO="$SRV/console.in"
SECONDS_UP="${1:-420}"
USER="${2:-}"
# Where a reviewer lands. On the arterial outside the outer gate, looking down it. Pass
# three more arguments to spawn somewhere else.
SPAWN_X="${3:--422}"; SPAWN_Y="${4:-67}"; SPAWN_Z="${5:--469}"

cleanup() {
  [ -p "$FIFO" ] && echo "stop" > "$FIFO" 2>/dev/null
  for _ in $(seq 1 90); do kill -0 "${MCPID:-0}" 2>/dev/null || break; sleep 1; done
  kill "${MCPID:-0}" "${FIFOPID:-0}" 2>/dev/null
  rm -f "$FIFO"
  # always put the mod back
  [ -f "$SRV/mods-off/gdmc_http_interface-1.8.4-1.21.11.jar" ] && \
    mv "$SRV/mods-off"/*.jar "$SRV/mods/" 2>/dev/null
  rmdir "$SRV/mods-off" 2>/dev/null
  echo "mod restored"
}
trap cleanup EXIT INT TERM

mkdir -p "$SRV/mods-off"
mv "$SRV/mods/gdmc_http_interface"*.jar "$SRV/mods-off/" 2>/dev/null
echo "GDMC-HTTP disabled for this session; Fabric API left in place (server-side only)"

rm -f "$FIFO"; mkfifo "$FIFO"
sleep infinity > "$FIFO" & FIFOPID=$!
cd "$SRV"
"$NIX_GLIBC/lib/ld-linux-x86-64.so.2" "$ROOT/run/jdk/bin/java" -Xms2G -Xmx8G \
  -jar fabric-server-launch.jar nogui < "$FIFO" > "$LOG" 2>&1 &
MCPID=$!

echo "waiting for server..."
for i in $(seq 1 180); do
  grep -q 'Done (' "$LOG" 2>/dev/null && break
  kill -0 $MCPID 2>/dev/null || { echo "server died"; tail -25 "$LOG"; exit 1; }
  sleep 1
done
grep -q 'Done (' "$LOG" || { echo "never started"; tail -25 "$LOG"; exit 1; }
echo "server up after ${i}s on 0.0.0.0:25565"

send() { echo "$1" > "$FIFO"; sleep 1; }
send "setworldspawn $SPAWN_X $SPAWN_Y $SPAWN_Z"
send "time set noon"
send "gamerule doDaylightCycle false"
send "gamerule doWeatherCycle false"
send "gamerule doMobSpawning false"
send "defaultgamemode creative"
[ -n "$USER" ] && { send "op $USER"; send "whitelist off"; }

IP=$(ip -4 addr show 2>/dev/null | grep -oP 'inet \K[\d.]+' | grep -v 127.0.0.1 | head -1)
echo "connect to ${IP:-<lan-ip>}:25565  — spawn at x=$SPAWN_X y=$SPAWN_Y z=$SPAWN_Z"
END=$(( $(date +%s) + SECONDS_UP ))
while [ "$(date +%s)" -lt "$END" ]; do
  sleep 15
  grep -hE "joined the game|left the game" "$LOG" | tail -3
done
