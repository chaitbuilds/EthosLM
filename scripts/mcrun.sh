#!/usr/bin/env bash
# mcrun.sh <python-script> [args...] Sandbox constraint: every shell invocation gets its
# own network namespace, so a server started in one call is unreachable from the next.
# Therefore one call owns the whole session: proxy shim -> server up -> run the build
# script against it -> flush chunks to disk -> shut down. Renders happen afterwards, off
# the saved world.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
SRV="$ROOT/run/server"
LOG="$SRV/console.log"
FIFO="$SRV/console.in"

cleanup() {
  [ -p "$FIFO" ] && echo "stop" > "$FIFO" 2>/dev/null
  for _ in $(seq 1 90); do kill -0 "${MCPID:-0}" 2>/dev/null || break; sleep 1; done
  kill "${MCPID:-0}" "${FIFOPID:-0}" "${PROXYPID:-0}" 2>/dev/null
  rm -f "$FIFO"
}
trap cleanup EXIT

# 0. Refuse a save that is already open. Starting a second here would have taken its
# console pipe and put two servers on one locked save (open thread 36). The world
# directory is shared between worktrees, so the check is on the process and the lock,
# not on this checkout.
if pgrep -f "fabric-server-launch.jar" > /dev/null 2>&1; then
  echo "mcrun.sh: a Minecraft server is already running on this host:" >&2
  pgrep -fa "fabric-server-launch.jar" | head -3 >&2
  echo "mcrun.sh: refusing to start a second one on the same save; stop it first" >&2
  exit 3
fi

# 1. auth-injecting proxy so the JVM can bootstrap without DNS or credentials
python3 "$ROOT/scripts/authproxy.py" 8888 > "$ROOT/run/authproxy.log" 2>&1 &
PROXYPID=$!
sleep 1

# 1b. the settings every live session needed by hand, the scripts' own: a tick that
# never times the server out under a 1.7M-block write. (The memory bound is on the build
# script in step 4 and not here: a `ulimit -v` set before the JVM starts refuses the JVM
# its own 12G heap.
if [ -f "$SRV/server.properties" ]; then
  if grep -q '^max-tick-time=' "$SRV/server.properties"; then
    sed -i 's/^max-tick-time=.*/max-tick-time=-1/' "$SRV/server.properties"
  else
    echo 'max-tick-time=-1' >> "$SRV/server.properties"
  fi
fi

# 2. server
rm -f "$FIFO"; mkfifo "$FIFO"
sleep infinity > "$FIFO" & FIFOPID=$!
cd "$SRV"
# The loader is a Nix host's business and is empty everywhere else; env.sh decides.
LOADER=(); [ -n "${ETHOSLM_JAVA_LOADER:-}" ] && LOADER=("$ETHOSLM_JAVA_LOADER")
${LOADER[@]+"${LOADER[@]}"} "$ETHOSLM_JAVA" \
  -Xms2G -Xmx12G \
  -Dhttp.proxyHost=127.0.0.1 -Dhttp.proxyPort=8888 \
  -Dhttps.proxyHost=127.0.0.1 -Dhttps.proxyPort=8888 \
  -Djava.net.preferIPv4Stack=true \
  -jar fabric-server-launch.jar nogui < "$FIFO" > "$LOG" 2>&1 &
MCPID=$!

# 3. wait for GDMC-HTTP on :9000
echo "waiting for GDMC-HTTP..."
UP=0
for i in $(seq 1 180); do
  if curl -s --noproxy '*' -m 2 -o /dev/null http://localhost:9000/blocks?x=0\&y=64\&z=0; then UP=1; break; fi
  kill -0 $MCPID 2>/dev/null || { echo "server died"; tail -30 "$LOG"; exit 1; }
  sleep 1
done
[ $UP -eq 1 ] || { echo "GDMC-HTTP never came up"; tail -40 "$LOG"; exit 1; }
echo "GDMC-HTTP up after ${i}s"

# 4. run the build script
if [ $# -gt 0 ]; then
  SCRIPT="$1"; shift
  # twelve gigabytes for a build script, twenty where it renders.
  case "$*" in *render*) LIM="${ETHOSLM_ULIMIT_KB:-20000000}";; *) LIM="${ETHOSLM_ULIMIT_KB:-12000000}";; esac
  ( cd "$ROOT" && ulimit -v "$LIM" 2>/dev/null; no_proxy='*' NO_PROXY='*' "$PY" "$SCRIPT" "$@" )
  RC=$?
else
  RC=0
fi

# 5. flush to disk so the renderer sees it
echo "save-all flush" > "$FIFO"
sleep 8
exit $RC
