#!/bin/sh
# PVPO chrome-headless-shell entrypoint.
#
# Runs chrome-headless-shell on an internal loopback port and forwards
# external 0.0.0.0:9222 → 127.0.0.1:9223 via socat. This sidesteps the
# chrome-headless-shell 147 bug where --remote-debugging-address=0.0.0.0 is
# accepted but not honored (binding stays on 127.0.0.1).
#
# tini is PID 1; this script is tini's argv[1]. tini reaps zombies and
# propagates SIGTERM correctly on ``docker stop``.
set -eu

INTERNAL_PORT="${PVPO_INTERNAL_PORT:-9223}"
EXTERNAL_PORT="${PVPO_EXTERNAL_PORT:-9222}"

chrome-headless-shell \
  --enable-begin-frame-control \
  --run-all-compositor-stages-before-draw \
  --disable-checker-imaging \
  --window-size=1280,900 \
  --no-sandbox \
  --disable-dev-shm-usage \
  "--remote-debugging-port=${INTERNAL_PORT}" &
CHROME_PID=$!

# Wait until chrome's CDP endpoint is actually up before starting the forwarder.
# socat will happily accept connections before chrome is ready and close them
# immediately, which shows up as "Connection reset by peer" on the host.
i=0
while [ $i -lt 100 ]; do
  if curl -fsS "http://127.0.0.1:${INTERNAL_PORT}/json/version" >/dev/null 2>&1; then
    break
  fi
  i=$((i + 1))
  sleep 0.1
done

socat "TCP-LISTEN:${EXTERNAL_PORT},bind=0.0.0.0,fork,reuseaddr" \
      "TCP:127.0.0.1:${INTERNAL_PORT}" &
SOCAT_PID=$!

# Block on chrome: if chrome dies, tear down socat and exit with chrome's code.
# dash (Debian /bin/sh) does not support ``wait -n``, so we wait on chrome
# directly — the common failure mode is chrome crashing, which this catches.
# tini (PID 1) reaps socat and handles signal forwarding from ``docker stop``.
wait "$CHROME_PID"
RC=$?
kill "$SOCAT_PID" 2>/dev/null || true
exit $RC
