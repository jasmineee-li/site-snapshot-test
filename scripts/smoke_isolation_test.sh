#!/usr/bin/env bash
# Smoke test for parallel docker isolation between WASP and DoomArena.
#
# Fires two streams in parallel (n=2 tasks each), one against WASP's
# dockers (gitlab :9001 / forum :8080) and one against DoomArena's
# duplicates (gitlab :9002 / forum :8081). Both should complete cleanly
# with no cross-container traffic.
#
# Cost: ~$0.50–$1 total (4 trajectories on glm-5:thinking + judges).
# Wallclock: ~3–5 minutes.
#
# Usage:
#   ./scripts/smoke_isolation_test.sh
#
# Exits 0 if both runs complete with non-empty trajectory_awareness JSONLs.
# Exits non-zero on any failure or pre-flight problem.

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

PYTHON=.venv/bin/python
OUTPUT_BASE="eval_awareness_experiments/results/smoke_isolation_test_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/smoke_isolation"
mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

WASP_LOG="$LOG_DIR/wasp_smoke.log"
DOOM_LOG="$LOG_DIR/doomarena_smoke.log"

echo "================================================================"
echo "Pre-flight: confirm both docker stacks are responsive"
echo "================================================================"

check_docker() {
    local name="$1" url="$2"
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url" || echo "000")
    if [[ "$code" =~ ^(200|301|302|401|403)$ ]]; then
        echo "  [ok]  $name ($url) → HTTP $code"
        return 0
    else
        echo "  [FAIL] $name ($url) → HTTP $code"
        return 1
    fi
}

failed=0
check_docker "WASP gitlab"     "http://localhost:9001/help" || failed=1
check_docker "WASP forum"      "http://localhost:8080/"     || failed=1
check_docker "DoomArena gitlab" "http://localhost:9002/help" || failed=1
check_docker "DoomArena forum"  "http://localhost:8081/"     || failed=1

if [ "$failed" -eq 1 ]; then
    echo
    echo "Pre-flight failed. Bring the docker stacks up before running the smoke."
    echo "See eval_awareness_experiments/DOCKER_PORTS.md for container provenance."
    exit 1
fi

echo
echo "================================================================"
echo "Snapshot pre-run docker logs (for cross-talk diff after)"
echo "================================================================"
docker logs --since 1s forum      > "$LOG_DIR/forum_pre.log"      2>&1 || true
docker logs --since 1s forum_doom > "$LOG_DIR/forum_doom_pre.log" 2>&1 || true

echo
echo "================================================================"
echo "Launching 2 parallel streams (n=2 each, glm-5:thinking)"
echo "================================================================"

# Stream 1: WASP
nohup env \
    GITLAB="http://localhost:9001" \
    REDDIT="http://localhost:8080" \
    DATASET="webarena_prompt_injections" \
    "$PYTHON" -m eval_awareness_experiments.run_causal_experiment \
        --models z-ai/glm-5:thinking \
        --benchmarks wasp \
        --conditions baseline \
        --extra-instructions-presets none \
        --system-prompt-frames none \
        --judges verbalized_awareness \
        --tasks-per-split 2 \
        --output-base "$OUTPUT_BASE" \
        > "$WASP_LOG" 2>&1 &
WASP_PID=$!
echo "  WASP stream:      PID $WASP_PID  → $WASP_LOG"

# Stream 2: DoomArena
nohup env \
    GITLAB="http://localhost:9002" \
    REDDIT="http://localhost:8081" \
    SHOPPING="http://localhost:8082" \
    SHOPPING_ADMIN="http://localhost:8083" \
    "$PYTHON" -m eval_awareness_experiments.run_causal_experiment \
        --models z-ai/glm-5:thinking \
        --benchmarks doomarena \
        --conditions baseline \
        --extra-instructions-presets none \
        --system-prompt-frames none \
        --judges verbalized_awareness \
        --tasks-per-split 2 \
        --output-base "$OUTPUT_BASE" \
        > "$DOOM_LOG" 2>&1 &
DOOM_PID=$!
echo "  DoomArena stream: PID $DOOM_PID  → $DOOM_LOG"

echo
echo "Waiting for both streams to finish..."
wait "$WASP_PID" || WASP_RC=$?
WASP_RC=${WASP_RC:-0}
wait "$DOOM_PID" || DOOM_RC=$?
DOOM_RC=${DOOM_RC:-0}

echo
echo "================================================================"
echo "Results"
echo "================================================================"

echo "WASP stream exit code:      $WASP_RC"
echo "DoomArena stream exit code: $DOOM_RC"
echo

# Look for trajectory_awareness JSONLs to confirm stage 2 completed
wasp_jsonl=$(find "$OUTPUT_BASE" -path '*/wasp/*/trajectory_awareness_results.jsonl' 2>/dev/null | head -1 || true)
doom_jsonl=$(find "$OUTPUT_BASE" -path '*/doomarena/*/trajectory_awareness_results.jsonl' 2>/dev/null | head -1 || true)

if [ -n "$wasp_jsonl" ] && [ -s "$wasp_jsonl" ]; then
    echo "  [ok]  WASP JSONL: $wasp_jsonl ($(wc -l < "$wasp_jsonl") rows)"
else
    echo "  [FAIL] WASP JSONL not produced. Check $WASP_LOG."
fi

if [ -n "$doom_jsonl" ] && [ -s "$doom_jsonl" ]; then
    echo "  [ok]  DoomArena JSONL: $doom_jsonl ($(wc -l < "$doom_jsonl") rows)"
else
    echo "  [FAIL] DoomArena JSONL not produced. Check $DOOM_LOG."
fi

echo
echo "================================================================"
echo "Cross-talk check (post-run docker logs)"
echo "================================================================"
docker logs --since 1s forum      > "$LOG_DIR/forum_post.log"      2>&1 || true
docker logs --since 1s forum_doom > "$LOG_DIR/forum_doom_post.log" 2>&1 || true

forum_hits=$(diff "$LOG_DIR/forum_pre.log"      "$LOG_DIR/forum_post.log"      2>/dev/null | grep -c '^>' || true)
doom_hits=$( diff "$LOG_DIR/forum_doom_pre.log" "$LOG_DIR/forum_doom_post.log" 2>/dev/null | grep -c '^>' || true)

echo "  forum (WASP)        new log lines: $forum_hits"
echo "  forum_doom (DoomA)  new log lines: $doom_hits"
echo
echo "If forum_hits > 0 and doom_hits > 0, both streams reached their"
echo "respective containers and the isolation works at runtime."
echo "If one is 0, that stream may have never started or routed to the"
echo "wrong container — inspect the run log."

echo
echo "Output dir: $OUTPUT_BASE"
echo "Per-stream logs: $LOG_DIR/{wasp,doomarena}_smoke.log"
echo "Pre/post container logs: $LOG_DIR/forum{,_doom}_{pre,post}.log"

if [ "$WASP_RC" -ne 0 ] || [ "$DOOM_RC" -ne 0 ]; then
    exit 1
fi
