# chrome-headless-shell Docker image for Paint-Verified Payload Oracle (PVPO) rigor runs.
#
# Why this container exists
#   macOS chrome-headless-shell does not support HeadlessExperimental.beginFrame
#   (confirmed in Chromium source headless/test/headless_compositor_browsertest.cc,
#   "BeginFrameControl is not supported on MacOS yet"). PVPO's I2 atomicity
#   invariant requires beginFrame. Running chrome-headless-shell inside a Linux
#   container on every host (including macOS dev boxes) gives uniform Linux
#   paint-pipeline behavior + full BeginFrame support, with zero code branches.
#
# Connect from Browser-Use / Playwright via:
#   page.context().newCDPSession() pointing at 127.0.0.1:9222
#
# See docs/handoffs/codex-handoff-paint-verified-oracle.md §3.1 for flag rationale
# and the research log for why the removed flags (--enable-surface-synchronization,
# --disable-threaded-scrolling, --disable-threaded-animation) are no-ops in 2026.

# Build note: an earlier draft pulled ``ghcr.io/browserless/chrome-headless-shell:stable``
# directly, but that registry path is gated (403 Forbidden for anonymous
# pulls as of 2026-04-19). We now stage the official Chrome-for-Testing
# ``chrome-headless-shell`` binary on top of a slim Debian base so the
# build is reproducible from the public Google JSON endpoint with no
# registry auth required. Keep this path primary unless a fully-public
# prebuilt image becomes available.
FROM debian:bookworm-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl jq unzip python3 socat tini \
      fonts-liberation libasound2 libatk-bridge2.0-0 libatk1.0-0 libcups2 \
      libdbus-1-3 libdrm2 libgbm1 libglib2.0-0 libgtk-3-0 libnspr4 libnss3 \
      libpango-1.0-0 libx11-6 libxcb1 libxcomposite1 libxdamage1 libxext6 \
      libxfixes3 libxkbcommon0 libxrandr2 libxshmfence1 wget xdg-utils \
  && rm -rf /var/lib/apt/lists/*
# Pin to the last-known-good Stable channel build from Chrome for Testing.
# The URL layout is stable across releases; we pick the linux64 bundle that
# matches our container arch (amd64 is the reference target; arm64 uses a
# different file name).
ARG TARGETARCH=amd64
RUN set -eux; \
    case "$TARGETARCH" in \
      amd64) CFT_PLATFORM=linux64 ;; \
      arm64) echo "chrome-for-testing does not publish an arm64 chrome-headless-shell binary; build on amd64 or rebuild against a Linux arm64 host" >&2; exit 2 ;; \
      *) echo "unsupported TARGETARCH=$TARGETARCH" >&2; exit 2 ;; \
    esac; \
    url="$(curl -fsSL https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json \
      | jq -r --arg p "$CFT_PLATFORM" '.channels.Stable.downloads["chrome-headless-shell"][] | select(.platform==$p) | .url')"; \
    test -n "$url"; \
    curl -fsSL "$url" -o /tmp/chs.zip; \
    unzip -q /tmp/chs.zip -d /opt; \
    rm /tmp/chs.zip; \
    ln -s /opt/chrome-headless-shell-linux64/chrome-headless-shell /usr/local/bin/chrome-headless-shell

EXPOSE 9222

# Flags rationale (do NOT change without re-verifying against Chromium source):
#   --enable-begin-frame-control           required for HeadlessExperimental.beginFrame
#   --run-all-compositor-stages-before-draw  closes the main Blink→compositor commit race
#   --disable-checker-imaging              prevents lazy-image-decode checker tiles in screenshots
#   --no-sandbox                           required inside an unprivileged container
#   --disable-dev-shm-usage                /dev/shm is often small in containers
#   --remote-debugging-port=9223           chrome binds CDP on 127.0.0.1:9223 inside the container
#
# Chrome bind address note: ``--remote-debugging-address=0.0.0.0`` is
# documented but is NOT honored by chrome-headless-shell 147 — the process
# binds only to 127.0.0.1 regardless. We work around that by running chrome
# on an internal port (9223) and forwarding external 0.0.0.0:9222 → internal
# 127.0.0.1:9223 via socat. The ``/json/version`` and websocket upgrade path
# both work transparently through the forwarder.
COPY worldsim/docker/entrypoint.sh /usr/local/bin/pvpo-entrypoint.sh
RUN chmod +x /usr/local/bin/pvpo-entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/pvpo-entrypoint.sh"]
