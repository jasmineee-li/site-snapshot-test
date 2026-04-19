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

FROM ghcr.io/browserless/chrome-headless-shell:stable AS base
# If the browserless/chrome-headless-shell image is unavailable, fall back to
# pulling the binary from Chrome for Testing and staging it manually. Swap the
# FROM line and uncomment the alternative block below.
#
# FROM debian:bookworm-slim AS base
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ca-certificates curl unzip \
#     fonts-liberation libasound2 libatk-bridge2.0-0 libatk1.0-0 libcups2 \
#     libdbus-1-3 libdrm2 libgbm1 libglib2.0-0 libgtk-3-0 libnspr4 libnss3 \
#     libpango-1.0-0 libx11-6 libxcb1 libxcomposite1 libxdamage1 libxext6 \
#     libxfixes3 libxkbcommon0 libxrandr2 libxshmfence1 wget xdg-utils \
#   && rm -rf /var/lib/apt/lists/*
# ARG CHROME_VERSION=stable
# RUN curl -fsSL "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json" \
#   | python3 -c "import json, sys, urllib.request; data=json.load(sys.stdin); \
#       url=next(b['url'] for b in data['channels']['Stable']['downloads']['chrome-headless-shell'] \
#               if b['platform']=='linux64'); urllib.request.urlretrieve(url, '/tmp/chs.zip')" \
#   && unzip -q /tmp/chs.zip -d /opt && rm /tmp/chs.zip \
#   && ln -s /opt/chrome-headless-shell-linux64/chrome-headless-shell /usr/local/bin/chrome-headless-shell

EXPOSE 9222

# Flags rationale (do NOT change without re-verifying against Chromium source):
#   --enable-begin-frame-control           required for HeadlessExperimental.beginFrame
#   --run-all-compositor-stages-before-draw  closes the main Blink→compositor commit race
#   --disable-checker-imaging              prevents lazy-image-decode checker tiles in screenshots
#   --no-sandbox                           required inside an unprivileged container
#   --disable-dev-shm-usage                /dev/shm is often small in containers
#   --remote-debugging-address=0.0.0.0     allow CDP connections from host over the exposed port
ENTRYPOINT ["chrome-headless-shell", \
    "--enable-begin-frame-control", \
    "--run-all-compositor-stages-before-draw", \
    "--disable-checker-imaging", \
    "--no-sandbox", \
    "--disable-dev-shm-usage", \
    "--remote-debugging-address=0.0.0.0", \
    "--remote-debugging-port=9222"]
