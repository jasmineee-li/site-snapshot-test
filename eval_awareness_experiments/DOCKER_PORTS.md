# Docker port assignments for safety-benchmark runs

Two sets of WebArena dockers running in parallel so DoomArena and WASP
don't collide. WASP is the only benchmark that mutates state during
setup (plants issues / posts via Playwright), so it gets its own
dirtied originals. DoomArena does runtime injection only — gets fresh
duplicates that stay clean for repeated runs.

Per-model stacks (one GitLab and Reddit pair per model) are listed in
`DOCKER_PORTS_MULTI.md`.

## Port layout

| service | for **WASP** | for **DoomArena** | image |
|---|---|---|---|
| gitlab | `:9001` (existing `gitlab`) | **`:9002` (new `gitlab_doom`)** | `gitlab-populated-final-port8023:latest` |
| reddit / forum | `:8080` (existing `forum`) | **`:8081` (new `forum_doom`)** | `postmill-populated-exposed-withimg:latest` |
| shopping | n/a (WASP doesn't use) | `:8082` (shared `shopping`) | `shopping_final_0712:latest` |
| shopping_admin | n/a (WASP doesn't use) | `:8083` (shared `shopping_admin`) | `shopping_admin_final_0719:latest` |

WASP only uses gitlab + reddit, so we don't duplicate shopping /
shopping_admin — DoomArena uses the existing ones directly.

## Env vars per benchmark

### For WASP runs (existing dockers)

```bash
export GITLAB="http://localhost:9001"
export REDDIT="http://localhost:8080"
export DATASET=webarena_prompt_injections
```

### For DoomArena runs (new duplicates + shared shopping)

```bash
export GITLAB="http://localhost:9002"
export REDDIT="http://localhost:8081"
export SHOPPING="http://localhost:8082"
export SHOPPING_ADMIN="http://localhost:8083"
```

Adjust `DOOMARENA_WEBARENA_BASE_URL` if needed (default is
`http://localhost`).

## Container provenance

- Both `gitlab` and `gitlab_doom` start from the **same image**
  (`gitlab-populated-final-port8023`). They share the read-only
  layers; only the writable layer (database state) diverges.
- Same for `forum` and `forum_doom`.
- The existing `gitlab` has ~22 GB of writable layer (10 days of WASP
  + DoomArena testing); the new `gitlab_doom` starts at ~0 GB and
  grows as DoomArena runs against it.
- WASP's currently-planted state (168 issues + posts + attacker
  accounts from the n=100 run) is in the existing `gitlab` / `forum`
  containers ONLY. The duplicates are clean.

## How they were created

```bash
# gitlab_doom — match the existing gitlab's runtime command
docker run -d --name gitlab_doom -p 9002:8023 --hostname localhost \
  gitlab-populated-final-port8023:latest \
  /opt/gitlab/embedded/bin/runsvdir-start

# forum_doom — image's default ENTRYPOINT/CMD work as-is
docker run -d --name forum_doom -p 8081:80 \
  postmill-populated-exposed-withimg:latest
```

## Verified responsive

```
$ curl -sI http://localhost:9002/help    →  200 OK
$ curl -sI http://localhost:8081/        →  200 OK
```

Internal HTML responses use the correct host port (`localhost:9002` /
`localhost:8081`) — no `external_url` rewrite was needed on
`gitlab_doom`. The image picks the host port up dynamically from the
request, unlike the original 2026-04-15 handoff doc's notes which
described a one-time external_url fix on the existing gitlab.

If something later breaks (URLs in HTML start pointing at wrong port,
links in emails are wrong, CI runners can't reach the server), refer
to the handoff doc's "Update — 2026-04-15 19:38" section for the
external_url + nginx-listen-port repair recipe.
