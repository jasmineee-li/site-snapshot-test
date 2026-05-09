"""wa_envctrl_patcher.py — Idempotent in-container patcher for WebArena
Verified env-ctrl site files.

Called by scripts/patch_webarena_containers.sh (with --on-ec2) via:
    docker cp wa_envctrl_patcher.py <container>:/tmp/
    docker exec <container> python3 /tmp/wa_envctrl_patcher.py <site_file.py>

What it does:
  1. Repairs any prior broken patch attempt where `import os\n` was
     prepended at line 0 above `from __future__ import annotations` or a
     module docstring (which produces a SyntaxError). This is the bug that
     bit us today: the earlier patch script inserted `import os\n` at the
     top of the file unconditionally, but a `from __future__` import must
     be the first non-docstring statement in the module. We strip the bad
     prepend before doing anything else.
  2. Inserts `import os` AFTER any `from __future__` imports / module
     docstring / top-of-file comments, if (and only if) the module does
     not already have a plain `import os`.
  3. Inserts the WA_ENV_CTRL_EXTERNAL_SITE_URL fallback into the `_init`
     body:
         if not base_url:
             base_url = os.environ.get("WA_ENV_CTRL_EXTERNAL_SITE_URL", "")
         if not base_url:
             raise ValueError("base_url is required")
     (only if the fallback isn't already there).
  4. Runs `compile()` on the final source so we exit non-zero if any
     prior run left the file in a SyntaxError state.

Idempotent: safe to run N times. On the second and later runs, it's a no-op
unless the file was left in a broken state by an earlier run.
"""

import pathlib
import re
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: wa_envctrl_patcher.py <path-to-site-file.py>", file=sys.stderr)
        return 2

    path = sys.argv[1]
    p = pathlib.Path(path)
    src = p.read_text()

    # --- Step 1: repair prior broken patch ---------------------------------
    # An earlier incarnation of patch_webarena_containers.sh prepended
    # `import os\n` at the start of the file, which is a SyntaxError when
    # the file begins with `from __future__ import annotations` (or a
    # module docstring -- `from __future__` must precede the docstring, but
    # `import os` above either of them is wrong either way).
    if src.startswith("import os\n"):
        rest_head = "\n".join(src.splitlines()[1:20])
        if (
            "from __future__" in rest_head
            or rest_head.lstrip().startswith('"""')
            or rest_head.lstrip().startswith("'''")
        ):
            src = src[len("import os\n") :]
            p.write_text(src)  # persist repair before doing anything else
            print(f"  repaired prior broken `import os` prepend in {path}")

    # --- Step 2: check if env-var guard already applied --------------------
    # If it is AND the file compiles, we're done. If it does NOT compile
    # (earlier run left it broken), fall through and re-run the os-import
    # placement + fallback insertion.
    if "WA_ENV_CTRL_EXTERNAL_SITE_URL" in src:
        try:
            compile(src, path, "exec")
            print(f"  already patched (env-var guard present, syntax OK) {path}")
            return 0
        except SyntaxError as exc:
            print(f"  env-var guard present but SYNTAX ERROR: {exc} ({path})")
            # Continue on: re-run the os-import placement / fallback insertion.

    # --- Step 3: insert `import os` in the right place ---------------------
    needs_os = not re.search(r"^\s*import os(\s|$)", src, re.MULTILINE)
    if needs_os:
        lines = src.splitlines(keepends=True)
        insert_at = 0
        i = 0
        while i < len(lines):
            line = lines[i]
            # Skip shebang, encoding declarations, module-level comments,
            # blank lines, and `from __future__` imports -- `import os`
            # must come after them.
            if line.startswith("#") or line.strip() == "" or line.startswith("from __future__"):
                insert_at = i + 1
                i += 1
                continue
            # Walk past a module docstring if present.
            stripped = line.lstrip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = stripped[:3]
                insert_at = i + 1
                # One-line docstring?
                if line.count(quote) >= 2:
                    i += 1
                    continue
                # Multi-line docstring: advance to its closing line.
                for j in range(i + 1, len(lines)):
                    insert_at = j + 1
                    if quote in lines[j]:
                        i = j + 1
                        break
                else:
                    # Unterminated docstring -- shouldn't happen; bail.
                    i = len(lines)
                continue
            break
        lines.insert(insert_at, "import os\n")
        src = "".join(lines)

    # --- Step 4: insert env-var fallback before the raise ------------------
    old = "        if not base_url:\n            raise ValueError"
    new = (
        "        if not base_url:\n"
        '            base_url = os.environ.get("WA_ENV_CTRL_EXTERNAL_SITE_URL", "")\n'
        "        if not base_url:\n"
        "            raise ValueError"
    )
    if "WA_ENV_CTRL_EXTERNAL_SITE_URL" not in src and old in src:
        src = src.replace(old, new, 1)

    # --- Step 5: write + compile sanity check ------------------------------
    p.write_text(src)
    try:
        compile(src, path, "exec")
    except SyntaxError as exc:
        print(f"  PATCHED but SYNTAX ERROR: {exc} ({path})")
        return 1
    print(f"  PATCHED (syntax OK) {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
