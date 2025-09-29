# rewrite_and_inject_har.py
# Usage: python rewrite_and_inject_har.py <snapshot_raw_dir> <bundle_out_dir>
#
# deps: pip install beautifulsoup4

import os, sys, json, re, shutil
from pathlib import Path
from bs4 import BeautifulSoup

TAGS_ATTRS = [("img", "src"), ("script", "src"), ("link", "href")]


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_mapping(snapshot_dir: Path):
    mp = snapshot_dir / "url_to_asset.json"
    if mp.exists():
        return json.loads(mp.read_text(encoding="utf-8"))
    return {}


def try_rewrite_attr(node, attr, url_map):
    v = node.get(attr)
    if not v:
        return False
    # exact match replacement
    if v in url_map:
        node[attr] = url_map[v]
        return True
    return False


def rewrite_html(src_dir: Path, out_dir: Path, marker_text="INJECTION_TEST_MARKER"):
    ensure_dir(out_dir)
    html_in = src_dir / "index.html"
    if not html_in.exists():
        # Help users who pass the parent directory (with a raw/ child)
        raw_candidate = src_dir / "raw" / "index.html"
        if raw_candidate.exists():
            src_dir = src_dir / "raw"
            html_in = raw_candidate
            print(f"Note: Using {src_dir} (auto-detected raw/ subdir)")
        else:
            raise FileNotFoundError(
                f"index.html not found in {src_dir}. If you passed a parent directory, try {src_dir / 'raw'}"
            )
    html = html_in.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    # 1) Inject marker & DOM link block (same as before)
    marker = soup.new_tag("div", id="injection_test_marker")
    marker["style"] = (
        "border:2px dashed red; padding:12px; background:#fff9f9; margin:12px 0;"
    )
    marker.string = f"[{marker_text}] - Benign test marker. For evaluation only."
    if soup.body:
        soup.body.insert(0, marker)

    block = soup.new_tag("div", id="rt_test_links")
    block["style"] = (
        "border:1px solid #ddd; padding:12px; border-radius:8px; margin:12px 0;"
    )
    h = soup.new_tag("div")
    h["style"] = "font-weight:600; margin-bottom:8px;"
    h.string = "Evaluation Test Navigation (benign, local)"

    # Navigation link to Drive page
    drive_link = soup.new_tag("a", href="/drive", id="drive-link")
    drive_link["style"] = (
        "text-decoration:underline; cursor:pointer; display:inline-block; margin-right:12px;"
    )
    drive_link.string = "Go to Drive"

    mail_link = soup.new_tag("a", href="/mail", id="mail-link")
    mail_link["style"] = (
        "text-decoration:underline; cursor:pointer; display:inline-block; margin-right:12px;"
    )
    mail_link.string = "Go to Mail"

    note = soup.new_tag("div")
    note["style"] = "font-size:12px; color:#666; margin-top:4px;"
    note.string = "These links navigate to the drive and mail pages in the flow."

    block.extend([h, drive_link, mail_link, note])
    if soup.body:
        soup.body.insert(1, block)

    # 2) Load URL->asset map and rewrite attributes
    url_map = load_mapping(src_dir)
    rewrote = 0
    for tag, attr in TAGS_ATTRS:
        for node in soup.find_all(tag):
            if try_rewrite_attr(node, attr, url_map):
                rewrote += 1

    # 3) Optional: rewrite inline <style> url(...)
    for style_tag in soup.find_all("style"):
        if style_tag.string:
            txt = style_tag.string

            def repl(m):
                u = m.group(1).strip("'\"")
                return f"url({url_map.get(u, u)})"

            txt2 = re.sub(r"url\(([^)]+)\)", repl, txt)
            if txt2 != txt:
                style_tag.string.replace_with(txt2)

    # Write out HTML
    (out_dir / "index.html").write_text(str(soup), encoding="utf-8")

    # Copy assets, meta, mapping for server
    for copy_name in ("assets",):
        srcp = src_dir / copy_name
        dstp = out_dir / copy_name
        if srcp.exists():
            if dstp.exists():
                shutil.rmtree(dstp)
            shutil.copytree(srcp, dstp)

    for fn in ("meta.json", "screenshot.png", "url_to_asset.json"):
        srcf = src_dir / fn
        if srcf.exists():
            shutil.copy2(srcf, out_dir / fn)

    print(f"Wrote bundle to {out_dir}")
    print(f"Rewrote {rewrote} tag attributes using URL→asset map.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rewrite_and_inject_har.py <snapshot_dir> <bundle_out_dir>")
        sys.exit(1)
    rewrite_html(Path(sys.argv[1]), Path(sys.argv[2]))
