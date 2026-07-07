"""Download Syzygy 3-4-5 endgame tablebases from the lichess mirror.

These cover every position with <= 5 pieces (~1 GB) and let the engine play
those endgames perfectly.

Usage:
    python scripts/download_syzygy.py C:\\path\\to\\syzygy

Resumable: files already present are skipped, so you can re-run after an
interruption.
"""
from __future__ import annotations

import os
import re
import sys
import urllib.request

BASE = "https://tablebase.lichess.ovh/tables/standard/3-4-5/"


def main() -> None:
    dest = sys.argv[1] if len(sys.argv) > 1 else "syzygy"
    os.makedirs(dest, exist_ok=True)

    print(f"Fetching file list from {BASE} ...")
    with urllib.request.urlopen(BASE) as r:
        html = r.read().decode("utf-8", "replace")
    files = sorted(set(re.findall(r'href="([^"?]+\.rtb[wz])"', html)))
    files = [f.split("/")[-1] for f in files]
    if not files:
        print("No tablebase files found in the index. The mirror layout may have changed.")
        sys.exit(1)

    print(f"Found {len(files)} files. Downloading to {os.path.abspath(dest)} ...")
    for i, name in enumerate(files, 1):
        out = os.path.join(dest, name)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print(f"[{i}/{len(files)}] skip {name}")
            continue
        url = BASE + name
        tmp = out + ".part"
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, out)
            print(f"[{i}/{len(files)}] {name}")
        except Exception as exc:  # noqa: BLE001
            if os.path.exists(tmp):
                os.remove(tmp)
            print(f"[{i}/{len(files)}] FAILED {name}: {exc}")

    print("Done. Set SYZYGY_PATH to this directory.")


if __name__ == "__main__":
    main()
