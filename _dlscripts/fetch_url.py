#!/usr/bin/env python3
"""Generic stdlib downloader with sha256.

Usage: python -I fetch_url.py head <url>
       python -I fetch_url.py get <url> <dest_path>
"""
import hashlib
import json
import os
import sys
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (research data fetch)"}


def main():
    mode, url = sys.argv[1], sys.argv[2]
    if mode == "head":
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=90) as r:
            print(json.dumps({"status": r.status, "url": r.url,
                              "content_length": r.headers.get("Content-Length"),
                              "content_type": r.headers.get("Content-Type"),
                              "content_disposition": r.headers.get("Content-Disposition")},
                             indent=2))
        return
    dest = sys.argv[3]
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    h = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        status, final, ctype = r.status, r.url, r.headers.get("Content-Type")
        clen = r.headers.get("Content-Length")
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            h.update(chunk)
            f.write(chunk)
            n += len(chunk)
    print(json.dumps({"requested_url": url, "final_url": final, "status": status,
                      "content_type": ctype, "content_length": clen,
                      "bytes": n, "sha256": h.hexdigest(), "path": dest}, indent=2))


if __name__ == "__main__":
    main()
