#!/usr/bin/env python3
"""Download CWRU Bearing Data Center .mat files. Pure stdlib.

Usage: python -I fetch_cwru.py <outdir> <mode: head|get> <num> [<num> ...]
"""
import hashlib
import json
import os
import sys
import urllib.request

BASE = "https://engineering.case.edu/sites/default/files/"
UA = {"User-Agent": "Mozilla/5.0 (research data fetch)"}


def head(num):
    url = BASE + f"{num}.mat"
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return url, r.status, r.headers.get("Content-Length"), r.headers.get("Content-Type")


def get(num, outdir):
    url = BASE + f"{num}.mat"
    req = urllib.request.Request(url, headers=UA)
    dest = os.path.join(outdir, f"{num}.mat")
    h = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        status = r.status
        ctype = r.headers.get("Content-Type")
        clen = r.headers.get("Content-Length")
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            h.update(chunk)
            f.write(chunk)
            n += len(chunk)
    return {"num": num, "url": url, "status": status, "content_type": ctype,
            "content_length": clen, "bytes": n, "sha256": h.hexdigest(), "path": dest}


def main():
    outdir, mode = sys.argv[1], sys.argv[2]
    nums = sys.argv[3:]
    if mode == "head":
        for num in nums:
            try:
                u, s, cl, ct = head(num)
                print(f"{num}\t{s}\t{cl}\t{ct}")
            except Exception as e:
                print(f"{num}\tERROR\t{type(e).__name__}: {e}")
            sys.stdout.flush()
        return
    os.makedirs(outdir, exist_ok=True)
    results = []
    for num in nums:
        try:
            rec = get(num, outdir)
        except Exception as e:
            rec = {"num": num, "url": BASE + f"{num}.mat", "error": f"{type(e).__name__}: {e}"}
        results.append(rec)
        print(json.dumps(rec))
        sys.stdout.flush()
    with open(os.path.join(outdir, "_download_log.json"), "a") as f:
        f.write(json.dumps(results) + "\n")


if __name__ == "__main__":
    main()
