#!/usr/bin/env python3
"""Probe GitHub repos for vendored dataset files. Usage: python -I gh_probe.py <owner/repo:branch> ..."""
import json
import sys
import urllib.request

EXT = (".mat", ".zip", ".csv", ".npz", ".7z", ".rar", ".tar.gz")


def api(url):
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "research-probe"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    for spec in sys.argv[1:]:
        if ":" in spec:
            repo, branch = spec.rsplit(":", 1)
        else:
            repo, branch = spec, None
        print("===", repo)
        try:
            if branch is None:
                branch = api(f"https://api.github.com/repos/{repo}")["default_branch"]
            tree = api("https://api.github.com/repos/%s/%s/trees/%s?recursive=1"
                       % (repo, "git", branch))
            hits = [t for t in tree.get("tree", []) if t["path"].lower().endswith(EXT)]
            if tree.get("truncated"):
                print("  (tree truncated)")
            for t in hits[:60]:
                print(f"  {t.get('size')}\t{t['path']}")
            if not hits:
                print("  no data files")
        except Exception as e:
            print("  ERROR", type(e).__name__, e)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
