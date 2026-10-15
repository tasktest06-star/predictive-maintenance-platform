#!/usr/bin/env python3
"""Checksum and provenance manifest for the measured datasets.

    python3 tools/dataset_manifest.py --write
    python3 tools/dataset_manifest.py --verify

Why this exists: this repository's standing admission is that no model in it had
been validated on measured data. That is now changing, and the moment a result
depends on 700 MB of bearing data sitting outside the repository, two things
become true that were not true before.

**A result is only reproducible if the bytes are pinned.** "Evaluated on CWRU"
is not a provenance statement. There are at least four CWRU redistributions in
circulation with different file subsets, different channel naming and, in some
mirrors, resampled or truncated records. A conclusion drawn from one of them is
not automatically a conclusion about another. So every file gets a SHA-256, and
every reported result should name the manifest it was computed against.

**The licences are not all the same, and one of them restricts what can be
shipped.** See `docs/10_dataset_provenance.md`. The manifest records the licence
per dataset root so that question is answerable from code rather than from
memory.

The datasets themselves are NOT committed and must not be: they are large, they
are not ours to redistribute, and in one case redistribution would impose a
share-alike obligation on a derivative. The manifest is committed; the data is
fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
#: Datasets live OUTSIDE the repository, deliberately: they are untrusted
#: downloads, they are large, and they are not ours to redistribute.
_DATA_ROOT = os.path.abspath(os.path.join(_REPO, ".."))
_MANIFEST = os.path.join(_REPO, "docs", "datasets", "manifest.json")

#: What we believe about each dataset root, recorded so it can be checked rather
#: than remembered. `source` is where the bytes came from in this environment,
#: which is not always the canonical home of the dataset.
ROOTS: Dict[str, Dict[str, object]] = {
    "data_raw/cwru": {
        "dataset": "CWRU Bearing Data Center",
        "owner": "Case Western Reserve University",
        "source": "https://engineering.case.edu/sites/default/files/ (official)",
        "canonical": "https://engineering.case.edu/bearingdatacenter",
        "licence": "No formal licence published; the Data Center asks for "
                   "acknowledgement of Case Western Reserve University when the "
                   "data is used. Treat as permission-to-use-with-attribution, "
                   "not as public domain.",
        "citation": "Case Western Reserve University Bearing Data Center, "
                    "https://engineering.case.edu/bearingdatacenter",
        "commercial_use": "unclear -- no licence grant is published, so a "
                          "commercial deployment trained on it needs its own "
                          "legal review",
        "notes": "Fetched from the official university host rather than a "
                 "third-party mirror, which is the point: mirrors differ.",
    },
    "data_raw/mfpt": {
        "dataset": "MFPT bearing fault data, as redistributed by MathWorks",
        "owner": "Eric Bechhoefer (data owner); Society for Machinery Failure "
                 "Prevention Technology (original publication)",
        "source": "https://github.com/mathworks/"
                  "RollingElementBearingFaultDiagnosis-Data",
        "canonical": "http://data-acoustics.com/measurements/bearing-faults/bearing-2/",
        "licence": "CC BY-NC-SA 4.0, per the license.txt and README.md shipped "
                   "in the source repository",
        "citation": "Bechhoefer, Eric. 'Condition Based Maintenance Fault "
                    "Database for Testing Diagnostics and Prognostic "
                    "Algorithms.' 2013.",
        "commercial_use": "PROHIBITED without permission from the data owner. "
                          "MathWorks obtained permission for its own commercial "
                          "use; that grant does not transfer. See "
                          "docs/10_dataset_provenance.md.",
        "notes": "This is a MODIFICATION of the original MFPT set -- the "
                 "MathWorks README says so explicitly. It is not "
                 "interchangeable with the full MFPT distribution, and a result "
                 "should say which one it used.",
    },
}

#: Additional roots that may exist from exploratory fetches. Recorded if present
#: so a stray second copy of a dataset cannot be silently mixed with the first.
OPTIONAL_ROOTS = ["dl_cwru_12kDE", "dl_mfpt"]

#: Files worth hashing. Archives are hashed too -- an extracted tree can be
#: regenerated, but only if the archive it came from is pinned.
_EXTENSIONS = (".mat", ".npz", ".csv", ".txt", ".md", ".tar.gz", ".zip")


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def scan_root(root_rel: str) -> Optional[Dict[str, object]]:
    root = os.path.join(_DATA_ROOT, root_rel)
    if not os.path.isdir(root):
        return None
    files: List[Dict[str, object]] = []
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if not name.lower().endswith(_EXTENSIONS):
                continue
            full = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            files.append({
                "path": os.path.relpath(full, root),
                "bytes": size,
                "sha256": sha256_file(full),
            })
            total += size
    files.sort(key=lambda f: f["path"])
    # A digest OF the manifest, so one short string identifies the whole
    # dataset state. This is what a result should quote.
    roll = hashlib.sha256()
    for f in files:
        roll.update(f["path"].encode("utf-8"))
        roll.update(str(f["sha256"]).encode("ascii"))
    return {
        "root": root_rel,
        "absolute_path_when_scanned": root,
        "n_files": len(files),
        "total_bytes": total,
        "digest": roll.hexdigest(),
        "files": files,
    }


def build() -> Dict[str, object]:
    out: Dict[str, object] = {
        "schema": 1,
        "note": ("Checksums of datasets held OUTSIDE this repository. The data "
                 "is not committed; this manifest is. Quote a root's `digest` "
                 "alongside any result computed from it."),
        "data_root": _DATA_ROOT,
        "datasets": {},
    }
    for rel, meta in ROOTS.items():
        scanned = scan_root(rel)
        if scanned is None:
            out["datasets"][rel] = {**meta, "present": False}  # type: ignore[index]
            continue
        out["datasets"][rel] = {**meta, "present": True, **scanned}  # type: ignore[index]
    extra = {}
    for rel in OPTIONAL_ROOTS:
        scanned = scan_root(rel)
        if scanned is not None:
            extra[rel] = {
                "n_files": scanned["n_files"],
                "total_bytes": scanned["total_bytes"],
                "digest": scanned["digest"],
                "notes": ("An additional fetch of the same dataset family. "
                          "Recorded so two copies cannot be mixed without "
                          "noticing -- compare digests before assuming they "
                          "are the same bytes."),
            }
    out["additional_fetches"] = extra
    return out


def report(man: Dict[str, object]) -> None:
    print(f"dataset manifest  (data root: {man['data_root']})")
    for rel, d in man["datasets"].items():            # type: ignore[union-attr]
        if not d.get("present"):
            print(f"  {rel:20s} ABSENT")
            continue
        print(f"  {rel:20s} {d['n_files']:4d} files, "
              f"{float(d['total_bytes']) / 1e6:7.1f} MB")
        print(f"  {'':20s} digest {str(d['digest'])[:16]}...")
        print(f"  {'':20s} licence: {str(d['licence'])[:66]}")
        if "PROHIBITED" in str(d.get("commercial_use", "")):
            print(f"  {'':20s} *** COMMERCIAL USE RESTRICTED -- see "
                  f"docs/10_dataset_provenance.md ***")
    for rel, d in (man.get("additional_fetches") or {}).items():   # type: ignore[union-attr]
        print(f"  {rel:20s} {d['n_files']:4d} files, "
              f"{float(d['total_bytes']) / 1e6:7.1f} MB  "
              f"digest {str(d['digest'])[:16]}...  (additional fetch)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true", help="scan and write the manifest")
    g.add_argument("--verify", action="store_true",
                   help="scan and compare against the committed manifest")
    ap.add_argument("--out", default=_MANIFEST)
    args = ap.parse_args(argv)

    if args.verify:
        if not os.path.isfile(args.out):
            print(f"no manifest at {args.out}; run --write first", file=sys.stderr)
            return 2
        with open(args.out, encoding="utf-8") as fh:
            committed = json.load(fh)
        current = build()
        bad = 0
        for rel, cd in committed.get("datasets", {}).items():
            nd = current["datasets"].get(rel, {})            # type: ignore[union-attr]
            if not cd.get("present"):
                continue
            if not nd.get("present"):
                print(f"  {rel}: MISSING -- the manifest records it but it is "
                      f"not on disk"); bad += 1
                continue
            if cd.get("digest") != nd.get("digest"):
                print(f"  {rel}: DIGEST MISMATCH\n"
                      f"      committed {cd.get('digest')}\n"
                      f"      on disk   {nd.get('digest')}\n"
                      f"      The data changed. Any result quoting the "
                      f"committed digest no longer describes these bytes.")
                bad += 1
            else:
                print(f"  {rel}: ok ({nd['n_files']} files)")
        print("verify: FAIL" if bad else "verify: all present roots match")
        return 1 if bad else 0

    man = build()
    report(man)
    if args.write:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=1)
        print(f"\nwrote {args.out}")
    else:
        print("\n(dry run -- pass --write to commit the manifest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
