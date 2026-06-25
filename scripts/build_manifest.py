"""Build a JSONL manifest from one or more audio directories.

Example (synthetic set, one domain per dir):
    python scripts/build_manifest.py --root data/synthetic/music --domain music \
        --root data/synthetic/environmental --domain environmental \
        --root data/synthetic/speech --domain speech \
        --out data/manifests/synthetic.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (sets up sys.path)

from tajepa.data.manifest import build_manifest, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True, dest="roots",
                    help="Audio directory. Repeatable; pairs positionally with --domain.")
    ap.add_argument("--domain", action="append", default=None, dest="domains",
                    help="Domain tag for the matching --root. Repeatable.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-probe", action="store_true", help="Skip duration/sr probing.")
    args = ap.parse_args()

    domains = args.domains or ["unknown"] * len(args.roots)
    if len(domains) != len(args.roots):
        ap.error("number of --domain must match number of --root (or omit --domain entirely)")

    entries = []
    for root, domain in zip(args.roots, domains):
        entries.extend(
            build_manifest([root], domain=domain, split=args.split, probe=not args.no_probe)
        )
    write_manifest(entries, args.out)
    print(f"Wrote {len(entries)} entries to {args.out}")


if __name__ == "__main__":
    main()
