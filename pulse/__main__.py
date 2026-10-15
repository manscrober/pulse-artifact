"""Command line of the PULSE reproduction: `python -m pulse <command>`."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pulse", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-cache", help="bin the sepsis cohort from a MIMIC-IV database into a cache")
    build.add_argument("--dsn", default="dbname=mimiciv host=localhost user=postgres",
                       help="PostgreSQL connection string")
    build.add_argument("--out", type=Path, default=Path("cache/sepsis.pt"))
    args = parser.parse_args(argv)
    if args.command == "build-cache":
        from pulse import cohort

        print(cohort.build_cache(args.dsn, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
