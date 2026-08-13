from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WARP Taskgen adapter-bearing 0.1.0 fixture")
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
