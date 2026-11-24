import argparse
from pathlib import Path
import sys

from .benchmark import Config, run_benchmark
from .data import load_csv, synthetic_load


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Leakage-aware hourly forecasting benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("demo", "evaluate"):
        child = commands.add_parser(command)
        child.add_argument("--out", type=Path, default=Path("runs/demo" if command == "demo" else "runs/csv"))
        child.add_argument("--horizons", nargs="+", type=int, default=[1, 6, 24], metavar="H")
        child.add_argument("--alpha", type=float, default=0.1)
        child.add_argument("--seed", type=int, default=42)
        if command == "demo":
            child.add_argument("--rows", type=int, default=3360)
        else:
            child.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = Config(tuple(args.horizons), args.alpha, args.seed)
        if args.command == "demo":
            frame = synthetic_load(args.rows, args.seed)
            source = "synthetic_hourly_load_v1"
        else:
            frame = load_csv(args.csv)
            source = "user_csv"  # Keep private local filenames out of shareable reports.
        report = run_benchmark(frame, args.out, config, source)
        if args.command == "demo":
            frame.to_csv(args.out / "input.csv", index=False)
        print(f"Report: {(args.out / 'report.html').resolve()}")
        for row in report["results"]:
            if row["selected_by_validation"]:
                print(f"h={row['horizon']:2} validation choice={row['model']:<16} test MAE={row['mae']:.3f} coverage={row['coverage']:.1%}")
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
