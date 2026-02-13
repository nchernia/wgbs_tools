#!/usr/bin/env python3
"""
resume_outlier_run.py - Salvage a partial outlier detector run.

Reads partial outliers_*.csv files, drops the last donor from each
(may be incomplete), moves finished .pat.gz files to done/ directory,
and writes safe results to outliers_done.csv.

Usage:
  python resume_outlier_run.py --dir /path/to/workdir [--pat_glob "*.pat.gz"]

Then rerun the detector on remaining files:
  python hypometh_outlier_detector.py --bed regions.bed.gz --glob "*.pat.gz" -np -j 8 -o outliers_new.csv
"""

import argparse
import csv
import glob
import os
import os.path as op
import shutil
import sys
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser(
        description="Salvage partial outlier run: keep completed donors, move them aside"
    )
    parser.add_argument("--dir", default=".", help="Working directory with outliers_*.csv and .pat.gz files")
    parser.add_argument("--pat_glob", default="*.pat.gz", help="Glob for pat files (default: *.pat.gz)")
    parser.add_argument("--csv_glob", default="outliers_*.csv", help="Glob for partial CSV files (default: outliers_*.csv)")
    parser.add_argument("--out", default="outliers_done.csv", help="Output CSV for safe results")
    parser.add_argument("--done_dir", default="done", help="Directory to move finished pat files into")
    parser.add_argument("--dry_run", action="store_true", help="Show what would happen without moving files")
    args = parser.parse_args()

    wdir = args.dir

    # Find partial CSV files
    csv_files = sorted(glob.glob(op.join(wdir, args.csv_glob)))
    if not csv_files:
        print(f"No CSV files matching {args.csv_glob} in {wdir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_files)} partial CSV files", file=sys.stderr)

    # For each CSV, collect donors in order and drop the last one
    safe_rows = []
    completed_donors = set()
    dropped_donors = set()

    for csv_path in csv_files:
        # Read all rows, tracking donor order
        rows_by_donor = OrderedDict()
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                donor = row["donor"]
                if donor not in rows_by_donor:
                    rows_by_donor[donor] = []
                rows_by_donor[donor].append(row)

        donors = list(rows_by_donor.keys())
        if not donors:
            print(f"  {op.basename(csv_path)}: empty, skipping", file=sys.stderr)
            continue

        # Drop last donor (potentially incomplete)
        last = donors[-1]
        dropped_donors.add(last)
        safe = donors[:-1]

        print(f"  {op.basename(csv_path)}: {len(donors)} donors, "
              f"keeping {len(safe)}, dropping last ({last})",
              file=sys.stderr)

        for donor in safe:
            completed_donors.add(donor)
            safe_rows.extend(rows_by_donor[donor])

    print(f"\nTotal completed donors: {len(completed_donors)}", file=sys.stderr)
    print(f"Dropped (last per worker, will redo): {dropped_donors}", file=sys.stderr)

    # Write safe results
    out_path = op.join(wdir, args.out)
    fieldnames = ["donor", "region", "chrom", "start", "end", "high_meth_reads"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in safe_rows:
            writer.writerow(row)
    print(f"\nWrote {len(safe_rows)} safe outlier rows to {out_path}", file=sys.stderr)

    # Move completed donor pat files to done/
    done_dir = op.join(wdir, args.done_dir)
    pat_files = glob.glob(op.join(wdir, args.pat_glob))
    moved = 0

    if not args.dry_run:
        os.makedirs(done_dir, exist_ok=True)

    for pf in pat_files:
        donor = op.basename(pf).replace(".pat.gz", "")
        if donor in completed_donors:
            dest = op.join(done_dir, op.basename(pf))
            if args.dry_run:
                print(f"  [dry run] would move {op.basename(pf)} -> {args.done_dir}/", file=sys.stderr)
            else:
                shutil.move(pf, dest)
                # Also move index if present
                for ext in [".tbi", ".csi"]:
                    idx = pf + ext
                    if op.exists(idx):
                        shutil.move(idx, op.join(done_dir, op.basename(idx)))
            moved += 1

    remaining = len(pat_files) - moved
    print(f"\n{'Would move' if args.dry_run else 'Moved'} {moved} pat files to {args.done_dir}/",
          file=sys.stderr)
    print(f"Remaining pat files to process: {remaining}", file=sys.stderr)

    if not args.dry_run:
        # Clean up partial CSVs
        for csv_path in csv_files:
            os.remove(csv_path)
            print(f"  Removed {op.basename(csv_path)}", file=sys.stderr)

    print(f"\nNext steps:", file=sys.stderr)
    print(f"  1. Rerun detector on remaining {remaining} donors:", file=sys.stderr)
    print(f"     python hypometh_outlier_detector.py --bed regions.bed.gz "
          f"--glob '{args.pat_glob}' -np -j 8 -o outliers_new.csv", file=sys.stderr)
    print(f"  2. Combine results:", file=sys.stderr)
    print(f"     cat {args.out} <(tail -n+2 outliers_new.csv) > outliers_all.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
