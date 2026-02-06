#!/usr/bin/env python3
"""
detect_hypometh_outliers.py

For hypomethylated regions, detect donor outliers that show unexpected high methylation.
Reads a BED file of regions and checks each donor for reads with >=70% methylation.
Outputs region-donor pairs where at least 4 reads are highly methylated.

Usage:
  python detect_hypometh_outliers.py \
    --bed regions.bed.gz \
    --files donor1.pat.gz donor2.pat.gz \
    --output outliers.csv \
    --min_reads 4 \
    --meth_threshold 0.7
    
  # Or with GCS files:
  export GCS_OAUTH_TOKEN=$(gcloud auth print-access-token)
  python detect_hypometh_outliers.py \
    --bed regions.bed.gz \
    --glob "gs://bucket/path/*.pat.gz" \
    --output outliers.csv
"""

import argparse
import subprocess
import glob
import gzip
import os
import os.path as op
import sys
import time
import csv

METHYLATED = set(list("CM"))
UNMETHYLATED = set(list("TU"))

MAX_PAT_LEN = 150
NANOPORE_EXTEND = 100000

# Token refresh tracking
_last_token_refresh = 0
_token_refresh_interval = 3000  # 50 minutes

def is_gcs_path(path):
    """Check if path is a Google Cloud Storage path"""
    return path.startswith('gs://')

def refresh_gcs_token_if_needed():
    """Refresh GCS OAuth token every 50 minutes"""
    global _last_token_refresh
    current_time = time.time()
    
    if current_time - _last_token_refresh > _token_refresh_interval:
        try:
            result = subprocess.check_output(['gcloud', 'auth', 'print-access-token'], 
                                            stderr=subprocess.PIPE)
            token = result.decode('utf-8').strip()
            os.environ['GCS_OAUTH_TOKEN'] = token
            _last_token_refresh = current_time
            print(f"Refreshed GCS OAuth token", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to refresh GCS token: {e}", file=sys.stderr)

def check_gcs_credentials():
    """Check if GCS credentials are set up"""
    if 'GCS_OAUTH_TOKEN' not in os.environ:
        try:
            result = subprocess.check_output(['gcloud', 'auth', 'print-access-token'],
                                            stderr=subprocess.PIPE)
            token = result.decode('utf-8').strip()
            os.environ['GCS_OAUTH_TOKEN'] = token
            global _last_token_refresh
            _last_token_refresh = time.time()
            print("Set GCS_OAUTH_TOKEN from gcloud", file=sys.stderr)
        except subprocess.CalledProcessError:
            print("ERROR: GCS files require authentication.", file=sys.stderr)
            print("Run: export GCS_OAUTH_TOKEN=$(gcloud auth print-access-token)", file=sys.stderr)
            return False
    return True

def get_gcs_files(pattern):
    """Get list of files from GCS using gsutil ls"""
    if not is_gcs_path(pattern):
        return []
    
    try:
        result = subprocess.check_output(['gsutil', 'ls', pattern], stderr=subprocess.PIPE)
        files = result.decode('utf-8').strip().split('\n')
        return [f for f in files if f.endswith('.pat.gz')]
    except subprocess.CalledProcessError as e:
        print(f"Error listing GCS files: {e}", file=sys.stderr)
        return []

def read_bed_file(bed_path):
    """
    Read BED file (supports gzip) and return list of regions.
    Expects wgbs_tools blocks format: chr, start, end, startCpG, endCpG [, name, ...]
    The startCpG/endCpG columns (4-5) are used for pat.gz tabix queries.
    If only 3 columns, raises an error (genomic coords can't query pat files).

    Returns list of dicts with CpG site indices as 'start'/'end'.
    """
    regions = []

    if bed_path.endswith('.gz'):
        f = gzip.open(bed_path, 'rt')
    else:
        f = open(bed_path, 'r')

    warned = False
    try:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue

            parts = line.strip().split('\t')
            if len(parts) < 5:
                if not warned:
                    print("ERROR: BED file must have at least 5 columns "
                          "(chr, start, end, startCpG, endCpG).\n"
                          "Run: wgbstools convert -L YOUR_BED -o OUTPUT.bed "
                          "to add CpG site columns.", file=sys.stderr)
                    warned = True
                    sys.exit(1)

            chrom = parts[0]
            genomic_start = int(parts[1])
            genomic_end = int(parts[2])
            site_start = int(parts[3])
            site_end = int(parts[4])
            name = parts[5] if len(parts) > 5 else f"{chrom}:{genomic_start}-{genomic_end}"

            regions.append({
                'chrom': chrom,
                'start': site_start,
                'end': site_end,
                'name': name,
                'genomic_start': genomic_start,
                'genomic_end': genomic_end,
            })
    finally:
        f.close()

    return regions

def pull_tabix(pat_file, chrom, start, end, extend_upstream=None):
    """
    Use tabix to extract lines from pat file for a region.
    start/end are CpG site indices (half-open: [start, end)).
    Extends upstream to capture reads that start before region.
    """
    if extend_upstream is None:
        extend_upstream = MAX_PAT_LEN
    if is_gcs_path(pat_file):
        refresh_gcs_token_if_needed()

    query_start = max(1, start - extend_upstream)
    region_str = f"{chrom}:{query_start}-{end - 1}"
    
    cmd = ["tabix", pat_file, region_str]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.PIPE)
        return out.decode('utf-8')
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else ""
        if 'Permission denied' in stderr:
            print(f"Warning: GCS permission denied for {pat_file}", file=sys.stderr)
        return ""

def count_high_meth_reads(pat_text, region_start, region_end, meth_threshold=0.7, min_informative=10):
    """
    Parse pat tabix output and count reads with methylation >= threshold.
    
    Returns: count of reads with >= meth_threshold methylation
    """
    if not pat_text:
        return 0
    
    high_meth_count = 0
    
    for line in pat_text.splitlines():
        if not line.strip():
            continue
        
        toks = line.split("\t")
        try:
            read_start = int(toks[1])
            patt = toks[-2]
            count = int(toks[-1])
        except Exception:
            continue
        
        read_end = read_start + len(patt)
        
        # Find overlap with region
        ov_s = max(read_start, region_start)
        ov_e = min(read_end, region_end)
        if ov_s >= ov_e:
            continue
        
        # Extract overlapping segment
        rel_s = ov_s - read_start
        rel_e = ov_e - read_start
        seg = patt[rel_s:rel_e]
        
        if len(seg) < 1:
            continue
        
        # Count methylated and unmethylated positions
        n_meth = sum(1 for ch in seg if ch in METHYLATED)
        n_unmeth = sum(1 for ch in seg if ch in UNMETHYLATED)
        n_inform = n_meth + n_unmeth
        
        if n_inform < min_informative:
            continue
        
        # Calculate methylation fraction
        frac = n_meth / n_inform
        
        # Check if this read is highly methylated
        if frac >= meth_threshold:
            high_meth_count += count
    
    return high_meth_count

def check_region_donor(pat_file, region, min_high_meth_reads=4, meth_threshold=0.7,
                       min_informative=10, extend_upstream=None):
    """
    Check if a region-donor pair has enough highly methylated reads.

    Returns: (is_outlier, high_meth_count)
    """
    pat_text = pull_tabix(pat_file, region['chrom'], region['start'], region['end'],
                          extend_upstream=extend_upstream)
    
    high_meth_count = count_high_meth_reads(
        pat_text, 
        region['start'], 
        region['end'],
        meth_threshold=meth_threshold,
        min_informative=min_informative
    )
    
    is_outlier = high_meth_count >= min_high_meth_reads
    
    return is_outlier, high_meth_count

def main():
    parser = argparse.ArgumentParser(
        description="Detect donors with unexpected high methylation in hypomethylated regions"
    )
    parser.add_argument('--bed', required=True, help='BED file with regions to check (can be .bed.gz)')
    parser.add_argument('--glob', default=None, help='Glob pattern for pat files (or gs://bucket/*.pat.gz)')
    parser.add_argument('--files', nargs='*', help='Explicit list of pat files (supports gs:// URLs)')
    parser.add_argument('--output', '-o', default='outliers.csv', help='Output CSV file')
    parser.add_argument('--min_reads', type=int, default=4, 
                        help='Minimum number of highly methylated reads to flag as outlier (default: 4)')
    parser.add_argument('--meth_threshold', type=float, default=0.7,
                        help='Methylation threshold for "high" methylation (default: 0.7)')
    parser.add_argument('--min_informative', type=int, default=10,
                        help='Minimum informative CpGs per read (default: 10)')
    parser.add_argument('-np', '--nanopore', action='store_true',
                        help='Pull very long reads starting before the requested region (nanopore/long-read data)')
    args = parser.parse_args()
    
    # Resolve pat files
    if args.files:
        pat_files = args.files
    elif args.glob:
        if is_gcs_path(args.glob):
            pat_files = get_gcs_files(args.glob)
        else:
            pat_files = sorted(glob.glob(args.glob))
    else:
        print("ERROR: Must provide --files or --glob", file=sys.stderr)
        sys.exit(1)
    
    if not pat_files:
        print("ERROR: No pat files found", file=sys.stderr)
        sys.exit(1)
    
    # Check GCS credentials if needed
    has_gcs = any(is_gcs_path(f) for f in pat_files)
    if has_gcs and not check_gcs_credentials():
        sys.exit(1)
    
    # Read regions
    print(f"Reading regions from {args.bed}...", file=sys.stderr)
    regions = read_bed_file(args.bed)
    print(f"Loaded {len(regions)} regions", file=sys.stderr)
    
    if not regions:
        print("ERROR: No regions found in BED file", file=sys.stderr)
        sys.exit(1)
    
    # Process each donor x region
    extend_upstream = NANOPORE_EXTEND if args.nanopore else MAX_PAT_LEN
    print(f"Processing {len(pat_files)} donors x {len(regions)} regions "
          f"(upstream_extend={extend_upstream})...", file=sys.stderr)

    fieldnames = ['donor', 'region', 'chrom', 'start', 'end', 'high_meth_reads']
    total_checks = len(pat_files) * len(regions)
    checked = 0
    total_outliers = 0

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for pat_file in pat_files:
            donor_name = op.basename(pat_file).replace('.pat.gz', '')
            print(f"\nProcessing {donor_name}...", file=sys.stderr)

            donor_outliers = 0

            for region in regions:
                checked += 1
                if checked % 100 == 0:
                    print(f"  Progress: {checked}/{total_checks} ({100*checked/total_checks:.1f}%)", file=sys.stderr)

                is_outlier, high_meth_count = check_region_donor(
                    pat_file,
                    region,
                    min_high_meth_reads=args.min_reads,
                    meth_threshold=args.meth_threshold,
                    min_informative=args.min_informative,
                    extend_upstream=extend_upstream
                )

                if is_outlier:
                    row = {
                        'donor': donor_name,
                        'region': region['name'],
                        'chrom': region['chrom'],
                        'start': region['genomic_start'],
                        'end': region['genomic_end'],
                        'high_meth_reads': high_meth_count
                    }
                    writer.writerow(row)
                    f.flush()
                    donor_outliers += 1
                    total_outliers += 1

            print(f"  Found {donor_outliers} outlier regions for {donor_name}", file=sys.stderr)

    print(f"\nDone! Found {total_outliers} region-donor outlier pairs in {args.output}.", file=sys.stderr)

if __name__ == '__main__':
    main()
