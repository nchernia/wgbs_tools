#!/usr/bin/env python3
"""
region_bimodal_vis.py - Enhanced with biallelic separation detection
"""
import argparse
import subprocess
import glob
import os
import os.path as op
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
import csv
import sys
from genomic_region import GenomicRegion
from sklearn.mixture import GaussianMixture

METHYLATED = set(list("CMcm"))
UNMETHYLATED = set(list("TUtu"))

def parse_region(region):
    """Parse a region string "chr:start-end" -> (chrom, start, end)"""
    m = re.match(r'^([^:]+):(\d+)(?:-(\d+))?$', region)
    if not m:
        raise ValueError(f"Invalid region string: {region}")
    chrom = m.group(1)
    start = int(m.group(2))
    end = int(m.group(3)) if m.group(3) else start + 1
    return chrom, start, end

def pull_tabix(pat_file, region_str):
    """Use tabix to extract lines for region_str"""
    cmd = ["tabix", pat_file, region_str]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return ""
    return out.decode('utf-8')

def parse_pat_lines(pat_text, region_start, region_end, strict=True, min_informative=10):
    """Parse tabix output into per-read entries"""
    if not pat_text:
        return [], []
    region_len = region_end - region_start
    read_rows = []
    fractions = []
    
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
        ov_s = max(read_start, region_start)
        ov_e = min(read_end, region_end)
        if ov_s >= ov_e:
            continue
        
        rel_s = ov_s - read_start
        rel_e = ov_e - read_start
        seg = patt[rel_s:rel_e]
        if len(seg) < 1:
            continue
        
        row = np.full((region_len,), np.nan, dtype=float)
        col_start = ov_s - region_start
        col_end = col_start + len(seg)
        
        vals = np.full((len(seg),), np.nan, dtype=float)
        for i, ch in enumerate(seg):
            if ch in METHYLATED:
                vals[i] = 1.0
            elif ch in UNMETHYLATED:
                vals[i] = 0.0
            else:
                vals[i] = np.nan
        
        row[col_start:col_end] = vals
        informative_mask = ~np.isnan(vals)
        n_inform = int(np.sum(informative_mask))
        if n_inform < min_informative:
            continue
        
        m = int(np.sum(vals[informative_mask] == 1.0))
        u = n_inform - m
        frac = m / (m + u) if (m + u) > 0 else np.nan
        
        read_rows.append((row, count))
        fractions.append((frac, count))
    
    return fractions, read_rows

def expand_counts_list(items):
    """Expand (value, count) list into flat list"""
    out = []
    for val, cnt in items:
        if cnt <= 1:
            out.append(val)
        else:
            out.extend([val] * cnt)
    return out

def expand_rows(rows):
    """Expand (row_array, count) list"""
    out = []
    for row, cnt in rows:
        if cnt <= 1:
            out.append(row)
        else:
            out.extend([row.copy() for _ in range(cnt)])
    return out

def assess_biallelic_separation(frac_array, low_thresh=0.3, high_thresh=0.7, min_reads_per_allele=3):
    """
    Assess if the data shows true biallelic separation.
    
    Returns dict with:
    - is_biallelic: bool, True if meets biallelic criteria
    - n_low: count of reads with meth < low_thresh
    - n_high: count of reads with meth > high_thresh
    - n_mid: count of reads in middle range
    - separation_score: metric of how well separated (0-1, higher = better separation)
    - gap_size: difference between 95th percentile of low and 5th percentile of high
    """
    if len(frac_array) < min_reads_per_allele * 2:
        return {
            'is_biallelic': False,
            'n_low': 0,
            'n_high': 0,
            'n_mid': 0,
            'separation_score': 0.0,
            'gap_size': 0.0,
            'reason': 'insufficient_reads'
        }
    
    n_low = np.sum(frac_array < low_thresh)
    n_high = np.sum(frac_array > high_thresh)
    n_mid = np.sum((frac_array >= low_thresh) & (frac_array <= high_thresh))
    
    # Check if we have enough reads in both extremes
    has_both_alleles = (n_low >= min_reads_per_allele) and (n_high >= min_reads_per_allele)
    
    if not has_both_alleles:
        return {
            'is_biallelic': False,
            'n_low': int(n_low),
            'n_high': int(n_high),
            'n_mid': int(n_mid),
            'separation_score': 0.0,
            'gap_size': 0.0,
            'reason': f'insufficient_alleles (low={n_low}, high={n_high})'
        }
    
    # Calculate separation quality
    low_reads = frac_array[frac_array < low_thresh]
    high_reads = frac_array[frac_array > high_thresh]
    
    # Gap between populations (95th percentile of low vs 5th percentile of high)
    gap_size = np.percentile(high_reads, 5) - np.percentile(low_reads, 95)
    
    # Separation score: penalize middle reads, reward gap
    total_reads = len(frac_array)
    middle_fraction = n_mid / total_reads
    separation_score = (1 - middle_fraction) * min(gap_size / 0.4, 1.0)  # gap_size normalized to 0.4 max
    
    # Final decision: is_biallelic if separation_score is high enough
    is_biallelic = separation_score > 0.5 and gap_size > 0.2
    
    return {
        'is_biallelic': bool(is_biallelic),
        'n_low': int(n_low),
        'n_high': int(n_high),
        'n_mid': int(n_mid),
        'separation_score': float(separation_score),
        'gap_size': float(gap_size),
        'reason': 'biallelic' if is_biallelic else 'poor_separation'
    }

def fit_gmms(frac_array, min_components=1, max_components=2):
    """Fit GaussianMixture with 1 and 2 components"""
    res = {}
    X = np.array(frac_array).reshape(-1, 1)
    data_std = np.std(X)
    
    models = {}
    bics = {}
    for k in range(min_components, max_components + 1):
        if k == 2 and data_std < 0.05:
            models[k] = None
            bics[k] = np.inf
            continue
        try:
            gm = GaussianMixture(n_components=k, covariance_type='full', random_state=0, 
                                n_init=10, max_iter=200)
            gm.fit(X)
            models[k] = gm
            bics[k] = gm.bic(X)
        except Exception:
            models[k] = None
            bics[k] = np.inf
    
    res['models'] = models
    res['bics'] = bics
    if 1 in bics and 2 in bics:
        res['delta_bic'] = bics[1] - bics[2]
    return res

def plot_heatmap_and_hist(read_rows_matrix, fractions_array, out_png, sample_name,
                          region_str, bic_delta=None, biallelic_info=None, expand_counts_used=False):
    """Plot heatmap and histogram with biallelic assessment"""
    nreads = read_rows_matrix.shape[0] if read_rows_matrix is not None else 0
    region_len = read_rows_matrix.shape[1] if read_rows_matrix is not None else 0
    
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.5], hspace=0.4)
    
    ax0 = fig.add_subplot(gs[0])
    if nreads == 0:
        ax0.text(0.5, 0.5, "No reads", ha='center', va='center', fontsize=14)
        ax0.set_axis_off()
    else:
        cmap = matplotlib.cm.get_cmap('RdYlBu_r').copy()
        cmap.set_bad(color='lightgray')
        im = ax0.imshow(read_rows_matrix, aspect='auto', interpolation='nearest',
                        cmap=cmap, vmin=0.0, vmax=1.0)
        ax0.set_ylabel('Reads (rows)')
        title = f"{sample_name}  {region_str}  reads={nreads}"
        if biallelic_info and biallelic_info.get('is_biallelic'):
            title += " ⭐ BIALLELIC"
        ax0.set_title(title)
        cbar = fig.colorbar(im, ax=ax0, orientation='vertical', fraction=0.03)
        cbar.set_label('methylation (1=C/M, 0=T/U)')
    
    ax1 = fig.add_subplot(gs[1])
    if len(fractions_array) == 0:
        ax1.text(0.5, 0.5, "No informative reads for histogram", ha='center', va='center')
    else:
        ax1.hist(fractions_array, bins=20, range=(0.0, 1.0), color='C0', alpha=0.7, edgecolor='k')
        ax1.set_xlim(0, 1)
        ax1.set_xlabel('Per-read methylation fraction (inside region)')
        ax1.set_ylabel('Count')
        
        # Add annotation box with metrics
        info_lines = []
        if bic_delta is not None:
            info_lines.append(f"Δ BIC = {bic_delta:.1f}")
        if biallelic_info:
            ba = biallelic_info
            info_lines.append(f"Low (<30%): {ba['n_low']}, High (>70%): {ba['n_high']}")
            info_lines.append(f"Sep score: {ba['separation_score']:.2f}, Gap: {ba['gap_size']:.2f}")
            if ba['is_biallelic']:
                info_lines.append("✓ BIALLELIC")
        
        if info_lines:
            info_text = "\n".join(info_lines)
            bbox_color = 'lightgreen' if (biallelic_info and biallelic_info.get('is_biallelic')) else 'white'
            ax1.text(0.98, 0.95, info_text, transform=ax1.transAxes, ha='right', va='top',
                     fontsize=8, bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_total_histogram(all_fractions_by_sample, out_png, region_str, chrom, site_start, site_end):
    """Create combined histogram with biallelic assessment"""
    all_fracs = []
    sample_means = {}
    for sample, fracs in all_fractions_by_sample.items():
        if len(fracs) > 0:
            all_fracs.extend(fracs)
            sample_means[sample] = np.mean(fracs)
    
    if len(all_fracs) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data for total histogram", ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
        return None, None, None
    
    all_fracs_arr = np.array(all_fracs)
    
    # Fit GMMs
    gmm_info = fit_gmms(all_fracs_arr) if len(all_fracs_arr) > 1 else {}
    delta_bic = gmm_info.get('delta_bic') if gmm_info else None
    is_unimodal = (delta_bic is not None and delta_bic < 0) or len(all_fracs_arr) <= 1
    
    # Assess biallelic separation
    biallelic_info = assess_biallelic_separation(all_fracs_arr)
    
    # Create figure
    if is_unimodal and len(sample_means) > 1:
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)
        ax_hist = fig.add_subplot(gs[0])
        ax_outlier = fig.add_subplot(gs[1])
    else:
        fig, ax_hist = plt.subplots(figsize=(10, 6))
        ax_outlier = None
    
    # Plot histogram
    ax_hist.hist(all_fracs_arr, bins=30, range=(0.0, 1.0), color='steelblue', 
                 alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel('Per-read methylation fraction', fontsize=12)
    ax_hist.set_ylabel('Count (all samples)', fontsize=12)
    
    title = f'Total Distribution: {region_str}\n{len(all_fracs)} reads from {len(all_fractions_by_sample)} samples'
    if biallelic_info['is_biallelic']:
        title += ' ⭐ BIALLELIC'
    ax_hist.set_title(title, fontsize=13)
    
    # Add metrics annotation
    metrics_lines = []
    if delta_bic is not None:
        modality_txt = "Unimodal" if is_unimodal else "Bimodal"
        metrics_lines.append(f"Δ BIC = {delta_bic:.1f} ({modality_txt})")
    metrics_lines.append(f"Low: {biallelic_info['n_low']}, High: {biallelic_info['n_high']}, Mid: {biallelic_info['n_mid']}")
    metrics_lines.append(f"Sep score: {biallelic_info['separation_score']:.2f}")
    if biallelic_info['is_biallelic']:
        metrics_lines.append("✓ TRUE BIALLELIC")
    
    bbox_color = 'lightgreen' if biallelic_info['is_biallelic'] else 'lightyellow'
    ax_hist.text(0.02, 0.98, "\n".join(metrics_lines), 
                 transform=ax_hist.transAxes, ha='left', va='top',
                 bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.8),
                 fontsize=9)
    
    # Outlier panel if unimodal
    outlier_info = None
    if is_unimodal and ax_outlier is not None and len(sample_means) > 1:
        means_arr = np.array(list(sample_means.values()))
        samples_list = list(sample_means.keys())
        overall_mean = np.mean(means_arr)
        overall_std = np.std(means_arr)
        
        outliers = []
        colors = []
        for sample, mean_val in zip(samples_list, means_arr):
            z_score = abs(mean_val - overall_mean) / overall_std if overall_std > 0 else 0
            is_outlier = z_score > 2.0
            if is_outlier:
                outliers.append(sample)
                colors.append('red')
            else:
                colors.append('steelblue')
        
        ax_outlier.barh(range(len(samples_list)), means_arr, color=colors, alpha=0.7, edgecolor='black')
        ax_outlier.axvline(overall_mean, color='black', linestyle='--', linewidth=1.5, label=f'Mean={overall_mean:.3f}')
        if overall_std > 0:
            ax_outlier.axvline(overall_mean - 2*overall_std, color='gray', linestyle=':', linewidth=1, alpha=0.6)
            ax_outlier.axvline(overall_mean + 2*overall_std, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        
        ax_outlier.set_yticks(range(len(samples_list)))
        ax_outlier.set_yticklabels([s[:25] for s in samples_list], fontsize=8)
        ax_outlier.set_xlabel('Mean methylation fraction', fontsize=10)
        ax_outlier.set_title('Per-sample means\n(outlier check)', fontsize=11)
        ax_outlier.set_xlim(0, 1)
        ax_outlier.legend(fontsize=8)
        
        if outliers:
            outlier_info = f"Potential outliers (>2σ): {', '.join(outliers)}"
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    
    return delta_bic, outlier_info, biallelic_info

def process_single_file(pat_file, chrom, site_start, site_end, strict=True, min_informative=10,
                        expand_counts_flag=False, out_dir=".", sample_name=None, min_reads_plot=1):
    """Process a single pat file"""
    if sample_name is None:
        sample_name = op.basename(pat_file)
     
    new_start = max(1, site_start - 1500)
    pat_region = f"{chrom}:{new_start}-{site_end - 1}"
    pat_text = pull_tabix(pat_file, pat_region)
    fracs, read_rows = parse_pat_lines(pat_text, site_start, site_end, strict=strict, min_informative=min_informative)
    
    if not fracs:
        return {
            'sample': sample_name, 
            'n_reads': 0, 
            'n_informative': 0, 
            'delta_bic': None,
            'is_biallelic': False,
            'separation_score': 0.0,
            'n_low': 0,
            'n_high': 0,
            'n_mid': 0,
            'png': None,
            'fractions': []
        }
    
    if expand_counts_flag:
        frac_list = expand_counts_list(fracs)
        rows_expanded = expand_rows(read_rows)
        rows_matrix = np.vstack(rows_expanded) if rows_expanded else np.empty((0, site_end - site_start))
    else:
        frac_list = []
        weights = []
        rows_list = []
        for (f, cnt), (row, cnt2) in zip(fracs, read_rows):
            frac_list.append(f)
            weights.append(cnt)
            rows_list.append(row)
        rows_matrix = np.vstack(rows_list) if rows_list else np.empty((0, site_end - site_start))
        total_w = sum(weights)
        if total_w > 100000:
            scale = max(1, int(total_w / 2000))
            frac_list = expand_counts_list([(f, max(1, int(w / scale))) for f, w in zip(frac_list, weights)])
        else:
            frac_list = expand_counts_list(list(zip(frac_list, weights)))
    
    fracs_arr = np.array([f for f in frac_list if not (f is None or np.isnan(f))], dtype=float)
    fractions_for_total = expand_counts_list(fracs)
    fractions_for_total = [f for f in fractions_for_total if not (f is None or np.isnan(f))]
    
    # Fit GMMs
    gmm_info = fit_gmms(fracs_arr) if fracs_arr.size > 1 else {}
    delta_bic = gmm_info.get('delta_bic') if gmm_info else None
    
    # Assess biallelic separation
    biallelic_info = assess_biallelic_separation(fracs_arr) if fracs_arr.size > 0 else {
        'is_biallelic': False, 'separation_score': 0.0, 'n_low': 0, 'n_high': 0, 'n_mid': 0
    }
    
    # Sort rows by fraction
    if rows_matrix.size > 0:
        try:
            if expand_counts_flag:
                order = np.argsort(fracs_arr)
                sorted_matrix = rows_matrix[order, :]
            else:
                unique_fracs = [f for (f, _) in fracs]
                order = np.argsort(unique_fracs)
                sorted_matrix = rows_matrix[order, :]
            plot_matrix = sorted_matrix
        except Exception:
            plot_matrix = rows_matrix
    else:
        plot_matrix = np.empty((0, site_end - site_start))
    
    safe_sample = re.sub(r'[^A-Za-z0-9_.-]', '_', sample_name)
    png_name = op.join(out_dir, f"{safe_sample}.{chrom}_{site_start}-{site_end}.png")
    n_reads_n = plot_matrix.shape[0]
    
    if n_reads_n >= min_reads_plot:
        plot_heatmap_and_hist(plot_matrix, fracs_arr, png_name, sample_name, 
                              f"{chrom}:{site_start}-{site_end}",
                              bic_delta=delta_bic, biallelic_info=biallelic_info,
                              expand_counts_used=expand_counts_flag)
        print(f"   Created plot with {n_reads_n} reads", file=sys.stderr)
    else:
        png_name = None
        print(f"   Skipping plot: only {n_reads_n} reads (min required: {min_reads_plot})", file=sys.stderr)
    
    return {
        'sample': sample_name,
        'n_reads': n_reads_n,
        'n_informative': len(fracs_arr),
        'delta_bic': delta_bic,
        'is_biallelic': biallelic_info['is_biallelic'],
        'separation_score': biallelic_info['separation_score'],
        'n_low': biallelic_info['n_low'],
        'n_high': biallelic_info['n_high'],
        'n_mid': biallelic_info['n_mid'],
        'png': png_name,
        'fractions': fractions_for_total
    }

def main():
    parser = argparse.ArgumentParser(description="Per-region per-read fraction visualization with biallelic detection")
    parser.add_argument('-r', '--region', required=True, help="Region string chr:start-end")
    parser.add_argument('--glob', default="*.pat.gz", help='Glob pattern to find .pat.gz files')
    parser.add_argument('--files', nargs='*', help='Explicit list of pat files')
    parser.add_argument('--out_dir', '-o', default='.', help='Directory to write PNGs and summaries')
    parser.add_argument('--strict', action='store_true', help='Truncate reads to the region')
    parser.add_argument('--min_informative', type=int, default=10,
                        help='Minimum informative CpGs per read (default 10)')
    parser.add_argument('--expand_counts', action='store_true', help='Expand per-read counts')
    parser.add_argument('--min_reads_plot', type=int, default=10, help='Minimum reads for plotting (default 10)')
    parser.add_argument('--summary_csv', default='summary.csv', help='CSV summary filename')
    parser.add_argument('--site_coords', action='store_true',
                        help='Interpret region as CpG site-index coordinates')
    parser.add_argument('--genome', default=None, help='Genome name (e.g., hg19)')
    args = parser.parse_args()
    
    if args.files:
        pat_files = args.files
    else:
        pat_files = sorted(glob.glob(args.glob))
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.site_coords:
        chrom, s1, s2 = parse_region(args.region)
    else:
        class _GArgs:
            def __init__(self, region=None, genome=None, sites=None, array_id=None):
                self.region = region
                self.genome = genome
                self.sites = sites
                self.array_id = array_id
            def __contains__(self, key):
                return hasattr(self, key)
        gargs = _GArgs(region=args.region, genome=args.genome, sites=None, array_id=None)
        try:
            gr = GenomicRegion(gargs)
        except Exception as e:
            raise RuntimeError(f"GenomicRegion conversion failed: {e}")
        chrom = gr.chrom
        s1, s2 = gr.sites
    
    summaries = []
    all_fractions_by_sample = {}
    
    if not pat_files:
        print("No pat files found. Exiting.", file=sys.stderr)
        return
    
    for pat in pat_files:
        print(f"Processing {pat} ...", file=sys.stderr)
        try:
            summary = process_single_file(pat, chrom, s1, s2,
                                          strict=args.strict,
                                          min_informative=args.min_informative,
                                          expand_counts_flag=args.expand_counts,
                                          out_dir=args.out_dir,
                                          sample_name=op.basename(pat),
                                          min_reads_plot=args.min_reads_plot)
            summaries.append(summary)
            if len(summary['fractions']) > 0:
                all_fractions_by_sample[summary['sample']] = summary['fractions']
            
            biallelic_flag = "⭐ BIALLELIC" if summary['is_biallelic'] else ""
            print(f" -> reads={summary['n_reads']}, Δ BIC={summary['delta_bic']}, "
                  f"sep_score={summary['separation_score']:.2f} {biallelic_flag}", file=sys.stderr)
        except Exception as e:
            print(f"Failed processing {pat}: {e}", file=sys.stderr)
    
    # Create total histogram
    if all_fractions_by_sample:
        total_png = op.join(args.out_dir, f"TOTAL.{chrom}_{s1}-{s2}.png")
        print(f"Creating total histogram...", file=sys.stderr)
        total_delta_bic, outlier_info, total_biallelic = plot_total_histogram(
            all_fractions_by_sample, total_png, args.region, chrom, s1, s2
        )
        print(f"Total histogram saved to {total_png}", file=sys.stderr)
        if total_biallelic and total_biallelic['is_biallelic']:
            print(f"⭐ TOTAL shows TRUE BIALLELIC separation (score={total_biallelic['separation_score']:.2f})", 
                  file=sys.stderr)
    
    # Write summary CSV with biallelic columns
    csv_path = op.join(args.out_dir, args.summary_csv)
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['sample', 'n_reads', 'n_informative', 'delta_bic', 
                        'is_biallelic', 'separation_score', 'n_low', 'n_high', 'n_mid', 'png'])
        for s in summaries:
            writer.writerow([s.get('sample'), s.get('n_reads'), s.get('n_informative'), 
                           s.get('delta_bic'), s.get('is_biallelic'), s.get('separation_score'),
                           s.get('n_low'), s.get('n_high'), s.get('n_mid'), s.get('png')])
    print(f"Wrote summary to {csv_path}", file=sys.stderr)

if __name__ == '__main__':
    main()
