#!/usr/bin/env python3
"""
region_hypometh_vis.py - Highlight one donor against all donors in a region

  Plot 1: Per-read methylation heatmap for highlighted donor (sorted high -> low)
  Plot 2: Per-CpG mean methylation across all donors; highlighted donor in blue
"""
import argparse
import glob
import os.path as op
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

from genomic_region import GenomicRegion
from utils_wgbs import MAX_PAT_LEN
from region_bimodal_vis import (
    parse_region,
    pull_tabix,
    parse_pat_lines,
    expand_counts_list,
    expand_rows,
    NANOPORE_EXTEND,
)


def get_genomic_positions(gr, site_start, site_end):
    """Convert CpG site indices to genomic bp positions."""
    positions = []
    for idx in range(site_start, site_end):
        _, locus = gr.index2locus(idx)
        positions.append(locus)
    return np.array(positions)


def plot_region_highlight_donor(
    region,
    highlight_pat,
    pat_files,
    genomic_pos=None,
    strict=True,
    min_informative=10,
    upstream_extend=MAX_PAT_LEN,
    out_png=None,
):
    """
    Visualization-only:
      1) Per-read heatmap for highlighted donor (sorted high->low methylation)
      2) Per-CpG mean methylation across all donors, highlight one donor
    """
    chrom, site_start, site_end = parse_region(region)
    region_len = site_end - site_start
    region_str = f"{chrom}:{site_start}-{site_end}"

    # -------------------------------------------------------
    # Helper: per-donor CpG means + read count
    # -------------------------------------------------------
    def donor_cpg_means(pat_file):
        new_start = max(1, site_start - upstream_extend)
        pat_region = f"{chrom}:{new_start}-{site_end - 1}"
        pat_text = pull_tabix(pat_file, pat_region)

        fracs, read_rows = parse_pat_lines(
            pat_text,
            site_start,
            site_end,
            strict=strict,
            min_informative=min_informative,
        )

        if not read_rows:
            return np.full(region_len, np.nan), 0

        rows = expand_rows(read_rows)
        mat = np.vstack(rows)
        return np.nanmean(mat, axis=0), mat.shape[0]

    # -------------------------------------------------------
    # 1) Highlight donor: per-read heatmap
    # -------------------------------------------------------
    new_start = max(1, site_start - upstream_extend)
    pat_region = f"{chrom}:{new_start}-{site_end - 1}"
    pat_text = pull_tabix(highlight_pat, pat_region)

    fracs, read_rows = parse_pat_lines(
        pat_text,
        site_start,
        site_end,
        strict=strict,
        min_informative=min_informative,
    )

    if read_rows:
        rows = expand_rows(read_rows)
        fracs_exp = expand_counts_list(fracs)

        fracs_arr = np.array(fracs_exp)
        rows_mat = np.vstack(rows)

        # sort HIGH -> LOW methylation
        order = np.argsort(-fracs_arr)
        rows_sorted = rows_mat[order, :]
    else:
        rows_sorted = np.empty((0, region_len))

    n_reads_highlight = rows_sorted.shape[0]

    # -------------------------------------------------------
    # 2) Per-CpG means across all donors
    # -------------------------------------------------------
    donor_names = []
    all_means = []
    donor_read_counts = []

    for pf in pat_files:
        donor_names.append(op.splitext(op.basename(pf))[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means, n_reads = donor_cpg_means(pf)
            all_means.append(means)
            donor_read_counts.append(n_reads)

    all_means = np.array(all_means)
    n_donors = len(donor_names)

    highlight_name = op.splitext(op.basename(highlight_pat))[0]
    idx_highlight = donor_names.index(highlight_name)

    mask = np.ones(n_donors, dtype=bool)
    mask[idx_highlight] = False

    mean_others = np.nanmean(all_means[mask], axis=0)

    # -------------------------------------------------------
    # X-axis: genomic positions or CpG index fallback
    # -------------------------------------------------------
    use_genomic = genomic_pos is not None and len(genomic_pos) == region_len
    if use_genomic:
        x_scatter = genomic_pos
        x_label = f"Genomic position ({chrom})"
    else:
        x_scatter = np.arange(region_len)
        x_label = "CpG index in region"

    # -------------------------------------------------------
    # Plot
    # -------------------------------------------------------
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

    # LEFT: per-read heatmap (discrete: 0=blue, 1=red, NaN=gray)
    ax0 = fig.add_subplot(gs[0])
    if n_reads_highlight == 0:
        ax0.text(0.5, 0.5, "No reads", ha="center", va="center")
        ax0.set_axis_off()
    else:
        cmap = ListedColormap(["#4575b4", "#d73027"])  # blue, red
        bounds = [-0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, cmap.N)
        cmap.set_bad(color="lightgray")

        im = ax0.imshow(
            rows_sorted,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
        ax0.set_ylabel("Reads (sorted high \u2192 low methylation)")
        ax0.set_title(f"{highlight_name}\n{region_str}  ({n_reads_highlight} reads)")

        # Genomic position x-axis on heatmap
        if use_genomic:
            n_ticks = min(6, region_len)
            tick_idx = np.linspace(0, region_len - 1, n_ticks, dtype=int)
            ax0.set_xticks(tick_idx)
            ax0.set_xticklabels(
                [f"{genomic_pos[i]:,}" for i in tick_idx],
                rotation=45, ha="right", fontsize=8,
            )
            ax0.set_xlabel(x_label)

        cbar = fig.colorbar(im, ax=ax0, fraction=0.03, ticks=[0, 1])
        cbar.ax.set_yticklabels(["Unmeth", "Meth"])

    # RIGHT: per-CpG donor means (beta_vis highlight style)
    ax1 = fig.add_subplot(gs[1])

    for i in np.where(mask)[0]:
        ax1.plot(x_scatter, all_means[i], color="lightgray", alpha=0.6,
                 linewidth=0.7, zorder=1)

    ax1.plot(x_scatter, mean_others, color="black", linewidth=2,
             label="Mean (others)", zorder=3)
    ax1.plot(x_scatter, all_means[idx_highlight], color="lightblue",
             linewidth=1.5, zorder=3.5)
    ax1.scatter(x_scatter, all_means[idx_highlight], color="blue", s=15,
                label=highlight_name, zorder=4)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Mean methylation (\u03b2)")
    ax1.set_title(f"Per-CpG methylation ({n_donors} donors)")
    ax1.legend(frameon=False, fontsize=8)
    if use_genomic:
        ax1.ticklabel_format(style="plain", axis="x")
        ax1.tick_params(axis="x", rotation=45, labelsize=8)

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Saved {out_png}", file=sys.stderr)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Highlight one donor: per-read heatmap + per-CpG means across donors"
    )
    parser.add_argument("-r", "--region", required=True,
                        help="Region string (genomic coords or CpG site indices)")
    parser.add_argument("--highlight", required=True,
                        help="Pat file for the donor to highlight")
    parser.add_argument("--files", nargs="*", help="Explicit list of pat files")
    parser.add_argument("--glob", default="*.pat.gz", help="Glob pattern for pat files")
    parser.add_argument("-o", "--out_png", default=None, help="Output PNG path")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--min_informative", type=int, default=10)
    parser.add_argument("--site_coords", action="store_true",
                        help="Region is already in CpG site-index coordinates")
    parser.add_argument("--genome", default=None, help="Genome name (e.g. hg19)")
    parser.add_argument("-np", "--nanopore", action="store_true",
                        help="Use large upstream extend for long reads")
    args = parser.parse_args()

    pat_files = args.files if args.files else sorted(glob.glob(args.glob))
    if not pat_files:
        print("No pat files found.", file=sys.stderr)
        sys.exit(1)

    if args.highlight not in pat_files:
        pat_files.append(args.highlight)

    # Convert region to CpG site indices
    gr = None
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
        gr = GenomicRegion(_GArgs(region=args.region, genome=args.genome))
        chrom = gr.chrom
        s1, s2 = gr.sites

    upstream_extend = NANOPORE_EXTEND if args.nanopore else MAX_PAT_LEN

    # Genomic positions for x-axis
    genomic_pos = None
    if gr is not None:
        try:
            genomic_pos = get_genomic_positions(gr, s1, s2)
        except Exception as e:
            print(f"Warning: could not get genomic positions: {e}", file=sys.stderr)

    region_str = f"{chrom}:{s1}-{s2}"
    plot_region_highlight_donor(
        region_str,
        args.highlight,
        pat_files,
        genomic_pos=genomic_pos,
        strict=args.strict,
        min_informative=args.min_informative,
        upstream_extend=upstream_extend,
        out_png=args.out_png,
    )


if __name__ == "__main__":
    main()
