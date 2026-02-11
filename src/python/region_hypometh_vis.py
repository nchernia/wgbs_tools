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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter
import numpy as np

from genomic_region import GenomicRegion
from utils_wgbs import MAX_PAT_LEN, load_beta_data, beta2vec
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
    beta_files,
    genomic_pos=None,
    strict=True,
    min_informative=10,
    upstream_extend=NANOPORE_EXTEND,
    out_png=None,
):
    """
    Visualization-only:
      1) Per-read heatmap for highlighted donor (sorted high->low methylation)
      2) Per-CpG mean methylation across all donors (from .beta files), highlight one
    """
    chrom, site_start, site_end = parse_region(region)
    region_len = site_end - site_start
    region_str = f"{chrom}:{site_start}-{site_end}"

    # -------------------------------------------------------
    # 1) Highlight donor: per-read heatmap (from .pat.gz)
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
    # 2) Per-CpG betas across all donors (from .beta files)
    # -------------------------------------------------------
    donor_names = []
    all_betas = []
    sites = (site_start, site_end)

    for bf in beta_files:
        stem = op.splitext(op.basename(bf))[0]
        # strip .pat if the beta was named donor.pat.beta
        if stem.endswith(".pat"):
            stem = stem[:-4]
        donor_names.append(stem)
        data = load_beta_data(bf, sites)
        all_betas.append(beta2vec(data))

    all_betas = np.array(all_betas)
    n_donors = len(donor_names)

    highlight_stem = op.splitext(op.basename(highlight_pat))[0]
    if highlight_stem.endswith(".pat"):
        highlight_stem = highlight_stem[:-4]
    try:
        idx_highlight = donor_names.index(highlight_stem)
    except ValueError:
        print(f"[Error] Highlight stem '{highlight_stem}' not found among beta files.",
              file=sys.stderr)
        return

    mask = np.ones(n_donors, dtype=bool)
    mask[idx_highlight] = False

    mean_others = np.nanmean(all_betas[mask], axis=0)

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
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3])

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
        ax0.set_xlabel("CpG index in region")
        ax0.set_title(f"{highlight_stem}\n{region_str}  ({n_reads_highlight} reads)")

        cbar = fig.colorbar(im, ax=ax0, fraction=0.03, ticks=[0, 1])
        cbar.ax.set_yticklabels(["Unmeth", "Meth"])

    # RIGHT: per-CpG donor means (beta_vis highlight style)
    ax1 = fig.add_subplot(gs[1])

    for i in np.where(mask)[0]:
        ax1.plot(x_scatter, all_betas[i], color="lightgray", alpha=0.6,
                 linewidth=0.7, zorder=1)

    ax1.plot(x_scatter, mean_others, color="black", linewidth=2,
             label="Mean (others)", zorder=3)
    ax1.plot(x_scatter, all_betas[idx_highlight], color="steelblue",
             linewidth=1.5, zorder=3.5)
    ax1.scatter(x_scatter, all_betas[idx_highlight], color="blue", s=15,
                label=highlight_stem, zorder=4)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Mean methylation (\u03b2)")
    ax1.set_title(f"Per-CpG methylation ({n_donors} donors)")
    ax1.legend(frameon=False, fontsize=8)
    if use_genomic:
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:.2f} Mb"))
        ax1.tick_params(axis="x", rotation=45, labelsize=8)

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Saved {out_png}", file=sys.stderr)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Highlight one donor: per-read heatmap + per-CpG means across donors"
    )
    parser.add_argument("-r", "--region", required=True,
                        help="Region string (genomic coords or CpG site indices)")
    parser.add_argument("--highlight", required=True,
                        help="Pat file (.pat.gz) or donor stem (e.g. 1001075)")
    parser.add_argument("--files", nargs="*", help="Explicit list of .beta files")
    parser.add_argument("--glob", default="*.beta", help="Glob pattern for .beta files")
    parser.add_argument("-o", "--out_png", default=None, help="Output PNG path")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--min_informative", type=int, default=10)
    parser.add_argument("--site_coords", action="store_true",
                        help="Region is already in CpG site-index coordinates")
    parser.add_argument("--genome", default=None, help="Genome name (e.g. hg19)")
    parser.add_argument("--illumina", action="store_true",
                        help="Use short upstream extend (Illumina reads; default is nanopore/long-read)")
    args = parser.parse_args()

    # Resolve highlight pat file from stem if needed
    highlight_pat = args.highlight
    if not highlight_pat.endswith(".pat.gz"):
        candidate = f"{highlight_pat}.pat.gz"
        if op.isfile(candidate):
            highlight_pat = candidate
        else:
            print(f"[Error] Cannot find pat file for '{highlight_pat}' "
                  f"(tried {candidate})", file=sys.stderr)
            sys.exit(1)

    beta_files = args.files if args.files else sorted(glob.glob(args.glob))
    if not beta_files:
        print("No beta files found.", file=sys.stderr)
        sys.exit(1)

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

    upstream_extend = MAX_PAT_LEN if args.illumina else NANOPORE_EXTEND

    # Genomic positions for x-axis
    genomic_pos = None
    if gr is not None:
        try:
            genomic_pos = get_genomic_positions(gr, s1, s2)
        except Exception as e:
            print(f"Warning: could not get genomic positions: {e}", file=sys.stderr)

    region_str = f"{chrom}:{s1}-{s2}"

    # Auto-generate output filename if not specified
    out_png = args.out_png
    if out_png is None:
        donor_stem = op.splitext(op.basename(highlight_pat))[0]
        if donor_stem.endswith(".pat"):
            donor_stem = donor_stem[:-4]
        safe_region = args.region.replace(":", "_").replace("-", "-")
        out_png = f"{donor_stem}_{safe_region}.png"

    plot_region_highlight_donor(
        region_str,
        highlight_pat,
        beta_files,
        genomic_pos=genomic_pos,
        strict=args.strict,
        min_informative=args.min_informative,
        upstream_extend=upstream_extend,
        out_png=out_png,
    )


if __name__ == "__main__":
    main()
