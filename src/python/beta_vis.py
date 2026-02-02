#!/usr/bin/python3 -u

import re
import os.path as op
import numpy as np
import warnings
from utils_wgbs import load_borders, load_beta_data, validate_file_list, color_text, \
        beta2vec, catch_BrokenPipeError, drop_dup_keep_order
from genomic_region import GenomicRegion

FULL_SQUARE = '\u25A0'
# FULL_SQUARE = '\u2588'
MISSING_VAL_SIGN = ' '
NR_CHARS_PER_FNAME = 50
MISSING_VAL = '.'


class BetaVis:
    def __init__(self, args):
        self.gr = GenomicRegion(args)
        self.start, self.end = self.gr.sites
        self.args = args

        # drop duplicated files, while keeping original order
        self.files = drop_dup_keep_order(args.input_files)

        # load raw data:
        self.dsets = self.load_data()

        # load borders:
        self.borders = load_borders(args.blocks_path, self.gr, args.genome)

        # Generate colors dictionary
        self.num2color_dict = generate_colors_dict(args.color_scheme)

        self.print_all()
        if self.args.plot:
            self.plot_all()

    def load_data(self):
        # raw table from *beta files:
        dsets = np.zeros((len(self.files), self.end - self.start, 2))
        for i, fpath in enumerate(self.files):
            dsets[i] = load_beta_data(fpath, self.gr.sites)
        return dsets

    def build_vals_line(self, data):
        # build a list of single character values.
        with np.errstate(divide='ignore', invalid='ignore'):
            vec = np.round((data[:, 0] / data[:, 1] * 10), 0).astype(int)  # normalize to range [0, 10)
        vec[vec == 10] = 9
        vec[data[:, 1] < self.args.min_cov] = -1
        vals = [MISSING_VAL if x == -1 else str(int(x)) for x in vec]

        # insert borders:
        if self.borders.size:
            vals = np.insert(vals, self.borders, '|')


        return self.color_vals(vals)

    def color_vals(self, vals):
        # join vals to a string line and color it:
        line = ''.join(vals)
        if not self.args.no_color:
            line = color_text(line, self.num2color_dict, scheme=self.args.color_scheme)
            if self.args.heatmap:
                line = re.sub('m[0-9]', 'm' + FULL_SQUARE * 1, line)
                line = re.sub('\.', MISSING_VAL_SIGN, line)
        return line

    def print_all(self):
        print(self.gr)

        # set the fixed number of characters for fpath names:
        fname_len = min(NR_CHARS_PER_FNAME, max([len(op.basename(op.splitext(f)[0])) for f in self.files])) + 1

        for dset, fpath in zip(self.dsets, self.files):
            line = self.build_vals_line(dset)
            adj_fname = op.splitext(op.basename(fpath))[0][:fname_len].ljust(fname_len)
            print(adj_fname + ': ' + line)

        if self.args.colorbar:
            digits = '0123456789'
            print('colorbar')
            print(self.color_vals(digits))
            if self.args.heatmap:
                print(digits)

    def plot_all(self):
        import matplotlib.pyplot as plt
        # Case 1: Highlight mode (scatter + mean line)
        if self.args.highlight is not None:
            highlight_path = self.args.highlight
            highlight_name = op.splitext(op.basename(highlight_path))[0]

            # Find index of highlighted file
            try:
                idx_highlight = [op.splitext(op.basename(f))[0] for f in self.files].index(highlight_name)
            except ValueError:
                print(f"[Error] Highlight file '{highlight_name}' not found among inputs.")
                return

            # Extract all beta values (shape: n_files x n_sites)
            betas = np.array([beta2vec(d) for d in self.dsets])  # each row = file, each col = CpG
            # Get genomic positions for x-axis
            positions = []
            for site_idx in range(self.start, self.end):
                _, locus = self.gr.index2locus(site_idx)
                positions.append(locus)
            x = np.array(positions)  # genomic positions

            #x = np.arange(betas.shape[1])  # CpG index for now

            # Compute mean of non-highlighted samples
            mask = np.ones(betas.shape[0], dtype=bool)
            mask[idx_highlight] = False
            mean_non_highlight = np.nanmean(betas[mask, :], axis=0)

            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
            # LEFT PLOT: Scatter plot with connected gray line
            # Plot gray line connecting the gray points first (so it's behind)
            for i in np.where(mask)[0]:
                ax1.plot(x, betas[i, :], color='lightgray', linewidth=0.5, alpha=0.6, zorder=1)
        
            # Plot gray dots for other samples
            ax1.scatter(
              np.tile(x, mask.sum()),
              betas[mask, :].flatten(),
              color='gray',
              s=8,
              alpha=0.4,
              label='Other samples',
              zorder=2
            )
            # Plot line for mean
            ax1.plot(x, mean_non_highlight, color='black', linewidth=2, label='Mean (others)', zorder=3)
            # Plot light blue line connecting the highlighted dots
            ax1.plot(x, betas[idx_highlight, :], color='lightblue', linewidth=1.5, alpha=0.8, zorder=3.5)
            # Plot blue dots for highlight
            ax1.scatter(x, betas[idx_highlight, :], color='blue', s=15, label=highlight_name, zorder=4)

            ax1.set_xlabel("Genomic position (bp)")
            ax1.set_ylabel("Methylation (β)")
            ax1.set_title(f"{highlight_name} {self.gr.region_str}")
        
            # RIGHT PLOT: Histogram of means
            # Calculate mean methylation for each sample in this region
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                sample_means = np.nanmean(betas, axis=1)
            # Create histogram
            # Filter out NaN means before plotting
            valid_means = sample_means[mask][~np.isnan(sample_means[mask])]
            highlight_mean = sample_means[idx_highlight]
            # Determine bins based on all valid data
            all_valid = np.concatenate([valid_means, [highlight_mean]]) if not np.isnan(highlight_mean) else valid_means
            bins = 20
            hist_range = (np.nanmin(all_valid), np.nanmax(all_valid))
        
            # Plot histogram for other samples
            ax2.hist(valid_means, bins=bins, range=hist_range, color='gray', alpha=0.7, edgecolor='black', label='Other samples')
        
            # Plot highlighted sample's mean as a blue bar
            ax2.hist([highlight_mean], bins=bins, range=hist_range, color='blue', alpha=0.8, edgecolor='black', label=f'{highlight_name}')
        
            ax2.set_xlabel("Mean methylation (β)")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Distribution of mean methylation")
            ax2.legend(frameon=False)

            plt.tight_layout()
        
            if self.args.output is not None:
                plt.savefig(self.args.output, bbox_inches='tight')
            plt.show()
            return

        # Case 2: Default heatmap plot (original behavior)
        fname_len = min(NR_CHARS_PER_FNAME, max([len(op.basename(op.splitext(f)[0])) for f in self.files]))
        ticks = [op.splitext(op.basename(f))[0][:fname_len].ljust(fname_len) for f in self.files]
        r = np.concatenate([beta2vec(d).reshape((1, -1)) for d in self.dsets])
        plt.imshow(1 - r, cmap='RdYlGn')
        if self.borders.size:
            plt.vlines(self.borders - .5, -.5, len(self.files) - .5)
        plt.yticks(np.arange(len(self.files)), ticks)
        if self.args.title:
            plt.title(self.args.title)
        if self.args.output is not None:
            plt.savefig(self.args.output)
        plt.show()


def generate_colors_dict(scheme=16):
    if scheme == 16:
        colors = [
            "01;92",  # bold light green
            "92",  # light green
            "32",  # green
            "32",  # green
            "34",  # blue
            "34",  # blue
            "02;31",  # dark red
            "02;31",  # dark red
            "31",  # red
            "01;31"  # bold red
        ]
    else:
        colors = [10, 47, 70, 28, 3, 3, 202, 204, 197, 196]
    return {str(i): colors[i] for i in range(10)}


def main(args):
    validate_file_list(args.input_files) #, '.beta')
    try:
        BetaVis(args)
    except BrokenPipeError:
        catch_BrokenPipeError()
