# plotter.py
"""
Plotting utilities for the swept-LO simulation.

Styling follows the spec2flux convention:
    - seaborn whitegrid
    - muted colour palette
    - clean axis labels and legends
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass

# Colour palette (muted tones matching spec2flux)
C = {
    'recon':    '#49506F',
    'analytic': '#C05746',
    'residual': '#416788',
    'lo':       '#5D7A3E',
    'signal':   '#DC965A',
    'mini_r':   '#8B1E3F',
    'mini_a':   '#BBB891',
}


class SimPlotter:
    """
    Handles all visualisation for the simulation.

    Arguments:
        save_dir: directory to save figures (if None, uses cwd)
    """

    def __init__(self, save_dir: str = '.'):
        self.save_dir = save_dir

    def _fig(self, title, xlabel, ylabel, figsize=(10, 5)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
        return fig, ax

    # ----- broadband stitched spectrum vs analytic ----- 
    def plot_stitched(self, f_opt, Ps_recon, Ps_analytic,
                      title='Reconstructed Broadband Spectrum',
                      fname='stitched.png', smooth_N=1,
                      xlim=None, psd_scale=1.0):
        """
        Two-panel plot: residual on top, recon vs analytic below.
        """
        from .helpers import boxcar_smooth as bcs

        x  = f_opt / 1e12
        yr = Ps_recon * psd_scale
        ya = Ps_analytic * psd_scale

        if smooth_N > 1:
            x  = bcs(x,  smooth_N)
            yr = bcs(yr, smooth_N)
            ya = bcs(ya, smooth_N)

        fig = plt.figure(figsize=(10, 7))

        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(x, yr - ya, lw=0.8, color=C['residual'])
        ax1.set_ylabel('Residual', fontsize=11)
        ax1.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(x, yr, lw=1, color=C['recon'],
                 label='Reconstructed', alpha=0.9)
        ax2.plot(x, ya, lw=1, color=C['analytic'],
                 label='Analytic', alpha=0.9)
        ax2.set_xlabel('Optical Frequency (THz)', fontsize=11)
        ax2.set_ylabel('Power Spectrum (a.u.)', fontsize=11)
        ax2.legend(fontsize=9, loc='upper left')
        ax2.grid(True, alpha=0.3)
        if xlim:
            ax2.set_xlim(xlim)

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{fname}', dpi=180,
                    bbox_inches='tight')
        plt.close()

    # ----- per-tooth mini-spectra ----- 
    def plot_mini_spectra(self, f_IF, recon_list, theory_list,
                          labels, title='Per-Tooth Mini-Spectra',
                          fname='mini_spectra.png'):
        n = len(recon_list)
        fig, axs = plt.subplots(1, n, figsize=(3.5*n, 3.5), sharey=True)
        if n == 1:
            axs = [axs]

        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

        for k in range(n):
            axs[k].plot(f_IF / 1e9, recon_list[k], lw=1,
                        color=C['mini_r'], label='Recon')
            axs[k].plot(f_IF / 1e9, theory_list[k], lw=1,
                        color=C['mini_a'], label='Analytic')
            axs[k].set_title(labels[k], fontsize=10)
            axs[k].set_xlim([-5, 5])
            axs[k].grid(True, alpha=0.3)
            axs[k].tick_params(labelsize=8)
            if k == 0:
                axs[k].set_ylabel('Mini-spectrum (a.u.)', fontsize=10)
                axs[k].legend(fontsize=7)
            axs[k].set_xlabel('IF (GHz)', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{fname}', dpi=180,
                    bbox_inches='tight')
        plt.close()

    # ----- raw PSD ----- 
    def plot_raw_psd(self, f_IF, P_psd, psd_units=1.0,
                     title='Raw PSD', fname='raw_psd.png'):
        fig, ax = self._fig(title, 'IF (GHz)', 'PSD (nW$^2$/MHz)')
        ax.plot(f_IF / 1e9, P_psd * psd_units, lw=0.8, color=C['lo'])
        ax.set_xlim([-5, 5])
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{fname}', dpi=180,
                    bbox_inches='tight')
        plt.close()