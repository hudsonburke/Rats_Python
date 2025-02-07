import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Optional, Union, Tuple
import numpy.typing as npt


def plot_sdv(ax: plt.Axes, time: npt.NDArray, data: npt.NDArray, std: npt.NDArray, color: str):
    """Plot standard deviation as a shaded area."""
    ax.fill_between(time, data - std, data + std, color=color, alpha=0.2)

def plot_spm(ax: plt.Axes, spm1: npt.NDArray, spm2: npt.NDArray):
    """Plot SPM analysis results."""
    ymin, ymax = ax.get_ylim()
    height = (ymax - ymin) * 0.05
    bottom = ymin - height
    
    # Plot SPM bars
    for i in range(spm1.shape[1]):
        if np.any(spm1[:, i]):
            ax.fill_between(np.arange(len(spm1)), 
                          [bottom]*len(spm1), 
                          [bottom + height]*len(spm1),
                          where=spm1[:, i].astype(bool),
                          color='red',
                          alpha=0.5)
    
    for i in range(spm2.shape[1]):
        if np.any(spm2[:, i]):
            ax.fill_between(np.arange(len(spm2)),
                          [bottom - height]*len(spm2),
                          [bottom]*len(spm2),
                          where=spm2[:, i].astype(bool),
                          color='blue',
                          alpha=0.5)
    
    ax.set_ylim(bottom - height, ymax)

def ratplot(ik_data: npt.NDArray, 
           id_data: npt.NDArray,
           grf_data: npt.NDArray,
           ik_labels: List[str],
           id_labels: List[str],
           grf_labels: List[str],
           fig: Optional[plt.Figure] = None,
           legend_labels: Optional[List[str]] = None,
           side: str = '',
           stance_swing: bool = False,
           normalized: bool = False,
           std_ik_data: Optional[npt.NDArray] = None,
           std_id_data: Optional[npt.NDArray] = None,
           std_grf_data: Optional[npt.NDArray] = None,
           spm_ik_data: Optional[npt.NDArray] = None,
           spm_id_data: Optional[npt.NDArray] = None,
           spm_grf_data: Optional[npt.NDArray] = None) -> Tuple[plt.Figure, GridSpec]:
    """
    Plot kinematic, kinetic and GRF data for rat gait analysis.
    
    Parameters similar to MATLAB version, but using Python types.
    Returns the figure and gridspec objects.
    """
    # Validate inputs
    if not (ik_data.shape[0] == id_data.shape[0] == grf_data.shape[0]):
        raise ValueError("ik_data, id_data, and grf_data must have the same number of rows")
    
    if not (ik_data.shape[2] == id_data.shape[2] == grf_data.shape[2]):
        raise ValueError("ik_data, id_data, and grf_data must have the same number of trials")

    # Create figure if not provided
    if fig is None:
        fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(4, 5, figure=fig)

    # Setup configuration
    side_prefix = f"{side}_" if side else ""
    
    tiles = [0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    labels = [
        "sacrum_pitch", "sacrum_roll", "sacrum_yaw",
        "GRF_x", "GRF_z", "GRF_y",
        f"hip_{side_prefix}flx", f"hip_{side_prefix}add", f"hip_{side_prefix}int",
        f"knee_{side_prefix}flx", f"ankle_{side_prefix}flx",
        f"hip_{side_prefix}flx_moment", f"hip_{side_prefix}add_moment",
        f"hip_{side_prefix}int_moment", f"knee_{side_prefix}flx_moment",
        f"ankle_{side_prefix}flx_moment"
    ]

    # Define y-axis limits
    ylims = {
        # ... existing ylims dictionary with the same values as MATLAB version ...
    }
    
    if normalized:
        # Update ylims for normalized data
        ylims.update({
            "GRF_x": [-1.25, 1.25],
            "GRF_z": [-2.25, 1],
            "GRF_y": [-0.1, 9],
            f"hip_{side_prefix}flx_moment": [-0.2, 0.1],
            f"hip_{side_prefix}add_moment": [-0.2, 0.1],
            f"hip_{side_prefix}int_moment": [-0.2, 0.2],
            f"knee_{side_prefix}flx_moment": [-0.2, 0.1],
            f"ankle_{side_prefix}flx_moment": [-0.2, 0.1]
        })

    titles = [
        "Sacrum Pitch", "Sacrum Roll", "Sacrum Yaw",
        "Anterior-Posterior GRF", "Mediolateral GRF", "Vertical GRF",
        "Hip Flexion", "Hip Adduction", "Hip Internal Rotation",
        "Knee Flexion", "Ankle Flexion",
        "Hip Flexion Moment", "Hip Adduction Moment", "Hip Internal Rotation Moment",
        "Knee Flexion Moment", "Ankle Flexion Moment"
    ]

    # Combine data and labels
    all_data = np.concatenate([ik_data, id_data, grf_data], axis=1)
    all_labels = ik_labels + id_labels + grf_labels
    
    # Handle standard deviation data if provided
    if std_ik_data is not None:
        std_data = np.concatenate([std_ik_data, std_id_data, std_grf_data], axis=1)
    else:
        std_data = None

    # Handle SPM data if provided
    if spm_ik_data is not None:
        spm_data = np.concatenate([spm_ik_data, spm_id_data, spm_grf_data], axis=1)
    else:
        spm_data = None

    # Time vector
    time = np.arange(all_data.shape[0])

    # Create plots
    for i, (tile, label, title) in enumerate(zip(tiles, labels, titles)):
        ax = fig.add_subplot(gs[tile])
        
        # Get data index
        idx = all_labels.index(label)
        
        for trial in range(all_data.shape[2]):
            if stance_swing:
                plot_data = np.concatenate([all_data[:101, idx, trial], all_data[102:, idx, trial]])
                plot_time = np.concatenate([time[:101], time[102:]])
            else:
                plot_data = all_data[:, idx, trial]
                plot_time = time

            line, = ax.plot(plot_time, plot_data, linewidth=1.5)
            
            if std_data is not None:
                if stance_swing:
                    std_plot = np.concatenate([std_data[:101, idx, trial], std_data[102:, idx, trial]])
                else:
                    std_plot = std_data[:, idx, trial]
                plot_sdv(ax, plot_time, plot_data, std_plot, line.get_color())

        # Set limits and title
        ax.set_ylim(ylims.get(label, [-1, 1]))
        ax.set_xlim([plot_time[0], plot_time[-1]])
        ax.set_title(title)

        if stance_swing:
            ax.set_xticks([0, 100, 200])
            ax.set_xticklabels(['0', '100  0', '100'])
            ax.axvline(x=100, color='k', linewidth=2)
            ax.set_xlabel('Percent Stance   |   Percent Swing')

        # Plot SPM data if provided
        if spm_data is not None:
            if stance_swing:
                spm_plot1 = np.concatenate([spm_data[:101, idx, :, 0], spm_data[102:, idx, :, 0]])
                spm_plot2 = np.concatenate([spm_data[:101, idx, :, 1], spm_data[102:, idx, :, 1]])
            else:
                spm_plot1 = spm_data[:, idx, :, 0]
                spm_plot2 = spm_data[:, idx, :, 1]
            plot_spm(ax, spm_plot1, spm_plot2)

    # Set labels for axes groups
    normalizing_unit = '/kg' if normalized else ''
    
    fig.text(0.04, 0.8, 'Pelvis angle (deg)', rotation=90)
    fig.text(0.04, 0.6, f'Force (N{normalizing_unit})', rotation=90)
    fig.text(0.04, 0.4, 'Joint angle (deg)', rotation=90)
    fig.text(0.04, 0.2, f'Joint moment (Nm{normalizing_unit})', rotation=90)

    if not stance_swing:
        fig.text(0.5, 0.04, 'Percent Gait Cycle', ha='center')

    if legend_labels:
        fig.legend(legend_labels, bbox_to_anchor=(0.95, 0.9))

    plt.tight_layout()
    
    return fig, gs
