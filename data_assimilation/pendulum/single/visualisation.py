"""
visualisation.py

Visualisation tools specific to the Single Pendulum (2D system).
Handles phase portraits, physical animations, and Bayesian update plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse

from .. import core
from . import physics as phys


# --- Static Plots ---


def plot_bayesian_analysis(X, Y, prior, likelihood, posterior, obs_val, obs_time):
    """
    Visualizes the Bayesian update step for the single pendulum.
    Displays Prior, Likelihood, and Posterior side-by-side.

    Args:
        X, Y: Meshgrids for Theta and Momentum.
        prior: 2D array of prior PDF values.
        likelihood: 2D array of likelihood values.
        posterior: 2D array of posterior PDF values.
        obs_val: The scalar observation value (theta).
        obs_time: The time at which observation occurred.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    def plot_panel(ax, data, title, cmap="viridis"):
        im = ax.contourf(X, Y, data, levels=30, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel(r"$\theta$ (rad)")
        ax.grid(True, alpha=0.3)
        return im

    # 1. Prior
    plot_panel(axes[0], prior, f"Prior PDF (t={obs_time:.1f}s)")
    axes[0].set_ylabel(r"$p_\theta$")

    # 2. Likelihood
    plot_panel(axes[1], likelihood, "Likelihood", cmap="plasma")
    # Draw observation line
    axes[1].axvline(
        obs_val, color="white", linestyle="--", lw=2, label=f"Obs: {obs_val:.2f}"
    )
    axes[1].legend(loc="upper right")

    # 3. Posterior
    plot_panel(axes[2], posterior, "Posterior PDF")

    plt.suptitle("Bayesian Analysis Step", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_phase_portrait(ensemble_trajectories, t_points, title=None):
    """
    Plots the ensemble distribution at initial and final times.

    Args:
        ensemble_trajectories: Array of shape (N_samples, 2, N_time).
        t_points: Time array.
    """
    # Extract states
    theta_0 = core.wrap_angle(ensemble_trajectories[:, 0, 0])
    p_0 = ensemble_trajectories[:, 1, 0]

    theta_f = core.wrap_angle(ensemble_trajectories[:, 0, -1])
    p_f = ensemble_trajectories[:, 1, -1]

    # Determine plot limits based on max momentum
    p_max = np.max(np.abs(ensemble_trajectories[:, 1, :])) * 1.1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot Initial
    ax1.scatter(theta_0, p_0, alpha=0.5, s=5, c="royalblue")
    ax1.set_title(f"Initial Distribution (t = {t_points[0]:.1f}s)")
    ax1.set_ylabel(r"$p_\theta$")

    # Plot Final
    ax2.scatter(theta_f, p_f, alpha=0.5, s=5, c="firebrick")
    ax2.set_title(f"Final Distribution (t = {t_points[-1]:.1f}s)")

    for ax in (ax1, ax2):
        ax.set_xlabel(r"$\theta$ (rad)")
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-p_max, p_max])
        ax.grid(True, alpha=0.3)

    if title:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()


def plot_ensemble_stats(ensemble_trajectories, t_points):
    """
    Plots initial and final distributions with Mean and 2-Sigma Confidence Ellipses.
    """

    def add_confidence_ellipse(ax, states, color="black", label=None):
        mean = np.mean(states, axis=0)
        cov = np.cov(states, rowvar=False)
        vals, vecs = np.linalg.eig(cov)

        # Calculate angle of ellipse
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 4 * np.sqrt(vals)  # 2-sigma width/height

        ell = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            edgecolor=color,
            facecolor="none",
            lw=2,
            linestyle="--",
            label=label,
        )
        ax.add_patch(ell)
        ax.plot(mean[0], mean[1], "k*", markersize=12, label="Mean")

    # Data Prep
    states_0 = ensemble_trajectories[:, :, 0]
    states_f = ensemble_trajectories[:, :, -1]

    # Note: wrapping applies to scatter visualization
    theta_0 = core.wrap_angle(states_0[:, 0])
    theta_f = core.wrap_angle(states_f[:, 0])

    p_max = np.max(np.abs(ensemble_trajectories[:, 1, :])) * 1.1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Initial
    ax1.scatter(theta_0, states_0[:, 1], alpha=0.3, s=5, c="royalblue")
    add_confidence_ellipse(ax1, states_0, label=r"2$\sigma$")
    ax1.set_title(f"Initial (t={t_points[0]:.1f}s)")

    # Final
    ax2.scatter(theta_f, states_f[:, 1], alpha=0.3, s=5, c="firebrick")
    add_confidence_ellipse(ax2, states_f, label=r"2$\sigma$")
    ax2.set_title(f"Final (t={t_points[-1]:.1f}s)")

    for ax in (ax1, ax2):
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-p_max, p_max])
        ax.set_xlabel(r"$\theta$ (rad)")
        ax.set_ylabel(r"$p_\theta$")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


# --- Animations ---


def animate_pendulum(t_points, solution, L=1.0):
    """
    Physical space animation (x, y) with a fading trail.
    """
    theta = solution[0, :]
    x, y = phys.get_coords(theta, L)

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = 1.2 * L
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Single Pendulum")

    # Objects
    (line,) = ax.plot(
        [], [], "o-", lw=2, color="k", markerfacecolor="royalblue", markersize=8
    )
    trail = LineCollection([], linewidths=1.5, cmap="Blues")
    ax.add_collection(trail)
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    # Trail setup
    trail_len = 20

    def update(i):
        # Current Rod
        line.set_data([0, x[i]], [0, y[i]])

        # Trail
        start = max(0, i - trail_len)
        if i - start > 1:
            pts = np.column_stack([x[start : i + 1], y[start : i + 1]])
            segments = np.stack([pts[:-1], pts[1:]], axis=1)
            trail.set_segments(segments)
            # Fade alpha
            trail.set_array(np.linspace(0, 1, len(segments)))

        time_text.set_text(f"t = {t_points[i]:.1f}s")
        return line, trail, time_text

    return FuncAnimation(fig, update, frames=len(t_points), interval=30, blit=True)


def animate_phase_portrait(ensemble_trajectories, t_points):
    """
    Animates the ensemble of particles moving in phase space (Theta vs P).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate limits
    p_vals = ensemble_trajectories[:, 1, :]
    p_max = np.max(np.abs(p_vals)) * 1.1

    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-p_max, p_max])
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$p_\theta$")
    ax.grid(True)
    ax.set_title("Phase Space Evolution")

    scatter = ax.scatter([], [], alpha=0.5, s=10, c="royalblue")
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    def update(i):
        # Extract frame i
        current_theta = core.wrap_angle(ensemble_trajectories[:, 0, i])
        current_p = ensemble_trajectories[:, 1, i]

        data = np.column_stack([current_theta, current_p])
        scatter.set_offsets(data)
        time_text.set_text(f"t = {t_points[i]:.2f}s")
        return scatter, time_text

    return FuncAnimation(fig, update, frames=len(t_points), interval=50, blit=True)


def animate_combined(t_points, solution, L=1.0, stride=1):
    """
    Side-by-side animation:
    Left: Physical Motion (Real space)
    Right: Phase Space Trajectory (Theta vs P)
    """
    theta = solution[0, :]
    p = solution[1, :]
    x, y = phys.get_coords(theta, L)

    fig, (ax_phys, ax_phase) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Left: Physical ---
    ax_phys.set_xlim(-1.2 * L, 1.2 * L)
    ax_phys.set_ylim(-1.2 * L, 1.2 * L)
    ax_phys.set_aspect("equal")
    ax_phys.grid(True, alpha=0.3)
    ax_phys.set_title("Physical Motion")

    (line,) = ax_phys.plot(
        [], [], "o-", lw=2, color="k", markerfacecolor="firebrick", markersize=10
    )

    # --- Right: Phase Space ---
    ax_phase.set_xlim(-np.pi, np.pi)
    range_p = np.ptp(p)
    ax_phase.set_ylim(np.min(p) - 0.1 * range_p, np.max(p) + 0.1 * range_p)
    ax_phase.set_xlabel(r"$\theta$")
    ax_phase.set_ylabel(r"$p$")
    ax_phase.grid(True)
    ax_phase.set_title("Trajectory")

    # Handling wrap-around lines in phase space
    # We plot the full history, but we need to insert NaNs where it wraps
    # so we don't draw horizontal lines across the plot.
    theta_wrapped = core.wrap_angle(theta)
    theta_plot = theta_wrapped.copy()
    diffs = np.abs(np.diff(theta_plot, prepend=theta_plot[0]))
    theta_plot[diffs > np.pi] = np.nan

    (trace,) = ax_phase.plot([], [], "-", color="royalblue", lw=1.5, alpha=0.6)
    (head,) = ax_phase.plot([], [], "o", color="royalblue")

    time_text = ax_phys.text(0.05, 0.9, "", transform=ax_phys.transAxes)

    def update(frame):
        # Physical
        line.set_data([0, x[frame]], [0, y[frame]])

        # Phase Space
        trace.set_data(theta_plot[: frame + 1], p[: frame + 1])
        head.set_data([theta_wrapped[frame]], [p[frame]])

        time_text.set_text(f"t = {t_points[frame]:.1f}s")
        return line, trace, head, time_text

    frames = range(0, len(t_points), stride)
    return FuncAnimation(fig, update, frames=frames, interval=30, blit=True)
