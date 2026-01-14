"""
visualise.py

A module for visualizing the pendulum systems defined in pendulum_physics.
Handles static plots, state space animations, and physical space animations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from IPython.display import HTML

# Import physics to access coords, wrapping, and solvers for on-the-fly animations
from . import physics as phys


def display_animation_html(anim):
    """Converts a FuncAnimation object to HTML for notebook display."""
    print("Rendering animation for notebook... (this may take a minute)")
    return HTML(anim.to_jshtml())


# --- Static State Space Plots ---


def plot_static_state_space_single(state_space_evolution, t_points):
    """Plots the single pendulum state space at t=0 and final time."""
    theta_0 = phys.wrap_angle(state_space_evolution[:, 0, 0])
    p_theta_0 = state_space_evolution[:, 1, 0]
    theta_final = phys.wrap_angle(state_space_evolution[:, 0, -1])
    p_theta_final = state_space_evolution[:, 1, -1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    ax1.scatter(theta_0, p_theta_0, alpha=0.5, s=5, c="blue")
    ax1.set_title(f"Initial Distribution (t = {t_points[0]:.1f}s)")
    ax1.set_xlabel(r"$\theta$ (rad)")
    ax1.set_ylabel(r"$p_\theta$")
    ax1.set_xlim([-np.pi, np.pi])
    ax1.grid(True)

    ax2.scatter(theta_final, p_theta_final, alpha=0.5, s=5, c="blue")
    ax2.set_title(f"Final Distribution (t = {t_points[-1]:.1f}s)")
    ax2.set_xlabel(r"$\theta$ (rad)")
    ax2.set_xlim([-np.pi, np.pi])
    ax2.grid(True)

    p_max = np.max(np.abs(state_space_evolution[:, 1, :])) * 1.1
    ax1.set_ylim([-p_max, p_max])
    plt.suptitle("Single Pendulum State Space Evolution", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_static_state_space_double(state_space_evolution, t_points):
    """Plots the double pendulum state space projections at t=0 and final time."""
    th1_0 = phys.wrap_angle(state_space_evolution[:, 0, 0])
    p1_0 = state_space_evolution[:, 2, 0]
    th2_0 = phys.wrap_angle(state_space_evolution[:, 1, 0])
    p2_0 = state_space_evolution[:, 3, 0]

    th1_f = phys.wrap_angle(state_space_evolution[:, 0, -1])
    p1_f = state_space_evolution[:, 2, -1]
    th2_f = phys.wrap_angle(state_space_evolution[:, 1, -1])
    p2_f = state_space_evolution[:, 3, -1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    ax1.scatter(th1_0, p1_0, alpha=0.5, s=5, c="blue", label=r"($\theta_1, p_1$)")
    ax1.scatter(th2_0, p2_0, alpha=0.5, s=5, c="green", label=r"($\theta_2, p_2$)")
    ax1.set_title(f"Initial Distribution (t = {t_points[0]:.1f}s)")
    ax1.set_xlabel(r"$\theta$ (rad)")
    ax1.set_ylabel(r"$p_\theta$")
    ax1.set_xlim([-np.pi, np.pi])
    ax1.grid(True)
    ax1.legend()

    ax2.scatter(th1_f, p1_f, alpha=0.5, s=5, c="blue", label=r"($\theta_1, p_1$)")
    ax2.scatter(th2_f, p2_f, alpha=0.5, s=5, c="green", label=r"($\theta_2, p_2$)")
    ax2.set_title(f"Final Distribution (t = {t_points[-1]:.1f}s)")
    ax2.set_xlabel(r"$\theta$ (rad)")
    ax2.set_xlim([-np.pi, np.pi])
    ax2.grid(True)
    ax2.legend()

    p_max = np.max(np.abs(state_space_evolution[:, 2:, :])) * 1.1
    ax1.set_ylim([-p_max, p_max])
    plt.suptitle("Double Pendulum State Space Projections", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_state_space_with_statistics(state_space_evolution, t_points):
    """Plots initial and final state space with overlaid statistics."""

    def get_stats(states):
        mean = np.mean(states, axis=0)
        cov = np.cov(states, rowvar=False)
        std = np.sqrt(np.diag(cov))
        return mean, cov, std

    def add_confidence_ellipse(ax, mean, cov, label=None, color="black"):
        vals, vecs = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 4 * np.sqrt(vals)
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

    all_p = state_space_evolution[:, 1, :]
    p_limit = np.max(np.abs(all_p)) * 1.1

    initial_states = state_space_evolution[:, :, 0]
    final_states = state_space_evolution[:, :, -1]

    init_mean, init_cov, init_std = get_stats(initial_states)
    final_mean, final_cov, final_std = get_stats(final_states)

    print(f"Initial: Mean={init_mean}, Std={init_std}")
    print(f"Final:   Mean={final_mean}, Std={final_std}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot Initial
    theta_0 = phys.wrap_angle(initial_states[:, 0])
    ax1.scatter(theta_0, initial_states[:, 1], alpha=0.5, s=10, c="royalblue")
    ax1.plot(
        init_mean[0],
        init_mean[1],
        "k*",
        markersize=15,
        markeredgecolor="white",
        label="Mean",
    )
    add_confidence_ellipse(ax1, init_mean, init_cov, label=r"$2\sigma$ Uncertainty")
    ax1.set_title(f"Initial (t = {t_points[0]:.1f}s)")
    ax1.set_xlabel(r"$\theta$ (rad)")
    ax1.set_ylabel(r"$p_\theta$")
    ax1.set_xlim([-np.pi, np.pi])
    ax1.set_ylim([-p_limit, p_limit])
    ax1.grid(True)
    ax1.legend()

    # Plot Final
    theta_f = phys.wrap_angle(final_states[:, 0])
    ax2.scatter(theta_f, final_states[:, 1], alpha=0.5, s=10, c="royalblue")
    ax2.plot(
        final_mean[0],
        final_mean[1],
        "k*",
        markersize=15,
        markeredgecolor="white",
        label="Mean",
    )
    add_confidence_ellipse(ax2, final_mean, final_cov, label=r"$2\sigma$ Uncertainty")
    ax2.set_title(f"Final (t = {t_points[-1]:.1f}s)")
    ax2.set_xlabel(r"$\theta$ (rad)")
    ax2.set_xlim([-np.pi, np.pi])
    ax2.set_ylim([-p_limit, p_limit])
    ax2.grid(True)
    ax2.legend()

    plt.suptitle("State Space Evolution with Statistics", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_pdf_advection_single(X, Y, pdf_initial, pdf_final, t_final):
    """Visualizes the initial and advected PDF (Static)."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), sharey=True, constrained_layout=True
    )

    ax1.contourf(X, Y, pdf_initial, levels=25, cmap="viridis")
    ax1.set_title("Prior PDF (t=0)")
    ax1.set_xlabel(r"$\theta$ ")
    ax1.set_ylabel(r"$p$")
    ax1.grid(True, alpha=0.3)

    c2 = ax2.contourf(X, Y, pdf_final, levels=25, cmap="viridis")
    ax2.set_title(f"Prior PDF (t={t_final:.1f}s)")
    ax2.set_xlabel(r"$\theta$")
    ax2.grid(True, alpha=0.3)

    cbar = fig.colorbar(c2, ax=[ax1, ax2], orientation="vertical", fraction=0.05)
    cbar.set_label("Probability Density")
    fig.suptitle("Push-forward of the prior distribution", fontsize=16)
    plt.show()


# --- Animations ---


def create_animation_state_space_single(state_space_evolution, t_points):
    """Animation of single pendulum state space ensemble."""
    fig, ax = plt.subplots(figsize=(8, 6))
    p_max = np.max(np.abs(state_space_evolution[:, 1, :])) * 1.1
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-p_max, p_max])
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$p_\theta$")
    ax.grid(True)

    scatter = ax.scatter([], [], alpha=0.5, s=5)
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes, fontsize=12)
    fig.suptitle("StateSpace Evolution (Single Pendulum)", fontsize=16)

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text("")
        return scatter, time_text

    def animate(i):
        data = np.stack(
            [
                phys.wrap_angle(state_space_evolution[:, 0, i]),
                state_space_evolution[:, 1, i],
            ]
        ).T
        scatter.set_offsets(data)
        time_text.set_text(f"Time t = {t_points[i]:.2f}s")
        return scatter, time_text

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=len(t_points), interval=40, blit=True
    )
    plt.close(fig)
    return anim


def create_physical_animation_single(t_points, solution, L1=1.0):
    """Animation of single pendulum with fading trail."""
    theta_t = solution[0, :]
    x_t, y_t = phys.get_single_coords(theta_t, L1)

    fig, ax = plt.subplots(figsize=(8, 8))
    lim = L1 * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Single Pendulum Animation")

    (line,) = ax.plot([], [], "o-", lw=2, markersize=12, c="blue", markeredgecolor="k")
    trail = LineCollection([], linewidths=1.5)
    ax.add_collection(trail)
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    trail_len = 20
    color_array = np.zeros((trail_len, 4))
    color_array[:, 0] = 1.0
    color_array[:, 1:] = np.linspace(1.0, 0.0, trail_len)[:, None]
    color_array[:, 3] = 0.8

    def update(i):
        line.set_data([0, x_t[i]], [0, y_t[i]])
        start = max(0, i - trail_len)
        if i - start > 1:
            pts = np.column_stack([x_t[start : i + 1], y_t[start : i + 1]])
            segments = np.stack([pts[:-1], pts[1:]], axis=1)
            trail.set_segments(segments)
            trail.set_color(color_array[-len(segments) :])
        else:
            trail.set_segments([])
        time_text.set_text(f"Time = {t_points[i]:.1f}s")
        return line, trail, time_text

    anim = FuncAnimation(fig, update, frames=len(t_points), interval=30, blit=True)
    plt.close(fig)
    return anim


def create_physical_animation_double(t_points, solution, L1, L2):
    """Animation of double pendulum with trails."""
    x1_t, y1_t, x2_t, y2_t = phys.get_double_coords(
        solution[0, :], solution[1, :], L1, L2
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    lim = (L1 + L2) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Double Pendulum Animation")

    (rods,) = ax.plot([], [], "-", lw=2, c="grey")
    (bob1,) = ax.plot([], [], "o", markersize=12, c="blue", markeredgecolor="k")
    (bob2,) = ax.plot([], [], "o", markersize=12, c="green", markeredgecolor="k")
    trail1 = LineCollection([], linewidths=1.5)
    ax.add_collection(trail1)
    trail2 = LineCollection([], linewidths=1.5)
    ax.add_collection(trail2)
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    trail_len = 150
    # Setup simple fading colors
    c_map1 = np.zeros((trail_len, 4))
    c_map1[:, 2] = 1.0
    c_map1[:, 3] = np.linspace(0, 0.6, trail_len)  # Fade in alpha
    c_map2 = np.zeros((trail_len, 4))
    c_map2[:, 1] = 0.5
    c_map2[:, 3] = np.linspace(0, 0.6, trail_len)

    def update(i):
        rods.set_data([0, x1_t[i], x2_t[i]], [0, y1_t[i], y2_t[i]])
        bob1.set_data([x1_t[i]], [y1_t[i]])
        bob2.set_data([x2_t[i]], [y2_t[i]])

        start = max(0, i - trail_len)
        if i - start > 1:
            # Trail 1
            pts1 = np.column_stack([x1_t[start : i + 1], y1_t[start : i + 1]])
            segs1 = np.stack([pts1[:-1], pts1[1:]], axis=1)
            trail1.set_segments(segs1)
            trail1.set_color(c_map1[-len(segs1) :])
            # Trail 2
            pts2 = np.column_stack([x2_t[start : i + 1], y2_t[start : i + 1]])
            segs2 = np.stack([pts2[:-1], pts2[1:]], axis=1)
            trail2.set_segments(segs2)
            trail2.set_color(c_map2[-len(segs2) :])
        else:
            trail1.set_segments([])
            trail2.set_segments([])
        time_text.set_text(f"Time = {t_points[i]:.1f}s")
        return rods, bob1, bob2, trail1, trail2, time_text

    anim = FuncAnimation(fig, update, frames=len(t_points), interval=30, blit=True)
    plt.close(fig)
    return anim


def animate_pdf(
    pdf_func, t_max, x_lim=(-np.pi, np.pi), y_lim=(-3, 3), res=100, frames=60
):
    """
    Animates PDF evolution using Eulerian approach.
    Re-solves the backwards integration for every frame to avoid grid distortion.
    """
    x_vals = np.linspace(x_lim[0], x_lim[1], res)
    y_vals = np.linspace(y_lim[0], y_lim[1], res)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Prepare solver args
    grid_flat = np.stack([X.ravel(), Y.ravel()])
    y_grid_vectorized = grid_flat.reshape(-1)

    # Define wrapper for vectorization (needs access to physics EOM)
    def vectorized_eom(t, y_flat):
        y_reshaped = y_flat.reshape(2, -1)
        dydt = phys.eom_single(t, y_reshaped)
        return np.concatenate(dydt).reshape(-1)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$p$")
    ax.set_title("PDF Evolution")

    Z = pdf_func(X, Y)
    mesh = ax.pcolormesh(X, Y, Z, cmap="viridis", shading="gouraud")
    fig.colorbar(mesh, ax=ax, label="Probability Density")
    time_text = ax.text(
        0.05, 0.9, "Time = 0.00s", transform=ax.transAxes, color="white"
    )

    t_points = np.linspace(0, t_max, frames)

    def update(frame):
        current_time = t_points[frame]
        if frame == 0:
            Z_new = pdf_func(X, Y)
        else:
            # Integrate backwards: current_time -> 0
            t_span = [current_time, 0.0]
            sol = phys.solve_trajectory_rk45(
                vectorized_eom, y_grid_vectorized, t_span, rtol=1e-4, atol=1e-4
            )
            origins = sol[:, -1].reshape(2, res, res)
            Z_new = pdf_func(origins[0], origins[1])

        mesh.set_array(Z_new.ravel())
        time_text.set_text(f"Time = {current_time:.2f}s")
        return mesh, time_text

    anim = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    plt.close(fig)
    return anim


def create_combined_animation_single(t_points, solution, L=1.0, stride=1):
    """
    Creates a side-by-side animation:
    Left: Physical pendulum motion.
    Right: Statespace trajectory.

    Args:
        stride (int): Only render every nth frame to speed up generation.
    """
    # Unpack solution
    theta = solution[0, :]
    p = solution[1, :]

    x_pend, y_pend = phys.get_single_coords(theta, L)

    fig, (ax_phys, ax_state) = plt.subplots(
        1, 2, figsize=(12, 6), constrained_layout=True
    )

    # --- Left Plot: Physical ---
    ax_phys.set_xlim(-1.5 * L, 1.5 * L)
    ax_phys.set_ylim(-1.5 * L, 1.5 * L)
    ax_phys.set_aspect("equal")
    ax_phys.set_title("Physical Motion")
    ax_phys.grid(True, alpha=0.3)
    ax_phys.plot(0, 0, "k+", markersize=10)
    (rod_line,) = ax_phys.plot([], [], "k-", lw=2)
    (bob_point,) = ax_phys.plot([], [], "o", color="firebrick", markersize=15, zorder=5)

    # --- Right Plot: StateSpace ---
    range_theta = np.ptp(theta)
    range_p = np.ptp(p)
    ax_state.set_xlim(
        np.min(theta) - 0.1 * range_theta, np.max(theta) + 0.1 * range_theta
    )
    ax_state.set_ylim(np.min(p) - 0.1 * range_p, np.max(p) + 0.1 * range_p)
    ax_state.set_title("State Space Trajectory")
    ax_state.set_xlabel(r"$\theta$")
    ax_state.set_ylabel(r"$p$")
    ax_state.grid(True)

    (state_curve,) = ax_state.plot([], [], "-", color="royalblue", lw=1.5, alpha=0.6)
    (state_head,) = ax_state.plot([], [], "o", color="royalblue", markersize=8)

    time_text = ax_phys.text(0.05, 0.9, "", transform=ax_phys.transAxes)

    def init():
        rod_line.set_data([], [])
        bob_point.set_data([], [])
        state_curve.set_data([], [])
        state_head.set_data([], [])
        time_text.set_text("")
        return rod_line, bob_point, state_curve, state_head, time_text

    def update(frame):
        # Update Physical Pendulum
        rod_line.set_data([0, x_pend[frame]], [0, y_pend[frame]])
        bob_point.set_data([x_pend[frame]], [y_pend[frame]])

        # Update State Space
        state_curve.set_data(theta[: frame + 1], p[: frame + 1])
        state_head.set_data([theta[frame]], [p[frame]])

        time_text.set_text(f"Time = {t_points[frame]:.1f}s")
        return rod_line, bob_point, state_curve, state_head, time_text

    # Generate indices based on stride
    frames = range(0, len(t_points), stride)

    # REMOVED: save_count argument to silence warning
    anim = FuncAnimation(
        fig, update, frames=frames, init_func=init, interval=40, blit=True
    )
    plt.close(fig)
    return anim


def create_combined_animation_double(t_points, solution, L1, L2, stride=1):
    """
    Creates a side-by-side animation for the Double Pendulum:
    Left: Physical motion.
    Right: State space trajectories (wrapped to [-pi, pi]).

    Args:
        stride (int): Only render every nth frame for speed.
    """
    # 1. Unpack and Wrap Angles to [-pi, pi]
    # We use the helper from the physics module
    th1 = phys.wrap_angle(solution[0, :])
    th2 = phys.wrap_angle(solution[1, :])
    p1 = solution[2, :]
    p2 = solution[3, :]

    # 2. Handle "Wrap Streaks"
    # When theta jumps from pi -> -pi (or vice versa), matplotlib draws a horizontal line.
    # We insert NaNs at these jumps to break the line for the plot.
    th1_plot = th1.copy()
    th2_plot = th2.copy()

    # Identify jumps (absolute difference > pi) and set those points to NaN
    th1_plot[np.abs(np.diff(th1, prepend=th1[0])) > np.pi] = np.nan
    th2_plot[np.abs(np.diff(th2, prepend=th2[0])) > np.pi] = np.nan

    # 3. Get physical coordinates
    # (Input angles to coords don't technically need wrapping, but it's safe)
    x1, y1, x2, y2 = phys.get_double_coords(solution[0, :], solution[1, :], L1, L2)

    # 4. Setup Figure
    fig, (ax_phys, ax_state) = plt.subplots(
        1, 2, figsize=(14, 6), constrained_layout=True
    )

    # --- Left: Physical ---
    lim = (L1 + L2) * 1.1
    ax_phys.set_xlim(-lim, lim)
    ax_phys.set_ylim(-lim, lim)
    ax_phys.set_aspect("equal")
    ax_phys.grid(True, alpha=0.3)
    ax_phys.set_title("Physical Motion")

    (rods,) = ax_phys.plot([], [], "k-", lw=2)
    (bob1,) = ax_phys.plot(
        [], [], "o", color="blue", markersize=10, zorder=5, label="Bob 1"
    )
    (bob2,) = ax_phys.plot(
        [], [], "o", color="green", markersize=10, zorder=5, label="Bob 2"
    )
    ax_phys.legend(loc="upper right", fontsize="small")

    # --- Right: State Space ---
    # Fixed limits for [-pi, pi]
    ax_state.set_xlim(-np.pi, np.pi)

    # Dynamic Y limits based on momentum
    all_p = np.concatenate([p1, p2])
    range_p = np.ptp(all_p)
    ax_state.set_ylim(np.min(all_p) - 0.1 * range_p, np.max(all_p) + 0.1 * range_p)

    ax_state.set_title(r"State Space (Wrapped $[-\pi, \pi]$)")
    ax_state.set_xlabel(r"$\theta$ (rad)")
    ax_state.set_ylabel(r"$p$")
    ax_state.grid(True)

    # Trails (using the NaN-injected arrays)
    (trail1,) = ax_state.plot(
        [], [], "-", color="blue", lw=1, alpha=0.5, label=r"Bob 1"
    )
    (head1,) = ax_state.plot([], [], "o", color="blue", markersize=6)

    (trail2,) = ax_state.plot(
        [], [], "-", color="green", lw=1, alpha=0.5, label=r"Bob 2"
    )
    (head2,) = ax_state.plot([], [], "o", color="green", markersize=6)
    ax_state.legend(loc="upper right", fontsize="small")

    time_text = ax_phys.text(0.05, 0.9, "", transform=ax_phys.transAxes)

    def init():
        rods.set_data([], [])
        bob1.set_data([], [])
        bob2.set_data([], [])
        trail1.set_data([], [])
        head1.set_data([], [])
        trail2.set_data([], [])
        head2.set_data([], [])
        time_text.set_text("")
        return rods, bob1, bob2, trail1, head1, trail2, head2, time_text

    def update(frame):
        # Update Physical
        rods.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
        bob1.set_data([x1[frame]], [y1[frame]])
        bob2.set_data([x2[frame]], [y2[frame]])

        # Update State Space
        # Plot the trail using the version with NaNs
        trail1.set_data(th1_plot[: frame + 1], p1[: frame + 1])
        head1.set_data([th1[frame]], [p1[frame]])

        trail2.set_data(th2_plot[: frame + 1], p2[: frame + 1])
        head2.set_data([th2[frame]], [p2[frame]])

        time_text.set_text(f"Time = {t_points[frame]:.1f}s")
        return rods, bob1, bob2, trail1, head1, trail2, head2, time_text

    frames = range(0, len(t_points), stride)

    anim = FuncAnimation(
        fig, update, frames=frames, init_func=init, interval=40, blit=True
    )
    plt.close(fig)
    return anim
