#!/usr/bin/env python
"""
Full Python script to simulate and visualize a phase space ensemble
for a Hamiltonian pendulum.

Includes:
1. Hamiltonian EOMs.
2. RK45 and Symplectic (Leapfrog) integrators.
3. Energy conservation comparison.
4. Gaussian ensemble sampling.
5. Static phase space plots.
6. Animation of phase space evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display  # Used to display the animation in notebooks
import time

# --- 1. Hamiltonian, EOMs, and Energy ---


def pendulum_eom(t, y):
    """
    Defines the Hamiltonian equations of motion for a simple pendulum.
    State vector y is [theta, p_theta].
    Units: m=1, L=1, g=1.
    """
    theta, p_theta = y
    dtheta_dt = p_theta
    dp_theta_dt = -np.sin(theta)  # F(q) = -∂H/∂q = -sin(theta)
    return [dtheta_dt, dp_theta_dt]


def calculate_energy(y):
    """
    Calculates the Hamiltonian (energy) for a state or trajectory.
    H = T + V = p^2 / 2 - cos(theta)

    Args:
        y (np.array): Can be a single state (2,) or a trajectory (2, n_times)
    """
    theta = y[0]
    p_theta = y[1]
    return p_theta**2 / 2.0 - np.cos(theta)


def wrap_angle(theta):
    """Wraps an angle or array of angles to the interval [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


# --- 2. Integrator Functions (The Solvers) ---


def solve_pendulum_rk45(initial_condition, t_points, rtol=1e-6, atol=1e-9):
    """
    Integrates the EOMs using SciPy's 'RK45' (Dormand-Prince) method.
    This is a general-purpose, non-symplectic adaptive solver.
    """
    t_span = (t_points[0], t_points[-1])
    sol = solve_ivp(
        pendulum_eom,
        t_span,
        initial_condition,
        t_eval=t_points,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    return sol.y


def solve_pendulum_symplectic(initial_condition, t_points):
    """
    Integrates the EOMs using a fixed-step 2nd-order Leapfrog integrator.
    This is a symplectic method, excellent for energy conservation.

    Note: t_points MUST be equally spaced for this simple implementation.
    """
    theta_0, p_theta_0 = initial_condition
    n_times = len(t_points)

    # Check for fixed time step
    dt_values = np.diff(t_points)
    if not np.allclose(dt_values, dt_values[0]):
        raise ValueError(
            "t_points must be an equally spaced array for this simple symplectic integrator."
        )
    dt = dt_values[0]

    # Pre-allocate solution array
    sol = np.zeros((2, n_times))
    sol[:, 0] = [theta_0, p_theta_0]

    # Current state
    theta = theta_0
    p_theta = p_theta_0

    for i in range(n_times - 1):
        # 1. Kick (half step) - update momentum
        p_half = p_theta - np.sin(theta) * (dt / 2.0)

        # 2. Drift (full step) - update position
        theta_new = theta + p_half * dt

        # 3. Kick (half step) - update momentum with new position's force
        p_new = p_half - np.sin(theta_new) * (dt / 2.0)

        # Store and update
        sol[:, i + 1] = [theta_new, p_new]
        theta, p_theta = theta_new, p_new

    return sol


# --- 3. Ensemble Sampling and Solving ---


def sample_initial_conditions(mean, cov, n_samples):
    """Draws n_samples from a 2D multivariate Gaussian distribution."""
    print(f"Sampling {n_samples} points from Gaussian distribution...")
    return np.random.multivariate_normal(mean, cov, n_samples)


def solve_for_distribution(initial_conditions, t_points, solver_func, **solver_kwargs):
    """
    Solves the EOM for every initial condition in the provided set
    using the specified solver function.

    Args:
        initial_conditions (np.array): Array of shape (n_samples, 2).
        t_points (np.array): Array of time points.
        solver_func (function): The integrator to use
                               (e.g., solve_pendulum_rk45 or solve_pendulum_symplectic).
        **solver_kwargs: Extra arguments to pass to the solver (e.g., rtol).

    Returns:
        np.array: A 3D array of shape (n_samples, 2, n_times)
    """
    n_samples = initial_conditions.shape[0]
    n_times = len(t_points)
    all_trajectories = np.zeros((n_samples, 2, n_times))

    print(f"Solving for {n_samples} trajectories using {solver_func.__name__}...")
    start_time = time.time()

    for i in range(n_samples):
        all_trajectories[i, :, :] = solver_func(
            initial_conditions[i], t_points, **solver_kwargs
        )

    end_time = time.time()
    print(f"Done. Solving took {end_time - start_time:.2f} seconds.")
    return all_trajectories


# --- 4. Visualization Functions ---


def compare_energy_conservation():
    """
    Solves for a single trajectory with both methods and
    plots the energy drift over a long time.
    """
    print("--- Running Energy Conservation Comparison ---")

    # --- FIX 1: Change Initial Condition ---
    # From [0.0, 2.0] (separatrix) to [1.5, 0.0] (a normal oscillator)
    IC = [1.5, 0.0]
    T_LONG = 1000.0  # Long simulation time

    # --- FIX 2: Increase N_STEPS ---
    # From 5000 to 50000. This makes dt = 0.02s, which is much better.
    N_STEPS = 50000

    t_long = np.linspace(0, T_LONG, N_STEPS)

    # Solve with RK45 (default tolerances)
    print("Solving with RK45 (default)...")
    sol_rk45_def = solve_pendulum_rk45(IC, t_long)
    energy_rk45_def = calculate_energy(sol_rk45_def)

    # Solve with RK45 (tight tolerances)
    print("Solving with RK45 (tight)...")
    sol_rk45_tight = solve_pendulum_rk45(IC, t_long, rtol=1e-12, atol=1e-12)
    energy_rk45_tight = calculate_energy(sol_rk45_tight)

    # Solve with Symplectic (Leapfrog)
    print("Solving with Symplectic Leapfrog...")
    sol_leapfrog = solve_pendulum_symplectic(IC, t_long)
    energy_leapfrog = calculate_energy(sol_leapfrog)

    # --- Plot the results ---
    fig, ax = plt.subplots(figsize=(12, 7))
    E0 = energy_rk45_def[0]

    print("Plotting...")
    ax.plot(t_long, energy_rk45_def - E0, label=f"RK45 (default tol)", color="red")
    ax.plot(
        t_long,
        energy_rk45_tight - E0,
        label=f"RK45 (tight tol, $10^{{-12}}$)",
        color="orange",
        linestyle="--",
    )
    ax.plot(
        t_long,
        energy_leapfrog - E0,
        label=f"Symplectic Leapfrog (dt={t_long[1]:.4f})",
        color="blue",
        alpha=0.8,
    )

    ax.set_title("Energy Conservation Comparison (Long-Term Drift)", fontsize=16)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy Drift $\Delta E = E(t) - E(0)$")
    ax.legend()
    ax.grid(True)
    # Set a tight y-limit to see the drift
    max_drift = np.max(np.abs(energy_rk45_def - E0)) * 1.1
    ax.set_ylim(-max_drift, max_drift)
    plt.tight_layout()
    plt.show()
    print("--- Energy Comparison Finished ---")


def plot_static(phase_space_evolution, t_points):
    """Plots the phase space distribution at t=0 and the final time."""
    print("Generating static plot...")

    theta_0 = wrap_angle(phase_space_evolution[:, 0, 0])
    p_theta_0 = phase_space_evolution[:, 1, 0]
    theta_final = wrap_angle(phase_space_evolution[:, 0, -1])
    p_theta_final = phase_space_evolution[:, 1, -1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    ax1.scatter(theta_0, p_theta_0, alpha=0.5, s=5, c="blue")
    ax1.set_title(f"Initial Distribution (t = {t_points[0]:.1f}s)")
    ax1.set_xlabel(r"Angle $\theta$ (rad)")
    ax1.set_ylabel(r"Momentum $p_\theta$")
    ax1.set_xlim([-np.pi, np.pi])
    ax1.grid(True)

    ax2.scatter(theta_final, p_theta_final, alpha=0.5, s=5, c="red")
    ax2.set_title(f"Final Distribution (t = {t_points[-1]:.1f}s)")
    ax2.set_xlabel(r"Angle $\theta$ (rad)")
    ax2.set_xlim([-np.pi, np.pi])
    ax2.grid(True)

    p_max = np.max(np.abs(phase_space_evolution[:, 1, :])) * 1.1
    ax1.set_ylim([-p_max, p_max])

    plt.suptitle("Phase Space Evolution of Pendulum Ensemble", fontsize=16)
    plt.tight_layout()
    plt.show()


def create_animation_object(phase_space_evolution, t_points):
    """
    Creates and returns the Matplotlib FuncAnimation object.
    This object can then be saved or converted to HTML.
    """
    print("Creating animation object...")
    fig, ax = plt.subplots(figsize=(8, 6))

    p_max = np.max(np.abs(phase_space_evolution[:, 1, :])) * 1.1
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-p_max, p_max])
    ax.set_xlabel(r"Angle $\theta$ (rad)")
    ax.set_ylabel(r"Momentum $p_\theta$")
    ax.grid(True)

    scatter = ax.scatter([], [], alpha=0.5, s=5)
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes, fontsize=12)
    fig.suptitle("Phase Space Evolution", fontsize=16)

    def init():
        """Initializes the animation."""
        scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text("")
        return scatter, time_text

    def animate(i):
        """Updates the animation for frame i."""
        theta_i = wrap_angle(phase_space_evolution[:, 0, i])
        p_theta_i = phase_space_evolution[:, 1, i]
        data = np.stack([theta_i, p_theta_i]).T
        scatter.set_offsets(data)
        time_text.set_text(f"Time t = {t_points[i]:.2f}s")
        return scatter, time_text

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(t_points),
        interval=40,  # 40ms = 25 frames per second
        blit=True,
    )

    plt.close(fig)  # Prevent the static figure from displaying
    return anim


def display_animation_html(anim):
    """Converts a FuncAnimation object to HTML for notebook display."""
    print("Rendering animation for notebook... (this may take a minute)")
    return HTML(anim.to_jshtml())
