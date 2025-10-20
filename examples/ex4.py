import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import time

# --- 1. Physics: Double Pendulum EOMs (Hamiltonian) ---
#
# State vector y = [theta1, theta2, p1, p2]
# Using simplified units: m1=m2=1, L1=L2=1, g=1
#
# The Hamiltonian is H = T + V
# V = -2*cos(theta1) - cos(theta2)
# T = (p1^2 + 2*p2^2 - 2*p1*p2*cos(t1-t2)) / (2 * (1 + sin(t1-t2)^2))
#
# Hamilton's Equations are:
# d(theta_i)/dt =  ∂H/∂p_i
# d(p_i)/dt     = -∂H/∂theta_i
#
# These get very complex. It's simpler to derive them from the
# Lagrangian and then express them using the momenta.
#
# d(theta1)/dt = (p1 - p2*cos(t1-t2)) / (1 + sin(t1-t2)^2)
# d(theta2)/dt = (2*p2 - p1*cos(t1-t2)) / (1 + sin(t1-t2)^2)
#
# d(p1)/dt = -d(th1)/dt * d(th2)/dt * sin(t1-t2) - 2*sin(t1)
# d(p2)/dt =  d(th1)/dt * d(th2)/dt * sin(t1-t2) - sin(t2)
#
# ---


def double_pendulum_eom(t, y):
    """
    Defines the Hamiltonian EOMs for a double pendulum.
    State vector y = [theta1, theta2, p1, p2]
    """
    th1, th2, p1, p2 = y

    # Helper terms
    c12 = np.cos(th1 - th2)
    s12 = np.sin(th1 - th2)
    den = 1 + s12**2  # Denominator: 1 + sin(th1-th2)^2

    # Equations for d(theta)/dt
    dth1_dt = (p1 - p2 * c12) / den
    dth2_dt = (2 * p2 - p1 * c12) / den

    # Common term for d(p)/dt
    dth1_dth2_s12 = dth1_dt * dth2_dt * s12

    # Equations for d(p)/dt
    dp1_dt = -dth1_dth2_s12 - 2 * np.sin(th1)
    dp2_dt = dth1_dth2_s12 - np.sin(th2)

    return [dth1_dt, dth2_dt, dp1_dt, dp2_dt]


def calculate_energy_double(y):
    """
    Calculates the total energy (Hamiltonian) for a double pendulum state.
    y can be (4,) or (4, n_times)
    """
    th1, th2, p1, p2 = y

    c12 = np.cos(th1 - th2)
    s12 = np.sin(th1 - th2)
    den = 2 * (1 + s12**2)

    # Kinetic Energy (T)
    T_num = p1**2 + 2 * p2**2 - 2 * p1 * p2 * c12
    T = T_num / den

    # Potential Energy (V)
    V = -2 * np.cos(th1) - np.cos(th2)

    return T + V


def wrap_angle(theta):
    """Wraps an angle or array of angles to the interval [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


# --- 2. Generic Solver Functions ---


def solve_trajectory(eom_func, initial_condition, t_points, rtol=1e-9, atol=1e-12):
    """
    Integrates any EOM function using SciPy's 'RK45'.
    We use tight tolerances because RK45 is not symplectic.
    """
    t_span = (t_points[0], t_points[-1])
    sol = solve_ivp(
        eom_func,
        t_span,
        initial_condition,
        t_eval=t_points,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    return sol.y


def sample_initial_conditions(mean, cov, n_samples):
    """Draws n_samples from a multivariate Gaussian distribution."""
    print(f"Sampling {n_samples} points from {len(mean)}D Gaussian...")
    return np.random.multivariate_normal(mean, cov, n_samples)


def solve_for_distribution(eom_func, initial_conditions, t_points):
    """
    Solves the EOM for every initial condition in the provided set.
    """
    n_samples = initial_conditions.shape[0]
    n_dim = initial_conditions.shape[1]
    n_times = len(t_points)
    all_trajectories = np.zeros((n_samples, n_dim, n_times))

    print(f"Solving for {n_samples} trajectories...")
    start_time = time.time()

    for i in range(n_samples):
        all_trajectories[i, :, :] = solve_trajectory(
            eom_func, initial_conditions[i], t_points
        )

    end_time = time.time()
    print(f"Done. Solving took {end_time - start_time:.2f} seconds.")
    return all_trajectories


# --- 3. Visualization for Double Pendulum ---


def compare_energy_conservation_double(ic, t_long):
    """Plots the energy drift for a single trajectory."""
    print("--- Running Energy Conservation Check ---")

    sol_rk45 = solve_trajectory(double_pendulum_eom, ic, t_long, rtol=1e-6, atol=1e-9)
    energy_rk45 = calculate_energy_double(sol_rk45)

    sol_rk45_tight = solve_trajectory(
        double_pendulum_eom, ic, t_long, rtol=1e-12, atol=1e-12
    )
    energy_rk45_tight = calculate_energy_double(sol_rk45_tight)

    fig, ax = plt.subplots(figsize=(12, 7))
    E0 = energy_rk45[0]

    ax.plot(t_long, energy_rk45 - E0, label=f"RK45 (default tol)", color="red")
    ax.plot(
        t_long,
        energy_rk45_tight - E0,
        label=f"RK45 (tight tol, $10^{{-12}}$)",
        color="orange",
        linestyle="--",
    )

    ax.set_title("Energy Conservation (RK45 on Double Pendulum)", fontsize=16)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy Drift $\Delta E = E(t) - E(0)$")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    print("--- Energy Check Finished ---")


def plot_static_double(phase_space_evolution, t_points):
    """
    Plots the phase space distribution at t=0 and final time.
    Overlays (th1, p1) and (th2, p2).
    """
    print("Generating static plot...")

    # State vector is [0: th1, 1: th2, 2: p1, 3: p2]

    # --- Initial Data (t=0) ---
    th1_0 = wrap_angle(phase_space_evolution[:, 0, 0])
    p1_0 = phase_space_evolution[:, 2, 0]
    th2_0 = wrap_angle(phase_space_evolution[:, 1, 0])
    p2_0 = phase_space_evolution[:, 3, 0]

    # --- Final Data (t=T_MAX) ---
    th1_f = wrap_angle(phase_space_evolution[:, 0, -1])
    p1_f = phase_space_evolution[:, 2, -1]
    th2_f = wrap_angle(phase_space_evolution[:, 1, -1])
    p2_f = phase_space_evolution[:, 3, -1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # --- Plot t=0 ---
    ax1.scatter(
        th1_0, p1_0, alpha=0.5, s=5, c="blue", label=r"Pendulum 1 ($\theta_1, p_1$)"
    )
    ax1.scatter(
        th2_0, p2_0, alpha=0.5, s=5, c="green", label=r"Pendulum 2 ($\theta_2, p_2$)"
    )
    ax1.set_title(f"Initial Distribution (t = {t_points[0]:.1f}s)")
    ax1.set_xlabel(r"Angle $\theta$ (rad)")
    ax1.set_ylabel(r"Momentum $p_\theta$")
    ax1.set_xlim([-np.pi, np.pi])
    ax1.legend()
    ax1.grid(True)

    # --- Plot t=T_MAX ---
    ax2.scatter(
        th1_f, p1_f, alpha=0.5, s=5, c="blue", label=r"Pendulum 1 ($\theta_1, p_1$)"
    )
    ax2.scatter(
        th2_f, p2_f, alpha=0.5, s=5, c="green", label=r"Pendulum 2 ($\theta_2, p_2$)"
    )
    ax2.set_title(f"Final Distribution (t = {t_points[-1]:.1f}s)")
    ax2.set_xlabel(r"Angle $\theta$ (rad)")
    ax2.set_xlim([-np.pi, np.pi])
    ax2.legend()
    ax2.grid(True)

    p_max = np.max(np.abs(phase_space_evolution[:, 2:, :])) * 1.1
    ax1.set_ylim([-p_max, p_max])

    plt.suptitle("Double Pendulum Phase Space Projections", fontsize=16)
    plt.tight_layout()
    plt.show()


def create_animation_object_double(phase_space_evolution, t_points):
    """
    Creates and returns the Matplotlib FuncAnimation object.
    """
    print("Creating animation object...")
    fig, ax = plt.subplots(figsize=(8, 6))

    p_max = np.max(np.abs(phase_space_evolution[:, 2:, :])) * 1.1
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-p_max, p_max])
    ax.set_xlabel(r"Angle $\theta$ (rad)")
    ax.set_ylabel(r"Momentum $p_\theta$")
    ax.grid(True)

    # Create two scatter artists
    scatter1 = ax.scatter(
        [], [], alpha=0.5, s=5, c="blue", label=r"Pendulum 1 ($\theta_1, p_1$)"
    )
    scatter2 = ax.scatter(
        [], [], alpha=0.5, s=5, c="green", label=r"Pendulum 2 ($\theta_2, p_2$)"
    )
    ax.legend()

    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes, fontsize=12)
    fig.suptitle("Phase Space Evolution (Projections)", fontsize=16)

    def init():
        """Initializes the animation."""
        scatter1.set_offsets(np.empty((0, 2)))
        scatter2.set_offsets(np.empty((0, 2)))
        time_text.set_text("")
        return scatter1, scatter2, time_text

    def animate(i):
        """Updates the animation for frame i."""
        # Data for Pendulum 1
        th1_i = wrap_angle(phase_space_evolution[:, 0, i])
        p1_i = phase_space_evolution[:, 2, i]
        data1 = np.stack([th1_i, p1_i]).T
        scatter1.set_offsets(data1)

        # Data for Pendulum 2
        th2_i = wrap_angle(phase_space_evolution[:, 1, i])
        p2_i = phase_space_evolution[:, 3, i]
        data2 = np.stack([th2_i, p2_i]).T
        scatter2.set_offsets(data2)

        time_text.set_text(f"Time t = {t_points[i]:.2f}s")
        return scatter1, scatter2, time_text

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


# --- 4. Main Execution Block ---

if __name__ == "__main__":

    # --- Part 1: Demonstrate Energy Conservation ---
    # We choose a "chaotic" initial condition to test energy
    IC_CHAOTIC = [0, np.pi / 2, 0.0, 0.0]  # Start at (0, 90) deg, at rest
    T_LONG = 100.0
    N_STEPS_LONG = 2000
    t_long = np.linspace(0, T_LONG, N_STEPS_LONG)

    # compare_energy_conservation_double(IC_CHAOTIC, t_long)

    # --- Part 2: Ensemble Simulation Configuration ---
    print("\n--- Starting Double Pendulum Ensemble Simulation ---")

    N_SAMPLES = 2500  # Fewer, as the EOMs are slower to solve
    T_MAX = 40.0  # Simulate for longer to see chaos
    N_TIMES = 800  # Number of frames

    # Initial distribution: A small blob around the chaotic IC
    MEAN_IC = [np.pi / 2, np.pi, 0.0, 0.0]

    # Small covariance in all 4 dimensions (no correlation)
    COV_IC = np.diag(
        [
            0.01**2,  # std dev of 0.05 rad in th1
            0.01**2,  # std dev of 0.05 rad in th2
            0.01**2,  # std dev of 0.05 in p1
            0.01**2,  # std dev of 0.05 in p2
        ]
    )

    # --- Part 3: Run the Ensemble Simulation ---

    t_points = np.linspace(0, T_MAX, N_TIMES)
    initial_conditions = sample_initial_conditions(MEAN_IC, COV_IC, N_SAMPLES)

    # We must use the double_pendulum_eom
    phase_space_evolution = solve_for_distribution(
        double_pendulum_eom, initial_conditions, t_points
    )

    # --- Part 4: Create Static Plots ---
    plot_static_double(phase_space_evolution, t_points)

    # --- Part 5: Create, Save, and Display Animation ---

    anim = create_animation_object_double(phase_space_evolution, t_points)

    try:
        print("Saving animation to double_pendulum_evolution.mp4...")
        anim.save("double_pendulum_evolution.mp4", writer="ffmpeg", fps=25, dpi=150)
        print("Animation saved successfully.")
    except Exception as e:
        print(f"\n*** Error saving animation: {e} ***")
        print(
            "Please ensure 'ffmpeg' is installed and accessible in your system's PATH."
        )

    # --- Part 6: Display in Notebook (optional) ---
    # print("Displaying animation in notebook:")
    # html_anim = display_animation_html(anim)
    # display(html_anim)
