"""
physics.py

A module containing the physics equations, Hamiltonian formulations,
coordinate transformations, and numerical solvers for single and double pendulums.
"""

import numpy as np
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# --- Constants & Utilities ---

G = 9.81  # Standard gravity (for reference or non-normalized defaults)


def wrap_angle(theta):
    """Wraps an angle or array of angles to the interval [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


# --- Single Pendulum Physics ---


def eom_single(t, y, L=1.0, m=1.0, g=1.0):
    """
    Hamilton's EOMs for a single pendulum: [theta, p_theta].
    Defaults (L=1, m=1, g=1) correspond to the normalized non-dimensional form.
    """
    theta, p_theta = y

    # d(theta)/dt = dH/dp = p / (m L^2)
    d_theta = p_theta / (m * L**2)

    # d(p)/dt = -dH/dtheta = -m g L sin(theta)
    d_p_theta = -m * g * L * np.sin(theta)

    return [d_theta, d_p_theta]


def calculate_energy_single(y, L=1.0, m=1.0, g=1.0):
    """Calculates the Hamiltonian (energy) for a single pendulum."""
    theta, p_theta = y[0], y[1]

    # Kinetic Energy T = p^2 / (2 m L^2)
    T = p_theta**2 / (2.0 * m * L**2)

    # Potential Energy V = -m g L cos(theta)
    V = -m * g * L * np.cos(theta)

    return T + V


def get_single_coords(theta, L=1.0):
    """Converts angle (theta) to (x, y) coordinates."""
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    return x, y


# --- Double Pendulum Physics ---


def eom_double(t, y, L1, L2, m1, m2, g):
    """
    Hamilton's EOMs for a double pendulum with variable parameters.
    State vector y = [th1, th2, p1, p2]
    """
    th1, th2, p1, p2 = y
    d_th = th1 - th2
    c_d_th, s_d_th = np.cos(d_th), np.sin(d_th)
    den = m1 + m2 * s_d_th**2
    dth1_dt = (L2 * p1 - L1 * p2 * c_d_th) / (L1**2 * L2 * den)
    dth2_dt = (L1 * (m1 + m2) * p2 - L2 * m2 * p1 * c_d_th) / (L1 * L2**2 * m2 * den)
    term1 = m2 * L1 * L2 * dth1_dt * dth2_dt * s_d_th
    term2 = (m1 + m2) * g * L1 * np.sin(th1)
    dp1_dt = -term1 - term2
    term3 = m2 * g * L2 * np.sin(th2)
    dp2_dt = term1 - term3
    return [dth1_dt, dth2_dt, dp1_dt, dp2_dt]


def calculate_energy_double(y, L1, L2, m1, m2, g):
    """Calculates the total energy for a double pendulum."""
    th1, th2, p1, p2 = y
    d_th = th1 - th2
    c_d_th = np.cos(d_th)
    den = m1 + m2 * np.sin(d_th) ** 2
    dth1_dt = (L2 * p1 - L1 * p2 * c_d_th) / (L1**2 * L2 * den)
    dth2_dt = (L1 * (m1 + m2) * p2 - L2 * m2 * p1 * c_d_th) / (L1 * L2**2 * m2 * den)
    T1 = 0.5 * m1 * (L1 * dth1_dt) ** 2
    T2 = (
        0.5
        * m2
        * (
            (L1 * dth1_dt) ** 2
            + (L2 * dth2_dt) ** 2
            + 2 * L1 * L2 * dth1_dt * dth2_dt * c_d_th
        )
    )
    V = -(m1 + m2) * g * L1 * np.cos(th1) - m2 * g * L2 * np.cos(th2)
    return T1 + T2 + V


def get_double_coords(theta1, theta2, L1, L2):
    """Converts angles to (x, y) coordinates with variable lengths."""
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2


# --- Linearized Systems ---


def eom_single_linear(t, y, L=1.0, m=1.0, g=1.0):
    """Linearized EOM for a single pendulum around equilibrium (0,0)."""
    # dtheta/dt = p / (m L^2)
    # dp/dt     = -m g L theta
    # Matrix form: [[0, 1/(m L^2)], [-m g L, 0]]

    a = 1.0 / (m * L**2)
    b = -m * g * L
    A = np.array([[0, a], [b, 0]])
    return A @ y


def get_propagator_matrix_single(t, L=1.0, m=1.0, g=1.0):
    """Returns analytical propagator P(t) for linearized single pendulum."""
    # Frequency omega = sqrt(g/L)
    omega = np.sqrt(g / L)
    cos_t = np.cos(omega * t)
    sin_t = np.sin(omega * t)

    # Terms for the mixed matrix elements
    # M_01 = (1 / (m L^2 * omega)) * sin
    # M_10 = (-m g L / omega) * sin

    m01 = sin_t / (m * L**2 * omega)
    m10 = -m * g * L * sin_t / omega

    return np.array([[cos_t, m01], [m10, cos_t]])


def get_linearized_system_matrix_double(L1, L2, m1, m2, g):
    """Computes system matrix 'A' for linearized double pendulum."""
    M = np.array([[(m1 + m2) * L1**2, m2 * L1 * L2], [m2 * L1 * L2, m2 * L2**2]])
    K = np.array([[(m1 + m2) * g * L1, 0], [0, m2 * g * L2]])
    M_inv = np.linalg.inv(M)
    A = np.zeros((4, 4))
    A[0:2, 2:4] = M_inv
    A[2:4, 0:2] = -K
    return A


def eom_double_linear(t, y, A):
    """Linearized EOM for a double pendulum given matrix A."""
    return A @ y


def get_propagator_matrix_double(t, A):
    """Computes propagator P(t) = exp(A*t) numerically."""
    return expm(A * t)


# --- Solvers ---


def solve_trajectory_rk45(
    eom_func, initial_condition, t_points, eom_args=(), rtol=1e-9, atol=1e-12
):
    """Integrates EOMs using SciPy's 'RK45'."""
    t_span = (t_points[0], t_points[-1])
    sol = solve_ivp(
        eom_func,
        t_span,
        initial_condition,
        t_eval=t_points,
        method="RK45",
        rtol=rtol,
        atol=atol,
        args=eom_args,
    )
    return sol.y


def solve_trajectory_symplectic_single(
    eom_func, initial_condition, t_points, eom_args=()
):
    """Integrates single pendulum EOMs using fixed-step 2nd-order Leapfrog."""
    theta_0, p_theta_0 = initial_condition
    n_times = len(t_points)
    dt_values = np.diff(t_points)
    if not np.allclose(dt_values, dt_values[0]):
        raise ValueError("t_points must be equally spaced for symplectic integrator.")
    dt = dt_values[0]

    # Note: Leapfrog requires splitting the Hamiltonian.
    # The standard implementation assumes H = p^2/2 + V(q).
    # If m != 1 or L != 1, H = p^2/(2mL^2) + V(q).
    # The velocity update becomes dtheta = (p - dt/2 * F) / (m L^2).
    # This specific simple function might need updates if we want to support
    # non-normalized units properly, but for now we leave it assuming normalization
    # or relying on eom_args if we enhanced it.
    # We will assume normalized inputs for this specific solver as requested.

    sol = np.zeros((2, n_times))
    sol[:, 0] = [theta_0, p_theta_0]
    theta, p_theta = theta_0, p_theta_0

    for i in range(n_times - 1):
        p_half = p_theta - np.sin(theta) * (dt / 2.0)
        theta_new = theta + p_half * dt
        p_new = p_half - np.sin(theta_new) * (dt / 2.0)
        sol[:, i + 1] = [theta_new, p_new]
        theta, p_theta = theta_new, p_new
    return sol


def sample_initial_conditions(mean, cov, n_samples):
    """Draws n_samples from a multivariate Gaussian distribution."""
    print(f"Sampling {n_samples} points from {len(mean)}D Gaussian...")
    return np.random.multivariate_normal(mean, cov, n_samples)


def solve_for_distribution(
    eom_func, initial_conditions, t_points, solver_func, eom_args=(), **solver_kwargs
):
    """Solves the EOM for an ensemble of initial conditions."""
    n_samples = initial_conditions.shape[0]
    n_dim = initial_conditions.shape[1]
    n_times = len(t_points)
    all_trajectories = np.zeros((n_samples, n_dim, n_times))

    print(f"Solving for {n_samples} trajectories using {solver_func.__name__}...")
    start_time = time.time()
    for i in range(n_samples):
        all_trajectories[i, :, :] = solver_func(
            eom_func,
            initial_conditions[i],
            t_points,
            eom_args=eom_args,
            **solver_kwargs,
        )
    print(f"Done. Took {time.time() - start_time:.2f}s.")
    return all_trajectories


def advect_pdf_single(pdf_func, t_final, x_lim=(-np.pi, np.pi), y_lim=(-3, 3), res=100):
    """
    Computes the advected PDF grid using Liouville's theorem.
    """
    x_vals = np.linspace(x_lim[0], x_lim[1], res)
    y_vals = np.linspace(y_lim[0], y_lim[1], res)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Prepare vectorized backwards integration
    flat_state = np.stack([X.ravel(), Y.ravel()])
    n_pixels = flat_state.shape[1]
    y0_vectorized = flat_state.reshape(-1)

    def vectorized_eom(t, y_flat):
        y_reshaped = y_flat.reshape(2, -1)
        # Uses default L=1, m=1, g=1 unless changed in eom_single default
        dydt = eom_single(t, y_reshaped)
        return np.concatenate(dydt).reshape(-1)

    t_span = np.array([t_final, 0.0])

    sol = solve_trajectory_rk45(
        vectorized_eom, y0_vectorized, t_span, rtol=1e-6, atol=1e-6
    )

    y_origins = sol[:, -1].reshape(2, n_pixels)
    pdf_initial = pdf_func(X, Y)
    pdf_advected = pdf_func(
        y_origins[0].reshape(res, res), y_origins[1].reshape(res, res)
    )
    return X, Y, pdf_initial, pdf_advected
