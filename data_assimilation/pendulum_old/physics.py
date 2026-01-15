"""
physics.py

A module containing the physics equations, Hamiltonian formulations,
coordinate transformations, and numerical solvers for single and double pendulums.
"""

import numpy as np
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


def solve_trajectory(
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
