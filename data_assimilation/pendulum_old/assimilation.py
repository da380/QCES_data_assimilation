"""
assimilation.py

A module containing the physics equations, Hamiltonian formulations,
coordinate transformations, and numerical solvers for single and double pendulums.
"""

import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid
from scipy.stats import norm
from scipy.stats import multivariate_normal

from . import physics as phys


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


def advect_pdf_single(
    pdf_func, t_final, x_lim=(-np.pi, np.pi), y_lim=(-3, 3), res=100, **physics_params
):
    """
    Computes the advected PDF grid using Liouville's theorem.

    Args:
        pdf_func: Function that takes (X, Y) grid and returns Z values.
        t_final: The target time to advect to.
        x_lim, y_lim: The bounds of the grid.
        res: Grid resolution.
        **physics_params: Optional overrides for physics (e.g., L=2.0, m=1.0).
    """
    x_vals = np.linspace(x_lim[0], x_lim[1], res)
    y_vals = np.linspace(y_lim[0], y_lim[1], res)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Prepare vectorized backwards integration
    flat_state = np.stack([X.ravel(), Y.ravel()])
    n_pixels = flat_state.shape[1]
    y0_vectorized = flat_state.reshape(-1)

    # Now captures physics_params to pass to the EOM
    def vectorized_eom(t, y_flat):
        y_reshaped = y_flat.reshape(2, -1)
        # Pass the kwargs (L, m, g) into the single pendulum EOM
        dydt = phys.eom_single(t, y_reshaped, **physics_params)
        return np.concatenate(dydt).reshape(-1)

    # Integrate backwards from t_final to 0
    t_span = np.array([t_final, 0.0])

    sol = phys.solve_trajectory(
        vectorized_eom, y0_vectorized, t_span, rtol=1e-6, atol=1e-6
    )

    y_origins = sol[:, -1].reshape(2, n_pixels)
    pdf_initial = pdf_func(X, Y)
    pdf_advected = pdf_func(
        y_origins[0].reshape(res, res), y_origins[1].reshape(res, res)
    )
    return X, Y, pdf_initial, pdf_advected


def get_pdf_from_grid(X, Y, Z, bounds_error=False, fill_value=0.0):
    """
    Creates a continuous PDF function by interpolating discrete grid values.
    Ensures axes are in correct order for scipy solvers.
    """
    # RegularGridInterpolator requires 1D coordinate arrays in strictly increasing order
    theta_axis = X[0, :]
    p_axis = Y[:, 0]

    # We provide Z such that Z[i, j] corresponds to (theta[i], p[j])
    # Since your grid Z is likely (p_index, theta_index), we transpose it
    interp = RegularGridInterpolator(
        (theta_axis, p_axis),
        Z.T,
        bounds_error=bounds_error,
        fill_value=fill_value,
        method="linear",
    )

    def pdf_func(theta, p):
        # advect_pdf_single often passes arrays of theta and p;
        # we must reshape them into (N, 2) for the interpolator
        pts = np.column_stack([np.atleast_1d(theta).ravel(), np.atleast_1d(p).ravel()])
        val = interp(pts)
        return val.reshape(np.shape(theta)) if np.ndim(theta) > 0 else val[0]

    return pdf_func


def compute_normalization(X, Y, func):
    """
    Computes the 2D integral of an unnormalized PDF using the trapezoidal rule.
    """
    # Identify spacing from the grid
    d_theta = X[0, 1] - X[0, 0]
    d_p = Y[1, 0] - Y[0, 0]

    # Integrate over p (axis 0) then theta (axis 1)
    integral = trapezoid(trapezoid(func, dx=d_p, axis=0), dx=d_theta)
    return integral


def gaussian_likelihood(X, obs_value, obs_std, measurement_func=None):
    """
    Computes a Gaussian likelihood grid given an observation.

    Args:
        X: The grid values for the state variable being observed.
        obs_value: The observed value (mean of the Gaussian).
        obs_std: The standard deviation of the observation noise.
        measurement_func: Optional function to transform X before comparison
                          (e.g., wrap_angle for theta observations).
    """
    if measurement_func:
        X_transformed = measurement_func(X)
    else:
        X_transformed = X

    return norm.pdf(X_transformed, loc=obs_value, scale=obs_std)


def bayesian_update(prior_pdf, likelihood_pdf, grid_X, grid_Y):
    """
    Performs the analysis step: Posterior ~ Likelihood * Prior.
    Returns the normalized posterior and the evidence (marginal likelihood).
    """
    posterior_unnorm = likelihood_pdf * prior_pdf

    # Calculate evidence (integral of the unnormalized posterior)
    evidence = compute_normalization(grid_X, grid_Y, posterior_unnorm)

    # Avoid division by zero
    if evidence == 0:
        print("Warning: Evidence is zero. Returning unnormalized posterior.")
        return posterior_unnorm, 0

    posterior = posterior_unnorm / evidence
    return posterior, evidence


def create_multivariate_prior(mean, cov):
    """
    Creates a callable PDF function for a general multivariate Gaussian.
    Allows for correlations between state variables (e.g., theta and p).

    Args:
        mean: Array of shape (2,) for [mu_theta, mu_p].
        cov: 2x2 covariance matrix.

    Returns:
        A function pdf(X, Y) that returns the probability density on the grid.
    """
    # Create the frozen random variable once for efficiency
    rv = multivariate_normal(mean, cov)

    def pdf_func(X, Y):
        # Stack the meshgrids into shape (N, M, 2) for scipy
        pos = np.dstack((X, Y))
        return rv.pdf(pos)

    return pdf_func


def create_independent_prior(mu_theta=0.0, mu_p=0.0, std_theta=1.0, std_p=1.0):
    """
    Wrapper for create_multivariate_prior that constructs a diagonal
    covariance matrix (assuming no initial correlation).
    """
    mean = np.array([mu_theta, mu_p])
    # Diagonal covariance: variances are squared std deviations
    cov = np.diag([std_theta**2, std_p**2])

    return create_multivariate_prior(mean, cov)
