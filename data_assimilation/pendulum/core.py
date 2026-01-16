"""
core.py

A dimension-agnostic engine for numerical integration, statistical analysis,
and Bayesian inference. This module handles N-dimensional grids and
generic differential equation solving.
"""

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
from IPython.display import HTML

# --- Math Utilities ---


def wrap_angle(theta):
    """Wraps an angle or array of angles to the interval [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


# --- Numerical Solvers (Dimension Agnostic) ---


def solve_trajectory(
    eom_func, y0, t_points, args=(), rtol=1e-9, atol=1e-12, method="RK45"
):
    """Integrates a single ODE trajectory over time."""
    t_span = (t_points[0], t_points[-1])
    sol = solve_ivp(
        eom_func,
        t_span,
        y0,
        t_eval=t_points,
        method=method,
        rtol=rtol,
        atol=atol,
        args=args,
    )
    return sol.y


def solve_ensemble(eom_func, initial_conditions, t_points, args=(), **solver_kwargs):
    """Propagates an ensemble of particles forward in time."""
    n_samples, n_dim = initial_conditions.shape
    n_times = len(t_points)
    trajectories = np.zeros((n_samples, n_dim, n_times))

    print(f"Propagating {n_samples} particles ({n_dim}D system)...")

    for i in range(n_samples):
        trajectories[i] = solve_trajectory(
            eom_func, initial_conditions[i], t_points, args=args, **solver_kwargs
        )

    return trajectories


# --- N-Dimensional Statistical Tools ---


def compute_normalization(grid_values, grid_axes):
    """Computes the total integral (volume) of an N-dimensional grid."""
    integral = grid_values
    for i, axis_vals in enumerate(reversed(grid_axes)):
        current_dim = len(grid_axes) - 1 - i
        integral = trapezoid(integral, x=axis_vals, axis=current_dim)
    return integral


def marginalise_grid(grid_values, grid_axes, keep_indices):
    """Computes the marginal PDF by integrating out all axes NOT in keep_indices."""
    ndim = len(grid_axes)
    all_indices = set(range(ndim))
    keep_set = set(keep_indices)
    integrate_indices = sorted(list(all_indices - keep_set), reverse=True)

    current_values = grid_values.copy()

    for i in integrate_indices:
        current_values = trapezoid(current_values, x=grid_axes[i], axis=i)

    new_axes = [grid_axes[i] for i in sorted(keep_indices)]
    return new_axes, current_values


def get_pdf_from_grid(grid_axes, grid_values, fill_value=0.0):
    """Creates a callable PDF function from a discrete N-dimensional grid."""
    interp = RegularGridInterpolator(
        grid_axes,
        grid_values,
        bounds_error=False,
        fill_value=fill_value,
        method="linear",
    )

    def pdf_func(*coords):
        if len(coords) != len(grid_axes):
            raise ValueError(f"PDF expects {len(grid_axes)} coordinates.")
        broadcasted = np.broadcast_arrays(*coords)
        result_shape = broadcasted[0].shape
        points = np.column_stack([b.ravel() for b in broadcasted])
        values = interp(points)
        return values.reshape(result_shape)

    return pdf_func


def advect_pdf_grid(eom_func, pdf_func, t_final, grid_limits, resolution, eom_args=()):
    """Generic Liouville Advection on a hyper-grid."""
    n_dim = len(grid_limits)
    axes = [np.linspace(l[0], l[1], resolution) for l in grid_limits]
    grids = np.meshgrid(*axes, indexing="ij")

    flat_state = np.stack([g.ravel() for g in grids])
    y0_vectorized = flat_state.reshape(-1)

    def vectorized_eom(t, y_flat):
        y_reshaped = y_flat.reshape(n_dim, -1)
        dydt = eom_func(t, y_reshaped, *eom_args)
        return np.concatenate(dydt).reshape(-1)

    t_span = [t_final, 0.0]
    sol = solve_trajectory(
        vectorized_eom, y0_vectorized, t_span, args=(), rtol=1e-9, atol=1e-9
    )

    origins_flat = sol[:, -1]
    grid_shape = (resolution,) * n_dim
    origins = origins_flat.reshape(n_dim, *grid_shape)

    Z_initial = pdf_func(*grids)
    Z_advected = pdf_func(*origins)

    norm_const = compute_normalization(Z_initial, axes)
    if norm_const == 0:
        norm_const = 1.0

    return axes, Z_initial / norm_const, Z_advected / norm_const


# --- Bayesian Tools ---


def get_independent_gaussian_func(means, stds):
    """
    Factory that returns a callable PDF function for independent Gaussians.

    Args:
        means: List or array of means [mu_1, mu_2, ...]
        stds:  List or array of standard deviations [sigma_1, sigma_2, ...]

    Returns:
        A function pdf(*args) that takes N arguments and returns the probability density.
    """
    means = np.array(means)
    stds = np.array(stds)

    def pdf_func(*args):
        # args is a tuple of grids or values: (theta, p, ...)
        # We assume independent variables, so P(x,y) = P(x)P(y)

        # Calculate prob for each dimension
        total_prob = 1.0
        for i, val in enumerate(args):
            mu = means[i]
            sigma = stds[i]
            # Standard Gaussian formula
            prob = np.exp(-0.5 * ((val - mu) / sigma) ** 2)
            total_prob *= prob

        return total_prob

    return pdf_func


def gaussian_likelihood(x_grid, obs_value, obs_std):
    """Computes Gaussian likelihood on a grid."""
    return norm.pdf(x_grid, loc=obs_value, scale=obs_std)


def bayesian_update(prior_grid, likelihood_grid, grid_axes):
    """Posterior = (Likelihood * Prior) / Evidence"""
    posterior_unnorm = likelihood_grid * prior_grid
    evidence = compute_normalization(posterior_unnorm, grid_axes)
    if evidence == 0:
        return posterior_unnorm, 0.0
    return posterior_unnorm / evidence, evidence


def assimilate_cycle(
    t_obs,
    observations,
    obs_std,
    prior_func,
    forecast_func,
    analysis_func,
    grid_to_func_wrapper,
):
    """
    Generic Grid-Based Assimilation Cycle.

    Args:
        prior_func: Callable f(grids) -> Z_initial.
        forecast_func: Callable f(current_pdf_func, dt) -> Z_forecast.
        analysis_func: Callable f(Z_forecast, obs_val, obs_std) -> (Z_posterior, evidence).
        grid_to_func_wrapper: Callable f(Z_grid) -> next_pdf_func.

    Returns:
        List of results dictionaries.
    """
    results = []
    current_pdf_func = prior_func
    t_prev = 0.0

    for i, t_now in enumerate(t_obs):
        obs_val = observations[i]
        dt = t_now - t_prev

        # 1. FORECAST (Generic)
        # Evolve probability from t_prev to t_now
        Z_forecast = forecast_func(current_pdf_func, dt)

        # 2. ANALYSIS (Generic)
        # Update belief based on new observation
        Z_posterior, evidence = analysis_func(Z_forecast, obs_val, obs_std)

        # 3. Store
        results.append({"time": t_now, "posterior": Z_posterior, "evidence": evidence})

        # 4. Prepare Next Step
        # Convert grid back to function for the next forecast
        current_pdf_func = grid_to_func_wrapper(Z_posterior)
        t_prev = t_now

    return results


# --- Generic Visualization Helpers ---


def display_animation_html(anim):
    """Standard helper to render animations in notebooks."""
    print("Rendering animation...")
    return HTML(anim.to_jshtml())
