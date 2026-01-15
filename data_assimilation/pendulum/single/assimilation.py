"""
assimilation.py

Single Pendulum specialization for Grid-based Data Assimilation.
Configures the generic core engine for the 2D pendulum state space.
"""

import numpy as np
from .. import core
from . import physics as phys


def advect_pdf(
    pdf_func, t_final, x_lim=(-np.pi, np.pi), y_lim=(-3, 3), res=100, **physics_params
):
    """
    Advects a PDF forward in time for the Single Pendulum using Liouville's theorem.

    Args:
        pdf_func: Callable(X, Y) -> Density (function defining the initial distribution).
        t_final: Target time (seconds).
        x_lim: (min, max) tuple for theta (radians).
        y_lim: (min, max) tuple for momentum.
        res: Grid resolution (res x res).
        **physics_params: Optional physics overrides (L=..., m=..., g=...).

    Returns:
        X, Y: Meshgrids of the state space at t_final.
        Z_initial: The initial PDF values on the grid.
        Z_advected: The advected PDF values on the grid.
    """
    # 1. Prepare Physics Arguments
    # We explicitly extract them to ensure the order matches phys.eom(t, y, L, m, g)
    # Using .get() allows defaults to persist if not provided.
    L = physics_params.get("L", 1.0)
    m = physics_params.get("m", 1.0)
    g = physics_params.get("g", 1.0)
    eom_args = (L, m, g)

    # 2. Define Domain Limits
    grid_limits = [x_lim, y_lim]

    # 3. Call Generic Core Engine
    # core.advect_pdf_grid handles the vectorization, integration, and normalization
    axes, Z_initial, Z_advected = core.advect_pdf_grid(
        eom_func=phys.eom,
        pdf_func=pdf_func,
        t_final=t_final,
        grid_limits=grid_limits,
        resolution=res,
        eom_args=eom_args,
    )

    # 4. Re-construct Meshgrids
    # The core returns 1D axes, but for 2D plotting (contourf etc.),
    # we typically want the full meshgrids X, Y.
    # indexing='ij' is used to match the matrix structure returned by the core.
    X, Y = np.meshgrid(*axes, indexing="ij")

    return X, Y, Z_initial, Z_advected
