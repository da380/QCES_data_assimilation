# data_assimilation/pendulum/__init__.py

"""
__init__.py for the pendulum sub-package.
"""

# Import from the local 'physics.py'
from .physics import (
    G,
    wrap_angle,
    eom_single,
    get_single_coords,
    eom_double,
    get_double_coords,
    solve_trajectory,
    eom_single_linear,
    get_propagator_matrix_single,
    get_linearized_system_matrix_double,
    eom_double_linear,
    get_propagator_matrix_double,
)

from .assimilation import (
    solve_for_distribution,
    sample_initial_conditions,
    advect_pdf_single,
    get_pdf_from_grid,
    compute_normalization,
    gaussian_likelihood,
    bayesian_update,
    create_multivariate_prior,
    create_independent_prior,
)

# Import from the local 'viz.py'
from .visualise import (
    display_animation_html,
    plot_static_state_space_single,
    plot_static_state_space_double,
    plot_state_space_with_statistics,
    plot_pdf_advection_single,
    create_animation_state_space_single,
    create_physical_animation_single,
    create_physical_animation_double,
    animate_pdf,
    create_combined_animation_single,
    create_combined_animation_double,
    plot_bayesian_analysis,
)

__all__ = [
    "G",
    "wrap_angle",
    "eom_single",
    "get_single_coords",
    "eom_double",
    "get_double_coords",
    "solve_for_distribution",
    "sample_initial_conditions",
    "advect_pdf_single",
    "eom_single_linear",
    "get_propagator_matrix_single",
    "get_linearized_system_matrix_double",
    "eom_double_linear",
    "get_propagator_matrix_double",
    "display_animation_html",
    "plot_static_state_space_single",
    "plot_static_state_space_double",
    "plot_state_space_with_statistics",
    "plot_pdf_advection_single",
    "create_animation_state_space_single",
    "create_physical_animation_single",
    "create_physical_animation_double",
    "animate_pdf",
    "create_combined_animation_single",
    "create_combined_animation_double",
    "get_pdf_from_grid",
    "compute_normalization",
    "bayesian_update",
    "gaussian_likelihood",
    "plot_bayesian_analysis",
    "create_multivariate_prior",
    "create_independent_prior",
]
