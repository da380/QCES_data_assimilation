"""
Tests for the CircleWave class.
"""

import pytest
import numpy as np
from numpy import pi
from data_assimilation.circle_wave import CircleWave


# Use pytest fixtures to create reusable CircleWave instances
@pytest.fixture
def homogeneous_wave():
    """Returns a CircleWave instance with constant properties."""
    return CircleWave(128, radius=1.0, density=1.5, rigidity=2.5)


@pytest.fixture
def inhomogeneous_wave():
    """Returns a CircleWave instance with variable properties."""
    density_func = lambda theta: 2.0 + 0.2 * np.sin(theta)
    rigidity_func = lambda theta: 3.0 + 0.3 * np.cos(theta)
    return CircleWave(128, density=density_func, rigidity=rigidity_func)


def test_initialization(homogeneous_wave, inhomogeneous_wave):
    """Tests that the CircleWave class initializes correctly."""
    # Test homogeneous case
    assert homogeneous_wave.npoints == 128
    assert homogeneous_wave.radius == 1.0
    assert homogeneous_wave.density_and_rigidity_constant is True
    assert homogeneous_wave._omega is not None

    # Test inhomogeneous case
    assert inhomogeneous_wave.npoints == 128
    assert inhomogeneous_wave.density_and_rigidity_constant is False
    assert inhomogeneous_wave._omega is None
    assert inhomogeneous_wave.density_values[0] != inhomogeneous_wave.density_values[10]


def test_project_initial_state(homogeneous_wave):
    """Tests the projection of functions onto the state vector."""
    # A simple cosine displacement and zero momentum
    disp_func = lambda theta: np.cos(theta)
    mom_func = lambda theta: 0.0
    state = homogeneous_wave.project_displacement_and_momentum(disp_func, mom_func)

    assert state.shape == (2 * homogeneous_wave.npoints,)
    expected_disp = np.cos(homogeneous_wave.angles)
    assert np.allclose(state[: homogeneous_wave.npoints], expected_disp)
    assert np.all(state[homogeneous_wave.npoints :] == 0.0)


def test_propagator_identity(homogeneous_wave):
    """The propagator for dt=0 should be the identity operator."""
    z0 = np.random.rand(2 * homogeneous_wave.npoints)
    propagator = homogeneous_wave.propagator(0.0, 0.0)
    z1 = propagator @ z0
    assert np.allclose(z0, z1)


def test_propagator_is_symplectic(homogeneous_wave, inhomogeneous_wave):
    """
    Test the symplectic property P^T J P = J.
    This is the correct check for a Hamiltonian system's propagator and
    validates the rmatvec implementation.
    """

    # Define the action of the symplectic matrix J
    def apply_J(z, n):
        q = z[:n]
        p = z[n:]
        return np.concatenate((p, -q))

    # --- Test homogeneous case (analytical propagator) ---
    z_h = np.random.rand(2 * homogeneous_wave.npoints)
    prop_h = homogeneous_wave.propagator(t0=0.0, t1=0.1)
    n_h = homogeneous_wave.npoints

    rhs_h = apply_J(z_h, n_h)
    lhs_h = prop_h.T @ apply_J(prop_h @ z_h, n_h)
    assert np.allclose(lhs_h, rhs_h, atol=1e-9)

    # --- Test inhomogeneous case (numerical propagator) ---
    z_i = np.random.rand(2 * inhomogeneous_wave.npoints)
    prop_i = inhomogeneous_wave.propagator(t0=0.0, t1=0.1)
    n_i = inhomogeneous_wave.npoints

    rhs_i = apply_J(z_i, n_i)
    lhs_i = prop_i.T @ apply_J(prop_i @ z_i, n_i)
    # The tolerance for the numerical integrator will be lower
    assert np.allclose(lhs_i, rhs_i, atol=1e-6)


def test_energy_conservation_homogeneous(homogeneous_wave):
    """
    Total energy (kinetic + potential) should be conserved for a
    homogeneous system. This is a fundamental physics check.
    """
    # Initial state: a single mode (k=2)
    disp_func = lambda theta: np.sin(2 * theta)
    mom_func = lambda theta: np.cos(2 * theta)
    z0 = homogeneous_wave.project_displacement_and_momentum(disp_func, mom_func)

    def get_total_energy(state):
        q = state[: homogeneous_wave.npoints]
        p = state[homogeneous_wave.npoints :]

        # Kinetic Energy = 0.5 * p^2 / rho
        kinetic_energy = 0.5 * np.sum(p**2) / homogeneous_wave._density

        # Potential Energy = 0.5 * tau * (dq/d_theta)^2
        q_hat = np.fft.rfft(q)
        dq_hat = homogeneous_wave._derivative_scaling * q_hat
        dq = np.fft.irfft(dq_hat, n=homogeneous_wave.npoints)
        potential_energy = 0.5 * homogeneous_wave._rigidity * np.sum(dq**2)

        return kinetic_energy + potential_energy

    initial_energy = get_total_energy(z0)

    # Evolve for a few time steps
    times = np.linspace(0, 1.0, 10)
    solution = homogeneous_wave.integrate(z0, times)

    final_energy = get_total_energy(solution[:, -1])

    # The energy should be conserved to a high degree of precision
    assert np.isclose(initial_energy, final_energy, rtol=1e-10)


def test_integrate_methods_consistency(homogeneous_wave):
    """
    The fast analytical `integrate` and the numerical `integrate` (via the
    propagator) should yield the same results for the homogeneous case.
    """
    z0 = np.random.rand(2 * homogeneous_wave.npoints)
    times = np.linspace(0, 0.5, 5)

    # Result from the fast, analytical method
    analytical_result = homogeneous_wave.integrate(z0, times)

    # Result from the numerical propagator (should be very similar)
    propagator = homogeneous_wave.propagator(0.0, times[-1])
    numerical_result_final_state = propagator @ z0

    # Check the final state
    assert np.allclose(
        analytical_result[:, -1], numerical_result_final_state, atol=1e-5
    )
