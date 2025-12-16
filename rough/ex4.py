"""
Task:
- Generates and solves for a phase space ensemble of SINGLE pendulums.
- Creates static before-and-after plots of the phase space.
- Creates and saves an animation of the phase space evolution.

Requires: data_assimilation.pendulum module
"""

import numpy as np
import data_assimilation.pendulum as psm

# --- Simulation Configuration ---
T_MAX = 20.0
N_FRAMES = 400
N_SAMPLES = 300
t_points = np.linspace(0, T_MAX, N_FRAMES)

# --- Initial Conditions for the Ensemble ---
# A small cloud of points around an initial state of (theta=1.5, p=0.0)
MEAN_IC_SINGLE = [1.5, 0.0]
COV_IC_SINGLE = np.array([[0.3**2, 0.0], [0.0, 0.3**2]])

print("--- Running Single Pendulum Phase Space Ensemble ---")

# 1. Sample the initial conditions
initial_conditions = psm.sample_initial_conditions(
    MEAN_IC_SINGLE, COV_IC_SINGLE, N_SAMPLES
)

# 2. Solve for the evolution of the ensemble
# For the single pendulum, we can and should use the superior symplectic solver.
phase_evo_single = psm.solve_for_distribution(
    eom_func=psm.eom_single,
    initial_conditions=initial_conditions,
    t_points=t_points,
    solver_func=psm.solve_trajectory_symplectic_single,
)

# 3. Generate the static before-and-after plots
psm.plot_static_phase_space_single(phase_evo_single, t_points)

# 4. Create the animation object
anim_single_phase = psm.create_animation_phase_space_single(phase_evo_single, t_points)

# 5. Save the animation to an MP4 file
try:
    anim_single_phase.save("single_pendulum_phase_space.mp4", writer="ffmpeg", fps=25)
    print("\nSuccessfully saved 'single_pendulum_phase_space.mp4'")
except Exception as e:
    print(f"\nError saving animation: {e}.")
    print("Please ensure 'ffmpeg' is installed and accessible in your system's PATH.")

print("\n--- Script finished ---")
