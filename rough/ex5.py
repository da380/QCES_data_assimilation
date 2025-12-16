"""
Task:
- Generates and solves for a phase space ensemble of double pendulums.
- Allows for variable physical parameters (masses, lengths).
- Creates static before-and-after plots of the phase space projections.
- Creates and saves an animation of the phase space evolution.

Requires: data_assimilation.pendulum module
"""

import numpy as np
import data_assimilation.pendulum as psm

# --- Simulation Configuration ---
T_MAX = 25.0
N_FRAMES = 300
N_SAMPLES = 400
t_points = np.linspace(0, T_MAX, N_FRAMES)

# --- Physical Parameters (Feel free to change these!) ---
L1 = 1.0
L2 = 1.0
m1 = 1.0
m2 = 1.0
g = psm.G
eom_args = (L1, L2, m1, m2, g)

# --- Initial Conditions for the Ensemble ---
MEAN_IC_DOUBLE = [np.pi / 2, np.pi / 2, 0.0, 0.0]
COV_IC_DOUBLE = np.diag([0.05**2, 0.05**2, 0.05**2, 0.05**2])

print("--- Starting Double Pendulum Phase Space Simulation ---")
print(f"Parameters: L1={L1}, L2={L2}, m1={m1}, m2={m2}, g={g:.2f}")

# 1. Sample the initial conditions
initial_conditions = psm.sample_initial_conditions(
    MEAN_IC_DOUBLE, COV_IC_DOUBLE, N_SAMPLES
)

# 2. Solve for the evolution of the ensemble
phase_space_evolution = psm.solve_for_distribution(
    eom_func=psm.eom_double,
    initial_conditions=initial_conditions,
    t_points=t_points,
    solver_func=psm.solve_trajectory_rk45,
    eom_args=eom_args,
)

# 3. Generate the static before-and-after plots
psm.plot_static_phase_space_double(phase_space_evolution, t_points)

# 4. Create the animation object
animation = psm.create_animation_phase_space_double(phase_space_evolution, t_points)

# 5. Save the animation
try:
    animation.save("double_pendulum_phase_space.mp4", writer="ffmpeg", fps=25)
    print("\nSuccessfully saved 'double_pendulum_phase_space.mp4'")
except Exception as e:
    print(f"\nError saving animation: {e}")

print("\n--- Script finished ---")
