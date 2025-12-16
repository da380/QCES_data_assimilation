"""
Task:
- Solves for single trajectories of both single and double pendulums.
- Allows for variable physical parameters.
- Creates and saves the physical "rods and bobs" animations of their motion.

Requires: data_assimilation.pendulum module
"""

import numpy as np
import data_assimilation.pendulum as psm

# --- General Configuration ---
T_MAX = 20.0
N_FRAMES = 800
t_points = np.linspace(0, T_MAX, N_FRAMES)

# =======================================================
# ==              SINGLE PENDULUM SECTION              ==
# =======================================================
print("--- Starting Single Pendulum Physical Animation ---")
L1_s = 1.5
IC_SINGLE = [0.0, 2.2]

solution_single = psm.solve_trajectory_rk45(psm.eom_single, IC_SINGLE, t_points)
animation_single = psm.create_physical_animation_single(
    t_points, solution_single, L1=L1_s
)

try:
    animation_single.save("single_pendulum_physical.mp4", writer="ffmpeg", fps=30)
    print("\nSuccessfully saved 'single_pendulum_physical.mp4'")
except Exception as e:
    print(f"\nError saving animation: {e}")

# =======================================================
# ==              DOUBLE PENDULUM SECTION              ==
# =======================================================
print("\n--- Starting Double Pendulum Physical Animation ---")
# Physical Parameters
L1_d, L2_d, m1_d, m2_d, g_d = 1.0, 1.0, 1.0, 1.0, psm.G
eom_args_d = (L1_d, L2_d, m1_d, m2_d, g_d)
print(f"Parameters: L1={L1_d}, L2={L2_d}, m1={m1_d}, m2={m2_d}, g={g_d:.2f}")

IC_DOUBLE = [np.pi / 2, np.pi, 0.0, 0.0]
solution_double = psm.solve_trajectory_rk45(
    psm.eom_double, IC_DOUBLE, t_points, eom_args=eom_args_d
)
animation_double = psm.create_physical_animation_double(
    t_points, solution_double, L1=L1_d, L2=L2_d
)

try:
    animation_double.save("double_pendulum_physical.mp4", writer="ffmpeg", fps=30)
    print("\nSuccessfully saved 'double_pendulum_physical.mp4'")
except Exception as e:
    print(f"\nError saving animation: {e}")

print("\n--- Script finished ---")
