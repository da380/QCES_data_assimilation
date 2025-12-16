from data_assimilation.pendulum import *


print("\n--- Starting Ensemble Simulation ---")

# CHOOSE YOUR SOLVER: 'symplectic' or 'rk45'
SOLVER_TO_USE = "rk45"

N_SAMPLES = 5000  # Number of pendulums in the distribution
T_MAX = 15.0  # Total time to simulate (seconds)
N_TIMES = 150  # Number of time steps (frames in animation)

# Initial distribution: A blob at (theta=1.5, p=0)
MEAN_IC = [1.5, 0.0]
COV_IC = np.array(
    [
        [0.1**2, 0.0],
        [0.0, 0.3**2],
    ]  # 0.3 std dev in theta  # 0.3 std dev in p_theta
)


t_points = np.linspace(0, T_MAX, N_TIMES)

initial_conditions = sample_initial_conditions(MEAN_IC, COV_IC, N_SAMPLES)


if SOLVER_TO_USE == "symplectic":
    solver_func = solve_pendulum_symplectic
    solver_kwargs = {}
elif SOLVER_TO_USE == "rk45":
    solver_func = solve_pendulum_rk45
    solver_kwargs = {"rtol": 1e-9, "atol": 1e-12}  # Use tighter tolerances
else:
    raise ValueError("SOLVER_TO_USE must be 'symplectic' or 'rk45'")

phase_space_evolution = solve_for_distribution(
    initial_conditions, t_points, solver_func, **solver_kwargs
)


plot_static(phase_space_evolution, t_points)


anim = create_animation_object(phase_space_evolution, t_points)

try:
    print("Saving animation to pendulum_evolution.mp4...")
    anim.save("pendulum_evolution.mp4", writer="ffmpeg", fps=25, dpi=150)
    print("Animation saved successfully.")
except Exception as e:
    print(f"\n***************************************************")
    print(f"Error saving animation: {e}")
    print("Could not save to 'pendulum_evolution.mp4'.")
    print("Please ensure 'ffmpeg' is installed and accessible in your system's PATH.")
    print(f"***************************************************\n")
