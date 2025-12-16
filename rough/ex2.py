import numpy as np
import matplotlib.pyplot as plt
from data_assimilation import (
    CircleWave,
)

from pygeoinf.symmetric_space.circle import Lebesgue

# --- Configuration ---
N_SAMPLES = 100  # Number of members in the ensemble
KMAX = 64  # Maximum Fourier degree
T_SPAN = (0, 10)  # Time interval for the simulation
N_FRAMES = 500  # Number of frames in the animation

# --- Setup the spatial grid and wave model ---
X = Lebesgue(KMAX)
cw = CircleWave(
    X.kmax,
)

# --- Generate the ensemble of initial conditions ---
print(f"Generating an ensemble of {N_SAMPLES} initial states...")

# Create a Gaussian measure based on the heat kernel
mu = X.heat_kernel_gaussian_measure(0.1)
mu = 0.05 * mu

nu = mu.affine_mapping(
    translation=X.project_function(lambda th: np.exp(-10 * (th - np.pi) ** 2))
)
pi = mu


# Use the .samples(n) method to get a list of n random fields
# np.array() converts the list of 1D arrays into a single 2D array
q_samples = np.array(nu.samples(N_SAMPLES))
p_samples = np.array(pi.samples(N_SAMPLES))

# Concatenate displacement and momentum samples into a single state matrix
# The result has shape (N_SAMPLES, 2 * n_points)
initial_states_ensemble = np.concatenate((q_samples, p_samples), axis=1)

# --- Generate the ensemble animation ---
print("Computing and creating ensemble animation...")
# Call the new animate_ensemble method with the collection of initial states
animation = cw.animate_ensemble(
    initial_states_ensemble, T_SPAN, n_frames=N_FRAMES, alpha=0.15
)

# --- Save the animation ---
try:
    print("Saving animation to wave_ensemble_animation.mp4...")
    animation.save("wave_ensemble_animation.mp4", writer="ffmpeg", fps=30)
    print("Done.")
except Exception as e:
    print(f"\nError saving animation: {e}")
    print("Please ensure 'ffmpeg' is installed and accessible in your system's PATH.")
