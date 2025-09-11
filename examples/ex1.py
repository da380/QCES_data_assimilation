import numpy as np
import matplotlib.pyplot as plt
from data_assimilation import (
    CircleWave,
)

# 1. Setup the simulation
wave_sim = CircleWave(256, rigidity=lambda th: 1 if th < 3 * np.pi / 2 else 2)


# 2. Create an initial condition (a Gaussian pulse)
def initial_displacement(theta):
    return np.exp(-((theta - np.pi) ** 2) * 20)


def initial_momentum(theta):
    return 0.0


z0 = wave_sim.project_displacement_and_momentum(initial_displacement, initial_momentum)


# 3. Generate the animation object
print("Computing and creating animation...")
animation = wave_sim.animate_solution(
    z0,
    (0, 10),
)

# print("Saving animation to wave_animation.mp4...")
animation.save("wave_animation.mp4", writer="ffmpeg", fps=30)
print("Done.")

# 4. Show the animation window
# This is the essential step for terminal execution!
# print("Displaying animation. Close the window to exit.")
# plt.show()
