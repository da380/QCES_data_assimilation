import numpy as np
import matplotlib.pyplot as plt
from data_assimilation import (
    CircleWave,
)

from pygeoinf.symmetric_space.circle import Lebesgue


X = Lebesgue(128)


# 1. Setup the simulation
cw = CircleWave(
    X.kmax,
    rigidity=lambda th: 1
    + 0.5 * np.sin(th)
    + 5 * (th - np.pi) * np.exp(-200 * (th - np.pi) ** 2),
)


mu = X.heat_kernel_gaussian_measure(0.1)

q0 = mu.sample()
p0 = mu.sample()
z0 = np.concatenate((q0, p0))

# z0 = cw.project_displacement_and_momentum(
#    lambda th: np.exp(-50 * (th - np.pi / 2) ** 2), lambda th: 0
# )


# 3. Generate the animation object
print("Computing and creating animation...")
animation = cw.animate_solution(z0, (0, 10), n_frames=500)

# print("Saving animation to wave_animation.mp4...")
animation.save("wave_animation.mp4", writer="ffmpeg", fps=30)
print("Done.")

# 4. Show the animation window
# This is the essential step for terminal execution!
# print("Displaying animation. Close the window to exit.")
# plt.show()
