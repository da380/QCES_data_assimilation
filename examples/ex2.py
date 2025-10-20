import numpy as np
import matplotlib.pyplot as plt
from data_assimilation import (
    CircleWave,
)

import pygeoinf as inf

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

# q0 = mu.sample()
# p0 = mu.sample()
# z0 = np.concatenate((q0, p0))

# z0 = cw.project_displacement_and_momentum(
#    lambda th: np.exp(-20 * (th - np.pi / 2) ** 2), lambda th: 0
# )


propagator_matrix = cw.propagator(0, 1)


z0 = cw.project_displacement_and_momentum(
    lambda th: np.exp(-20 * (th - np.pi / 2) ** 2), lambda th: 0
)

z1 = cw.project_displacement_and_momentum(
    lambda th: np.exp(-20 * (th - np.pi) ** 2), lambda th: 0
)


(
    fig0,
    ax0,
) = cw.plot_state(z0)
(
    fig1,
    ax1,
) = cw.plot_state(z1)

plt.show()
