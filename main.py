import _gravity
import numpy as np

NSTEPS = 10
NBODIES = 100
R = 200
V = 100
G = 1e5
DT = 1e-4
DAMPING = 1 - 1e-7
SOFTENING = 10


def get_space():
    space = np.random.rand(NBODIES, 3, 2) * 2 - 1
    space[:, :, 0] *= R
    space[:, :, 1] *= V
    space[:, :, 0] -= space[:, :, 0].mean(0)
    space[:, :, 1] -= space[:, :, 1].mean(0)
    return space


space = get_space()
print(space[0])
_gravity.run(space, NSTEPS, G, DT, DAMPING, SOFTENING)
print(space[0])
