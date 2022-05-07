from math import pi

import _gravity
import numpy as np

from display3d import Display3d

NSTEPS = 100000
NBODIES = 100
R = 200
V = 100
G = 1e5
DT = 1e-4
# DAMPING = 1 - 1e-8
DAMPING = 1
SOFTENING = 0.01
WRITE_INTERVAL = 10


def get_space():
    polar = np.random.rand(NBODIES, 2, 2) * 2 * pi
    space = np.zeros((NBODIES, 3, 2))
    space[:, 0] = np.cos(polar[:, 0]) * np.cos(polar[:, 1])
    space[:, 1] = np.sin(polar[:, 0]) * np.cos(polar[:, 1])
    space[:, 2] = np.sin(polar[:, 1])
    # comment this to have all stars on the sphere
    # space *= np.random.rand(NBODIES, 1, 2)
    space[:, :, 0] *= R
    space[:, :, 1] *= V
    space[:, :, 0] -= space[:, :, 0].mean(0)
    space[:, :, 1] -= space[:, :, 1].mean(0)
    return space


def parse_results():
    with open("result.data", "r") as f:
        text_list = f.readlines()
    space_list = []
    space = []
    for text in text_list:
        if text == "\n":
            space_list.append(space)
            space = []
        else:
            space.append([float(x) for x in text[:-2].split(" ")])
    res = np.array(space_list).astype(float)
    return res


space = get_space()
_gravity.run(space, NSTEPS, G, DT, DAMPING, SOFTENING, WRITE_INTERVAL)
trajectories = parse_results()
app = Display3d(trajectories / 10, camera_position=[0, R / 10 * 5, 0], object_scale=1)
app.run()
