import sys
from math import pi

import _gravity
import numpy as np

from display3d import Display3d

NSTEPS = 1
NBODIES = 30000
R = 10
V = 20
G = 2e3
DT = 1e-4
# DAMPING = 1 - 1e-8
DAMPING = 1
SOFTENING = 0.01
WRITE_INTERVAL = 10
USE_THREADS = 1
SQRTNT = 3

n = 0
while n * (SQRTNT**2) * 2 < NBODIES:
    n += 1
NBODIES = n * (SQRTNT**2) * 2
print("Setting NBODIES to:", NBODIES)


def get_sphere():
    theta = np.random.rand(NBODIES) * 2 * pi
    phi = np.arccos(np.random.rand(NBODIES) * 2 - 1)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    res = np.stack([x, y, z], 1)
    return res


def get_space():
    sphere = get_sphere()
    # comment this to have all stars on the sphere
    # space *= np.random.rand(NBODIES, 1, 2)
    space = np.stack([sphere * R, sphere * V], 2)
    space[:, :, 0] -= space[:, :, 0].mean(0)
    space[:, :, 1] -= space[:, :, 1].mean(0)
    return space


def parse_results(result_path="result.data"):
    with open(result_path, "r") as f:
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


if len(sys.argv) > 1:
    trajectories = parse_results(sys.argv[1])
else:
    space = get_space()
    _gravity.run(
        space, NSTEPS, G, DT, DAMPING, SOFTENING, WRITE_INTERVAL, USE_THREADS, SQRTNT
    )
    # trajectories = parse_results()
# app = Display3d(trajectories, camera_position=[0, R + V, 0], object_scale=1.5)
# app.run()
