from time import time

import torch

NBODIES = 50
NSTEPS = int(1e5)
DAMPING = 1 - 1e-7
SOFTENING = 10
R = 200
V = 100
G = 1e5
DT = 1e-4


def get_space():
    space = torch.rand(NBODIES, 3, 2) * 2 - 1
    space[:, :, 0] *= R
    space[:, :, 1] *= V
    space[:, :, 0] -= space[:, :, 0].mean(0)
    space[:, :, 1] -= space[:, :, 1].mean(0)
    return space


def forward_physics(space):
    d = space[:, :, 0].reshape(-1, 1, 3) - space[:, :, 0]
    r = torch.norm(d) + SOFTENING
    f = G / r.pow(2)
    space[:, :, 1] += (DT * f * d / r).sum(1)
    return space


def run():
    space = get_space()
    t0 = time()
    for i in range(NSTEPS):
        space = forward_physics(space)
        print(i)
    print(f"Torch: {time() - t0} seconds to execute")
