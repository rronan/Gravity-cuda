from time import time

import torch


def forward_physics(space, G, DT, DAMPING, SOFTENING):
    d = space[:, :, 0].reshape(-1, 1, 3) - space[:, :, 0]
    r = torch.norm(d) + SOFTENING
    f = G / r.pow(2)
    space[:, :, 1] += (DT * f * d / r).sum(1)
    return space


def run(space, NSTEPS, G, DT, DAMPING, SOFTENING):
    t0 = time()
    for i in range(NSTEPS):
        space = forward_physics(space, G, DT, DAMPING, SOFTENING)
        print(i)
    print(f"Torch: {time() - t0} seconds to execute")
