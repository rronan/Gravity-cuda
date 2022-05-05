import _gravity
import numpy as np

NSTEPS = 100000
NBODIES = 100
R = 200
V = 100
G = 1e5
DT = 1e-4
DAMPING = 1 - 1e-7
SOFTENING = 10
WRITE_INTERVAL = 10


def get_space():
    space = np.random.rand(NBODIES, 3, 2) * 2 - 1
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


# space = get_space()
# np.save("space0.npy", space)
space = np.load("space0.npy")
_gravity.run(space, NSTEPS, G, DT, DAMPING, SOFTENING, WRITE_INTERVAL)
trajectories = parse_results()
print(trajectories)
