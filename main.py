from argparse import ArgumentParser
from math import pi

import numpy as np

import gravity_cpu
import gravity_gpu
from display3d import Display3d


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--nsteps", type=int, default=100000)
    parser.add_argument("--nbodies", type=int, default=1000)
    parser.add_argument("--r", type=float, default=10)
    parser.add_argument("--v", type=float, default=300)
    parser.add_argument("--G", type=float, default=2e3)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--damping", type=float, default=1)
    parser.add_argument("--softening", type=float, default=0.01)
    parser.add_argument("--object_scale", type=float, default=1.5)
    parser.add_argument("--write_interval", type=int, default=10)
    parser.add_argument("--trajectories", default=None)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    print(args)
    return args


def get_sphere(args):
    theta = np.random.rand(args.nbodies) * 2 * pi
    phi = np.arccos(np.random.rand(args.nbodies) * 2 - 1)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    res = np.stack([x, y, z], 1)
    return res


def get_space(args):
    sphere = get_sphere(args)
    # comment this to have all stars on the sphere
    # space *= np.random.rand(NBODIES, 1, 2)
    space = np.stack([sphere * args.r, sphere * args.v], 2)
    space[:, :, 0] -= space[:, :, 0].mean(0)
    space[:, :, 1] -= space[:, :, 1].mean(0)
    return space


def parse_results(result_path="trajectories/result.data"):
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


def main():
    args = parse_args()
    if args.trajectories is not None:
        trajectories = parse_results(args.trajectories)
    else:
        space = get_space(args)
        if args.gpu:
            gravity_gpu.run(
                space,
                args.nsteps,
                args.G,
                args.dt,
                args.damping,
                args.softening,
                args.write_interval,
            )
        else:
            gravity_cpu.run(
                space,
                args.nsteps,
                args.G,
                args.dt,
                args.damping,
                args.softening,
                args.write_interval,
            )
        trajectories = parse_results()
    try:
        app = Display3d(
            trajectories,
            camera_position=[0, args.r + args.v, 0],
            object_scale=args.object_scale,
        )
        app.run()
    except:
        pass


main()
