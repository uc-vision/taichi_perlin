import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

import taichi as ti

from perlin_3d import NoiseGenerator3D


def main():
  ti.init(arch=ti.gpu)

  parser = argparse.ArgumentParser()
  parser.add_argument('filename', type=Path)
  parser.add_argument('--weight', type=float, default=0.05)
  parser.add_argument('--scale', type=float, default=0.5)


  args = parser.parse_args()

  pcd = o3d.io.read_point_cloud(str(args.filename))
  points = np.asarray(pcd.points)


  gen = NoiseGenerator3D()
  scales = tuple([(s * args.scale, w * args.weight) 
            for s, w in [(1, 1), (0.5, 0.5), (0.25, 0.25)]])


  offsets = np.zeros_like(points)

  print("Running..")
  gen.sample_vec_kernel(points, offsets, scales)

  print("Done.")

  pcd.points = o3d.utility.Vector3dVector(points + offsets)
  o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
  main()