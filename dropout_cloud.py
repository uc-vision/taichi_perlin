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
  parser.add_argument('--scale', type=float, default=0.05)


  args = parser.parse_args()

  pcd = o3d.io.read_point_cloud(str(args.filename))
  points = np.asarray(pcd.points)

  gen = NoiseGenerator3D()
  scales = tuple([(s * args.scale, w) 
            for s, w in [(1, 1), (0.5, 0.5), (0.25, 0.5), (0.125, 0.5)]])

  noise = np.zeros(points.shape[0], dtype=np.float32)

  print("Running..")
  gen.sample_kernel(points, noise, scales)

  print("Done.")

  dropped = pcd.select_by_index(np.where(noise < 0.6)[0])
  o3d.visualization.draw_geometries([dropped])


if __name__ == '__main__':
  main()