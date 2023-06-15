import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

import taichi as ti

from perlin_3d import NoiseGenerator3D

def vector_pow(v, p):
    mag = np.linalg.norm(v, axis=1)
    mean = np.mean(mag)

    unit = v / mag[:, None]
    mag = (mag ** p)
    mag *= mean / np.mean(mag)

    return mag[:, None] * unit

def main():
  ti.init(arch=ti.gpu)

  parser = argparse.ArgumentParser()
  parser.add_argument('filename', type=Path)

  parser.add_argument('--scale', type=float, default=2.0)
  parser.add_argument('--magnitude', type=float, default=0.004)

  
  args = parser.parse_args()

  pcd = o3d.io.read_point_cloud(str(args.filename))
  points = np.asarray(pcd.points)

  gen = NoiseGenerator3D()

  scales = ((1, 1), (0.5, 1), (0.25, 1), (0.125, 1))
  scales = tuple([(args.scale * s, w) for s, w in scales])

  args = parser.parse_args()

  pcd = o3d.io.read_point_cloud(str(args.filename))
  points = np.asarray(pcd.points)



  offsets = np.zeros_like(points)
  gen.sample_vec_kernel(points, offsets, scales)

  pcd.points = o3d.utility.Vector3dVector(points + vector_pow(offsets, 2) * args.magnitude )
  o3d.visualization.draw([pcd])


if __name__ == '__main__':
  main()