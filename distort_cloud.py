import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

import taichi as ti

from perlin_3d import NoiseGenerator3D

def vector_pow(v, p):
    mag = np.linalg.norm(v, axis=1)
    unit = v / mag[:, None]
    return (mag ** p)[:, None] * unit


def main():
  ti.init(arch=ti.gpu)


  parser = argparse.ArgumentParser()
  parser.add_argument('filename', type=Path)

  parser.add_argument('--scale', type=float, default=0.1)
  parser.add_argument('--octaves', type=int, default=8)
  parser.add_argument('--octave_scale', type=float, default=2)

  parser.add_argument('--exponent', type=float, default=1)

  # Actions on point cloud
  parser.add_argument('--colorize', action='store_true')

  parser.add_argument('--dropout', type=float, default=0.0)
  parser.add_argument('--distort', type=float, default=0.0)

  
  args = parser.parse_args()

  pcd = o3d.io.read_point_cloud(str(args.filename))
  points = np.asarray(pcd.points)

  gen = NoiseGenerator3D()
  scales = tuple([ 
    (args.scale * 1/(2**i), 1/(args.octave_scale**i)) for i in range(args.octaves)])

  args = parser.parse_args()

  pcd = o3d.io.read_point_cloud(str(args.filename))
  points = np.asarray(pcd.points)

  if args.colorize:
    noise = np.zeros(points.shape[0], dtype=np.float32)
    gen.sample_kernel(points, noise, scales)

    green = np.array([0, 1, 0], dtype=np.float32)
    red = np.array([1, 0, 0], dtype=np.float32)

    noise = np.expand_dims(noise, axis=1)
    colors = green * noise + red * (1 - noise)
    pcd.colors = o3d.utility.Vector3dVector(colors)
  
  if args.dropout > 0.0:
    
    noise = np.zeros(points.shape[0], dtype=np.float32)
    gen.sample_kernel(points, noise, scales)

    threshold = np.quantile(noise, args.dropout)

    keep = np.where(noise > threshold)[0]
    pcd = pcd.select_by_index(keep)

    # dropped = np.where(noise <= threshold)[0]
    # o3d.visualization.draw_geometries([  
    #   pcd.select_by_index(dropped).paint_uniform_color([1, 0, 0]),
    #   pcd.select_by_index(keep)
    # ])


  if args.distort > 0.0:

    offsets = np.zeros_like(points)
    gen.sample_vec_kernel(points, offsets, scales)

    mag = np.linalg.norm(offsets, axis=1)
    unit = offsets / mag[:, None]
    

    offsets = (mag ** args.exponent)[:, None] * unit

    pcd.points = o3d.utility.Vector3dVector(points + offsets * args.distort)

  o3d.visualization.draw([pcd])


if __name__ == '__main__':
  main()