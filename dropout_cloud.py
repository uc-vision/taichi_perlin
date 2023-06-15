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

  parser.add_argument('--noise_scale', type=float, default=0.1)
  parser.add_argument('--octaves', type=int, default=6)
  parser.add_argument('--octave_scale', type=float, default=1.2)

  parser.add_argument('--dir_scale', type=float, default=0.2)

  # Actions on point cloud
  parser.add_argument('--colorize', action='store_true')
  parser.add_argument('--dropout', type=float, default=0.1)
  parser.add_argument('--noisy', type=float, default=0.4)


  parser.add_argument('--distort', type=float, default=0.005)

  
  args = parser.parse_args()

  pcd = o3d.io.read_point_cloud(str(args.filename))
  points = np.asarray(pcd.points)

  gen = NoiseGenerator3D()
  scales = tuple([ 
    (args.noise_scale * 1/(2**i), 1/(args.octave_scale**i)) for i in range(args.octaves)])

  gen_vec = NoiseGenerator3D()

  pcd = o3d.io.read_point_cloud(str(args.filename))
  points = np.asarray(pcd.points)
  
  noise = np.zeros(points.shape[0], dtype=np.float32)
  gen.sample_kernel(points, noise, scales)

  t_noisy, t_drop = np.quantile(noise, [args.dropout + args.noisy, args.dropout])

  keep = np.where(noise > t_drop)[0]
  pcd = pcd.select_by_index(keep)

  noise = 1 - np.clip(0, 1, (noise[keep] - t_drop) / (t_noisy - t_drop))
  
  points = np.asarray(pcd.points)
  dirs = np.zeros_like(points)
  gen_vec.sample_vec_kernel(points, dirs, ((args.dir_scale, 1),) )
  dirs /= np.linalg.norm(dirs, axis=1).reshape(-1, 1)
  
  vecs = dirs * noise.reshape(-1, 1) 
  pcd.points = o3d.utility.Vector3dVector(points + vecs * args.distort)


  colors = np.asarray(pcd.colors).reshape(-1, 3)
  noise = noise.reshape(-1, 1)
  colors = colors * (1 - noise) + noise * np.array([1, 0.0, 0.0]).reshape(1, 3)

  pcd.colors = o3d.utility.Vector3dVector(colors)

  o3d.visualization.draw([pcd])


if __name__ == '__main__':
  main()