import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

import taichi as ti
from tqdm import tqdm

import torch

from taichi_perlin.dropout_3d import DropoutParams, PointDropout

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('filename', type=Path)

  parser.add_argument('--noise_scale', type=float, default=0.2)
  parser.add_argument('--octaves', type=int, default=8)
  parser.add_argument('--freq_multiplier', type=float, default=1.2)


  # Actions on point cloud
  parser.add_argument('--colorize', action='store_true')
  parser.add_argument('--dropout', type=float, default=0.2)
  parser.add_argument('--peturb', type=float, default=0.6)


  parser.add_argument('--peturb_distance', type=float, default=0.005)
  parser.add_argument('--peturb_bias', type=float, default=0.2)

  args = parser.parse_args()

  params = DropoutParams(
    noise_scale=args.noise_scale,
    octaves=args.octaves,
    freq_multiplier=args.freq_multiplier,
    
    dropout=args.dropout,
    peturb=args.peturb,

    peturb_bias=args.peturb_bias,
    peturb_distance=args.peturb_distance
  )

  return args.filename, params

  
def main():
  ti.init(arch=ti.gpu)

  filename, params = parse_args()
  dropout = PointDropout(params)
  
  pcd = o3d.io.read_point_cloud(str(filename))
  points = torch.tensor(np.asarray(pcd.points), device='cuda:0')

  for i in tqdm(range(1000)):
    noisy_points, keep = dropout(points, i)
    torch.cuda.synchronize()


  pcd = pcd.select_by_index(keep.cpu().numpy())
  pcd.points = o3d.utility.Vector3dVector(noisy_points.cpu().numpy())


  o3d.visualization.draw([pcd])


if __name__ == '__main__':
  main()