import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

import taichi as ti
from tqdm import tqdm

from taichi_perlin.perlin_3d import NoiseGenerator3D
import torch

@dataclass 
class DropoutParams:
  noise_scale: float

  octaves: int
  freq_multiplier: float

  dropout_proportion: float
  peturb_proportion: float

  peturb_scale: float
  peturb_distance: float

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
  parser.add_argument('--peturb_scale', type=float, default=0.2)

  args = parser.parse_args()

  params = DropoutParams(
    noise_scale=args.noise_scale,

    octaves=args.octaves,
    freq_multiplier=args.freq_multiplier,
    dropout_proportion=args.dropout,

    peturb_proportion=args.peturb,
    peturb_scale=args.peturb_scale,
    peturb_distance=args.peturb_distance
  )


  return args.filename, params


class PointDropout:
  def __init__(self, params: DropoutParams):
    self.params = params

    scales = [ 
      (params.noise_scale * 1/(2**i), 1/(params.freq_multiplier**i)) for i in range(params.octaves)]

    self.gen = NoiseGenerator3D(scales)
    self.gen_vec = NoiseGenerator3D( [(params.peturb_scale, 1)])

  def __call__(self, points: torch.Tensor, seed: int = None):

    if seed is None:
      seed = np.random.randint(0, 2**32 - 1)

    noise = torch.zeros((points.shape[0],), dtype=torch.float32, device=points.device)
    self.gen.sample_kernel(points, noise, seed=seed)

    params = self.params
    quantiles = torch.tensor([min(1, params.dropout_proportion + params.peturb_proportion), params.dropout_proportion]).to(device=points.device)
    t_noisy, t_drop = torch.quantile(noise, quantiles)

    keep = torch.nonzero(noise > t_drop).squeeze()
    points = points[keep]

    noise = 1 - torch.clamp((noise[keep] - t_drop) / (t_noisy - t_drop), 0, 1)

    dirs = torch.zeros_like(points)
    self.gen_vec.sample_vec_kernel(points, dirs, seed=seed)
    dirs /= torch.norm(dirs, dim=1).reshape(-1, 1)

    points = points + (dirs * noise.reshape(-1, 1)) * params.peturb_distance

    return points, keep
  
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