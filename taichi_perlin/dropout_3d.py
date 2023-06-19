from dataclasses import dataclass
import numpy as np

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
  
