from dataclasses import dataclass, replace
from typing import Tuple
import numpy as np

from taichi_perlin.perlin_3d import NoiseGenerator3D
import torch
from beartype import beartype

RangeValue = Tuple[float, float] | float

@beartype
@dataclass 
class DropoutParams:
  # (range) controls the scale/base frequency of noise - larger = bigger areas, smaller
  noise_scale: RangeValue
  octaves: int

  # controls the frequency of the noise 0.5 - uniform dropout 2.0 smooth coherent dropout
  freq_multiplier: float  

  dropout: RangeValue # (range) proportion of points to drop completely
  peturb: RangeValue  # (range) proportion of points to peturb

  # controls the frequency which the peturbed directions change
  # use smaller number to give 0 mean noise, larger to give biased noise
  peturb_bias: float  
  peturb_distance: RangeValue # (range) controls the magnitude of the peturbation

  def _randomize(self, x):
    if isinstance(x, Tuple):
      return np.random.uniform(x[0], x[1])
    else:
      return x

  def randomize(self) -> 'DropoutParams':
    return replace(self, 
                  noise_scale=self._randomize(self.noise_scale),
                  dropout=self._randomize(self.dropout),
                  peturb=self._randomize(self.peturb),
                  peturb_distance=self._randomize(self.peturb_distance)
                )
  

    
class PointDropout:
  def __init__(self, params: DropoutParams):
    self.params = params

    scales = [ 
      (1/(2**i), 1/(params.freq_multiplier**i)) for i in range(params.octaves)]

    self.gen = NoiseGenerator3D(scales)
    self.gen_vec = NoiseGenerator3D( [(params.peturb_bias, 1)])

  def __call__(self, points: torch.Tensor, seed: int = None):

    if seed is None:
      seed = np.random.randint(0, 2**32 - 1)

    params = self.params.randomize()

    noise = torch.zeros((points.shape[0],), dtype=torch.float32, device=points.device)
    self.gen.sample_kernel(points, noise, seed=seed, noise_scale=params.noise_scale)

    quantiles = torch.tensor([min(1, params.dropout + params.peturb), params.dropout]).to(device=points.device)
    t_noisy, t_drop = torch.quantile(noise, quantiles)

    keep = torch.nonzero(noise > t_drop).squeeze()
    points = points[keep]

    noise = 1 - torch.clamp((noise[keep] - t_drop) / (t_noisy - t_drop), 0, 1)

    dirs = torch.zeros_like(points)
    self.gen_vec.sample_vec_kernel(points, dirs, seed=seed)
    dirs /= torch.norm(dirs, dim=1).reshape(-1, 1)

    points = points + (dirs * noise.reshape(-1, 1)) * params.peturb_distance

    return points, keep
  
