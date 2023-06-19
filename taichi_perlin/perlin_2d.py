from typing import Sequence, Tuple
import taichi as ti
import taichi.math as tm
import numpy as np


from .noise_table import noise_table


@ti.data_oriented
class NoiseGenerator2D():
  def __init__(self, scales:Sequence[Tuple[float, float]]):
    self.table = noise_table(2)
    self.scales = tuple(scales)

  @ti.func
  def grad_to(self, p:tm.vec2, i:tm.ivec2, seed:ti.i32) -> ti.f32:
    v = self.table.rand_unit(i, seed)
    return v.dot(p - ti.cast(i, ti.f32))
    
  @ti.func
  def perlin_2d(self, p:tm.vec2, seed:ti.i32) -> ti.f32:
    i = ti.floor(p, dtype=ti.i32)
    f = p - i
    u = f * f * (3 - 2 * f)
    
    a00 = self.grad_to(p, i + tm.ivec2(0, 0), seed)
    a01 = self.grad_to(p, i + tm.ivec2(0, 1), seed)
    a10 = self.grad_to(p, i + tm.ivec2(1, 0), seed)
    a11 = self.grad_to(p, i + tm.ivec2(1, 1), seed)

    return tm.mix(
       tm.mix(a00, a01, u.y), tm.mix(a10, a11, u.y), u.x)  

  @ti.kernel
  def image_kernel(self, image:ti.types.ndarray(dtype=ti.f32, ndim=2), scales:ti.template(), seed:ti.i32):

    for i in ti.grouped(ti.ndrange(*image.shape)):
      out = 0.0
      for (scale, weight) in ti.static(scales):
        p = tm.vec2(ti.f32(i.x), ti.f32(i.y)) / scale
        out += self.perlin_2d(p, seed) * weight
      
      image[i] = out / 2 + 0.5






if __name__ == "__main__":
  import cv2

  ti.init(arch=ti.gpu, offline_cache=False)
  np.random.seed(0)

  scales = ((64, 1), (32, 1), (16, 0.5), (8, 0.25), (4, 0.125), (2, 0.0625))

  image = np.zeros((512, 512), dtype=np.float32)
  noise = NoiseGenerator2D(scales)


  noise.image_kernel(image, 0) 
  cv2.imshow('image', image)
  cv2.waitKey(0)

