import taichi as ti
import taichi.math as tm
import numpy as np

import cv2

from noise_table import noise_table


@ti.data_oriented
class NoiseGenerator3D():
  def __init__(self):
    self.table = noise_table(3)

  @ti.func
  def grad_to(self, p:tm.vec3, i:tm.ivec3) -> ti.f32:
    v = self.table.rand_unit(i)
    return v.dot(p - ti.cast(i, ti.f32))

  @ti.func
  def grad(self, i:tm.ivec3) -> tm.vec3:
    return self.table.rand_unit(i)
    
  @ti.func
  def perlin_grad_3d(self, p:tm.vec3) -> tm.vec3:
    i = ti.cast(ti.floor(p), ti.i32)
    f = p - ti.cast(i, ti.f32)
    u = f * f * (3 - 2 * f)

    a000 = self.grad(i + tm.ivec3(0, 0, 0))
    a001 = self.grad(i + tm.ivec3(0, 0, 1))
    a010 = self.grad(i + tm.ivec3(0, 1, 0))
    a011 = self.grad(i + tm.ivec3(0, 1, 1))

    a100 = self.grad(i + tm.ivec3(1, 0, 0))
    a101 = self.grad(i + tm.ivec3(1, 0, 1))
    a110 = self.grad(i + tm.ivec3(1, 1, 0))
    a111 = self.grad(i + tm.ivec3(1, 1, 1))

    a0 = tm.mix(tm.mix(a000, a001, u.z), tm.mix(a010, a011, u.z), u.y)
    a1 = tm.mix(tm.mix(a100, a101, u.z), tm.mix(a110, a111, u.z), u.y)

    return tm.mix(a0, a1, u.x)

  @ti.func
  def perlin_3d(self, p:tm.vec3) -> ti.f32:
    i = ti.cast(ti.floor(p), ti.i32)
    f = p - ti.cast(i, ti.f32)
    u = f * f * (3 - 2 * f)

    a000 = self.grad_to(p, i + tm.ivec3(0, 0, 0))
    a001 = self.grad_to(p, i + tm.ivec3(0, 0, 1))
    a010 = self.grad_to(p, i + tm.ivec3(0, 1, 0))
    a011 = self.grad_to(p, i + tm.ivec3(0, 1, 1))

    a100 = self.grad_to(p, i + tm.ivec3(1, 0, 0))
    a101 = self.grad_to(p, i + tm.ivec3(1, 0, 1))
    a110 = self.grad_to(p, i + tm.ivec3(1, 1, 0))
    a111 = self.grad_to(p, i + tm.ivec3(1, 1, 1))

    a0 = tm.mix(tm.mix(a000, a001, u.z), tm.mix(a010, a011, u.z), u.y)
    a1 = tm.mix(tm.mix(a100, a101, u.z), tm.mix(a110, a111, u.z), u.y)

    return tm.mix(a0, a1, u.x)

  
  @ti.kernel
  def volume_vec_kernel(self, image:ti.types.ndarray(dtype=tm.vec3, ndim=3), scales:ti.template()):
    for i in ti.grouped(ti.ndrange(*image.shape)):
      out = tm.vec3(0.0)
      for (scale, weight) in ti.static(scales):
        p = tm.vec3(ti.f32(i.x), ti.f32(i.y), ti.f32(i.z)) / scale
        out += self.perlin_grad_3d(p) * weight
      image[i] = out 

  @ti.kernel
  def sample_vec_kernel(self, points:ti.types.ndarray(dtype=tm.vec3, ndim=1),
                     vectors:ti.types.ndarray(dtype=tm.vec3, ndim=1), 
                     scales:ti.template()):
    for i in range(points.shape[0]):
      vectors[i] = tm.vec3(0.0)
      for (scale, weight) in ti.static(scales):
        p = points[i] / scale
        vectors[i] += self.perlin_grad_3d(p) * weight
    
  @ti.kernel
  def sample_kernel(self, points:ti.types.ndarray(dtype=tm.vec3, ndim=1),
                     out:ti.types.ndarray(dtype=ti.f32, ndim=1), 
                     scales:ti.template()):
    
    for i in range(points.shape[0]):
      x:ti.f32 = 0.0
      
      for (scale, weight) in ti.static(scales):
        
        p = points[i] / scale
        x += self.perlin_3d(p) * weight
        
      out[i] = x / 2 + 0.5

if __name__ == "__main__":
  ti.init(arch=ti.gpu, offline_cache=False)

  image = np.zeros((512, 512, 1, 3), dtype=np.float32)
  noise = NoiseGenerator3D()

  scales = ((32, 1), (16, 0.5), (8, 0.25), (4, 0.125), (2, 0.0625))

  noise.volume_vec_kernel(image, scales) 
  cv2.imshow('image', image.reshape(512, 512, 3) / 2 + 0.5 )
  cv2.waitKey(0)
