import taichi as ti
import taichi.math as tm
import numpy as np
from functools import cache



prime_sets = [
  (1, 2654435761, 805459861),
  (1, 1958374283, 2654435761),
]

@cache
def _noise_table(dim):
  vec = ti.types.vector(dim, ti.f32)
  ivec = ti.types.vector(dim, ti.i32)
  
  @ti.data_oriented
  class NoiseTable():

    def __init__(self, table_size=1024, primes=prime_sets[0]):
      self.table = ti.field(dtype=vec, shape=table_size)
      values = np.random.rand(table_size, dim).astype(np.float32)

      self.table.from_numpy(values * 2 - 1) 
      self.primes = primes

    @ti.func
    def hash(self, p:ivec) -> ti.int32:
        result = ti.int32(0)
        for i in ti.static(range(dim)):
            result ^= ti.cast(ti.uint64(p[i]) * ti.uint64(self.primes[i]), ti.int32)
        return result % self.table.shape[0]


    @ti.func
    def rand_vec(self, p:ivec) -> vec:
      return self.table[self.hash(p)]
        

    @ti.func
    def rand_unit(self, p:ivec) -> vec:
      v = self.table[self.hash(p)]
      return tm.normalize(v)
    
  return NoiseTable


def noise_table(dim, table_size=1024, primes=prime_sets[0]):
  return _noise_table(dim)(table_size, primes)