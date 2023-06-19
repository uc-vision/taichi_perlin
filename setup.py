from setuptools import find_packages, setup

setup(name='taichi_perlin',
      version='0.1',
      install_requires=[
          'taichi',
          'beartype',
          'tqdm'
      ],
      packages=find_packages(),
      entry_points={}
)
