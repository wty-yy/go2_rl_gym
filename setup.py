from setuptools import find_packages
from distutils.core import setup

setup(name='go2_rl_gym',
      version='1.0.0',
      author='Wu Tianyang',
      license="MIT",
      packages=find_packages(),
      author_email='993660140@qq.com',
      description='RL environments for Unitree Go2 Robots',
      install_requires=[
            'isaacgym',
            'rsl-rl',
            'matplotlib',
            'numpy==1.20',
            'tensorboard',
            'mujoco==3.2.3',
            'pyyaml',
            'onnx==1.17.0',
            'pygame'
      ])
