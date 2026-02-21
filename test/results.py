import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import advection_diffusion_simulator

path = "biscayne_oxygen/result.pkl"

with open(path, "rb") as f:
    result = pickle.load(f)

print(type(result))
print(result.solutions.shape)
print(result.mesh_mask.shape)
print(len(result.time_steps))
print(len(result.meshgrid))

print(np.max(result.meshgrid[0]), np.min(result.meshgrid[0]))
print(np.max(result.meshgrid[1]), np.min(result.meshgrid[1]))
print(result.meshgrid[0].shape)
print(result.config)
