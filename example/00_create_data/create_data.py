#!/usr/bin/env python2

import sys
import numpy as np
import time
import os
from glob import glob
import random
import multiprocessing
import urllib
import zipfile

sys.path.append('../../py/')
import pyoctnet


random.seed(42)
np.random.seed(42)

vx_res = 64
n_threads = 10


# create oc from voxel grid
dense = np.zeros((vx_res,vx_res,vx_res), dtype=np.float32)
dense[:,0,1:vx_res-1] = 1
dense[:,vx_res-1,1:vx_res-1] = 1
dense[:,:,1] = 1
dense[:,:,vx_res-2] = 1
val_range = np.array([(0.9, 1.1)], dtype=np.float32)
oc_from_dense1 = pyoctnet.Octree.create_from_dense(dense, val_range, n_threads=n_threads)
oc_from_dense2 = pyoctnet.Octree.create_from_dense2(dense, dense[np.newaxis,...], n_threads=n_threads)

# create from point cloud
xyz = np.random.normal(0, 1, (100, 3)).astype(np.float32)
features = np.ones_like(xyz)
oc_from_pcl = pyoctnet.Octree.create_from_pc(xyz, features, vx_res,vx_res,vx_res, normalize=True, n_threads=n_threads)

for idx in range(3):
  xyz[...,idx] = (xyz[...,idx] - xyz[...,idx].min()) / (xyz[...,idx].max() - xyz[...,idx].min())
  xyz[...,idx] *= vx_res
oc_from_pcl = pyoctnet.Octree.create_from_pc(xyz, features, vx_res,vx_res,vx_res, normalize=False, n_threads=n_threads)

