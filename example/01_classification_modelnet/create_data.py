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
pad = 2
out_root= './preprocessed'
n_rots = 1

n_processes = 10
n_threads = 1


# get MN10 data
if not os.path.exists('mn10.zip'):
  print('downloading ModelNet10')
  mn10 = urllib.URLopener()
  mn10.retrieve("http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip", "mn10.zip")

in_root = 'mn10'
if not os.path.isdir(in_root):
  print('unzipping ModelNet10')
  mn10 = zipfile.ZipFile('mn10.zip', 'r')
  mn10.extractall(in_root)
  mn10.close()



# create out directory
if not os.path.isdir(out_root):
  os.makedirs(out_root)


# list all off files
off_paths = []
for root, dirs, files in os.walk(in_root):
  off_paths.extend(glob(os.path.join(root, '*.off')))
off_paths.sort()


# fix off header for MN meshes
print('fixing off headers')
for path in off_paths:
  f = open(path, 'r')
  lines = f.readlines()
  f.close()

  # parse header
  if lines[0].strip().lower() != 'off':
    print(path)
    print(lines[0])

    splits = lines[0][3:].strip().split(' ')
    n_verts = int(splits[0])
    n_faces = int(splits[1])
    n_other = int(splits[2])

    f = open(path, 'w')
    f.write('OFF\n')
    f.write('%d %d %d\n' % (n_verts, n_faces, n_other))
    for line in lines[1:]:
      f.write(line)
    f.close()



# create grid-octree from off mesh
def worker(rot_idx, rot, off_idx, off_path):
  print('%d/%d - %d/%d - %s' % (rot_idx+1, n_rots, off_idx+1, len(off_paths), off_path))

  phi = rot / 180.0 * np.pi
  R = np.array([
      [np.cos(phi), -np.sin(phi), 0],
      [np.sin(phi), np.cos(phi), 0],
      [0, 0, 1]
    ], dtype=np.float32)
  rot_out_dir = os.path.join(out_root, 'rot%03d' % np.round(rot))


  basename, ext = os.path.splitext(os.path.basename(off_path))
  train_test_prefix = os.path.basename(os.path.dirname(off_path))

  print('create octree')
  t = time.time()
  grid = pyoctnet.Octree.create_from_off(off_path, vx_res,vx_res,vx_res, R, pad=pad, n_threads=n_threads)
  print('  took %f[s]' % (time.time() - t))

  oc_out_path = os.path.join(rot_out_dir, '%s_%s.oc' % (train_test_prefix, basename))
  print('write bin - %s' % oc_out_path)
  t = time.time()
  grid.write_bin(oc_out_path)
  print('  took %f[s]' % (time.time() - t))




start_t = time.time()
if n_processes > 1:
  pool = multiprocessing.Pool(processes=n_processes)
for rot_idx, rot in enumerate(np.linspace(0, 360, n_rots, endpoint=False)):
  rot_out_dir = os.path.join(out_root, 'rot%03d' % np.round(rot))
  if not os.path.isdir(rot_out_dir):
    os.makedirs(rot_out_dir)


  for off_idx, off_path in enumerate(off_paths):
    print('%d pool.apply_async' % off_idx)
    if n_processes > 1:
      pool.apply_async(worker, args=(rot_idx, rot, off_idx, off_path,))
    else:
      worker(rot_idx, rot, off_idx, off_path)

if n_processes > 1:
  pool.close()
  pool.join()

print('create_data took %f[s]' % (time.time() - start_t))
