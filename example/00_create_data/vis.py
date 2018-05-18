import numpy as np

def write_ply_pcl(out_path, xyz, color=[128,128,128], color_array=None):
  if xyz.shape[1] != 3:
    raise Exception('xyz has to be Nx3')

  f = open(out_path, 'w')
  f.write('ply\n')
  f.write('format ascii 1.0\n')
  f.write('element vertex %d\n' % xyz.shape[0])
  f.write('property float32 x\n')
  f.write('property float32 y\n')
  f.write('property float32 z\n')
  f.write('property uchar red\n')
  f.write('property uchar green\n')
  f.write('property uchar blue\n')
  f.write('end_header\n')

  for row in range(xyz.shape[0]):
    xyz_row = xyz[row]
    if color_array is not None:
      c = color_array[row]
    else:
      c = color
    f.write('%f %f %f %d %d %d\n' % (xyz_row[0],xyz_row[1],xyz_row[2], c[0],c[1],c[2]))
  f.close()


def write_ply_boxes(out_path, bxs, binary=False):
  if binary:
    f = open(out_path, 'wb')
  else:
    f = open(out_path, 'w')
  f.write("ply\n");
  if binary:
    f.write("format binary_little_endian 1.0\n")
  else:
    f.write("format ascii 1.0\n")
  f.write("element vertex %d\n" % (24 * len(bxs)))
  f.write("property float32 x\n")
  f.write("property float32 y\n")
  f.write("property float32 z\n")
  f.write("property uchar red\n")
  f.write("property uchar green\n")
  f.write("property uchar blue\n")
  f.write("element face %d\n" % (12 * len(bxs)))
  f.write("property list uchar int32 vertex_indices\n")
  f.write("end_header\n")

  if binary:
    write_fcn = lambda x,y,z,c0,c1,c2: f.write(struct.pack('<fffBBB', x,y,z, c0,c1,c2))
  else:
    write_fcn = lambda x,y,z,c0,c1,c2: f.write('%f %f %f %d %d %d\n' % (x,y,z, c0,c1,c2))
  for box in bxs:
    x0, x1 = box[0], box[1]
    y0, y1 = box[2], box[3]
    z0, z1 = box[4], box[5]
    pts = [
      [x1, y1, z1],
      [x1, y1, z0],
      [x1, y0, z1],
      [x1, y0, z0],
      [x0, y1, z1],
      [x0, y1, z0],
      [x0, y0, z1],
      [x0, y0, z0]
    ];

    write_fcn(pts[6][0],pts[6][1],pts[6][2], box[6],box[7],box[8])
    write_fcn(pts[7][0],pts[7][1],pts[7][2], box[6],box[7],box[8])
    write_fcn(pts[3][0],pts[3][1],pts[3][2], box[6],box[7],box[8])
    write_fcn(pts[2][0],pts[2][1],pts[2][2], box[6],box[7],box[8])
    write_fcn(pts[4][0],pts[4][1],pts[4][2], box[6],box[7],box[8])
    write_fcn(pts[5][0],pts[5][1],pts[5][2], box[6],box[7],box[8])
    write_fcn(pts[1][0],pts[1][1],pts[1][2], box[6],box[7],box[8])
    write_fcn(pts[0][0],pts[0][1],pts[0][2], box[6],box[7],box[8])
    write_fcn(pts[4][0],pts[4][1],pts[4][2], box[6],box[7],box[8])
    write_fcn(pts[6][0],pts[6][1],pts[6][2], box[6],box[7],box[8])
    write_fcn(pts[7][0],pts[7][1],pts[7][2], box[6],box[7],box[8])
    write_fcn(pts[5][0],pts[5][1],pts[5][2], box[6],box[7],box[8])
    write_fcn(pts[0][0],pts[0][1],pts[0][2], box[6],box[7],box[8])
    write_fcn(pts[2][0],pts[2][1],pts[2][2], box[6],box[7],box[8])
    write_fcn(pts[3][0],pts[3][1],pts[3][2], box[6],box[7],box[8])
    write_fcn(pts[1][0],pts[1][1],pts[1][2], box[6],box[7],box[8])
    write_fcn(pts[6][0],pts[6][1],pts[6][2], box[6],box[7],box[8])
    write_fcn(pts[2][0],pts[2][1],pts[2][2], box[6],box[7],box[8])
    write_fcn(pts[0][0],pts[0][1],pts[0][2], box[6],box[7],box[8])
    write_fcn(pts[4][0],pts[4][1],pts[4][2], box[6],box[7],box[8])
    write_fcn(pts[7][0],pts[7][1],pts[7][2], box[6],box[7],box[8])
    write_fcn(pts[3][0],pts[3][1],pts[3][2], box[6],box[7],box[8])
    write_fcn(pts[1][0],pts[1][1],pts[1][2], box[6],box[7],box[8])
    write_fcn(pts[5][0],pts[5][1],pts[5][2], box[6],box[7],box[8])

  if binary:
    write_fcn = lambda i0,i1,i2: f.write(struct.pack('<Biii', 3,i0,i1,i2))
  else:
    write_fcn = lambda i0,i1,i2: f.write('3 %d %d %d\n' % (i0,i1,i2))
  vidx = 0
  for box in bxs:
    write_fcn(vidx+0, vidx+1, vidx+2)
    write_fcn(vidx+0, vidx+3, vidx+2)
    vidx = vidx + 4;
    write_fcn(vidx+0, vidx+1, vidx+2)
    write_fcn(vidx+0, vidx+3, vidx+2)
    vidx = vidx + 4;
    write_fcn(vidx+0, vidx+1, vidx+2)
    write_fcn(vidx+0, vidx+3, vidx+2)
    vidx = vidx + 4;
    write_fcn(vidx+0, vidx+1, vidx+2)
    write_fcn(vidx+0, vidx+3, vidx+2)
    vidx = vidx + 4;
    write_fcn(vidx+0, vidx+1, vidx+2)
    write_fcn(vidx+0, vidx+3, vidx+2)
    vidx = vidx + 4;
    write_fcn(vidx+0, vidx+1, vidx+2)
    write_fcn(vidx+0, vidx+3, vidx+2)
    vidx = vidx + 4;

  f.close()

def write_ply_voxels(out_path, grid, color=[128,128,128], color_fcn=None, explode=1, binary=False):
  bxs = []
  width = 1
  for d in range(grid.shape[0]):
    for h in range(grid.shape[1]):
      for w in range(grid.shape[2]):
        if grid[d,h,w] == 0:
          continue
        x = w * explode
        y = h * explode
        z = d * explode
        if color_fcn is not None:
          color = color_fcn(d,h,w)
        bxs.append((x,x+width, y,y+width, z,z+width, color[0], color[1], color[2]))
  write_ply_boxes(out_path, bxs, binary)
