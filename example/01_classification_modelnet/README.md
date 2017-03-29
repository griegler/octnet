# ModelNet10 Classificaiton

This examples shows how to create grid-octree data from meshes and how to train a simple 3D CNN to classify those shapes:

1. Run `python2 create_data.py`. 
   This will download the ModelNet10 dataset, unzip the archive and then create for each mesh an grid-octree structure, which is stored to the disk.
2. Run `th train_mn10_r64.lua`.  
   This will train the network on the generated grid-octrees for 20 epochs. 

You can easily modify the create and train script to increase the resolution.
