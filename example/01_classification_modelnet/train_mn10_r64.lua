#!/usr/bin/env th

local common = dofile('classification_common.lua')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('oc')

local opt = {}
opt.ex_data_root = 'preprocessed'
opt.ex_data_ext = 'oc'
opt.out_root = 'results'
opt.vx_size = 64
opt.n_classes = 10
opt.batch_size = 32

opt.weightDecay = 0.0001
opt.learningRate = 1e-3
opt.n_epochs = 20
opt.learningRate_steps = {}
opt.learningRate_steps[15] = 0.1
opt.optimizer = optim['adam']

local n_grids = 4096
opt.net = nn.Sequential()
  :add( oc.OctreeConvolutionMM(1,8, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(8,14, n_grids) )
  :add( oc.OctreeReLU(true) )
  
  :add( oc.OctreeConvolutionMM(14,14, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(14,20, n_grids) )
  :add( oc.OctreeReLU(true) )
  
  :add( oc.OctreeConvolutionMM(20,20, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeConvolutionMM(20,26, n_grids) )
  :add( oc.OctreeReLU(true) )
  :add( oc.OctreeGridPool2x2x2('max') )
  :add( oc.OctreeToCDHW() )
  
  :add( cudnn.VolumetricConvolution(26,26, 3,3,3, 1,1,1, 1,1,1) )
  :add( cudnn.ReLU(true) )
  :add( cudnn.VolumetricConvolution(26,32, 3,3,3, 1,1,1, 1,1,1) )
  :add( cudnn.ReLU(true) )
  :add( cudnn.VolumetricMaxPooling(2,2,2, 2,2,2) )

  :add( cudnn.VolumetricConvolution(32,32, 3,3,3, 1,1,1, 1,1,1) )
  :add( cudnn.ReLU(true) )
  :add( cudnn.VolumetricConvolution(32,32, 3,3,3, 1,1,1, 1,1,1) )
  :add( cudnn.ReLU(true) )
  :add( cudnn.VolumetricMaxPooling(2,2,2, 2,2,2) )

  :add( nn.View(32*8*8*8) )
  :add( nn.Dropout(0.5) )
  :add( nn.Linear(32*8*8*8, 512) )
  :add( cudnn.ReLU(true) )
  :add( nn.Linear(512, opt.n_classes) )
common.net_he_init(opt.net)
opt.net:cuda()
opt.criterion = nn.CrossEntropyCriterion()
opt.criterion:cuda()

common.classification_worker(opt)
