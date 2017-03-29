local common = dofile('../common.lua')
local dataloader = dofile('dataloader.lua')

function common.fname_to_labels(ps, name2label, err_on_new_label)
  if not name2label then
    name2label = {}
  end
  local max_label = 1
  for k, v in pairs(name2label) do
    if v > max_label then max_label = v end
  end

  local labels = torch.FloatTensor(#ps)

  for idx, path in ipairs(ps) do
    local name
    for n in paths.basename(path):gmatch('[train|test]_(.*)_[0-9]+') do
      name = n
    end

    if not name2label[name] then
      if err_on_new_label then 
        error('new label: '..name)
      else
        name2label[name] = max_label
        max_label = max_label + 1
      end
      -- print(name..' is a NEW label')
    -- else
      -- print(name..' is NOT a new label')
    end

    labels[idx] = name2label[name]
  end 
  return labels
end

function common.fname_train_test_split_mn(ps)
  local train = {}
  local test = {}

  for _, path in ipairs(ps) do
    p = paths.basename(path)
    if p:sub(1, 4) == 'test' and string.find(path, 'rot000') then
      table.insert(test, path)
    elseif p:sub(1, 5) == 'train' then
      table.insert(train, path)
    else
      -- error('invalid path: '..path)
    end
  end
  return train, test
end

function common.classification_worker(opt)
  print(string.format('out_root: %s', opt.out_root))
  -- create out root dir
  paths.mkdir(opt.out_root)

  -- load data_paths
  print('[INFO] load data_paths')
  local t = torch.Timer()
  local data_paths = common.walk_paths_cached(opt.ex_data_root, opt.ex_data_ext)
  table.sort(data_paths)
  print('[INFO] load data_paths took '..t:time().real..'[s], '..(#data_paths))

  print('[INFO] train test split')
  local t = torch.Timer()
  opt.train_paths, opt.test_paths = common.fname_train_test_split_mn(data_paths)
  print('[INFO] train test split took '..t:time().real..'[s], '..(#opt.train_paths)..', '..(#opt.test_paths))

  -- get labels
  opt.name2label = {}
  print('[INFO] get train_labels')
  local t = torch.Timer()
  opt.train_labels = common.fname_to_labels(opt.train_paths, opt.name2label)
  print('[INFO] get train_labels took '..t:time().real..'[s]')
  print(string.format('opt.n_classes=%d, #opt.name2label=%d', opt.n_classes, common.table_length(opt.name2label)))
  assert(opt.n_classes == common.table_length(opt.name2label))

  print('[INFO] get test_labels')
  local t = torch.Timer()
  opt.test_labels = common.fname_to_labels(opt.test_paths, opt.name2label)
  print('[INFO] get test_labels took '..t:time().real..'[s]')
  if opt.n_classes ~= common.table_length(opt.name2label) then
    for k, v in pairs(opt.name2label) do
      print(k, v)
    end
    error(string.format('number of classes in opt.n_classes %d ~= table_length(opt.name2label) %d', opt.n_classes, table_length(opt.name2label)))
  end
  
  -- get data loaders
  local train_data_loader = dataloader.DataLoader(opt.train_paths, opt.train_labels, opt.batch_size, opt.vx_size, opt.ex_data_ext, true)
  local test_data_loader = dataloader.DataLoader(opt.test_paths, opt.test_labels, opt.batch_size, opt.vx_size, opt.ex_data_ext, false)

  -- train
  common.worker(opt, train_data_loader, test_data_loader)
end

return common
