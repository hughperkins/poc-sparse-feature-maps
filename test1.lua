-- might as well start in lua first....

-- this tensor is 'per-plane', so each plane is fully dense, 2d, but we have a lookup table for the planes we 
-- are actually storing.  all other planes are zero.

-- this is just one of many possible sparse representations.  good points are:
-- - we can continue to use stndard convolution code
-- - dont need to mess around too much with coordinates, can reuse a lot of existing function sfor addition etc

-- possible steps:
-- - create basic per-plane sparse tensor
-- - add add, mul
-- - create a sparse convolutional layer
-- - create some sort of basic sparse convolutonal network, just a layer or two
-- - demonstrate that it learns better than a dense network with same number of parameters (this is obviously the hardest bit :-P)

local SparseTensor = torch.class('torch.SparseTensor')

function torch.SparseTensor:__init(size)
   self.dims = size:size()
   assert(self.dims >= 3)
   self.size =  torch.LongStorage(self.dims)
   self.size:copy(size)
--   self.planes = planes
   self.rows = size[self.dims-1]
   self.cols = size[self.dims]
   self.denseBySparse = {} -- if eg 4d, the lookup will be the linear position within the storage, if one suddenly shurnk the last two dimensions to 1, and 
                           -- assuming close-packed contiguous (can probably phrase this better somehow)
   self.sparseByDense = {}
   self.planes = {}
end

function torch.SparseTensor:set3d(p, row, col, value)
   local sparse = self.sparseByDense[p]
   if sparse == nil then
      sparse = #self.planes + 1
      self.denseBySparse[sparse] = p
      self.sparseByDense[p] = sparse
      self.planes[sparse] = torch.Tensor(self.rows, self.cols)
      self.planes[sparse]:fill(0)
   end
   self.planes[sparse][row][col] = value
end

function torch.SparseTensor:get1d(p)
   local sparse = self.sparseByDense[p]
   if sparse == nil then
      return 0
   end
   return self.planes[sparse]
end

function torch.SparseTensor:_pCoordToLinear(p_coord)
   print('p_coord', p_coord)
   local sparse_dims = p_coord:size()
   print('sparse_dims', sparse_dims)
   if sparse_dims == 1 then
      return p_coord[1]
   elseif sparse_dims == 2 then
      return p_coord[1] * self.size[2] + p_coord[2]
   else
      error("not implemented")
   end
--   assert(p_coord:size() + 2 == self.size:size())
--   local linear = 0
--   for d=1,p_coord:size() do
--      linear = linear * self.size[d]
--      linear = linear + p_coord[d]
--   end
--   return linear
end

function torch.SparseTensor:linearToPcoord(linear)
   local pcoord_dims = self.size:size() - 2
   local pcoord = torch.LongStorage(pcoord_dims)
   for d=pcoord_dims,1,-1 do
      local thiscoord = (linear - 1) % self.size[d] + 1
      pcoord[d] = thiscoord
      linear = (linear-1) / self.size[d]
   end
   return pcoord
end

function torch.SparseTensor:addPlane(p_coord, plane)  -- p_coord is the cooridnates without hte last two cordinates, eg if this tensor is 4d, p_coord will be 2d
   print('addPlane p_coord', p_coord, 'plane', plane)
   local linearIndex = self:_pCoordToLinear(p_coord)
   local s = #self.planes + 1
   self.planes[s] = plane:clone()
   self.denseBySparse[s] = linearIndex
   self.sparseByDense[linearIndex] = s
   return self
end

function torch.SparseTensor:getPlane(p_coord)
   local linearIndex = self:_pCoordToLinear(p_coord)
   local s = self.sparseByDense[linearIndex]
   if s == nil then
      return nil -- ? or return empty plane?
   end
   return self.planes[s]
end

function torch.SparseTensor:get3d(p, row, col)
   local sparse = self.sparseByDense[p]
   if sparse == nil then
      return 0
   end
   return self.planes[sparse][row][col]
end

function torch.SparseTensor:__tostring__()
   res = ''
   local sparse_dims = self.size:size() - 2
   if sparse_dims == 1 then
      for s, d in ipairs(self.denseBySparse) do
         res = res .. 'feature plane [' .. d .. ']\n'
         res = res .. self.planes[s]:__tostring__()
      end
   elseif sparse_dims == 2 then
      for s, dlinear in ipairs(self.denseBySparse) do
         local dcoords = torch.LongStorage(2)
         dcoords[1] = dlinear / self.size[2]
         dcoords[2] = dlinear % self.size[2]
         res = res .. 'feature plane [' .. dcoords[1] .. '][' .. dcoords[2] .. ']\n'
         res = res .. self.planes[s]:__tostring__()
      end
   else
      error("not implemented")
   end
   return res
end

--function torch.SparseTensor.fromDense(dense, tolerance)  -- if a plane has any value at least tolerance, then it's included, otherwise excluded
--   tolerance = tolerance or 0.000001
--   local dims = dense:dim()
--   assert(dims >= 3)
--   local sparse = torch.SparseTensor(dense:size())
--   local sparse_dims = dense:size():size() - 2
--   local sparse_coords = torch.LongStorage(sparse_dims)
--   for d=1,sparse_dims do
--      sparse_coords[d] = 1
--   end
--   local finished = false
--   local dense_linear = 1
--   local dense_by_d = {}
--   dense_by_d[1] = dense[1]
--   for d=2,sparse_dims do
--      dense_by_d[d] = dense_by_d[d-1][1]
--   end
--   while not finished do
--      print('fromDense sparse_dims', sparse_dims)
--      if dense_by_d[sparse_dims]:clone():abs():max() >= tolerance then
--         sparse:addPlane(sparse_coords,dense_by_d[sparse_dims])
--      end
--      local inced_d = sparse_dims
--      sparse_coords[inced_d] = sparse_coords[inced_d] + 1
--      if inced_d > 1 then
--         dense_by_d[inced_d] = dense_by_d[inced_d - 1][inced_d]
--      else
--         dense_by_d[inced_d] = dense[inced_d]
--      end
--      while inced_d > 1 and sparse_coords[inced_d] > dense:size(inced_d) do
--         sparse_coords[inced_d] = 1
--         inced_d = inced_d - 1
--         sparse_coords[inced_d] = sparse_coords[inced_d] + 1
--      end
--      if sparse_coords[inced_d] > dense:size(inced_d) then
--         finished = true
--      end
--      dense_linear = dense_linear + 1
--   end
----   for d=1,dense:size(1) do
----      if dense[d]:clone():abs():max() >= tolerance then
----         sparse:addPlane(d,dense[d])
----      end
----   end
--   return sparse
--end

function torch.SparseTensor.fromDense(dense, tolerance)  -- if a plane has any value at least tolerance, then it's included, otherwise excluded
   tolerance = tolerance or 0.000001
   local sparse = torch.SparseTensor(dense:size())
   local dims = dense:dim()
   assert(dims >= 3)
   if dims == 3 then
      for d=1,dense:size(1) do
         if dense[d]:clone():abs():max() >= tolerance then
            sparse:addPlane(torch.LongStorage({d}),dense[d])
         end
      end      
   elseif dims == 4 then
      local denseLinear = 1
      for d=1,dense:size(1) do
         local dense1 = dense[d]
         for e=1,dense:size(2) do
            local dense2 = dense1[e]
            if dense2:clone():abs():max() >= tolerance then
               sparse:addPlane(torch.LongStorage({d,e}),dense2)
            end
            denseLinear = denseLinear + 1
          end
      end      
   else
      error("not implemented")
   end
--   local sparse = torch.SparseTensor(dense:size())
--   local sparse_dims = dense:size():size() - 2
--   local sparse_coords = torch.LongStorage(sparse_dims)
--   for d=1,sparse_dims do
--      sparse_coords[d] = 1
--   end
--   local finished = false
--   local dense_linear = 1
--   local dense_by_d = {}
--   dense_by_d[1] = dense[1]
--   for d=2,sparse_dims do
--      dense_by_d[d] = dense_by_d[d-1][1]
--   end
--   while not finished do
--      print('fromDense sparse_dims', sparse_dims)
--      if dense_by_d[sparse_dims]:clone():abs():max() >= tolerance then
--         sparse:addPlane(sparse_coords,dense_by_d[sparse_dims])
--      end
--      local inced_d = sparse_dims
--      sparse_coords[inced_d] = sparse_coords[inced_d] + 1
--      if inced_d > 1 then
--         dense_by_d[inced_d] = dense_by_d[inced_d - 1][inced_d]
--      else
--         dense_by_d[inced_d] = dense[inced_d]
--      end
--      while inced_d > 1 and sparse_coords[inced_d] > dense:size(inced_d) do
--         sparse_coords[inced_d] = 1
--         inced_d = inced_d - 1
--         sparse_coords[inced_d] = sparse_coords[inced_d] + 1
--      end
--      if sparse_coords[inced_d] > dense:size(inced_d) then
--         finished = true
--      end
--      dense_linear = dense_linear + 1
--   end
--   for d=1,dense:size(1) do
--      if dense[d]:clone():abs():max() >= tolerance then
--         sparse:addPlane(d,dense[d])
--      end
--   end
   return sparse
end

function torch.SparseTensor:add(second)
   for s, d in ipairs(self.denseBySparse) do
      local second_s = second.sparseByDense[d]
      if second_s ~= nil then
         self.planes[s]:add(second.planes[second_s])
      end
   end
   return self
end

function torch.SparseTensor:cmul(second)
   for s, d in ipairs(self.denseBySparse) do
      local second_s = second.sparseByDense[d]
      if second_s ~= nil then
         self.planes[s]:cmul(second.planes[second_s])
      end
   end
   return self
end

local a = torch.SparseTensor(torch.LongStorage({3,8,3}))
a:set3d(2,4,1, 4.5)
print('a[2]', a:get1d(2))
print('a[2][4][1]', a:get3d(2,4,1))

-- things we should be able to do with these:
-- add them
-- multiply them by elemenet
-- SpatialConvolutionMM them
-- (print them...)

-- print
print('a', a)

-- add
local b = torch.SparseTensor(torch.LongStorage({3,8,3}))
b:set3d(2,4,1, 7.2)
print('b', b)
a:add(b)
print('a', a)

assert(b:_pCoordToLinear(torch.LongStorage({2})) == 2)

-- multiply by element
a:cmul(b)
print('a', a)

require 'nn'
-- lets make a normal, non-sparse, 3d tensor, and another
-- then, we'll fill it with random nubmers ('uniform'),then
-- set some planes to zero
-- then convolve those
-- then do the same with a sparse version
-- and compare

a = torch.Tensor(5,3,4):uniform()
a[2]:zero()
a[4]:zero()
--b = torch.Tensor(10,12,15):uniform()
--a[3]:zero()
--a[4]:zero()
--a[6]:zero()
--a[9]:zero()

mlp = nn.SpatialConvolutionMM(5,5,3,3,1,1,1,1)
-- hmmm.... b is ... weights?
local b = mlp.weight
b[1]:zero()
b[2]:zero()
b[5]:zero()
mlp.bias:zero()
-- remove bias, for simplicity

local out = mlp:forward(a)
print('out', out)

a_sparse = torch.SparseTensor.fromDense(a)
print('a_sparse', a_sparse)

print('b:size()', b:size())
print('b:numel()', b:numel())
print('5*5*3*3', 5*5*3*3)
b_view = b:view(5,5,3,3)
print('b_view:size()', b_view:size())
b_sparse = torch.SparseTensor.fromDense(b_view)
print('=========')
print('b_view', b_view)
print('=========')
print('b_sparse', b_sparse)

function sparse_convolve(input, filters)
   local outplanes = filters.size[1]
   local inplanes = filters.size[2]
   local kH = filters.size[3]
   local kW = filters.size[4]
   print(outplanes .. '->' .. inplanes .. ' ' .. kH .. 'x' .. kW)
   local output = torch.SparseTensor(torch.LongStorage({outplanes, kH, kW}))
   -- brute force for now...
   -- so we have:
   -- input is 3d tensor, where first dimension is number of input planes
   -- we will create an output tensor, 3d, where first dimension is number of output planes
   -- weights is 4 dimensional:
   -- - first dimension is number of output planes
   -- - second dimension is number of input planes
   -- for each output plane:
   --  - we should convolve each of the input planes with one of the filter planes
   --    ..... and add them together
   -- let's start by listing the coordinates of the available output planes, in the weights:
   local lastOutPlane = filters:linearToPcoord(filters.denseBySparse[1])[1]
   print('lastOutPlane', lastOutPlane)

   local convolver = nn.SpatialConvolutionMM(1, 1, kH, kW, 1, 1, kH/2, kW/2)
   convolver.bias:zero()
   for p_s, p_d in ipairs(filters.denseBySparse) do
      print('')
      print('============================================')
--      print('p_s', p_s, 'p_d', p_d)
      local pcoord = filters:linearToPcoord(p_d)
      print('p_s', p_s, 'p_d', p_d, 'pcoord ' .. pcoord[1] .. ',' .. pcoord[2])
      local currentOutputPlaneIdx = pcoord[1]
      local inputPlaneIdx = pcoord[2]
      if currentOutputPlaneIdx ~= lastOldPlane then
         -- finished the output feature, so write it to output, and then create new empty
         -- out plane
         local outPlane = filters:getPlane(pcoord):clone()
         output:addPlane(torch.LongStorage({currentOutputPlaneIdx}), outPlane)
      end
      local outPlane = output:getPlane(torch.LongStorage({currentOutputPlaneIdx}))
      print('outPlane', outPlane)
      -- convolve input with filter plane, and add to output plane
      local weightPlane = filters:getPlane(pcoord)
      print('weightPlane', weightPlane)
      local weightPlaneView = weightPlane:view(torch.LongStorage({1,1*kH*kW}))
      print('weightPlaneView', weightPlaneView)
      convolver.weight = weightPlaneView
      local inputPlane = input:getPlane(torch.LongStorage({inputPlaneIdx}))
      print('inputPlane', inputPlane)
      local inputPlane_view = inputPlane:view(torch.LongStorage({1,1,inputPlane:size(1), inputPlane:size(2)}))
      print('inputPlane_view', inputPlane_view)
      local thisConvRes = convolver:forward(inputPlane_view)
      local lastOldPlane = currentOutputPlaneIdx
   end
   return output
end

c_sparse = sparse_convolve(a_sparse, b_sparse)

