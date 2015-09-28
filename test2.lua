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

require 'nn'
require 'sparseplanar'
print('imported sparseplanar')

local a = torch.SparsePlanarTensor(torch.LongStorage({3,8,3}))
print('a', a)
a:set3d(2,4,1, 4.5)
print('a[2][4][1]', a:get3d(2,4,1))
print('a[2]', a:get1d(2))

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

a_sparse = torch.SparsePlanarTensor.fromDense(a)
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

c_sparse = sparse_convolve(a_sparse, b_sparse)

