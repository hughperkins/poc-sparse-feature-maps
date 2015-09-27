-- might as well start in lua first....


local SparseTensor = torch.class('torch.SparseTensor')

function torch.SparseTensor:__init(planes, rows, cols)
   self.planes = planes
   self.rows = rows
   self.cols = cols
   self.denseBySparse = {}
   self.sparseByDense = {}
   self.planes = {}
end

function torch.SparseTensor:set(p, row, col, value)
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

function torch.SparseTensor:get3d(p, row, col)
   local sparse = self.sparseByDense[p]
   if sparse == nil then
      return 0
   end
   return self.planes[sparse][row][col]
end

function torch.SparseTensor:__tostring__()
   res = ''
   for s, d in ipairs(self.denseBySparse) do
      res = res .. 'feature plane ' .. d .. '\n'
      res = res .. self.planes[s]:__tostring__()
   end
   return res
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

local a = torch.SparseTensor(3,8,3)
a:set(2,4,1, 4.5)
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
local b = torch.SparseTensor(3,8,3)
b:set(2,4,1, 7.2)
print('b', b)
a:add(b)
print('a', a)

-- multiply by element
a:cmul(b)
print('a', a)


