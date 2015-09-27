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



local a = torch.SparseTensor(3,8,3)
a:set(2,4,1, 4.5)
print('a[2]', a:get1d(2))
print('a[2][4][1]', a:get3d(2,4,1))

