#include "TH.h"

extern "C" {
  #include "lua.h"
  #include "utils.h"
  #include "luaT.h"
}

#include <iostream>
#include <vector>
#include <map>
#include <sstream>
using namespace std;

class SPT {
public:
  int refCount;
  vector< THFloatTensor * > planes;
  map< int, int > denseBySparse;
  map< int, int > sparseByDense;
  int dims;
  vector<int> size;
  SPT() {
  }
  ~SPT() {
  }
};

static int SPT_pcoordToLinear(SPT *self, THLongStorage *pcoord);

static THLongStorage *getLongStorage(lua_State *L, int index) {
  void *longStorageVoid = luaT_checkudata(L, index, "torch.LongStorage");
  return (THLongStorage *)longStorageVoid;
}
static THLongStorage *getLongStorageNoCheck(lua_State *L, int index) {
  void *longStorageVoid = luaT_toudata(L, index, "torch.LongStorage");
  return (THLongStorage *)longStorageVoid;
}
static THFloatTensor *getFloatTensor(lua_State *L, int index) {
  void *tensorVoid = luaT_checkudata(L, index, "torch.FloatTensor");
  return (THFloatTensor *)tensorVoid;
}
static THFloatTensor *getFloatTensorNoCheck(lua_State *L, int index) {
  void *tensorVoid = luaT_toudata(L, index, "torch.FloatTensor");
  return (THFloatTensor *)tensorVoid;
}
static THDoubleTensor *getDoubleTensorNoCheck(lua_State *L, int index) {
  void *tensorVoid = luaT_toudata(L, index, "torch.DoubleTensor");
  return (THDoubleTensor *)tensorVoid;
}
static THDoubleTensor *getDoubleTensor(lua_State *L, int index) {
  void *tensorVoid = luaT_checkudata(L, index, "torch.DoubleTensor");
  return (THDoubleTensor *)tensorVoid;
}
static void push(lua_State *L, SPT *spt) {
  luaT_pushudata(L, spt, "torch.SparsePlanarTensor");
}
static void push(lua_State *L, THDoubleTensor *tensor) {
  luaT_pushudata(L, tensor, "torch.DoubleTensor");
}
static void push(lua_State *L, THFloatTensor *tensor) {
  luaT_pushudata(L, tensor, "torch.FloatTensor");
}

static SPT *getSPT(lua_State *L, int index) {
  SPT *self = (SPT *)luaT_checkudata(L, index, "torch.SparsePlanarTensor");
  return self;
}
static void SPT_rawInit(SPT *self) {
  self->refCount = 1;
}
static int SPT_new(lua_State *L) {
  SPT *self = (SPT*)THAlloc(sizeof(SPT));
  self = new(self) SPT();
  SPT_rawInit(self);

  if(getLongStorageNoCheck(L, 1) != 0) {
    THLongStorage *size = getLongStorageNoCheck(L, 1);
    cout << "got size" << endl;
    int dims = THLongStorage_size(size);
    cout << "size.size " << dims << endl;
    self->dims = dims;
    for(int d=0; d<dims; d++) {
       self->size.push_back(THLongStorage_get(size, d));
    }
  } else {
    THError("First parameter to torch.SparsePlanarTensor should be a longstorage");
  }

  luaT_pushudata(L, self, "torch.SparsePlanarTensor");
  return 1;
}
static int SPT_free(lua_State *L) {
  SPT *self = (SPT*)THAlloc(sizeof(SPT));
  if(!self) {
    return 0;
  }
  if(THAtomicDecrementRef(&self->refCount))
  {
    self->~SPT();
    THFree(self);
  }
  return 0;
}
static int SPT_factory(lua_State *L) {
  THError("SPT_factory not implemented");
  return 0;
}
static string floatTensorToString(THFloatTensor *tensor) {  // since this is implemented in lua in torch7
  ostringstream oss;
  int dims = THFloatTensor_nDimension(tensor);
  if(dims != 2) {
    THError("not implemented");
  }
  int rows = THFloatTensor_size(tensor, 0);
  int cols = THFloatTensor_size(tensor, 1);
  for(int y = 0; y < rows; y++ ) {
    for(int x = 0; x < cols; x++ ) {
      oss << " " << THFloatTensor_get2d(tensor, y, x);
    }
    oss << "\n";
  }
  return oss.str();
}
static int SPT_tostring(lua_State *L) {
  SPT *self = getSPT(L, 1);
  ostringstream oss;
  if(self->dims == 3) {
    for(map<int, int>::iterator it = self->denseBySparse.begin(); it != self->denseBySparse.end(); it++) {
      int s = it->first;
      int d = it->second;
      oss << "(" << (d+1) << ",.,.) =\n";
      oss << floatTensorToString(self->planes[s]);
    }
    oss << "[torch.SparsePlanarTensor of size ";
    for(int d=0; d < self->dims; d++) {
      if(d > 0) {
        oss << "x";
      }
      oss << self->size[d];
    }
    oss << "]";
  } else if(self->dims == 4){
    for(map<int, int>::iterator it = self->denseBySparse.begin(); it != self->denseBySparse.end(); it++) {
      int s = it->first;
      int d = it->second;
      THLongStorage *dcoords = THLongStorage_newWithSize(2);
      int x0 = d / self->size[1];
      int x1 = d % self->size[1];
//      THLongStorage_set(dcoords, 0, x0);
//      THLongStorage_set(dcoords, 1, x1);
      THLongStorage_free(dcoords);
      oss << "(" << (x0+1) << "," << (x1+1) << ",.,.) =\n";
      oss << floatTensorToString(self->planes[s]);
    }
// lua:
//      for s, dlinear in ipairs(self.denseBySparse) do
//         local dcoords = torch.LongStorage(2)
//         dcoords[1] = dlinear / self.size[2]
//         dcoords[2] = dlinear % self.size[2]
//         res = res .. 'feature plane [' .. dcoords[1] .. '][' .. dcoords[2] .. ']\n'
//         res = res .. self.planes[s]:__tostring__()
//      end
  } else {
    THError("not implemented");
  }
  lua_pushstring(L, oss.str().c_str());
  return 1;
}
//static int SPT_set3d(lua_State *L, int x1, int x2, int x3, float value) {
static int SPT_set3d(lua_State *L) {
  SPT *self = getSPT(L, 1);
  int x1 = luaL_checkint(L, 2)-1;
  int x2 = luaL_checkint(L, 3)-1;
  int x3 = luaL_checkint(L, 4)-1;
  float value = luaL_checknumber(L,5);

  int d = x1;
  if(self->dims != 3) {
    THError("cant use set3d on non-3d tensor");
  }
  if(self->sparseByDense.find(d) == self->sparseByDense.end()) {
    // create new plane
    int s = self->planes.size();
    self->denseBySparse[s] = d;
    self->sparseByDense[d] = s;
    THFloatTensor *newTensor = THFloatTensor_newWithSize2d(self->size[1], self->size[2]);
    THFloatTensor_zero(newTensor);
    self->planes.push_back(newTensor);
  }
  int s = self->sparseByDense.at(x1);
  THFloatTensor_set2d(self->planes[s], x2+1, x3+1, value);
  return 0;
}
static int SPT_get3d(lua_State *L) {
  SPT *self = getSPT(L, 1);
  int x1 = luaL_checkint(L, 2)-1;
  int x2 = luaL_checkint(L, 3)-1;
  int x3 = luaL_checkint(L, 4)-1;

  int d = x1;
  if(self->dims != 3) {
    THError("cant use get3d on non-3d tensor");
  }
  if(self->sparseByDense.find(d) == self->sparseByDense.end()) {
    lua_pushnumber(L, 0);
    return 1;
  }
  int s = self->sparseByDense.at(x1);
  lua_pushnumber(L, THFloatTensor_get2d(self->planes[s], x2+1, x3+1));
  return 1;
}
static void SPT_addPlane(SPT *self, THLongStorage *pcoord, THFloatTensor *second) {
  int linearIndex = SPT_pcoordToLinear(self, pcoord);
  int s = self->planes.size();
  self->denseBySparse[s] = linearIndex;
  self->sparseByDense[linearIndex] = s;
  int dims = self->dims;
  int H = self->size[dims-2];
  int W = self->size[dims-1];
//  THFloatTensor *plane = THFloatTensor_newWithSize2d(H, W);
  THFloatTensor_retain(second);
  self->planes.push_back(second);
}
static int SPT_addPlane(lua_State *L) {
  SPT *self = getSPT(L, 1);
  THLongStorage *pcoord = getLongStorage(L, 2);
  THFloatTensor *second = getFloatTensor(L, 3); // probalby should support float too
  SPT_addPlane(self, pcoord, second);
  push(L, self);
}
static int SPT_copy(lua_State *L) {
  // for now, both the num elements, and the size must match
  // might loosen this restriction in the future
  SPT *self = getSPT(L, 1);
  float tolerance = 0.000001f; // = tolerance or 0.000001
  if(getFloatTensor(L, 2) != 0) {
    THFloatTensor *second = getFloatTensor(L, 2);
    for(int p=0; p < (int)self->planes.size(); p++) {
      THFloatTensor_free(self->planes[p]);
    }
    self->planes.clear(); // hmmm, do we need to delete these planes?
    self->denseBySparse.clear();
    self->sparseByDense.clear();
    if(self->dims != THFloatTensor_nDimension(second)) {
      THError("Num dimensions doesnt match");
    }
    int dims = self->dims;
    for(int d=0; d < dims; d++) {
      if(THFloatTensor_size(second,d) != self->size[d]) {
        THError("tensor sizes dont match");
        return 0;
      }
    }
    if(dims == 3) {
      for(int d=0; d < dims; d++) {
        float maxvalue = THFloatTensor_maxall(second); // saves cloning, or writing our own method...
        float minvalue = THFloatTensor_minall(second);
        if(maxvalue >= tolerance || minvalue <= -tolerance) {
          THLongStorage *coord = THLongStorage_newWithSize(1);
          THLongStorage_set(coord, 0, d);
          THFloatTensor *plane = THFloatTensor_newSelect(second, 0, d);
          SPT_addPlane(self, coord, plane);
          THFloatTensor_free(plane);
          THLongStorage_free(coord);
        }
      }
      push(L, self);
      return 1;
    } else if(dims == 4) {
      int denseLinear = 0;
      int size0 = THFloatTensor_size(second, 0);
      int size1 = THFloatTensor_size(second, 1);
      for(int d=0; d < size0; d++) {
        THFloatTensor *dense0 = THFloatTensor_newSelect(second, 0, d);
        for(int e=0; e < size1; e++) {
          THFloatTensor *dense1 = THFloatTensor_newSelect(dense0, 0, e);
          float maxvalue = THFloatTensor_maxall(dense1); // saves cloning, or writing our own method...
          float minvalue = THFloatTensor_minall(dense1);
          if(maxvalue >= tolerance || minvalue <= -tolerance) {
            THLongStorage *coord = THLongStorage_newWithSize(2);
            THLongStorage_set(coord, 0, d);
            THLongStorage_set(coord, 1, e);
            SPT_addPlane(self, coord, dense1);
            THLongStorage_free(coord);
          }
          THFloatTensor_free(dense1);
        }
        THFloatTensor_free(dense0);
      }
      return 0;
    } else {
      THError("Not implemented");
      return 0;
    }
  } else {
    THError("Not implemented");
    return 0;
  }
}
// input: pcoord, a longstorage.  output: an integer, representing
// the linear position of the plane iwthin the tensor, if last two dimensions
// of tensor lopped off (reduced to size one, then removed)
static int SPT_pcoordToLinear(SPT *self, THLongStorage *pcoord) {
  int pcoord_dims = THLongStorage_size(pcoord);
  if(pcoord_dims + 2 != self->dims) {
    THError("pcoord dimnesions must be 2 less than sparse tensor dimensions");
  }
  if(pcoord_dims == 1) {
    return THLongStorage_get(pcoord, 0);
  } else if(pcoord_dims == 2){
    return THLongStorage_get(pcoord, 0) * self->size[1] + THLongStorage_get(pcoord, 1);
  } else {
    THError("not implemented");
  }
}
static int SPT_pcoordToLinear(lua_State *L) {
  SPT *self = getSPT(L, 1);
  THLongStorage *pcoord = getLongStorage(L, 2);
  int linear = SPT_pcoordToLinear(self, pcoord);
  lua_pushnumber(L, linear);
  return 1;
}
static int SPT_get1d(lua_State *L) {
  SPT *self = getSPT(L, 1);
  int x1 = luaL_checkint(L, 2)-1;

  int d = x1;
  if(self->dims != 3) {
    THError("Not implemented");
  }
  if(self->sparseByDense.find(d) == self->sparseByDense.end()) {
    lua_pushnil(L); // not sure if this is the best way, but has its good points
    return 1;
  }
  int s = self->sparseByDense.at(x1);
  luaT_pushudata(L, self->planes[s], "torch.FloatTensor");
  return 1;
}
static int SPT_add(lua_State *L) {
  SPT *self = getSPT(L, 1);
  SPT *second = getSPT(L, 2);
  for(map<int, int>::iterator it = self->denseBySparse.begin(); it != self->denseBySparse.end(); it++) {
    int s = it->first;
    int d = it->second;
    if(second->sparseByDense.find(d) != second->sparseByDense.end()) {
      int second_s = second->sparseByDense[d];
      THFloatTensor_cadd(self->planes[s], self->planes[s], 1, second->planes[second_s]);
    }
  }

  luaT_pushudata(L, self, "torch.SparsePlanarTensor");
  return 1;
}
static int SPT_cmul(lua_State *L) {
  SPT *self = getSPT(L, 1);
  SPT *second = getSPT(L, 2);
  for(map<int, int>::iterator it = self->denseBySparse.begin(); it != self->denseBySparse.end(); it++) {
    int s = it->first;
    int d = it->second;
    if(second->sparseByDense.find(d) != second->sparseByDense.end()) {
      int second_s = second->sparseByDense[d];
      THFloatTensor_cmul(self->planes[s], self->planes[s], second->planes[second_s]);
    }
  }

  luaT_pushudata(L, self, "torch.SparsePlanarTensor");
  return 1;
}
static const struct luaL_Reg SPT_funcs [] = {
  {"__tostring__", SPT_tostring},
  {"set3d", SPT_set3d},
  {"get3d", SPT_get3d},
  {"get1d", SPT_get1d},
  {"copy", SPT_copy},
  {"add", SPT_add},
  {"cmul", SPT_cmul},
  {"addPlane", SPT_addPlane},
  {"pcoordToLinear", SPT_pcoordToLinear},
  {0,0}
};
void SparsePlanarTensor_init(lua_State *L) {
  cout << "SparsePlanarTensor_init()" << endl;
  luaT_newmetatable(L, "torch.SparsePlanarTensor", NULL,
                    SPT_new, SPT_free, SPT_factory);
  luaL_setfuncs(L, SPT_funcs, 0);
  lua_pop(L, 1);
}

