#include "TH.h"
//#include "luaT.h"

extern "C" {
  #include "lua.h"
  //#include <lualib.h>
  //#include <lauxlib.h>
  #include "utils.h"
  #include "luaT.h"
}

#include <iostream>
#include <vector>
#include <map>
#include <sstream>
using namespace std;

static THLongStorage *getLongStorageNoCheck(lua_State *L, int index) {
  void *longStorageVoid = luaT_toudata(L, index, "torch.LongStorage");
  return (THLongStorage *)longStorageVoid;
}
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
static int SPT_tostring(lua_State *L) {
  SPT *self = getSPT(L, 1);
  ostringstream oss;
  oss << "[torch.SparsePlanarTensor of size ";
  for(int d=0; d < self->dims; d++) {
    if(d > 0) {
      oss << "x";
    }
    oss << self->size[d];
  }
  oss << "]";
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
    self->planes.push_back(THFloatTensor_newWithSize2d(self->size[1], self->size[2]));
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
static const struct luaL_Reg SPT_funcs [] = {
  {"__tostring__", SPT_tostring},
  {"set3d", SPT_set3d},
  {"get3d", SPT_get3d},
//  {"print", ClKernel_print},
//  {"getRenderedKernel", ClKernel_getRenderedKernel},
//  {"getRawKernel", ClKernel_getRawKernel},
//  {"run", ClKernel_run},
  {0,0}
};
void SparsePlanarTensor_init(lua_State *L) {
  cout << "SparsePlanarTensor_init()" << endl;
  luaT_newmetatable(L, "torch.SparsePlanarTensor", NULL,
                    SPT_new, SPT_free, SPT_factory);
  luaL_setfuncs(L, SPT_funcs, 0);
  lua_pop(L, 1);
}

