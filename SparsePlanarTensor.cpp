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

static const struct luaL_Reg SPT_funcs [] = {
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

