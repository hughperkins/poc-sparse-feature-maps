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
using namespace std;

class SPT {
public:
  int refCount;
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

  if(lua_type(L, 1) == -1) { // not sure what the type should be right now :-P
  } else {
    THError("First parameter to torch.SparsePlanarTensor should be a longstorage");
  }

  luaT_pushudata(L, self, "torch.ClKernel");
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

