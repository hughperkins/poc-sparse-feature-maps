#include "TH.h"
#include "luaT.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libsparseplanar(lua_State *L);

void SparsePlanarTensor_init(lua_State *L);

int luaopen_libsparseplanar(lua_State *L) {
  SparsePlanarTensor_init(L);
  return 0;
}

