#include "TH.h"
#include "luaT.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libsparseplanar(lua_State *L);

void nn_PlanarSparseTensor_init(lua_State *L);

int luaopen_libsparseplanar(lua_State *L) {
  nn_PlanarSparseTensor_init(L);
}

