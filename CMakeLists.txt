cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE(Torch REQUIRED)

LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

SET(src init.cpp SparsePlanarTensor.cpp)

FILE(GLOB luasrc *.lua)

ADD_TORCH_PACKAGE(sparseplanar "${src}" "${luasrc}")

TARGET_LINK_LIBRARIES(sparseplanar luaT TH)

#INSTALL(DIRECTORY "doc" DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/PlanarSparseTensor")
INSTALL(FILES "README.md" DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/PlanarSparseTensor")

