package = "sparseplanar"
version = "scm-1"

source = {
   url = "git://github.com/hughperkins/poc-sparse-feature-maps.git",
}

description = {
   summary = "Planar Sparse Tensors",
   detailed = [[
   ]],
   homepage = "https://github.com/hughperkins/poc-sparse-feature-maps",
   license = "BSD2"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}
