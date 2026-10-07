cmake_minimum_required(VERSION 3.18...3.30)
project(pysurfaceevolver LANGUAGES C CXX)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# Link-time optimization: Evolver's tiny element accessors (inline.h) are
# compiled out of line in skeleton.c; LTO inlines them across files.
# Measured: ~20% fewer instructions per iteration, identical results.
option(PYSE_LTO "Build with link-time optimization" ON)
if(PYSE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT PYSE_IPO_SUPPORTED OUTPUT PYSE_IPO_MESSAGE LANGUAGES C CXX)
  if(NOT PYSE_IPO_SUPPORTED)
    message(STATUS "LTO not supported, building without it: ${PYSE_IPO_MESSAGE}")
    set(PYSE_LTO OFF)
  endif()
endif()

# ---------------------------------------------------------------------------
# Surface Evolver itself, built headless (nulgraph) as a static library.
# The source list mirrors OBJ in src/Makefile.
set(EVOLVER_SOURCES
  calcforc variable trirevis stringl stringq model fixvol query matrix
  grapher painter filml filmq torvol lexinit graphgen modify userio boundary
  curtest display yexparse lexyy ytab hessian evaltree cnstrnt verpopst
  popfilm machine veravg pixgraph tmain tordup wulff help psgraph check
  utility skeleton storage dump iterate filgraph zoom softimag mvgraph
  diffuse sqcurve klein command hidim simplex metric torus quotient alice
  sdrv odrv userfunc kusner simequi2 geomgraph symtable exprint quantity
  meanint mindeg dodecGroup registry khyp gauss knot1 eval_all lexinit2
  evalmore knot2 knot3 teix sqcurve2 hessian2 hessian3 method1 method2
  method3 bk method4 method5 eval_sec sqcurve3 metis lagrange
  nulgraph)
list(TRANSFORM EVOLVER_SOURCES PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/)
list(TRANSFORM EVOLVER_SOURCES APPEND .c)

add_library(evolver_core STATIC ${EVOLVER_SOURCES})
# SYSTEM: code that includes Evolver's headers doesn't get their warnings.
target_include_directories(evolver_core SYSTEM PUBLIC src)
target_include_directories(evolver_core PRIVATE bindings)  # pyse_hooks.h
# LINUX selects the generic Unix configuration in include.h.  PYSE enables
# the library hooks.
target_compile_definitions(evolver_core PUBLIC LINUX PYSE)
set(EVOLVER_C_PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  INTERPROCEDURAL_OPTIMIZATION ${PYSE_LTO}
  C_VISIBILITY_PRESET hidden
  # The code uses K&R-style definitions, which C23 (GCC 15's default) removed.
  C_STANDARD 99
  C_EXTENSIONS ON)
set_target_properties(evolver_core PROPERTIES ${EVOLVER_C_PROPERTIES})
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  # 2013-era C: silence the flood of warnings, and keep newer compilers from
  # turning old idioms into hard errors.
  target_compile_options(evolver_core PRIVATE -w
    -Wno-error=implicit-function-declaration -Wno-error=implicit-int
    -Wno-error=int-conversion -Wno-error=incompatible-pointer-types)
endif()
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
  target_link_libraries(evolver_core PUBLIC ${MATH_LIBRARY})
endif()

# ---------------------------------------------------------------------------
# Our C glue, compiled with warnings on.  An OBJECT library, because the
# glue and Evolver call each other: as two static archives, the linker
# would not resolve the hooks Evolver calls.
add_library(pyse_glue OBJECT bindings/pyse_api.c)
target_include_directories(pyse_glue PUBLIC bindings)
target_link_libraries(pyse_glue PUBLIC evolver_core)
set_target_properties(pyse_glue PROPERTIES ${EVOLVER_C_PROPERTIES})
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pyse_glue PRIVATE -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Python extension.
# PYSE_NOSTRIP keeps symbols in the module, for profiling (callgrind, perf).
option(PYSE_NOSTRIP "Keep symbols in the extension module" OFF)
if(PYSE_NOSTRIP)
  nanobind_add_module(_core NB_STATIC NOSTRIP bindings/module.cpp)
else()
  nanobind_add_module(_core NB_STATIC bindings/module.cpp)
endif()
target_link_libraries(_core PRIVATE pyse_glue evolver_core)
set_target_properties(_core PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${PYSE_LTO})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(_core PRIVATE -Wall -Wextra)
endif()
install(TARGETS _core LIBRARY DESTINATION pysurfaceevolver)
# Evolver's sample datafiles and command scripts (see pysurfaceevolver.examples)
install(DIRECTORY fe/ DESTINATION pysurfaceevolver/data/fe)

# Type stubs (_core.pyi), generated from the installed module.
nanobind_add_stub(
  _core_stub
  INSTALL_TIME
  MODULE pysurfaceevolver._core
  OUTPUT pysurfaceevolver/_core.pyi
  PYTHON_PATH "."
)
