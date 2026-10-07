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
  method3 bk method4 method5 eval_sec sqcurve3 metis lagrange fasthess
  mumpsfactor nulgraph fastloops)
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
# OpenMP for the parallel facet loops (fastloops.c); serial without it.
option(PYSE_OPENMP "Use OpenMP for parallel facet loops" ON)
if(PYSE_OPENMP)
  find_package(OpenMP COMPONENTS C)
  if(OpenMP_C_FOUND)
    target_link_libraries(evolver_core PUBLIC OpenMP::OpenMP_C)
  else()
    message(STATUS "OpenMP not found: facet loops run serially")
  endif()
endif()
# MUMPS (sequential, no MPI) for factoring Newton-step Hessians; downloaded
# and built from source with the scivision/mumps CMake build. Optional: it
# needs a Fortran compiler, LAPACK/BLAS and the network at build time;
# without them Evolver's own factoring is used.
option(PYSE_MUMPS "Build with MUMPS for Hessian factoring" ON)
set(PYSE_MUMPS_FOUND OFF)
if(PYSE_MUMPS)
  include(CheckLanguage)
  check_language(Fortran)
  if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
    # MUMPS spends its time in BLAS: prefer OpenBLAS (the reference BLAS
    # makes it several times slower)
    if(NOT DEFINED BLA_VENDOR)
      set(BLA_VENDOR OpenBLAS)
      find_package(LAPACK QUIET)
      # MUMPS's own FindLAPACK also wants the OpenBLAS header, without
      # looking in subdirectories (RHEL/manylinux: include/openblas)
      find_path(PYSE_OPENBLAS_INCLUDE_DIR NAMES openblas_config.h
        PATH_SUFFIXES openblas openblas-pthread openblas-openmp openblas-serial)
      if(LAPACK_FOUND AND PYSE_OPENBLAS_INCLUDE_DIR)
        set(LAPACK_VENDOR OpenBLAS CACHE STRING "LAPACK vendor for MUMPS")
        set(LAPACK_INCLUDE_DIR ${PYSE_OPENBLAS_INCLUDE_DIR} CACHE PATH "OpenBLAS headers")
      else()
        unset(BLA_VENDOR)
        set(LAPACK_FOUND FALSE)
      endif()
    endif()
    if(NOT LAPACK_FOUND)
      find_package(LAPACK)
    endif()
    if(LAPACK_FOUND)
      message(STATUS "pySE: LAPACK/BLAS for MUMPS: ${LAPACK_LIBRARIES}")
    endif()
  endif()
  if(NOT CMAKE_Fortran_COMPILER)
    message(WARNING "No Fortran compiler: building without MUMPS (Evolver's own Hessian factoring)")
  elseif(NOT LAPACK_FOUND)
    message(WARNING "No LAPACK/BLAS: building without MUMPS (Evolver's own Hessian factoring)")
  else()
    include(FetchContent)
    set(MUMPS_parallel OFF CACHE BOOL "" FORCE)
    set(MUMPS_scalapack OFF CACHE BOOL "" FORCE)
    set(MUMPS_openmp ON CACHE BOOL "" FORCE)
    set(MUMPS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_SINGLE OFF CACHE BOOL "" FORCE)
    set(BUILD_DOUBLE ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(mumps
      URL https://github.com/scivision/mumps/archive/refs/tags/v5.9.1.2.tar.gz
      URL_HASH SHA256=021e7b34d0d15aa134d758f84c4ede69bbc714b1c1fd15d5323be63a37704237)
    FetchContent_MakeAvailable(mumps)
    if(TARGET MUMPS::MUMPS)
      target_link_libraries(evolver_core PRIVATE MUMPS::MUMPS)
      target_compile_definitions(evolver_core PUBLIC PYSE_MUMPS)
      set(PYSE_MUMPS_FOUND ON)
    else()
      message(WARNING "MUMPS build not available: building without it")
    endif()
  endif()
endif()
message(STATUS "pySE: MUMPS Hessian factoring: ${PYSE_MUMPS_FOUND}")

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
find_package(Threads REQUIRED)   # the SIGINT handler forwards between threads
target_link_libraries(pyse_glue PUBLIC evolver_core Threads::Threads)
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
# Our files are the "python" install component (pyproject.toml installs only
# that), so the MUMPS subproject's install rules stay out of the wheel.
install(TARGETS _core LIBRARY DESTINATION pysurfaceevolver COMPONENT python)
# Evolver's sample datafiles and command scripts (see pysurfaceevolver.examples)
install(DIRECTORY fe/ DESTINATION pysurfaceevolver/data/fe COMPONENT python)

# Type stubs (_core.pyi), generated from the installed module.
nanobind_add_stub(
  _core_stub
  INSTALL_TIME
  MODULE pysurfaceevolver._core
  COMPONENT python
  OUTPUT pysurfaceevolver/_core.pyi
  PYTHON_PATH "."
)
