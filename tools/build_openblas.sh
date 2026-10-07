#!/usr/bin/env bash
# Build a recent OpenBLAS for the manylinux wheels (cibuildwheel before-all).
#
# OpenMP build: inside MUMPS's parallel tree regions it runs serially (no
# nested OpenMP), above them it uses the threads, which MUMPS expects; it is
# also safe to call from several OpenMP threads at once (a USE_THREAD=0 build
# needs USE_LOCKING=1 for that, or Newton steps go wrong). Measured against
# serial+locking: Newton steps 4-13% faster. With LAPACK, and with kernels for
# every x86-64 CPU picked at run time (DYNAMIC_ARCH; baseline TARGET=PRESCOTT
# keeps the library itself portable). Installed to /usr/local, where CMake finds it;
# auditwheel bundles it into the wheel.
set -euo pipefail
version=0.3.34
sha256=cd7e129868320cc2d033afa920e31202dfe0b8066a5b66661900ccc0f197dfed
work=$(mktemp -d)
cd "$work"
curl -fsSL -o openblas.tar.gz \
  "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${version}/OpenBLAS-${version}.tar.gz"
echo "${sha256}  openblas.tar.gz" | sha256sum -c -
tar xzf openblas.tar.gz
cd "OpenBLAS-${version}"
opts="DYNAMIC_ARCH=1 TARGET=PRESCOTT BINARY=64 USE_OPENMP=1 NUM_THREADS=256 NO_AFFINITY=1 NO_STATIC=1"
make -j"$(nproc)" $opts libs netlib shared > build.log 2>&1 || { tail -50 build.log; exit 1; }
make $opts PREFIX=/usr/local install > install.log 2>&1 || { tail -50 install.log; exit 1; }
# the loader (stub generation, tests) and auditwheel must find it
echo /usr/local/lib > /etc/ld.so.conf.d/openblas.conf
ldconfig
echo "OpenBLAS ${version} installed in /usr/local"
