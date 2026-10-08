#!/usr/bin/env bash
# Build the stock Evolver program with AddressSanitizer and UBSan, run every
# sample datafile in fe/ through a sequence of commands, and fail on any
# sanitizer report. Then the same with OpenMP on a few refined samples, large
# enough for the parallel loops (they start at 4096 elements), with Newton
# steps.
#
# usage: tools/run_sanitizers.sh [--quick] [build-dir]
#   --quick: only the OpenMP build, on two refined samples (normal and quantity
#   mode), built incrementally in build/sanitize-quick: about a minute after
#   the first build. The full run (all samples, then OpenMP) is CI's.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
quick=0
if [ "${1:-}" = "--quick" ]; then quick=1; shift; fi
if [ "$quick" = 1 ]; then build=${1:-$root/build/sanitize-quick}; else build=${1:-$(mktemp -d)}; fi
flags="-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined"

# build <dir> <extra flags>
build_evolver() {
  mkdir -p "$1"
  # cp -u: unchanged sources keep their times, so make rebuilds only what changed
  cp -u "$root"/src/*.c "$root"/src/*.h "$root"/src/Makefile "$1"/
  make -C "$1" -j"$(nproc)" CC="gcc $flags $2" CFLAGS="-DLINUX -w" GRAPH=nulgraph.o GRAPHLIB= evolver >"$1/build.log" 2>&1 \
    || { tail -20 "$1/build.log"; exit 1; }
}

export ASAN_OPTIONS=detect_leaks=0
failures=0

# run <build dir> <script> <label> <datafiles...>
run() {
  local dir=$1 script=$2 label=$3 f log
  shift 3
  for f in "$@"; do
    log="$dir/$f.$label.log"
    # SIGKILL on timeout: SIGTERM runs Evolver's handler, which dumps the
    # surface from inside the signal handler and can trip ASan by itself
    local status=0
    printf "$script" | timeout -s KILL 1500 "$dir/evolver" "$f" >"$log" 2>&1 || status=$?
    if [ "$status" = 137 ]; then
      echo "FAIL $f ($label): timed out after 1500 s"
      failures=$((failures + 1))
    elif grep -q "ERROR: AddressSanitizer\|runtime error" "$log"; then
      echo "FAIL $f ($label):"
      grep -A12 "ERROR: AddressSanitizer\|runtime error" "$log" | head -16
      failures=$((failures + 1))
    else
      echo "ok   $f ($label)"
    fi
  done
}

# Each script exercises iteration, refinement, all element models, vertex
# averaging, equiangulation, weeding, listing and dumping; one adds Newton.
scripts=(
  'g 5\nr\ng 5\nquadratic\ng 3\nlagrange 3\ng 3\nlinear\nV\nu\nw 0.001\ng 3\nlist vertices\ndump "/dev/null"\nq\n'
  'g 5\nr\ng 5\nhessian\nquadratic\ng 3\nlagrange 3\ng 3\nlinear\nV\nu\ng 3\nlist vertices\ndump "/dev/null"\nq\n'
)
cd "$root/fe"
if [ "$quick" = 0 ]; then
  build_evolver "$build/stock" ""
  for i in "${!scripts[@]}"; do
    run "$build/stock" "${scripts[$i]}" "script $i" *.fe
  done
fi

# OpenMP: refined five times, Newton steps in the linear and Lagrange models
omp_script='g 5\nr\ng 5\nr\ng 5\nr\ng 5\nr\ng 5\nr\ng 5\nu\nV\nu\ng 5\nhessian\nhessian\nlagrange 2\ng 2\nhessian\nq\n'
# the same in quantity mode (as after hessian_seek with constraint integrals)
omp_quantities='g 5\nr\ng 5\nr\ng 5\nr\ng 5\nr\ng 5\nr\nconvert_to_quantities\ng 5\nu\nV\nu\ng 5\nhessian\nhessian\nq\n'
build_evolver "$build/openmp" "-fopenmp"
cd "$root/fe"
if [ "$quick" = 1 ]; then
  samples=(cube.fe mound.fe)
else
  samples=(cube.fe mound.fe catbody.fe column.fe sphere.fe twointor.fe phelanc.fe)
fi
OMP_NUM_THREADS=8 run "$build/openmp" "$omp_script" "openmp" "${samples[@]}"
OMP_NUM_THREADS=8 run "$build/openmp" "$omp_quantities" "openmp-quantities" "${samples[@]::3}"

echo "$failures sanitizer failure(s)"
[ "$failures" -eq 0 ]
