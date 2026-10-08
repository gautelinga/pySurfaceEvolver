#!/usr/bin/env bash
# Build the stock Evolver program with AddressSanitizer and UBSan, run every
# sample datafile in fe/ through a sequence of commands, and fail on any
# sanitizer report. Then the same with OpenMP on a few refined samples, large
# enough for the parallel loops (they start at 4096 elements), with Newton
# steps.
#
# usage: tools/run_sanitizers.sh [build-dir]
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
build=${1:-$(mktemp -d)}
flags="-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined"

# build <dir> <extra flags>
build_evolver() {
  mkdir -p "$1"
  cp "$root"/src/*.c "$root"/src/*.h "$root"/src/Makefile "$1"/
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
    printf "$script" | timeout 600 "$dir/evolver" "$f" >"$log" 2>&1 || true
    if grep -q "ERROR: AddressSanitizer\|runtime error" "$log"; then
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
build_evolver "$build/stock" ""
cd "$root/fe"
for i in "${!scripts[@]}"; do
  run "$build/stock" "${scripts[$i]}" "script $i" *.fe
done

# OpenMP: refined five times, Newton steps in the linear and Lagrange models
omp_script='g 5\nr\ng 5\nr\ng 5\nr\ng 5\nr\ng 5\nr\ng 5\nu\nV\nu\ng 5\nhessian\nhessian\nlagrange 2\ng 2\nhessian\nq\n'
build_evolver "$build/openmp" "-fopenmp"
OMP_NUM_THREADS=8 run "$build/openmp" "$omp_script" "openmp" \
  cube.fe mound.fe catbody.fe column.fe sphere.fe twointor.fe phelanc.fe

echo "$failures sanitizer failure(s)"
[ "$failures" -eq 0 ]
