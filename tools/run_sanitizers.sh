#!/usr/bin/env bash
# Build the stock Evolver program with AddressSanitizer and UBSan, run every
# sample datafile in fe/ through a sequence of commands, and fail on any
# sanitizer report.
#
# usage: tools/run_sanitizers.sh [build-dir]
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
build=${1:-$(mktemp -d)}
mkdir -p "$build"
cp "$root"/src/*.c "$root"/src/*.h "$root"/src/Makefile "$build"/
flags="-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined"
make -C "$build" -j"$(nproc)" CC="gcc $flags" CFLAGS="-DLINUX -w" GRAPH=nulgraph.o GRAPHLIB= evolver >"$build/build.log" 2>&1 \
  || { tail -20 "$build/build.log"; exit 1; }

# Each script exercises iteration, refinement, all element models, vertex
# averaging, equiangulation, weeding, listing and dumping; one adds Newton.
scripts=(
  'g 5\nr\ng 5\nquadratic\ng 3\nlagrange 3\ng 3\nlinear\nV\nu\nw 0.001\ng 3\nlist vertices\ndump "/dev/null"\nq\n'
  'g 5\nr\ng 5\nhessian\nquadratic\ng 3\nlagrange 3\ng 3\nlinear\nV\nu\ng 3\nlist vertices\ndump "/dev/null"\nq\n'
)
export ASAN_OPTIONS=detect_leaks=0
failures=0
cd "$root/fe"
for f in *.fe; do
  for i in "${!scripts[@]}"; do
    log="$build/$f.$i.log"
    printf "${scripts[$i]}" | timeout 600 "$build/evolver" "$f" >"$log" 2>&1 || true
    if grep -q "ERROR: AddressSanitizer\|runtime error" "$log"; then
      echo "FAIL $f (script $i):"
      grep -A12 "ERROR: AddressSanitizer\|runtime error" "$log" | head -16
      failures=$((failures + 1))
    else
      echo "ok   $f (script $i)"
    fi
  done
done
echo "$failures sanitizer failure(s)"
[ "$failures" -eq 0 ]
