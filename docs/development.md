# Development

## Building and testing

```bash
pip install ".[test]"
pytest
python bench/benchmark.py --levels 6 8 --threads 1 4   # linear at 98k and 1.6M facets,
                                                       # Lagrange 2/4/6 at 6k and 24k
pip install ".[docs]"
sphinx-build -W docs docs/_build/html   # API reference + tutorial (docs/tutorial.ipynb,
                                       # executed during the build)
```

CI (`.github/workflows/`) runs the tests and mypy on several Python versions, runs
every sample datafile through the stock program under AddressSanitizer and UBSan,
plus refined samples with Newton steps through an OpenMP build (`tools/run_sanitizers.sh`), builds the docs (published to GitHub Pages from `main`), and builds manylinux wheels
(x86_64, aarch64) plus an sdist as workflow artifacts. The build uses link-time optimization (`PYSE_LTO`);
`-C cmake.define.PYSE_NOSTRIP=ON -C install.strip=false` keeps symbols for profiling.
The version lives in `python/pysurfaceevolver/__init__.py` only.

## Layout

- `src/` – the Surface Evolver C sources, maintained as a fork of 2.70a (the first git
  commit is the untouched original). Bug fixes apply to every build, and the stock
  `Makefile` still builds the standalone program (`GRAPH=nulgraph.o` for headless);
  hooks for the Python library are guarded by `#ifdef PYSE`.
- `bindings/` – C glue (`pyse_api.c`) that contains Evolver's `setjmp`/`longjmp`
  error handling, and the nanobind module (`module.cpp`).
- `python/pysurfaceevolver/` – the Python API; type stubs for the extension are
  generated at build time.
- `tests/` – the pytest suite.
- `fe/`, `doc/`, `manual270.pdf` – Evolver's sample datafiles and manual.
