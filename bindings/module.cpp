// module.cpp -- nanobind extension pysurfaceevolver._core.
//
// A thin layer over pyse_api.h.  Every guarded call takes its output and
// input callbacks as arguments and returns a CallResult holding everything
// about that call (status, value, error, warnings, data), all collected
// while one mutex is held.  So concurrent calls from several threads can't
// see or overwrite each other's callbacks or results.  The GIL is released
// while Evolver runs and re-acquired only to call the Python callbacks.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pyse_api.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Evolver is one global state machine: serialize every access.  try_lock
// makes a re-entrant call from a callback fail instead of deadlocking.
std::mutex evolver_mutex;

struct Lock {
  std::unique_lock<std::mutex> lock{evolver_mutex, std::try_to_lock};
  Lock() {
    if (!lock.owns_lock())
      throw std::runtime_error(
          "Surface Evolver is busy: calls are neither re-entrant nor "
          "thread-safe");
  }
};

// Callbacks of the call in progress.  Only touched while evolver_mutex is
// held.
nb::object output_callback;  // callable(stream: int, text: str) or None
nb::object input_callback;   // callable(prompt: str) -> str | None, or None

void output_trampoline(int stream, const char *text, void *) {
  nb::gil_scoped_acquire gil;
  if (!output_callback.is_valid() || output_callback.is_none()) return;
  try {
    output_callback(stream, nb::str(text));
  } catch (nb::python_error &e) {
    e.discard_as_unraisable("pysurfaceevolver output callback");
  }
}

int input_trampoline(const char *prompt, char *buf, int max, void *) {
  nb::gil_scoped_acquire gil;
  if (!input_callback.is_valid() || input_callback.is_none()) return -1;
  try {
    nb::object reply = input_callback(nb::str(prompt));
    if (reply.is_none()) return 0;  // EOF
    std::string line = nb::cast<std::string>(reply);
    std::strncpy(buf, line.c_str(), max - 2);
    buf[max - 2] = 0;
    return 1;
  } catch (nb::python_error &e) {
    e.discard_as_unraisable("pysurfaceevolver input callback");
    return 0;
  }
}

struct CallResult {
  int status = PYSE_OK;
  double value = NAN;
  int errnum = 0;
  std::string message;
  std::vector<std::string> warnings;
  int exit_code = 0;
  nb::object data = nb::none();
};

// Holds the lock and installs the callbacks for one call.
struct CallScope {
  Lock lock;
  CallScope(nb::object out, nb::object in, bool sigint) {
    output_callback = std::move(out);
    input_callback = std::move(in);
    pyse_set_handle_sigint(sigint);
  }
  ~CallScope() {
    output_callback = nb::object();
    input_callback = nb::object();
    pyse_set_handle_sigint(0);
  }
};

// Run a guarded C call with the GIL released, and collect its results
// while the lock is still held.
template <typename F>
CallResult run_guarded(F &&f) {
  int status;
  {
    nb::gil_scoped_release release;
    status = f();
  }
  CallResult r;
  r.status = status;
  r.errnum = pyse_last_errnum();
  r.message = pyse_last_errmsg();
  r.exit_code = pyse_exit_code();
  for (int i = 0; i < pyse_warning_count(); i++) r.warnings.emplace_back(pyse_warning(i));
  return r;
}

// Hand a heap buffer to NumPy without copying.
template <typename T>
nb::object to_numpy(std::vector<T> &&data, std::initializer_list<size_t> shape) {
  auto *heap = new std::vector<T>(std::move(data));
  nb::capsule owner(heap, [](void *p) noexcept {
    delete static_cast<std::vector<T> *>(p);
  });
  return nb::cast(nb::ndarray<nb::numpy, T>(heap->data(), shape, owner));
}

}  // namespace

// Every guarded function takes the call's callbacks as its last arguments.
#define CALLBACK_ARGS nb::object out, nb::object in, bool sigint
#define CALL_SCOPE CallScope scope(std::move(out), std::move(in), sigint)
#define CALLBACK_NAMES "out"_a.none(), "input"_a.none(), "sigint"_a

NB_MODULE(_core, m) {
  m.doc() = "Low-level bindings to Surface Evolver (use pysurfaceevolver.Evolver)";

  m.attr("OK") = int(PYSE_OK);
  m.attr("ERROR") = int(PYSE_ERROR);
  m.attr("EXIT") = int(PYSE_EXIT);
  m.attr("INTERRUPT") = int(PYSE_INTERRUPT);
  m.attr("FATAL") = int(PYSE_FATAL);
  m.attr("BUSY") = int(PYSE_BUSY);
  m.attr("INVALID") = int(PYSE_INVALID);
  m.attr("ERR_NO_DATAFILE") = int(PYSE_ERR_NO_DATAFILE);
  m.attr("ERR_INVALID_SURFACE") = int(PYSE_ERR_INVALID_SURFACE);
  m.attr("ERR_BAD_VALUE") = int(PYSE_ERR_BAD_VALUE);
  m.attr("ERR_BAD_ARGUMENT") = int(PYSE_ERR_BAD_ARGUMENT);
  m.attr("VERTEX") = int(PYSE_VERTEX);
  m.attr("EDGE") = int(PYSE_EDGE);
  m.attr("FACET") = int(PYSE_FACET);
  m.attr("BODY") = int(PYSE_BODY);

  pyse_set_output_callback(output_trampoline, nullptr);
  pyse_set_input_callback(input_trampoline, nullptr);

  nb::class_<CallResult>(m, "CallResult",
                         "Everything about one guarded call into Evolver.")
      .def_ro("status", &CallResult::status)
      .def_ro("value", &CallResult::value)
      .def_ro("errnum", &CallResult::errnum)
      .def_ro("message", &CallResult::message)
      .def_ro("warnings", &CallResult::warnings)
      .def_ro("exit_code", &CallResult::exit_code)
      .def_ro("data", &CallResult::data);

  // ---- running Evolver -----------------------------------------------------
  m.def("initialize", [](CALLBACK_ARGS) {
    CALL_SCOPE;
    return run_guarded([] { return pyse_initialize(); });
  }, CALLBACK_NAMES);

  m.def("load", [](const std::string &path, CALLBACK_ARGS) {
    CALL_SCOPE;
    return run_guarded([&] { return pyse_load(path.c_str()); });
  }, "path"_a, CALLBACK_NAMES);

  m.def("command", [](const std::string &text, CALLBACK_ARGS) {
    CALL_SCOPE;
    return run_guarded([&] { return pyse_command(text.c_str()); });
  }, "text"_a, CALLBACK_NAMES);

  m.def("eval", [](const std::string &expr, CALLBACK_ARGS) {
    CALL_SCOPE;
    double value = NAN;
    CallResult r = run_guarded([&] { return pyse_eval(expr.c_str(), &value); });
    r.value = value;
    return r;
  }, "expr"_a, CALLBACK_NAMES);

  m.def("values", [](int type, const std::string &expr, CALLBACK_ARGS) {
    CALL_SCOPE;
    long n = pyse_count(type);
    std::vector<double> result(n > 0 ? size_t(n) : 0);
    CallResult r = run_guarded([&] { return pyse_values(type, expr.c_str(), result.data(), n); });
    if (r.status == PYSE_OK) {
      size_t size = result.size();
      r.data = to_numpy(std::move(result), {size});
    }
    return r;
  }, "type"_a, "expr"_a, CALLBACK_NAMES);

  m.def("set_vertex_coords",
        [](nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu> xyz,
           CALLBACK_ARGS) {
          CALL_SCOPE;
          return run_guarded([&] {
            return pyse_set_vertex_coords(xyz.data(), long(xyz.shape(0)),
                                          int(xyz.shape(1)));
          });
        }, "xyz"_a, CALLBACK_NAMES);

  m.def("fast_attribute", [](int type, const std::string &attribute) {
    Lock lock;
    return bool(pyse_fast_attribute(type, attribute.c_str()));
  }, "type"_a, "attribute"_a);

  m.def("set_values",
        [](int type, const std::string &attribute,
           nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> values,
           std::optional<nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>> mask,
           CALLBACK_ARGS) {
          CALL_SCOPE;
          const uint8_t *m = mask ? mask->data() : nullptr;
          return run_guarded([&] {
            return pyse_set_values(type, attribute.c_str(), values.data(), m,
                                   long(values.shape(0)));
          });
        }, "type"_a, "attribute"_a, "values"_a, "mask"_a.none(), CALLBACK_NAMES);

  // ---- surface snapshots (guarded) -------------------------------------------
  m.def("vertices", [](CALLBACK_ARGS) {
    CALL_SCOPE;
    long n = pyse_count(PYSE_VERTEX);
    int sdim = pyse_sdim();
    std::vector<double> xyz(size_t(n) * sdim);
    std::vector<int64_t> ids(n);
    std::vector<uint8_t> fixed(n);
    CallResult r = run_guarded([&] {
      return pyse_get_vertices(xyz.data(), ids.data(), fixed.data(), n, sdim);
    });
    if (r.status == PYSE_OK)
      r.data = nb::make_tuple(to_numpy(std::move(xyz), {size_t(n), size_t(sdim)}),
                              to_numpy(std::move(ids), {size_t(n)}),
                              to_numpy(std::move(fixed), {size_t(n)}));
    return r;
  }, CALLBACK_NAMES, "data: (coords (n, sdim), ids (n,), fixed (n,) uint8)");

  m.def("edges", [](CALLBACK_ARGS) {
    CALL_SCOPE;
    long n = pyse_count(PYSE_EDGE);
    std::vector<int64_t> verts(size_t(n) * 2), ids(n);
    CallResult r = run_guarded([&] { return pyse_get_edges(verts.data(), ids.data(), n); });
    if (r.status == PYSE_OK)
      r.data = nb::make_tuple(to_numpy(std::move(verts), {size_t(n), 2}),
                              to_numpy(std::move(ids), {size_t(n)}));
    return r;
  }, CALLBACK_NAMES, "data: (vertex rows (m, 2), ids (m,))");

  m.def("facets", [](CALLBACK_ARGS) {
    CALL_SCOPE;
    long n = pyse_count(PYSE_FACET);
    std::vector<int64_t> verts(size_t(n) * 3), ids(n), bodies(size_t(n) * 2);
    CallResult r = run_guarded([&] {
      return pyse_get_facets(verts.data(), ids.data(), bodies.data(), n);
    });
    if (r.status == PYSE_OK)
      r.data = nb::make_tuple(to_numpy(std::move(verts), {size_t(n), 3}),
                              to_numpy(std::move(ids), {size_t(n)}),
                              to_numpy(std::move(bodies), {size_t(n), 2}));
    return r;
  }, CALLBACK_NAMES, "data: (vertex rows (k, 3), ids (k,), bodies (k, 2))");

  m.def("bodies", [](CALLBACK_ARGS) {
    CALL_SCOPE;
    long n = pyse_count(PYSE_BODY);
    std::vector<int64_t> ids(n);
    std::vector<double> volume(n), target(n), pressure(n);
    std::vector<uint8_t> fixed(n);
    CallResult r = run_guarded([&] {
      return pyse_get_bodies(ids.data(), volume.data(), target.data(),
                             pressure.data(), fixed.data(), n);
    });
    if (r.status == PYSE_OK)
      r.data = nb::make_tuple(to_numpy(std::move(ids), {size_t(n)}),
                              to_numpy(std::move(volume), {size_t(n)}),
                              to_numpy(std::move(target), {size_t(n)}),
                              to_numpy(std::move(pressure), {size_t(n)}),
                              to_numpy(std::move(fixed), {size_t(n)}));
    return r;
  }, CALLBACK_NAMES, "data: (ids, volume, target (NaN if free), pressure, fixed)");

  m.def("mesh", [](CALLBACK_ARGS) {
    CALL_SCOPE;
    pyse_mesh_arrays a{};
    long nv = pyse_count(PYSE_VERTEX), ne = pyse_count(PYSE_EDGE), nf = pyse_count(PYSE_FACET);
    int sdim = pyse_sdim();
    bool soapfilm = pyse_representation() == 2;
    int edge_per = pyse_element_node_count(PYSE_EDGE);
    int facet_per = soapfilm ? pyse_element_node_count(PYSE_FACET) : -1;
    std::vector<double> xyz(size_t(nv) * sdim);
    std::vector<int64_t> vids(nv), edges(size_t(ne) * 2), eids(ne);
    std::vector<uint8_t> fixed(nv);
    std::vector<int64_t> faces, fids, fbodies, enodes, fnodes;
    a.xyz = xyz.data(); a.vertex_ids = vids.data(); a.fixed = fixed.data();
    a.nv = nv; a.sdim = sdim;
    a.edges = edges.data(); a.edge_ids = eids.data(); a.ne = ne;
    if (soapfilm) {
      faces.resize(size_t(nf) * 3); fids.resize(nf); fbodies.resize(size_t(nf) * 2);
      a.faces = faces.data(); a.face_ids = fids.data(); a.face_bodies = fbodies.data();
      a.nf = nf;
    }
    if (edge_per > 0) {
      enodes.resize(size_t(ne) * edge_per);
      a.edge_nodes = enodes.data(); a.edge_nodes_per = edge_per;
    }
    if (facet_per > 0) {
      fnodes.resize(size_t(nf) * facet_per);
      a.facet_nodes = fnodes.data(); a.facet_nodes_per = facet_per;
    }
    CallResult r = run_guarded([&] { return pyse_get_mesh(&a); });
    if (r.status != PYSE_OK) return r;
    auto layout = [](int type, int per) -> nb::object {
      if (per <= 0) return nb::none();
      int dim = (type == PYSE_EDGE) ? 1 : 2;
      std::vector<int32_t> index(size_t(per) * (dim + 1));
      pyse_node_layout(type, index.data(), per);
      return to_numpy(std::move(index), {size_t(per), size_t(dim + 1)});
    };
    nb::object none = nb::none();
    r.data = nb::make_tuple(
        to_numpy(std::move(xyz), {size_t(nv), size_t(sdim)}),
        to_numpy(std::move(vids), {size_t(nv)}), to_numpy(std::move(fixed), {size_t(nv)}),
        to_numpy(std::move(edges), {size_t(ne), 2}), to_numpy(std::move(eids), {size_t(ne)}),
        soapfilm ? to_numpy(std::move(faces), {size_t(nf), 3}) : none,
        soapfilm ? to_numpy(std::move(fids), {size_t(nf)}) : none,
        soapfilm ? to_numpy(std::move(fbodies), {size_t(nf), 2}) : none,
        edge_per > 0 ? to_numpy(std::move(enodes), {size_t(ne), size_t(edge_per)}) : none,
        layout(PYSE_EDGE, edge_per),
        facet_per > 0 ? to_numpy(std::move(fnodes), {size_t(nf), size_t(facet_per)}) : none,
        layout(PYSE_FACET, facet_per),
        pyse_element_order(), bool(pyse_bezier()));
    return r;
  }, CALLBACK_NAMES, "data: everything Mesh needs, in one call (see _evolver.py)");

  m.def("element_nodes", [](int type, CALLBACK_ARGS) {
    CALL_SCOPE;
    int per = pyse_element_node_count(type);
    if (per < 0) return CallResult{};  // no node layout: data is None
    int dim = (type == PYSE_EDGE) ? 1 : 2;
    long n = pyse_count(type);
    std::vector<int64_t> nodes(size_t(n) * per);
    std::vector<int32_t> layout(size_t(per) * (dim + 1));
    pyse_node_layout(type, layout.data(), per);
    int order = pyse_element_order();
    bool bezier = pyse_bezier();
    CallResult r = run_guarded([&] {
      return pyse_get_element_nodes(type, nodes.data(), n, per);
    });
    if (r.status == PYSE_OK)
      r.data = nb::make_tuple(to_numpy(std::move(nodes), {size_t(n), size_t(per)}),
                              to_numpy(std::move(layout), {size_t(per), size_t(dim + 1)}),
                              order, bezier);
    return r;
  }, "type"_a, CALLBACK_NAMES,
     "data: (vertex rows (n, nodes), barycentric index (nodes, dim+1), order, "
     "bezier), or None if the model has no node layout for this element type");

  m.def("parameters", [](CALLBACK_ARGS) {
    CALL_SCOPE;
    long n = pyse_parameter_count();
    std::vector<char> names(size_t(n) * PYSE_NAME_SIZE);
    std::vector<double> values(n);
    std::vector<uint8_t> optimizing(n);
    CallResult r = run_guarded([&] {
      return pyse_get_parameters(reinterpret_cast<char(*)[PYSE_NAME_SIZE]>(names.data()),
                                 values.data(), optimizing.data(), n);
    });
    if (r.status == PYSE_OK) {
      nb::list out;
      for (long i = 0; i < n; i++)
        out.append(nb::make_tuple(std::string(&names[size_t(i) * PYSE_NAME_SIZE]),
                                  values[i], bool(optimizing[i])));
      r.data = out;
    }
    return r;
  }, CALLBACK_NAMES, "data: [(name, value, optimizing)] for datafile parameters");

  m.def("quantities", [](CALLBACK_ARGS) {
    CALL_SCOPE;
    long n = pyse_quantity_count();
    std::vector<char> names(size_t(n) * PYSE_NAME_SIZE);
    std::vector<double> value(n), target(n), modulus(n), pressure(n);
    std::vector<int> kind(n);
    CallResult r = run_guarded([&] {
      return pyse_get_quantities(reinterpret_cast<char(*)[PYSE_NAME_SIZE]>(names.data()),
                                 value.data(), target.data(), modulus.data(),
                                 pressure.data(), kind.data(), n);
    });
    if (r.status == PYSE_OK) {
      nb::list out;
      for (long i = 0; i < n; i++)
        out.append(nb::make_tuple(std::string(&names[size_t(i) * PYSE_NAME_SIZE]),
                                  value[i], target[i], modulus[i], pressure[i], kind[i]));
      r.data = out;
    }
    return r;
  }, CALLBACK_NAMES,
     "data: [(name, value, target, modulus, pressure, kind)]; "
     "kind 0 energy, 1 fixed, 2 info, 3 conserved");

  // ---- unguarded state: plain field reads -----------------------------------
  m.def("is_initialized", []() { Lock lock; return bool(pyse_is_initialized()); });
  m.def("surface_valid", []() { Lock lock; return bool(pyse_surface_valid()); });
  m.def("surface_version", []() { Lock lock; return pyse_surface_version(); });
  m.def("count", [](int type) { Lock lock; return pyse_count(type); }, "type"_a);
  m.def("sdim", []() { Lock lock; return pyse_sdim(); });
  m.def("representation", []() { Lock lock; return pyse_representation(); });
  m.def("modeltype", []() { Lock lock; return pyse_modeltype(); });
  m.def("lagrange_order", []() { Lock lock; return pyse_lagrange_order(); });
  m.def("torus", []() { Lock lock; return bool(pyse_torus()); });
  m.def("total_energy", []() { Lock lock; return pyse_total_energy(); });
  m.def("total_area", []() { Lock lock; return pyse_total_area(); });
  m.def("datafile", []() { Lock lock; return std::string(pyse_datafilename()); });
  m.def("set_datafile", [](const std::string &name) {
    Lock lock;
    pyse_set_datafilename(name.c_str());
  }, "name"_a);
}
