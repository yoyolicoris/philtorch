#include <Python.h>
#include <torch/script.h>
#include <torch/torch.h>

/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so/.pyd associated with this extension
   built from this file, so that all the TORCH_LIBRARY calls below are run.
   PyMODINIT_FUNC (rather than a bare `extern "C" PyObject *`) is required so
   the symbol is exported from the DLL on Windows (adds __declspec(dllexport)). */
PyMODINIT_FUNC PyInit__C(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C", /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,   /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        NULL, /* methods */
    };
    return PyModule_Create(&module_def);
}

TORCH_LIBRARY(philtorch, m)
{
    m.def("philtorch::recur2(Tensor A, Tensor zi, Tensor x) -> Tensor", {at::Tag::pt2_compliant_tag});
    m.def("philtorch::recurN(Tensor A, Tensor zi, Tensor x) -> Tensor", {at::Tag::pt2_compliant_tag});

    m.def("philtorch::lti_recur2(Tensor A, Tensor zi, Tensor x) -> Tensor", {at::Tag::pt2_compliant_tag});
    m.def("philtorch::lti_recurN(Tensor A, Tensor zi, Tensor x) -> Tensor", {at::Tag::pt2_compliant_tag});

    m.def("philtorch::lti_recur(Tensor A, Tensor zi, Tensor x) -> Tensor", {at::Tag::pt2_compliant_tag});
}
