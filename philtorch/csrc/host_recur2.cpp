#include <Python.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <execution>
#include <utility>
#include <vector>

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so associated with this extension
   built from this file, so that all the TORCH_LIBRARY calls below are run.*/
PyObject *PyInit__C(void) {
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
}

template <typename T>
using host_sqm2_pair = std::tuple<T, T, T, T, T, T>;

template <typename T>
host_sqm2_pair<T> recur2_binary_op(const host_sqm2_pair<T> &a,
                                   const host_sqm2_pair<T> &b) {
    auto a_a = std::get<0>(a);
    auto a_b = std::get<1>(a);
    auto a_c = std::get<2>(a);
    auto a_d = std::get<3>(a);
    auto a_e = std::get<4>(a);
    auto a_f = std::get<5>(a);
    auto b_a = std::get<0>(b);
    auto b_b = std::get<1>(b);
    auto b_c = std::get<2>(b);
    auto b_d = std::get<3>(b);
    auto b_e = std::get<4>(b);
    auto b_f = std::get<5>(b);

    return std::make_tuple(b_a * a_a + b_b * a_c, b_a * a_b + b_b * a_d,
                           b_c * a_a + b_d * a_c, b_c * a_b + b_d * a_d,
                           b_a * a_e + b_b * a_f + b_e,
                           b_c * a_e + b_d * a_f + b_f);
}

template <typename scalar_t>
void host_batch_mat_recur_second_order(const scalar_t *A, const scalar_t *x,
                                       scalar_t *out, int n_steps) {
    std::vector<host_sqm2_pair<scalar_t>> buffer(n_steps);

    at::parallel_for(0, n_steps, 1, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
            auto a = A[i];
            auto b = A[i + n_steps];
            auto c = A[i + 2 * n_steps];
            auto d = A[i + 3 * n_steps];
            auto e = x[i];
            auto f = x[i + n_steps];
            buffer[i] = std::make_tuple(a, b, c, d, e, f);
        }
    });

    std::inclusive_scan(std::execution::par, buffer.begin(), buffer.end(),
                        buffer.begin(), recur2_binary_op<scalar_t>);

    at::parallel_for(0, n_steps, 1, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
            auto &result = buffer[i];
            out[i] = std::get<4>(result);
            out[i + n_steps] = std::get<5>(result);
        }
    });
}

template <typename scalar_t>
void host_share_mat_recur_second_order(const scalar_t *A, const scalar_t *x,
                                       scalar_t *out, int n_steps,
                                       int n_batches) {
    auto total_steps = n_steps * n_batches;
    std::vector<host_sqm2_pair<scalar_t>> buffer(total_steps);

    at::parallel_for(0, total_steps, 1, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
            auto step_index = i % n_steps;
            auto a = A[step_index];
            auto b = A[step_index + n_steps];
            auto c = A[step_index + 2 * n_steps];
            auto d = A[step_index + 3 * n_steps];
            auto e = x[i];
            auto f = x[i + total_steps];
            buffer[i] = std::make_tuple(a, b, c, d, e, f);
        }
    });

    std::inclusive_scan(std::execution::par, buffer.begin(), buffer.end(),
                        buffer.begin(), recur2_binary_op<scalar_t>);

    at::parallel_for(0, total_steps, 1, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
            auto &result = buffer[i];
            out[i] = std::get<4>(result);
            out[i + total_steps] = std::get<5>(result);
        }
    });
}

at::Tensor mat_recur_second_order_cpu_impl(const at::Tensor &A,
                                           const at::Tensor &zi,
                                           const at::Tensor &x) {
    TORCH_CHECK(x.is_floating_point() || x.is_complex(),
                "Input must be floating point or complex");
    TORCH_CHECK(zi.scalar_type() == zi.scalar_type(),
                "zi must have the same scalar type as input");
    TORCH_CHECK(A.scalar_type() == A.scalar_type(),
                "A must have the same scalar type as input");
    TORCH_CHECK(A.dim() == 3 || A.dim() == 4, "A must be a 3D or 4D tensor");
    TORCH_CHECK(x.size(2) == 2, "Input x must have a last dimension of size 2");
    TORCH_CHECK(A.size(-1) == 2 && A.size(-2) == 2,
                "Last two dimensions of A must be of size 2");

    auto n_steps = x.size(1) + 1;  // +1 for the initial state
    auto n_batches = x.size(0);

    auto A_contiguous =
        at::pad(A, {0, 0, 0, 0, 1, 0}).view({-1, 4}).t().contiguous();
    auto x_contiguous =
        at::cat({zi.unsqueeze(1), x}, 1).view({-1, 2}).t().contiguous();
    auto out = at::empty_like(x_contiguous);

    if (A.dim() == 4) {
        // Batch
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
            x.scalar_type(), "host_batch_mat_recur_second_order", [&] {
                host_batch_mat_recur_second_order<scalar_t>(
                    A_contiguous.const_data_ptr<scalar_t>(),
                    x_contiguous.const_data_ptr<scalar_t>(),
                    out.mutable_data_ptr<scalar_t>(), n_steps * n_batches);
            });
    } else {
        // Shared
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
            x.scalar_type(), "host_share_mat_recur_second_order", [&] {
                host_share_mat_recur_second_order<scalar_t>(
                    A_contiguous.const_data_ptr<scalar_t>(),
                    x_contiguous.const_data_ptr<scalar_t>(),
                    out.mutable_data_ptr<scalar_t>(), n_steps, n_batches);
            });
    }
    return out.t()
        .reshape({n_batches, n_steps, 2})
        .slice(1, 1, n_steps)
        .contiguous();  // Remove the initial state from the output
}

TORCH_LIBRARY(philtorch, m) {
    m.def("philtorch::recur2(Tensor A, Tensor zi, Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(philtorch, CPU, m) {
    m.impl("recur2", &mat_recur_second_order_cpu_impl);
}