#include <Python.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <utility>
#include <vector>

extern "C"
{
    /* Creates a dummy empty _C module that can be imported from Python.
       The import from Python will load the .so associated with this extension
       built from this file, so that all the TORCH_LIBRARY calls below are run.*/
    PyObject *PyInit__C(void)
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
}

template <typename T>
using host_sqm2_pair = std::array<T, 6>;

template <typename T>
host_sqm2_pair<T> recur2_binary_op(const host_sqm2_pair<T> &a,
                                   const host_sqm2_pair<T> &b)
{
    auto [a_a, a_b, a_c, a_d, a_e, a_f] = a;
    auto [b_a, b_b, b_c, b_d, b_e, b_f] = b;

    return {b_a * a_a + b_b * a_c, b_a * a_b + b_b * a_d,
            b_c * a_a + b_d * a_c, b_c * a_b + b_d * a_d,
            b_a * a_e + b_b * a_f + b_e, b_c * a_e + b_d * a_f + b_f};
}

template <typename scalar_t>
void host_batch_mat_recur_second_order(const scalar_t *A, const scalar_t *x,
                                       scalar_t *out, int n_steps)
{
    std::vector<host_sqm2_pair<scalar_t>> buffer(n_steps);

    at::parallel_for(0, n_steps, 1, [&](int64_t start, int64_t end)
                     {
        for (auto i = start; i < end; i++) {
            auto offset = i * 2;
            auto e = x[offset++];
            auto f = x[offset];
            offset = i * 4;
            auto a = A[offset++];
            auto b = A[offset++];
            auto c = A[offset++];
            auto d = A[offset];
            buffer[i] = {a, b, c, d, e, f};
        } });

    std::inclusive_scan(buffer.begin(), buffer.end(), buffer.begin(),
                        recur2_binary_op<scalar_t>);

    at::parallel_for(0, n_steps, 1, [&](int64_t start, int64_t end)
                     {
        for (auto i = start; i < end; i++) {
            auto &result = buffer[i];
            auto offset = i * 2;
            out[offset++] = std::get<4>(result);
            out[offset] = std::get<5>(result);
        } });
}

template <typename scalar_t>
void host_share_mat_recur_second_order(const scalar_t *A, const scalar_t *x,
                                       scalar_t *out, int n_steps,
                                       int n_batches)
{
    auto total_steps = n_steps * n_batches;
    std::vector<host_sqm2_pair<scalar_t>> buffer(total_steps);

    at::parallel_for(0, total_steps, 1, [&](int64_t start, int64_t end)
                     {
        for (auto i = start; i < end; i++) {
            auto offset = i % n_steps * 4;
            auto a = A[offset++];
            auto b = A[offset++];
            auto c = A[offset++];
            auto d = A[offset];
            offset = i * 2;
            auto e = x[offset++];
            auto f = x[offset];
            buffer[i] = {a, b, c, d, e, f};
        } });

    std::inclusive_scan(buffer.begin(), buffer.end(), buffer.begin(),
                        recur2_binary_op<scalar_t>);

    at::parallel_for(0, total_steps, 1, [&](int64_t start, int64_t end)
                     {
        for (auto i = start; i < end; i++) {
            auto &result = buffer[i];
            auto offset = i * 2;
            out[offset++] = std::get<4>(result);
            out[offset] = std::get<5>(result);
        } });
}

at::Tensor mat_recur_second_order_cpu_impl(const at::Tensor &A,
                                           const at::Tensor &zi,
                                           const at::Tensor &x)
{
    TORCH_CHECK(zi.scalar_type() == x.scalar_type(),
                "zi must have the same scalar type as input");
    TORCH_CHECK(A.scalar_type() == x.scalar_type(),
                "A must have the same scalar type as input");
    TORCH_CHECK(A.dim() == 3 || A.dim() == 4, "A must be a 3D or 4D tensor");
    TORCH_CHECK(x.size(2) == 2, "Input x must have a last dimension of size 2");
    TORCH_CHECK(A.size(-1) == 2 && A.size(-2) == 2,
                "Last two dimensions of A must be of size 2");

    auto n_steps = x.size(1) + 1; // +1 for the initial state
    auto n_batches = x.size(0);

    auto A_contiguous = at::pad(A, {0, 0, 0, 0, 1, 0}).contiguous();
    auto x_contiguous = at::cat({zi.unsqueeze(1), x}, 1).contiguous();
    auto out = at::empty_like(x_contiguous);

    if (A.dim() == 4)
    {
        // Batch
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
            at::kLong, x.scalar_type(), "host_batch_mat_recur_second_order",
            [&]
            {
                host_batch_mat_recur_second_order<scalar_t>(
                    A_contiguous.const_data_ptr<scalar_t>(),
                    x_contiguous.const_data_ptr<scalar_t>(),
                    out.mutable_data_ptr<scalar_t>(), n_steps * n_batches);
            });
    }
    else
    {
        // Shared
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
            at::kLong, x.scalar_type(), "host_share_mat_recur_second_order",
            [&]
            {
                host_share_mat_recur_second_order<scalar_t>(
                    A_contiguous.const_data_ptr<scalar_t>(),
                    x_contiguous.const_data_ptr<scalar_t>(),
                    out.mutable_data_ptr<scalar_t>(), n_steps, n_batches);
            });
    }
    return out.reshape({n_batches, n_steps, 2})
        .slice(1, 1, n_steps)
        .contiguous(); // Remove the initial state from the output
}

TORCH_LIBRARY(philtorch, m)
{
    m.def("philtorch::recur2(Tensor A, Tensor zi, Tensor x) -> Tensor");
    m.def("philtorch::recurN(Tensor A, Tensor zi, Tensor x) -> Tensor");

    m.def("philtorch::lti_recur2(Tensor A, Tensor zi, Tensor x) -> Tensor");
    m.def("philtorch::lti_recurN(Tensor A, Tensor zi, Tensor x) -> Tensor");

    m.def("philtorch::lti_recur(Tensor A, Tensor zi, Tensor x) -> Tensor");
}
