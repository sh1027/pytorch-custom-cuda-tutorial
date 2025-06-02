#include <torch/extension.h>
#include <cmath>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void linear_forward_kernel(const float* x, float a, float b, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i] = a * x[i] + b;
    }
}

__global__ void linear_backward_kernel(const float* x, const float* grad_out, float* grad_a, float* grad_b, int N) {
    __shared__ float shared_a[BLOCK_SIZE], shared_b[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local_a = 0.0f, local_b = 0.0f;

    if (i < N) {
        local_a = grad_out[i] * x[i];
        local_b = grad_out[i];
    }
    shared_a[threadIdx.x] = local_a;
    shared_b[threadIdx.x] = local_b;
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum_a = 0.0f, sum_b = 0.0f;
        for (int j = 0; j < blockDim.x && (blockIdx.x * blockDim.x + j) < N; ++j) {
            sum_a += shared_a[j];
            sum_b += shared_b[j];
        }
        atomicAdd(grad_a, sum_a);
        atomicAdd(grad_b, sum_b);
    }
}

void linear_forward(torch::Tensor x, float a, float b, torch::Tensor y) {
    int N = x.size(0);
    linear_forward_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        x.data_ptr<float>(), a, b, y.data_ptr<float>(), N
    );
}

void linear_backward(torch::Tensor x, torch::Tensor grad_out, torch::Tensor grad_a, torch::Tensor grad_b) {
    int N = x.size(0);
    linear_backward_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        x.data_ptr<float>(), grad_out.data_ptr<float>(),
        grad_a.data_ptr<float>(), grad_b.data_ptr<float>(), N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "Linear forward");
    m.def("linear_backward", &linear_backward, "Linear backward");
}
