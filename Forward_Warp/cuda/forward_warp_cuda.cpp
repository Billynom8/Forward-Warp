#include <torch/torch.h> // Should be first for PyTorch C++ extensions
#include <vector>

// Explicitly include the header that is known to define the enum.
// Based on the previous nvcc error log, this was:
// C:/AI2/StereoCrafter/venv/Lib/site-packages/torch/include\ATen/native/GridSamplerUtils.h
// Let's try including ATen/native/GridSampler.h as it's a more public-facing header
// and should pull in GridSamplerUtils.h.
#include <ATen/native/GridSampler.h>

#include "forward_warp.h" // Contains helper macros like CUDA_KERNEL_LOOP

// The CUDA kernel (.cu file) expects at::native::detail::GridSamplerInterpolation.
// We need to ensure the function declarations in this .cpp file match that.

// Declarations for functions defined in forward_warp_cuda_kernel.cu
at::Tensor forward_warp_cuda_forward(
    const at::Tensor im0,
    const at::Tensor flow,
    const at::native::detail::GridSamplerInterpolation interpolation_mode); // Match .cu
std::vector<at::Tensor> forward_warp_cuda_backward(
    const at::Tensor grad_output,
    const at::Tensor im0,
    const at::Tensor flow,
    const at::native::detail::GridSamplerInterpolation interpolation_mode); // Match .cu

// Use TORCH_CHECK for modern PyTorch
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x) // Not used, can be removed

// Wrapper functions exposed to Python
at::Tensor forward_warp_forward(
    const at::Tensor im0,
    const at::Tensor flow,
    const int interpolation_mode_int){ // Python will pass an int (0 for Bilinear, 1 for Nearest)

  // CHECK_INPUT(im0); // Optional: Add checks if desired
  // CHECK_INPUT(flow);

  // Cast the integer from Python to the correct enum type
  // The enum values Bilinear=0, Nearest=1 are assumed.
  return forward_warp_cuda_forward(
      im0,
      flow,
      static_cast<at::native::detail::GridSamplerInterpolation>(interpolation_mode_int)
  );
}

std::vector<at::Tensor> forward_warp_backward(
    const at::Tensor grad_output,
    const at::Tensor im0,
    const at::Tensor flow,
    const int interpolation_mode_int){ // Python will pass an int

  // CHECK_INPUT(grad_output); // Optional
  // CHECK_INPUT(im0);
  // CHECK_INPUT(flow);

  return forward_warp_cuda_backward(
      grad_output,
      im0,
      flow,
      static_cast<at::native::detail::GridSamplerInterpolation>(interpolation_mode_int)
  );
}

// Python bindings
PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m){
  m.def("forward", &forward_warp_forward, "forward warp forward (CUDA)");
  m.def("backward", &forward_warp_backward, "forward warp backward (CUDA)");
}