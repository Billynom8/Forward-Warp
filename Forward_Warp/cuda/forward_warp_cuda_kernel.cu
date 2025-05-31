#include <ATen/ATen.h>
#include <ATen/native/GridSampler.h> 
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_KERNEL_LAUNCH_CHECK

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "forward_warp.h"

// Kernels and launchers use at::native::detail::GridSamplerInterpolation

static __forceinline__ __device__ 
int get_im_index(
    const int b,
    const int c,
    const int h,
    const int w,
    const size_t C,
    const size_t H,
    const size_t W) {
  return b*C*H*W + c*H*W + h*W + w;
}

template <typename scalar_t>
__global__ void forward_warp_cuda_forward_kernel(
    const int total_step,
    const scalar_t* im0,
    const scalar_t* flow,
    scalar_t* im1,
    const int B,
    const int C,
    const int H,
    const int W,
    const at::native::detail::GridSamplerInterpolation interpolation_mode) {
  CUDA_KERNEL_LOOP(index, total_step) {
    const int b = index / (H * W);
    const int h = (index / W) % H; 
    const int w = index % W;

    const scalar_t u = flow[index*2+0];
    const scalar_t v = flow[index*2+1];
    const scalar_t x = (scalar_t)w + u;
    const scalar_t y = (scalar_t)h + v;

    if (interpolation_mode == at::native::detail::GridSamplerInterpolation::Bilinear) {
      const int x_f = static_cast<int>(::floor(x));
      const int y_f = static_cast<int>(::floor(y));
      const int x_c = x_f + 1;
      const int y_c = y_f + 1;

      if(x_f >= 0 && x_c < W && y_f >= 0 && y_c < H) {
        const scalar_t nw_k = (x_c - x) * (y_c - y);
        const scalar_t ne_k = (x - x_f) * (y_c - y);
        const scalar_t sw_k = (x_c - x) * (y - y_f);
        const scalar_t se_k = (x - x_f) * (y - y_f);

        const scalar_t* im0_channel_base = im0 + get_im_index(b, 0, h, w, C, H, W);

        for (int c_idx = 0; c_idx < C; ++c_idx) {
          const scalar_t val_im0 = im0_channel_base[c_idx * H * W]; 

          atomicAdd(im1 + get_im_index(b, c_idx, y_f, x_f, C, H, W), nw_k * val_im0);
          atomicAdd(im1 + get_im_index(b, c_idx, y_f, x_c, C, H, W), ne_k * val_im0);
          atomicAdd(im1 + get_im_index(b, c_idx, y_c, x_f, C, H, W), sw_k * val_im0);
          atomicAdd(im1 + get_im_index(b, c_idx, y_c, x_c, C, H, W), se_k * val_im0);
        }
      }
    } 
    else if (interpolation_mode == at::native::detail::GridSamplerInterpolation::Nearest) {
      const int x_nearest = static_cast<int>(::round(x));
      const int y_nearest = static_cast<int>(::round(y));

      if(x_nearest >= 0 && x_nearest < W && y_nearest >= 0 && y_nearest < H) {
        const scalar_t* im0_p = im0 + get_im_index(b, 0, h, w, C, H, W);
        scalar_t* im1_p = im1 + get_im_index(b, 0, y_nearest, x_nearest, C, H, W);
        for (int c_idx = 0; c_idx < C; ++c_idx) {
          im1_p[c_idx * H * W] = im0_p[c_idx * H * W]; 
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void forward_warp_cuda_backward_kernel(
    const int total_step,
    const scalar_t* grad_output,
    const scalar_t* im0,
    const scalar_t* flow,
    scalar_t* im0_grad,
    scalar_t* flow_grad,
    const int B,
    const int C,
    const int H,
    const int W,
    const at::native::detail::GridSamplerInterpolation interpolation_mode) {
  CUDA_KERNEL_LOOP(index, total_step) {
    const int b = index / (H * W);
    const int h = (index / W) % H; 
    const int w = index % W;

    const scalar_t u = flow[index*2+0];
    const scalar_t v = flow[index*2+1];
    const scalar_t x = (scalar_t)w + u;
    const scalar_t y = (scalar_t)h + v;

    scalar_t flow_grad_x_acc = 0;
    scalar_t flow_grad_y_acc = 0;

    if (interpolation_mode == at::native::detail::GridSamplerInterpolation::Bilinear) {
      const int x_f = static_cast<int>(::floor(x));
      const int y_f = static_cast<int>(::floor(y));
      const int x_c = x_f + 1;
      const int y_c = y_f + 1;

      if(x_f >= 0 && x_c < W && y_f >= 0 && y_c < H) {
        const scalar_t nw_k = (x_c - x) * (y_c - y); 
        const scalar_t ne_k = (x - x_f) * (y_c - y); 
        const scalar_t sw_k = (x_c - x) * (y - y_f); 
        const scalar_t se_k = (x - x_f) * (y - y_f); 

        for (int c_idx = 0; c_idx < C; ++c_idx) {
          const scalar_t p_val = im0[get_im_index(b, c_idx, h, w, C, H, W)]; 

          const scalar_t grad_val_nw = grad_output[get_im_index(b, c_idx, y_f, x_f, C, H, W)];
          const scalar_t grad_val_ne = grad_output[get_im_index(b, c_idx, y_f, x_c, C, H, W)];
          const scalar_t grad_val_sw = grad_output[get_im_index(b, c_idx, y_c, x_f, C, H, W)];
          const scalar_t grad_val_se = grad_output[get_im_index(b, c_idx, y_c, x_c, C, H, W)];
          
          scalar_t current_im0_grad_val = 0;
          current_im0_grad_val += nw_k * grad_val_nw;
          current_im0_grad_val += ne_k * grad_val_ne;
          current_im0_grad_val += sw_k * grad_val_sw;
          current_im0_grad_val += se_k * grad_val_se;
          atomicAdd(im0_grad + get_im_index(b, c_idx, h, w, C, H, W) , current_im0_grad_val);

          flow_grad_x_acc += p_val * ( -(y_c-y)*grad_val_nw + (y_c-y)*grad_val_ne - (y-y_f)*grad_val_sw + (y-y_f)*grad_val_se );
          flow_grad_y_acc += p_val * ( -(x_c-x)*grad_val_nw - (x-x_f)*grad_val_ne + (x_c-x)*grad_val_sw + (x-x_f)*grad_val_se );
        }
      }
    } 
    else if (interpolation_mode == at::native::detail::GridSamplerInterpolation::Nearest) {
      const int x_nearest = static_cast<int>(::round(x));
      const int y_nearest = static_cast<int>(::round(y));

      if(x_nearest >= 0 && x_nearest < W && y_nearest >= 0 && y_nearest < H) {
        for (int c_idx = 0; c_idx < C; ++c_idx) {
          im0_grad[get_im_index(b, c_idx, h, w, C, H, W)] = grad_output[get_im_index(b, c_idx, y_nearest, x_nearest, C, H, W)];
        }
      }
    }
    flow_grad[index*2+0] = flow_grad_x_acc;
    flow_grad[index*2+1] = flow_grad_y_acc;
  }
}

at::Tensor forward_warp_cuda_forward(
    const at::Tensor im0, 
    const at::Tensor flow,
    const at::native::detail::GridSamplerInterpolation interpolation_mode) {
  TORCH_CHECK(im0.is_cuda(), "im0 must be a CUDA tensor");
  TORCH_CHECK(flow.is_cuda(), "flow must be a CUDA tensor");
  TORCH_CHECK(im0.scalar_type() == flow.scalar_type(), "im0 and flow must have the same scalar type");
  TORCH_CHECK(im0.dim() == 4, "im0 must be a 4D tensor (B, C, H, W)");
  TORCH_CHECK(flow.dim() == 4, "flow must be a 4D tensor (B, H, W, 2)");
  TORCH_CHECK(im0.size(0) == flow.size(0), "Batch size of im0 and flow must match");
  TORCH_CHECK(im0.size(2) == flow.size(1), "Height of im0 and flow must match");
  TORCH_CHECK(im0.size(3) == flow.size(2), "Width of im0 and flow must match");
  TORCH_CHECK(flow.size(3) == 2, "flow channels (dim 3) must be 2 for (u,v)");

  auto im1 = at::zeros_like(im0);
  const int B = im0.size(0);
  const int C = im0.size(1);
  const int H = im0.size(2);
  const int W = im0.size(3);
  const int total_step = B * H * W;

  if (total_step == 0) {
    return im1;
  }

  AT_DISPATCH_FLOATING_TYPES(im0.scalar_type(), "forward_warp_forward_cuda", ([&] {
    forward_warp_cuda_forward_kernel<scalar_t>
    <<<GET_BLOCKS(total_step), CUDA_NUM_THREADS>>>(
      total_step,
      im0.const_data_ptr<scalar_t>(),
      flow.const_data_ptr<scalar_t>(),
      im1.data_ptr<scalar_t>(),
      B, C, H, W,
      interpolation_mode);
  }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return im1;
}

std::vector<at::Tensor> forward_warp_cuda_backward(
    const at::Tensor grad_output,
    const at::Tensor im0, 
    const at::Tensor flow,
    const at::native::detail::GridSamplerInterpolation interpolation_mode) {
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
  TORCH_CHECK(im0.is_cuda(), "im0 must be a CUDA tensor");
  TORCH_CHECK(flow.is_cuda(), "flow must be a CUDA tensor");
  TORCH_CHECK(grad_output.scalar_type() == im0.scalar_type() && im0.scalar_type() == flow.scalar_type(), 
              "grad_output, im0, and flow must have the same scalar type");

  auto im0_grad = at::zeros_like(im0);
  auto flow_grad = at::zeros_like(flow); 

  const int B = im0.size(0);
  const int C = im0.size(1);
  const int H = im0.size(2);
  const int W = im0.size(3);
  const int total_step = B * H * W;

  if (total_step == 0) {
    return {im0_grad, flow_grad};
  }

  AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "forward_warp_backward_cuda", ([&] {
    forward_warp_cuda_backward_kernel<scalar_t>
    <<<GET_BLOCKS(total_step), CUDA_NUM_THREADS>>>(
      total_step,
      grad_output.const_data_ptr<scalar_t>(),
      im0.const_data_ptr<scalar_t>(),
      flow.const_data_ptr<scalar_t>(),
      im0_grad.data_ptr<scalar_t>(),
      flow_grad.data_ptr<scalar_t>(),
      B, C, H, W,
      interpolation_mode);
  }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {im0_grad, flow_grad};
}