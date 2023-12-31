#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

void check_inputs(const Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(
      input1.dim() <= 4 && input2.dim() <= 4,
      "Vulkan only supports tensors <= 4 dimensions");

  // check if the shapes of input tensors are broadcastable
  // see https://pytorch.org/docs/stable/notes/broadcasting.html
  // for broadcasting semantics
  const std::string broadcast_error_msg =
      "The shapes of input and mask are not broadcastable!";
  TORCH_CHECK(
      get_dim<Dim4D::Batch>(input1) == get_dim<Dim4D::Batch>(input2) ||
          get_dim<Dim4D::Batch>(input1) == 1 ||
          get_dim<Dim4D::Batch>(input2) == 1,
      broadcast_error_msg);
  TORCH_CHECK(
      get_dim<Dim4D::Channel>(input1) == get_dim<Dim4D::Channel>(input2) ||
          get_dim<Dim4D::Channel>(input1) == 1 ||
          get_dim<Dim4D::Channel>(input2) == 1,
      broadcast_error_msg);
  TORCH_CHECK(
      get_dim<Dim4D::Height>(input1) == get_dim<Dim4D::Height>(input2) ||
          get_dim<Dim4D::Height>(input1) == 1 ||
          get_dim<Dim4D::Height>(input2) == 1,
      broadcast_error_msg);
  TORCH_CHECK(
      get_dim<Dim4D::Width>(input1) == get_dim<Dim4D::Width>(input2) ||
          get_dim<Dim4D::Width>(input1) == 1 ||
          get_dim<Dim4D::Width>(input2) == 1,
      broadcast_error_msg);
}

// compute the output shape by broadcasting the shapes of t1 and t2
std::vector<int64_t> broadcast_size(const Tensor& t1, const Tensor& t2) {
  int64_t t1_size = t1.dim();
  int64_t t2_size = t2.dim();

  std::vector<int64_t> out;
  if (t1_size > t2_size) {
    for (int64_t i = 0; i < t1_size; i++) {
      out.push_back(t1.sizes()[i]);
    }
  } else {
    for (int64_t i = 0; i < t2_size; i++) {
      out.push_back(t2.sizes()[i]);
    }
  }

  if (!out.empty()) {
    out[out.size() - 1] =
        std::max(get_dim<Dim4D::Width>(t1), get_dim<Dim4D::Width>(t2));
  }
  if (out.size() > 1) {
    out[out.size() - 2] =
        std::max(get_dim<Dim4D::Height>(t1), get_dim<Dim4D::Height>(t2));
  }
  if (out.size() > 2) {
    out[out.size() - 3] =
        std::max(get_dim<Dim4D::Channel>(t1), get_dim<Dim4D::Channel>(t2));
  }
  if (out.size() > 3) {
    out[out.size() - 4] =
        std::max(get_dim<Dim4D::Batch>(t1), get_dim<Dim4D::Batch>(t2));
  }

  return out;
}

Tensor masked_fill_scalar(
    const Tensor& self_arg,
    const Tensor& mask_arg,
    const Scalar& value) {
  check_inputs(self_arg, mask_arg);

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();

  const Tensor mask = mask_arg.is_vulkan() ? mask_arg : mask_arg.vulkan();
  const vTensor& v_mask = convert(mask);

  // compute the output shape by broadcasting the shapes of self and mask
  auto in_ndims = safe_downcast<uint32_t>(self_arg.dim());
  auto in_sizes = self_arg.sizes();
  auto mask_sizes = mask_arg.sizes();
  std::vector<int64_t> out_sizes = broadcast_size(self_arg, mask_arg);
  TORCH_INTERNAL_ASSERT(!out_sizes.empty(), "output shape is empty!");

  // generalize the shape of output and mask to 4D
  uvec4 generalized_out_sizes{1u, 1u, 1u, 1u},
      generalized_mask_sizes{1u, 1u, 1u, 1u};
  int add_out_ndims = static_cast<int>(4 - out_sizes.size());
  for (int i = 0; (unsigned)i < out_sizes.size(); i++) {
    generalized_out_sizes.data[i + add_out_ndims] = out_sizes[i];
  }
  int add_mask_ndims = static_cast<int>(4 - mask_sizes.size());
  for (int i = 0; (unsigned)i < mask_sizes.size(); i++) {
    generalized_mask_sizes.data[i + add_mask_ndims] = mask_sizes[i];
  }

  auto out_ndims = safe_downcast<uint32_t>(out_sizes.size());

  // channels of mask and output after padding to nearest multiple of 4
  uint32_t mask_c_aligned =
      api::utils::align_up(generalized_mask_sizes.data[1u], 4u);
  uint32_t out_c_aligned =
      api::utils::align_up(generalized_out_sizes.data[1u], 4u);

  // compute the repeats needed to output a tensor of out_sizes by doing
  // repeat operation on self
  auto add_ndims = out_ndims - in_ndims;
  std::vector<int64_t> repeats;
  for (int i = 0; (unsigned)i < out_ndims; i++) {
    if ((unsigned)i < add_ndims || in_sizes[i - add_ndims] == 1) {
      repeats.push_back(out_sizes[i]);
    } else {
      repeats.push_back(1);
    }
  }

  // generate the output of out_sizes by doing repeat operation on self
  at::Tensor out = self.repeat(repeats);
  vTensor& v_out = convert(out);

  const struct Block final {
    ivec3 outExtents;
    int32_t fill0;
    ivec3 maskExtents;
    int32_t fill1;
    uvec4 outTensorSize;
    uvec4 maskTensorSize;
    uvec2 alignedChannelInfo;
    float value;
  } block{
      api::utils::make_ivec3(v_out.extents()),
      0,
      api::utils::make_ivec3(v_mask.extents()),
      0,
      generalized_out_sizes,
      generalized_mask_sizes,
      {out_c_aligned, mask_c_aligned},
      value.to<float>(),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  // One possible implementation of masked_fill is to do repeat operation on
  // mask and generate a broadcasted mask of the same shape as the output, and
  // then fill elements of the output with value where mask is True. However the
  // repeat operation on mask would cause extra time and space overhead.
  // Instead, in the shader file we traverse through the original mask and
  // compute the corresponding broadcasted positions in the output tensor when a
  // mask value is True.
  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(masked_fill),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_mask.extents(),
      // local work group size
      adaptive_work_group_size(v_mask.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_out.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_mask.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_out);
}

Tensor masked_fill_tensor(
    const Tensor& self_arg,
    const Tensor& mask_arg,
    const Tensor& value) {
  TORCH_CHECK(
      value.dim() == 0,
      "masked_fill only supports a 0-dimensional value tensor, but got tensor with ",
      value.dim(),
      " dimension(s).");
  return masked_fill_scalar(self_arg, mask_arg, value.item<float>());
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::masked_fill.Scalar"),
      TORCH_FN(masked_fill_scalar));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::masked_fill.Tensor"),
      TORCH_FN(masked_fill_tensor));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
