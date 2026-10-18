#include <xpool/ffnagent/router.hpp>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <utility>

#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>

namespace xpool::ffnagent {
namespace {

void require_success(cublasStatus_t status, const char *operation) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "xpool BF16 Router ", operation, " failed: ", static_cast<int>(status));
}

using MatmulDescriptor =
    std::unique_ptr<std::remove_pointer_t<cublasLtMatmulDesc_t>, decltype(&cublasLtMatmulDescDestroy)>;
using MatrixLayout =
    std::unique_ptr<std::remove_pointer_t<cublasLtMatrixLayout_t>, decltype(&cublasLtMatrixLayoutDestroy)>;
using MatmulPreference =
    std::unique_ptr<std::remove_pointer_t<cublasLtMatmulPreference_t>, decltype(&cublasLtMatmulPreferenceDestroy)>;

MatrixLayout matrix_layout(std::int64_t rows, std::int64_t columns, std::int64_t stride) {
  cublasLtMatrixLayout_t result = nullptr;
  require_success(cublasLtMatrixLayoutCreate(&result, CUDA_R_16BF, rows, columns, stride), "layout creation");
  return MatrixLayout(result, &cublasLtMatrixLayoutDestroy);
}

std::uint32_t tensor_alignment(const at::Tensor &tensor) {
  const auto address = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
  auto alignment = std::uint32_t{1};
  while (alignment < 256 && address % (alignment * 2) == 0) {
    alignment *= 2;
  }
  return alignment;
}

using AlgorithmKey =
    std::tuple<int, std::int64_t, std::int64_t, std::int64_t, std::uint32_t, std::uint32_t, std::uint32_t>;

cublasLtMatmulAlgo_t prepared_algorithm(const AlgorithmKey &key, cublasLtHandle_t handle,
                                        cublasLtMatmulDesc_t descriptor, cublasLtMatrixLayout_t weight_layout,
                                        cublasLtMatrixLayout_t input_layout, cublasLtMatrixLayout_t output_layout,
                                        cublasLtMatmulPreference_t preference) {
  // Only host algorithm values are retained. Warmup fixes each value before
  // stream capture; replay never queries the library or this cache.
  static auto algorithms = std::map<AlgorithmKey, cublasLtMatmulAlgo_t>{};
  static auto mutex = std::mutex{};
  const auto lock = std::lock_guard(mutex);
  if (const auto found = algorithms.find(key); found != algorithms.end()) {
    return found->second;
  }
  auto capture_status = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(c10::cuda::getCurrentCUDAStream(), &capture_status));
  TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
              "xpool BF16 Router algorithm must be warmed before CUDA Graph capture");
  auto algorithm = cublasLtMatmulHeuristicResult_t{};
  auto returned = int{0};
  require_success(cublasLtMatmulAlgoGetHeuristic(handle, descriptor, weight_layout, input_layout, output_layout,
                                                 output_layout, preference, 1, &algorithm, &returned),
                  "algorithm selection");
  TORCH_CHECK(returned == 1 && algorithm.state == CUBLAS_STATUS_SUCCESS,
              "xpool BF16 Router has no algorithm within its explicit workspace");
  return algorithms.emplace(key, algorithm.algo).first->second;
}

} // namespace

void compute_bf16_router_logits(const at::Tensor &hidden_states, const at::Tensor &weight, const at::Tensor &bias,
                                const at::Tensor &output, const at::Tensor &workspace) {
  for (const auto *tensor : {&hidden_states, &weight, &bias, &output}) {
    TORCH_CHECK(tensor->is_cuda() && tensor->scalar_type() == at::kBFloat16 && tensor->is_contiguous() &&
                    tensor->numel() != 0 && tensor->device() == hidden_states.device(),
                "xpool BF16 Router tensors must be nonempty contiguous BF16 on one CUDA device");
  }
  TORCH_CHECK(hidden_states.dim() == 2 && weight.dim() == 2 && bias.dim() == 1 && output.dim() == 2 &&
                  hidden_states.size(1) == weight.size(1) && bias.size(0) == weight.size(0) &&
                  output.size(0) == hidden_states.size(0) && output.size(1) == weight.size(0),
              "xpool BF16 Router tensor shapes disagree");
  TORCH_CHECK(workspace.is_cuda() && workspace.device() == hidden_states.device() && workspace.is_contiguous() &&
                  workspace.scalar_type() == at::kByte && workspace.dim() == 1 &&
                  workspace.nbytes() == kBf16RouterWorkspaceBytes &&
                  reinterpret_cast<std::uintptr_t>(workspace.data_ptr()) % kBf16RouterWorkspaceAlignment == 0,
              "xpool BF16 Router requires its exact aligned byte workspace on the input device");
  for (const auto *tensor : {&hidden_states, &weight, &bias}) {
    at::assert_no_overlap(output, *tensor);
    at::assert_no_overlap(workspace, *tensor);
  }
  at::assert_no_overlap(output, workspace);
  const auto guard = c10::cuda::CUDAGuard(hidden_states.device());

  cublasLtMatmulDesc_t raw_descriptor = nullptr;
  require_success(cublasLtMatmulDescCreate(&raw_descriptor, CUBLAS_COMPUTE_32F, CUDA_R_32F), "descriptor creation");
  const auto descriptor = MatmulDescriptor(raw_descriptor, &cublasLtMatmulDescDestroy);
  const auto transpose = CUBLAS_OP_T;
  require_success(
      cublasLtMatmulDescSetAttribute(descriptor.get(), CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)),
      "transpose configuration");
  const auto epilogue = CUBLASLT_EPILOGUE_BIAS;
  require_success(
      cublasLtMatmulDescSetAttribute(descriptor.get(), CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)),
      "bias configuration");
  auto *bias_pointer = bias.data_ptr();
  require_success(cublasLtMatmulDescSetAttribute(descriptor.get(), CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer,
                                                 sizeof(bias_pointer)),
                  "bias pointer configuration");
  const auto hidden_size = hidden_states.size(1);
  const auto expert_count = weight.size(0);
  const auto rows = hidden_states.size(0);
  const auto weight_layout = matrix_layout(hidden_size, expert_count, hidden_size);
  const auto input_layout = matrix_layout(hidden_size, rows, hidden_size);
  const auto output_layout = matrix_layout(expert_count, rows, expert_count);

  cublasLtMatmulPreference_t raw_preference = nullptr;
  require_success(cublasLtMatmulPreferenceCreate(&raw_preference), "preference creation");
  const auto preference = MatmulPreference(raw_preference, &cublasLtMatmulPreferenceDestroy);
  const auto workspace_bytes = kBf16RouterWorkspaceBytes;
  require_success(cublasLtMatmulPreferenceSetAttribute(preference.get(), CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &workspace_bytes, sizeof(workspace_bytes)),
                  "workspace configuration");
  const auto alignments = std::array{
      std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, tensor_alignment(weight)},
      std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, tensor_alignment(hidden_states)},
      std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, tensor_alignment(output)},
      std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, tensor_alignment(output)},
  };
  for (const auto &[attribute, alignment] : alignments) {
    require_success(cublasLtMatmulPreferenceSetAttribute(preference.get(), attribute, &alignment, sizeof(alignment)),
                    "alignment configuration");
  }
  const auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  const auto algorithm = prepared_algorithm(
      AlgorithmKey{hidden_states.get_device(), rows, expert_count, hidden_size, tensor_alignment(weight),
                   tensor_alignment(hidden_states), tensor_alignment(output)},
      handle, descriptor.get(), weight_layout.get(), input_layout.get(), output_layout.get(), preference.get());
  const auto alpha = 1.0F;
  const auto beta = 0.0F;
  require_success(cublasLtMatmul(handle, descriptor.get(), &alpha, weight.data_ptr(), weight_layout.get(),
                                 hidden_states.data_ptr(), input_layout.get(), &beta, output.data_ptr(),
                                 output_layout.get(), output.data_ptr(), output_layout.get(), &algorithm,
                                 workspace.data_ptr(), workspace_bytes, c10::cuda::getCurrentCUDAStream()),
                  "GEMM launch");
}

} // namespace xpool::ffnagent
