/// \file tests/suites/cext/ffnagent/executor_test.cu
/// \brief Device behavior tests for the typed FfnAgent execution boundary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cooperative_groups.h>

#include <xpool/abi.hpp>
#include <xpool/debug/options.hpp>
#include <xpool/ffnagent/executor.cuh>

namespace {

constexpr auto kCaseCount = std::size_t{6};
constexpr auto kValueStride = std::size_t{4};

::testing::AssertionResult cuda_succeeded(cudaError_t error) {
  if (error == cudaSuccess) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << cudaGetErrorString(error);
}

class FfnExecutorCudaTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto device_count = int{0};
    const auto error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA device is not available: " << cudaGetErrorString(error);
    }
    ASSERT_TRUE(cuda_succeeded(cudaSetDevice(0)));
  }
};

__global__ void execute_invocations(const xpool::fabric::FfnInvocation *invocations,
                                    const xpool::fabric::FabricModelLayout *models,
                                    const xpool::fabric::FabricLayerLayout *layers, const float *input,
                                    float *output, std::uint32_t *results) {
  const auto index = static_cast<std::size_t>(blockIdx.x);
  results[index] = xpool::ffnagent::executor::execute(
                       cooperative_groups::this_thread_block(), invocations[index], models[index], layers[index],
                       input + index * kValueStride, output + index * kValueStride)
                       .value();
}

xpool::fabric::FfnInvocation valid_invocation() {
  return {
      .key = {.model_index = 0, .invocation_sequence = 1},
      .layer_ordinal = 0,
      .payload_rows = 1,
      .input_pe = 0,
      .execution_mode = xpool::fabric::FfnExecutionMode::Decode,
      .result_handoff = xpool::abi::FfnResultHandoff::ReplicatedFull,
      .dp_padding_mode = xpool::abi::DpPaddingMode::None,
  };
}

xpool::fabric::FabricModelLayout valid_model() {
  return {
      .dtype = xpool::abi::TensorDType::Fp32,
      .hidden_size = 2,
      .atn_tp_size = 1,
      .atn_dp_size = 1,
      .layer_begin = 0,
      .layer_count = 1,
      .decode_payload_offset = 0,
      .decode_payload_capacity_bytes = xpool::arena::kPayloadAlignment,
      .prefill_payload_capacity_bytes = xpool::arena::kPayloadAlignment,
  };
}

xpool::fabric::FabricLayerLayout valid_layer() {
  return {
      .layer_id = 0,
      .kind = xpool::fabric::FfnLayerKind::Dense,
  };
}

} // namespace

TEST_F(FfnExecutorCudaTest, ValidatesTypedInvocationAndLayerBeforeLoopbackExecution) {
  auto debug_options = xpool::debug::DebugOptions{};
  debug_options.loopback = xpool::debug::LoopbackOptions{true, xpool::debug::LoopbackSite::FfnAgent};
  xpool::debug::configure(debug_options, 0);

  auto *invocations = static_cast<xpool::fabric::FfnInvocation *>(nullptr);
  auto *models = static_cast<xpool::fabric::FabricModelLayout *>(nullptr);
  auto *layers = static_cast<xpool::fabric::FabricLayerLayout *>(nullptr);
  auto *input = static_cast<float *>(nullptr);
  auto *output = static_cast<float *>(nullptr);
  auto *results = static_cast<std::uint32_t *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&invocations, kCaseCount * sizeof(*invocations))));
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&models, kCaseCount * sizeof(*models))));
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&layers, kCaseCount * sizeof(*layers))));
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&input, kCaseCount * kValueStride * sizeof(*input))));
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&output, kCaseCount * kValueStride * sizeof(*output))));
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&results, kCaseCount * sizeof(*results))));

  for (auto index = std::size_t{0}; index < kCaseCount; ++index) {
    invocations[index] = valid_invocation();
    models[index] = valid_model();
    layers[index] = valid_layer();
    input[index * kValueStride] = 1.0F;
    input[index * kValueStride + 1] = 0.0F;
  }
  invocations[1].result_handoff = 99;
  invocations[2].execution_mode = 99;
  layers[3].kind = 99;
  models[4].hidden_size = 3;
  invocations[5].result_handoff = xpool::abi::FfnResultHandoff::ReduceScatterInput;
  invocations[5].dp_padding_mode = xpool::abi::DpPaddingMode::MaxLen;

  execute_invocations<<<kCaseCount, 1, 0, nullptr>>>(invocations, models, layers, input, output, results);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ(results[0], xpool::abi::FfnResultCode::Ok);
  EXPECT_EQ(results[1], xpool::abi::FfnResultCode::ProtocolMismatch);
  EXPECT_EQ(results[2], xpool::abi::FfnResultCode::ProtocolMismatch);
  EXPECT_EQ(results[3], xpool::abi::FfnResultCode::ProtocolMismatch);
  EXPECT_EQ(results[4], xpool::abi::FfnResultCode::NotImplemented);
  EXPECT_EQ(results[5], xpool::abi::FfnResultCode::Ok);
  EXPECT_NEAR(output[0], std::sqrt(0.5F), 1e-6F);
  EXPECT_NEAR(output[1], std::sqrt(0.5F), 1e-6F);

  EXPECT_TRUE(cuda_succeeded(cudaFree(results)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(output)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(input)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(layers)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(models)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(invocations)));
}
