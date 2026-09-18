# Supported Models

The following concrete model IDs have SGLang adapters and explicit CrossPool
model-qualification suites:

- ✅ `Qwen/Qwen2.5-0.5B`
- ✅ `Qwen/Qwen3-0.6B`
- ✅ `Qwen/Qwen3-14B`
- ✅ `Qwen/Qwen3-30B-A3B`
- ✅ `deepseek-ai/DeepSeek-V2-Lite-Chat`
- ✅ `zai-org/GLM-4.7-Flash`

Run the relevant `xtest run --suite <model-id> --strict-requirements` suite
when changes invalidate that model's qualification evidence. The complete
test-selection rules are in [Test Architecture](../tests/README.md).
