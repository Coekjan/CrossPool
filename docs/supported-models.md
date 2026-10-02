# Supported Models

This table lists qualified models and coverage candidates. Supported models have
SGLang adapters and explicit CrossPool model-qualification suites. Support is
qualified per model ID; sharing a family does not establish support.

Family names identify the complete checkpoint architecture, including multimodal
wrappers. Model ID links point to the checkpoint configuration.

| Model ID | Family | Support |
| --- | --- | --- |
| [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json) | `Qwen2ForCausalLM` | ✅ Supported |
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) | `Qwen2ForCausalLM` | ❌ Not supported |
| [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json) | `Qwen2ForCausalLM` | ❌ Not supported |
| [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json) | `Qwen2ForCausalLM` | ❌ Not supported |
| [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/config.json) | `Qwen2ForCausalLM` | ❌ Not supported |
| [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/main/config.json) | `Qwen2ForCausalLM` | ❌ Not supported |
| [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json) | `Qwen2ForCausalLM` | ❌ Not supported |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/blob/main/config.json) | `Qwen2ForCausalLM` | ❌ Not supported |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json) | `Qwen3ForCausalLM` | ✅ Supported |
| [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json) | `Qwen3ForCausalLM` | ❌ Not supported |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json) | `Qwen3ForCausalLM` | ❌ Not supported |
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) | `Qwen3ForCausalLM` | ❌ Not supported |
| [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json) | `Qwen3ForCausalLM` | ✅ Supported |
| [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) | `Qwen3ForCausalLM` | ❌ Not supported |
| [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json) | `Qwen3MoeForCausalLM` | ✅ Supported |
| [Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json) | `Qwen3MoeForCausalLM` | ❌ Not supported |
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json) | `Qwen3_5ForConditionalGeneration` | ❌ Not supported |
| [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json) | `Qwen3_5ForConditionalGeneration` | ❌ Not supported |
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json) | `Qwen3_5ForConditionalGeneration` | ❌ Not supported |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json) | `Qwen3_5ForConditionalGeneration` | ❌ Not supported |
| [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json) | `Qwen3_5ForConditionalGeneration` | ❌ Not supported |
| [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json) | `GptOssForCausalLM` | ❌ Not supported |
| [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json) | `GptOssForCausalLM` | ❌ Not supported |
| [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json) | `Gemma4ForConditionalGeneration` | ❌ Not supported |
| [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json) | `Gemma4ForConditionalGeneration` | ❌ Not supported |
| [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json) | `Gemma4ForConditionalGeneration` | ❌ Not supported |
| [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json) | `Gemma4ForConditionalGeneration` | ❌ Not supported |
| [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it/blob/main/config.json) | `Gemma4UnifiedForConditionalGeneration` | ❌ Not supported |
| [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b/blob/main/config.json) | `LlamaForCausalLM` | ❌ Not supported |
| [huggyllama/llama-13b](https://huggingface.co/huggyllama/llama-13b/blob/main/config.json) | `LlamaForCausalLM` | ❌ Not supported |
| [huggyllama/llama-30b](https://huggingface.co/huggyllama/llama-30b/blob/main/config.json) | `LlamaForCausalLM` | ❌ Not supported |
| [huggyllama/llama-65b](https://huggingface.co/huggyllama/llama-65b/blob/main/config.json) | `LlamaForCausalLM` | ❌ Not supported |
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json) | `LlamaForCausalLM` | ❌ Not supported |
| [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json) | `LlamaForCausalLM` | ❌ Not supported |
| [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json) | `LlamaForCausalLM` | ❌ Not supported |
| [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json) | `LlamaForCausalLM` | ❌ Not supported |
| [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json) | `LlamaForCausalLM` | ❌ Not supported |
| [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json) | `LlamaForCausalLM` | ❌ Not supported |
| [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json) | `LlamaForCausalLM` | ❌ Not supported |
| [facebook/opt-125m](https://huggingface.co/facebook/opt-125m/blob/main/config.json) | `OPTForCausalLM` | ❌ Not supported |
| [facebook/opt-350m](https://huggingface.co/facebook/opt-350m/blob/main/config.json) | `OPTForCausalLM` | ❌ Not supported |
| [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b/blob/main/config.json) | `OPTForCausalLM` | ❌ Not supported |
| [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b/blob/main/config.json) | `OPTForCausalLM` | ❌ Not supported |
| [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b/blob/main/config.json) | `OPTForCausalLM` | ❌ Not supported |
| [facebook/opt-13b](https://huggingface.co/facebook/opt-13b/blob/main/config.json) | `OPTForCausalLM` | ❌ Not supported |
| [facebook/opt-30b](https://huggingface.co/facebook/opt-30b/blob/main/config.json) | `OPTForCausalLM` | ❌ Not supported |
| [facebook/opt-66b](https://huggingface.co/facebook/opt-66b/blob/main/config.json) | `OPTForCausalLM` | ❌ Not supported |
| [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json) | `FalconForCausalLM` | ❌ Not supported |
| [tiiuae/falcon-40b](https://huggingface.co/tiiuae/falcon-40b/blob/main/config.json) | `FalconForCausalLM` | ❌ Not supported |
| [tiiuae/falcon-11B](https://huggingface.co/tiiuae/falcon-11B/blob/main/config.json) | `FalconForCausalLM` | ❌ Not supported |
| [tiiuae/falcon-180B](https://huggingface.co/tiiuae/falcon-180B/blob/main/config.json) | `FalconForCausalLM` (provisional) | ❌ Not supported |
| [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR/blob/main/config.json) | `DeepseekOCRForCausalLM` | ❌ Not supported |
| [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) | `DeepseekV3ForCausalLM` | ❌ Not supported (deferred) |
| [deepseek-ai/DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json) | `DeepseekV2ForCausalLM` | ✅ Supported |
| [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/config.json) | `Glm4MoeLiteForCausalLM` | ✅ Supported |

Falcon-180B's family remains provisional until its gated checkpoint configuration
is verified. DeepSeek-V3 is deferred from the current adaptation scope.

Run the relevant `xtest run --suite <model-id> --strict-requirements` suite
when changes invalidate that model's qualification evidence. The complete
test-selection rules are in [Test Architecture](../tests/README.md).
