"""Register custom model capabilities with litellm."""

import litellm

litellm.model_cost["Qwen/Qwen3.5-4B"] = {"supports_reasoning": True}
litellm.model_cost["qwen3.5:4b"] = {"supports_reasoning": True}
