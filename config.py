# TODO
# https://openrouter.ai/meta-llama/llama-4-scout?quantization=bf16

# When executing, check github state and if True will break if git not committed
# use this if I'm recording experiments
BREAK_IF_NOT_CHECKED_IN = False

providers = {
    # Llama 4 Scout is multimodal (text and image)
    "openrouter/meta-llama/llama-4-scout":
    # on OpenRouter limit to certain providers as the variety that's available
    # will use various quantisations context sizes
    # provider https://openrouter.ai/docs/features/provider-routing#example-specifying-providers-with-fallbacks-disabled
    {
        "order": [
            # quantisation, input context size, output context size
            # https://openrouter.ai/meta-llama/llama-4-scout?quantization=bf16
            "CentML",  # bf16, 1.05M, 1.05M,
            "GMICloud",  # bf16, 1.05M, 1.05M
            # "DeepInfra", # bf16, 328k, 16k - unusual config, not sure if model is much different?
            # https://openrouter.ai/meta-llama/llama-4-scout?quantization=fp8
            # "Lambda",  # fp8, 1.05M, 1.05M
            # "Parasail",  # fp8, 158k, 158k
        ],
        "allow_fallbacks": False,  # only use specified providers
    },
    # DeepSeek V3 0324 is probably better than DeepSeek R1
    "openrouter/deepseek/deepseek-chat-v3-0324": {
        "order": [
            "deepinfra/fp8",  # "DeepInfra",  # fp8, 164k, 164k
            "lambda/fp8",  # "Lambda",  # fp8, 164k, 164k
            "nebius/fp8",  # "Nebius AI Studio",  # fp8 164k, 164k
        ],
        "allow_fallbacks": False,
    },
    # Opus 4 is multimodal (text and image)
    "openrouter/anthropic/claude-opus-4": {
        "order": [
            "anthropic",  # 200k, 32k
            "google-vertex",  # 200k, 32k
        ],
        "allow_fallbacks": False,
    },
    # GPT 4o, multimodal
    # https://openrouter.ai/openai/gpt-4o/api
    # GPT o3 - needs private (2nd) OpenAI key so I'm ignoring it
    # GPT o1-pro is crazy expensive! Multimodal
    # https://openrouter.ai/openai/o1-pro
    # GPT o1 is more like Opus 4 on pricing, multimodal
    # https://openrouter.ai/openai/o1
}
