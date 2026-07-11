# vLLM server for Qwen3.6-27B-NVFP4 (Unsloth quant).
#
# Requires an NVIDIA Blackwell GPU (RTX 50-series / DGX Spark / B200 / B300).
# NVFP4 uses FP4 tensor cores and will NOT run on Ampere/Hopper.
#
# Why vLLM (not SGLang): Unsloth's model card recommends vLLM for this NVFP4
# checkpoint. SGLang's compressed-tensors loader can't load the model's
# "nvfp4-pack-quantized" MoE scheme yet (NotImplementedError, even in nightly),
# because the MoE gate_up_proj dispatches to the linear scheme path.
#
# Base image bundles everything: vLLM 0.24.0, nvidia-cutlass-dsl 4.5.2, CUDA
# 13.0 (safely below the 13.2 gibberish threshold Unsloth warns about). No pip
# installs needed.
#
# Memory: weights ~22 GiB. On a 32 GiB card we cap context at 98304 (near the
# ~115K-token KV-cache ceiling), use FP8 KV cache, disable CUDA graphs
# (--enforce-eager) and enable expandable_segments to fit. This leaves ~1.17x
# concurrency; lower --max-model-len for more concurrent requests, or raise it
# on a larger GPU.

FROM vllm/vllm-openai:v0.24.0

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV MAX_MODEL_LEN=98304 \
    HF_TOKEN= \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Weights are mounted from the host at run time, not baked in:
#   -v ~/.cache/huggingface:/root/.cache/huggingface
EXPOSE 30000

HEALTHCHECK --interval=30s --timeout=10s --start-period=420s --retries=3 \
  CMD curl -sf http://127.0.0.1:30000/health || exit 1

# Override the base `vllm serve` ENTRYPOINT so the wrapper can expand env vars
# (e.g. MAX_MODEL_LEN). Override at run time: -e MAX_MODEL_LEN=65536
ENTRYPOINT ["/entrypoint.sh"]
