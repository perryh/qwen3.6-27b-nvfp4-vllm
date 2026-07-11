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
# Memory: weights ~22 GiB. On a 32 GiB card we cap context at 32768, use FP8 KV
# cache, disable CUDA graphs (--enforce-eager) and enable expandable_segments to
# fit. Raise --max-model-len only if your GPU has headroom.

FROM vllm/vllm-openai:v0.24.0

ENV HF_TOKEN= \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Weights are mounted from the host at run time, not baked in:
#   -v ~/.cache/huggingface:/root/.cache/huggingface
EXPOSE 30000

HEALTHCHECK --interval=30s --timeout=10s --start-period=420s --retries=3 \
  CMD curl -sf http://127.0.0.1:30000/health || exit 1

# Base ENTRYPOINT is `vllm serve`, so CMD begins with the model path.
CMD ["unsloth/Qwen3.6-27B-NVFP4", \
     "--host", "0.0.0.0", \
     "--port", "30000", \
     "--enforce-eager", \
     "--max-model-len", "32768", \
     "--kv-cache-dtype", "fp8_e4m3", \
     "--speculative-config", "{\"method\":\"mtp\",\"num_speculative_tokens\":2}"]
