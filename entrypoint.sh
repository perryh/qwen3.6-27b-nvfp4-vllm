#!/bin/sh
set -e
exec vllm serve unsloth/Qwen3.6-27B-NVFP4 \
  --host 0.0.0.0 \
  --port 30000 \
  --enforce-eager \
  --max-model-len "${MAX_MODEL_LEN:-98304}" \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  "$@"
