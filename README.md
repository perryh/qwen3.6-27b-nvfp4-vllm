# Qwen3.6-27B-NVFP4 on vLLM (Docker)

Serves [`unsloth/Qwen3.6-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4)
behind a [vLLM](https://github.com/vllm-project/vllm) OpenAI-compatible server
with Multi-Token Prediction (MTP) speculative decoding, per the
[Unsloth NVFP4 guide](https://unsloth.ai/docs/models/qwen3.6#nvfp4).

## Why vLLM (not SGLang)

The Unsloth model card recommends **vLLM** for this NVFP4 checkpoint. SGLang
can't load it yet: its `compressed_tensors` loader raises
`NotImplementedError: No compressed-tensors compatible scheme was found` on the
model's `nvfp4-pack-quantized` MoE layers (the `gate_up_proj` dispatches to the
linear scheme path instead of the MoE path). This holds even in SGLang's latest
nightly, so we use vLLM (`vllm/vllm-openai:v0.24.0`), which loads it natively
via the `FlashInferCutlassNvFp4LinearKernel`.

## Requirements

- **NVIDIA Blackwell GPU** (RTX 50-series / DGX Spark / B200 / B300) with ~32GB.
  NVFP4 uses FP4 tensor cores and will **not** run on Ampere/Hopper. The model
  weights alone take ~22 GiB.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  on the host (so Docker can pass GPUs through).
- Docker with Compose v2.

## Run

```bash
docker compose up -d --build
```

The server starts on `http://localhost:30000`. First launch downloads the
weights (~21GB) into `~/.cache/huggingface` (mounted from the host); subsequent
starts reuse them. Model load + warmup takes ~3 minutes.

### Selecting the GPU

The compose file pins the container to a single GPU via `GPU_DEVICE_ID`
(defaults to `0`). Pick the id from `nvidia-smi` and override it:

```bash
GPU_DEVICE_ID=1 docker compose up -d --build
```

You can also set it in a `.env` file next to the compose file (Compose loads
that automatically) or export it in your shell.

## Memory configuration

Weights use ~22 GiB. On a 32 GiB card (e.g. RTX 5090) the default 256K context
+ CUDA graphs OOM, so the `Dockerfile` `CMD` sets:

- `--max-model-len 98304` — cap context near the KV-cache ceiling (~115K tokens
  on a 32 GiB card; gives ~1.17x concurrency). Lower for more concurrency.

The context length is configurable via the `MAX_MODEL_LEN` env var (default
98304):

```bash
MAX_MODEL_LEN=65536 docker compose up -d      # 64K context, 1.66x concurrency
# or: docker run -e MAX_MODEL_LEN=32768 ...
```
- `--kv-cache-dtype fp8_e4m3` — FP8 KV cache (~2x the context for the same RAM)
- `--enforce-eager` — skip CUDA-graph capture to free VRAM
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduce fragmentation

On larger GPUs (B200/96GB), drop `--enforce-eager` and raise `--max-model-len`
for higher throughput.

## Smoke test

```bash
curl http://localhost:30000/v1/models
curl http://localhost:30000/health        # 200 + empty body == healthy
```

Chat (thinking mode is on by default):

```bash
curl http://localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "unsloth/Qwen3.6-27B-NVFP4",
    "messages": [{"role":"user","content":"What is 2+2? One word."}],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

Disable thinking for direct answers:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "unsloth/Qwen3.6-27B-NVFP4",
    "messages": [{"role":"user","content":"What is 2+2? One word."}],
    "max_tokens": 50,
    "temperature": 0.7,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## Equivalent raw `docker run`

```bash
docker run -d --name qwen36-vllm --gpus '"device=0"' --shm-size 32g --ipc=host \
  -p 30000:30000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  qwen36-27b-nvfp4-vllm:latest
```

## CUDA note

Unsloth reports that **CUDA 13.2 produces gibberish** for Qwen3.6. The vLLM
0.24.0 image ships CUDA 13.0, which is safe. Output was verified coherent.

## Files

- `Dockerfile` — thin layer over `vllm/vllm-openai:v0.24.0`; sets the serve
  command, healthcheck, and memory flags.
- `docker-compose.yml` — GPU pin via `GPU_DEVICE_ID` (default 0), shared HF cache
  volume, `shm_size 32g`, `ipc: host`, port mapping.
- `.dockerignore` — keeps the build context small.
