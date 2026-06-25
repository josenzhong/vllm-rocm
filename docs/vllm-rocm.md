# vLLM ROCm for Unraid

This template runs the official `vllm/vllm-openai-rocm:latest` container on Unraid for AMD ROCm-capable GPUs.

## Requirements

- AMD ROCm-compatible GPU
- `/dev/kfd` available on the Unraid host
- `/dev/dri` available on the Unraid host
- Radeon Top or equivalent AMD GPU support configured on Unraid
- Enough VRAM for the selected model

## Default configuration

The template starts with:

```text
--model Qwen/Qwen3-0.6B
```

This is intended as a small smoke-test model. After confirming the container works, change the **Model Arguments** field to your desired model.

Example for a 7B-class model:

```text
--model Qwen/Qwen2.5-7B-Instruct --dtype auto --gpu-memory-utilization 0.90 --max-model-len 8192
```

## API endpoint

vLLM exposes an OpenAI-compatible API:

```text
http://UNRAID_IP:8000/v1
```

Test with:

```bash
curl http://UNRAID_IP:8000/v1/models
```

## Chat UI

vLLM is an API server, not a full chat web interface. For a browser chat interface, connect Open WebUI or another OpenAI-compatible frontend to:

```text
http://UNRAID_IP:8000/v1
```

## Troubleshooting

If the container cannot see the GPU, check that the host exposes the AMD device nodes:

```bash
ls -l /dev/kfd
ls -l /dev/dri
ls -l /dev/dri/render*
```

If `/dev/kfd` is missing, fix AMD GPU support on the Unraid host before troubleshooting vLLM.
