# vLLM ROCm for Unraid

This template runs the official `vllm/vllm-openai-rocm:latest` container on Unraid for AMD ROCm-capable GPUs.

## What this app is

vLLM is an OpenAI-compatible inference API server. It is best used as a backend for applications such as Open WebUI, LiteLLM, Lobe Chat, custom scripts, or other tools that can talk to an OpenAI-compatible `/v1` API.

## What this app is not

This container does **not** include a full browser WebUI for downloading models, browsing local models, changing settings at runtime, or chatting directly. The Unraid WebUI controls the Docker container only. The template's WebUI button opens `/v1/models` as a simple API health check.

To get a full browser chat interface, install Open WebUI or another frontend and point it at:

```text
http://UNRAID_IP:8000/v1
```

## Requirements

- AMD ROCm-compatible GPU
- `/dev/kfd` available on the Unraid host
- `/dev/dri` available on the Unraid host
- Radeon Top or equivalent AMD GPU support configured on Unraid
- Enough VRAM for the selected model

Consumer Radeon GPU compatibility can depend on the ROCm/vLLM image version and selected model. The template follows the official vLLM ROCm Docker run pattern, but it cannot guarantee every AMD GPU/model combination will run.

## Default configuration

The template starts with:

```text
--model Qwen/Qwen3-0.6B
```

This is intended as a small smoke-test model. After confirming the container works, change the **Model Arguments** field to your desired model.

Example for a Hugging Face model:

```text
--model Qwen/Qwen2.5-7B-Instruct --dtype auto --gpu-memory-utilization 0.90 --max-model-len 8192
```

## Local models

The template maps this host path by default:

```text
/mnt/user/appdata/vllm/models -> /models
```

Put a complete Hugging Face-format model folder inside that directory, then set **Model Arguments** to a container path, for example:

```text
--model /models/my-local-model --dtype auto --gpu-memory-utilization 0.90 --max-model-len 8192
```

vLLM reads the model path at startup. To switch models, edit the container's **Model Arguments** and restart the container.

## API endpoint

vLLM exposes an OpenAI-compatible API:

```text
http://UNRAID_IP:8000/v1
```

Test with:

```bash
curl http://UNRAID_IP:8000/v1/models
```

## Troubleshooting

If the container cannot see the GPU, check that the host exposes the AMD device nodes:

```bash
ls -l /dev/kfd
ls -l /dev/dri
ls -l /dev/dri/render*
```

If `/dev/kfd` is missing, fix AMD GPU support on the Unraid host before troubleshooting vLLM.
