# vLLM ROCm Unraid Community App

This repository contains an Unraid Community Applications Docker template for running the official vLLM ROCm OpenAI-compatible API server on AMD GPUs.

## Included app

- **vLLM ROCm**: runs `vllm/vllm-openai-rocm:latest` with AMD ROCm device mappings for `/dev/kfd` and `/dev/dri`.

## Repository layout

```text
ca_profile.xml
templates/vllm-rocm.xml
docs/vllm-rocm.md
icon.svg
LICENSE
```

## Requirements

- Unraid host with AMD GPU support exposed as `/dev/kfd` and `/dev/dri`
- ROCm-capable AMD GPU
- Enough VRAM for the selected model
- Optional Hugging Face token for gated/private models

## Usage

The template defaults to a small smoke-test model:

```text
--model Qwen/Qwen3-0.6B
```

After confirming the container starts, edit **Model Arguments** to use your preferred model and runtime settings.

The vLLM API endpoint is:

```text
http://UNRAID_IP:8000/v1
```

vLLM is an API server, not a full browser chat UI. For chat, connect Open WebUI or another OpenAI-compatible frontend to the `/v1` endpoint.

## Submission notes

After meaningful XML changes, use the Community Applications submission page to run **Validate** and then **Scan** before submitting for review.

## Support

Please open an issue in this repository for template-specific problems. For vLLM runtime issues, also check the upstream vLLM project documentation and issue tracker.
