# vLLM ROCm Manager for Unraid

This project builds a custom Unraid-friendly Docker image based on the official `vllm/vllm-openai-rocm:latest` image. It adds a lightweight modern WebUI for selecting models, configuring vLLM launch settings, starting/stopping/restarting the vLLM process, and viewing logs.

## Ports

- `8080`: vLLM ROCm Manager WebUI
- `8000`: vLLM OpenAI-compatible API, available after vLLM is started from the WebUI

## What this app does

The manager WebUI lets you:

- Scan local models mounted under `/models`
- Select a local Hugging Face-format model folder containing `config.json`
- Enter a Hugging Face model ID such as `Qwen/Qwen3-0.6B`
- Configure common and advanced vLLM settings
- Preview the generated `vllm serve` command
- Save configuration under `/config/config.json`
- Start, stop, and restart the vLLM subprocess
- View manager and vLLM logs
- Use light or dark mode

## Requirements

- AMD ROCm-compatible GPU
- `/dev/kfd` available on the Unraid host
- `/dev/dri` available on the Unraid host
- Radeon Top or equivalent AMD GPU support configured on Unraid
- Enough VRAM for the selected model

Consumer Radeon GPU compatibility can depend on the ROCm/vLLM image version and selected model. The template follows the official vLLM ROCm Docker run pattern, but it cannot guarantee every AMD GPU/model combination will run.

## Paths

Default Unraid mappings:

```text
/mnt/user/appdata/vllm/config -> /config
/mnt/user/appdata/vllm/cache  -> /root/.cache/huggingface
/mnt/user/appdata/vllm/models -> /models
```

Put local Hugging Face-format models inside:

```text
/mnt/user/appdata/vllm/models
```

For example:

```text
/mnt/user/appdata/vllm/models/Qwen2.5-7B-Instruct/config.json
```

The WebUI will scan for directories containing `config.json` and offer them in the local model picker.

## Default model

The default launch profile uses:

```text
Qwen/Qwen3-0.6B
```

This is intended as a small smoke-test model. After confirming the container and GPU mapping work, change the model and advanced settings in the WebUI.

## Advanced settings

The first manager version exposes:

- dtype
- GPU memory utilization
- max model length
- tensor parallel size
- pipeline parallel size
- max batched tokens
- max concurrent sequences
- KV cache dtype, including fp8 and int8 options exposed by vLLM
- model quantization field
- prefix caching
- CPU offload GB
- swap space GB
- trust remote code
- disable request logs
- speculative/MTP config JSON
- extra raw vLLM args

Most vLLM engine settings require restarting vLLM. Use the WebUI **Restart** button after changing model or engine settings.

## MTP / speculative decoding

Use the **Speculative / MTP config JSON** box for vLLM's `--speculative-config` value.

Example:

```json
{"method":"mtp","num_speculative_tokens":1}
```

MTP only works with models supported by vLLM for MTP. For unsupported models, use another speculative decoding method supported by your vLLM version or leave the field blank.

## API endpoint

After vLLM starts, the OpenAI-compatible API is:

```text
http://UNRAID_IP:8000/v1
```

Test with:

```bash
curl http://UNRAID_IP:8000/v1/models
```

Use this endpoint with Open WebUI, LiteLLM, Lobe Chat, or other OpenAI-compatible clients.

## Troubleshooting

If the container cannot see the GPU, check that the host exposes the AMD device nodes:

```bash
ls -l /dev/kfd
ls -l /dev/dri
ls -l /dev/dri/render*
```

If `/dev/kfd` is missing, fix AMD GPU support on the Unraid host before troubleshooting vLLM.

If the WebUI starts but vLLM fails, open the WebUI logs panel. Common causes include an unsupported model, insufficient VRAM, incompatible quantization settings, invalid speculative config JSON, or ROCm support issues for the selected GPU/model combination.
