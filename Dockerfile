FROM vllm/vllm-openai-rocm:latest

LABEL org.opencontainers.image.title="vLLM ROCm Manager for Unraid"
LABEL org.opencontainers.image.description="Modern WebUI manager for vLLM ROCm on Unraid"
LABEL org.opencontainers.image.source="https://github.com/josenzhong/vllm-rocm"

WORKDIR /app
COPY app/ /app/

ENV VLLM_MANAGER_HOST=0.0.0.0 \
    VLLM_MANAGER_PORT=8080 \
    VLLM_API_HOST=0.0.0.0 \
    VLLM_API_PORT=8000 \
    VLLM_MANAGER_CONFIG=/config/config.json \
    VLLM_MODEL_ROOTS=/models

EXPOSE 8080 8000

ENTRYPOINT ["python3", "/app/server.py"]
