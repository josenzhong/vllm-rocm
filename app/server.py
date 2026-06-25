#!/usr/bin/env python3
"""vLLM ROCm Manager for Unraid.

A small WebUI/controller that launches vLLM as a subprocess and can search/download
models from Hugging Face Hub into a local /models mount.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.parse
from collections import deque
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import HfApi, snapshot_download
except Exception:  # noqa: BLE001
    HfApi = None  # type: ignore[assignment]
    snapshot_download = None  # type: ignore[assignment]

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
CONFIG_PATH = Path(os.environ.get("VLLM_MANAGER_CONFIG", "/config/config.json"))
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
DEFAULT_MODEL = "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive"
LOG_LIMIT = 1000
LOGS: deque[str] = deque(maxlen=LOG_LIMIT)
PROCESS: subprocess.Popen[str] | None = None
PROCESS_LOCK = threading.RLock()
DOWNLOAD_LOCK = threading.RLock()
DOWNLOADS: dict[str, dict[str, Any]] = {}
STARTED_AT: float | None = None


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def append_log(line: str) -> None:
    LOGS.append(f"[{now()}] {line.rstrip()}")


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def model_roots() -> list[Path]:
    return [Path(p) for p in os.environ.get("VLLM_MODEL_ROOTS", "/models").split(":") if p]


def primary_model_root() -> Path:
    roots = model_roots()
    root = roots[0] if roots else Path("/models")
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_model_dir_name(repo_id: str, override: str | None = None) -> str:
    value = (override or repo_id).strip().strip("/")
    value = value.replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = value.strip(".-_")
    if not value:
        raise ValueError("Invalid model directory name")
    return value[:180]


def default_config() -> dict[str, Any]:
    return {
        "model": DEFAULT_MODEL,
        "served_model_name": "",
        "host": os.environ.get("VLLM_API_HOST", "0.0.0.0"),
        "port": int(os.environ.get("VLLM_API_PORT", "8000")),
        "dtype": "auto",
        "gpu_memory_utilization": 0.90,
        "max_model_len": "",
        "max_num_batched_tokens": "",
        "max_num_seqs": "",
        "kv_cache_dtype": "auto",
        "quantization": "",
        "enable_prefix_caching": False,
        "cpu_offload_gb": "",
        "swap_space": "",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "trust_remote_code": False,
        "disable_log_requests": True,
        "speculative_config": "",
        "extra_args": "",
        "auto_start": False,
    }


def load_config() -> dict[str, Any]:
    cfg = default_config()
    if CONFIG_PATH.exists():
        try:
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cfg.update(loaded)
        except Exception as exc:  # noqa: BLE001
            append_log(f"Failed to load config: {exc}")
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")


def split_extra_args(value: str) -> list[str]:
    import shlex

    if not value.strip():
        return []
    return shlex.split(value)


def add_optional_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    command.extend([flag, str(value)])


def build_command(cfg: dict[str, Any]) -> list[str]:
    model = str(cfg.get("model") or "").strip()
    if not model:
        raise ValueError("Model is required")

    command = ["vllm", "serve", model]

    add_optional_arg(command, "--host", cfg.get("host") or "0.0.0.0")
    add_optional_arg(command, "--port", cfg.get("port") or 8000)

    served_name = str(cfg.get("served_model_name") or "").strip()
    if served_name:
        command.extend(["--served-model-name", served_name])

    dtype = str(cfg.get("dtype") or "auto").strip()
    if dtype and dtype != "default":
        command.extend(["--dtype", dtype])

    add_optional_arg(command, "--gpu-memory-utilization", cfg.get("gpu_memory_utilization"))
    add_optional_arg(command, "--max-model-len", cfg.get("max_model_len"))
    add_optional_arg(command, "--max-num-batched-tokens", cfg.get("max_num_batched_tokens"))
    add_optional_arg(command, "--max-num-seqs", cfg.get("max_num_seqs"))

    kv_dtype = str(cfg.get("kv_cache_dtype") or "").strip()
    if kv_dtype and kv_dtype != "auto":
        command.extend(["--kv-cache-dtype", kv_dtype])

    add_optional_arg(command, "--quantization", cfg.get("quantization"))
    add_optional_arg(command, "--cpu-offload-gb", cfg.get("cpu_offload_gb"))
    add_optional_arg(command, "--swap-space", cfg.get("swap_space"))
    add_optional_arg(command, "--tensor-parallel-size", cfg.get("tensor_parallel_size"))
    add_optional_arg(command, "--pipeline-parallel-size", cfg.get("pipeline_parallel_size"))

    if cfg.get("enable_prefix_caching"):
        command.append("--enable-prefix-caching")
    if cfg.get("trust_remote_code"):
        command.append("--trust-remote-code")
    if cfg.get("disable_log_requests"):
        command.append("--disable-log-requests")

    speculative_config = str(cfg.get("speculative_config") or "").strip()
    if speculative_config:
        json.loads(speculative_config)
        command.extend(["--speculative-config", speculative_config])

    command.extend(split_extra_args(str(cfg.get("extra_args") or "")))
    return command


def process_reader(proc: subprocess.Popen[str]) -> None:
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            append_log(line)
    except Exception as exc:  # noqa: BLE001
        append_log(f"Log reader stopped: {exc}")


def is_running() -> bool:
    with PROCESS_LOCK:
        return PROCESS is not None and PROCESS.poll() is None


def start_vllm(cfg: dict[str, Any]) -> dict[str, Any]:
    global PROCESS, STARTED_AT
    with PROCESS_LOCK:
        if PROCESS is not None and PROCESS.poll() is None:
            return {"ok": False, "message": "vLLM is already running"}
        command = build_command(cfg)
        save_config(cfg)
        append_log("Starting vLLM: " + " ".join(command))
        PROCESS = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        STARTED_AT = time.time()
        threading.Thread(target=process_reader, args=(PROCESS,), daemon=True).start()
        return {"ok": True, "message": "vLLM started", "command": command}


def stop_vllm() -> dict[str, Any]:
    global PROCESS, STARTED_AT
    with PROCESS_LOCK:
        if PROCESS is None or PROCESS.poll() is not None:
            PROCESS = None
            STARTED_AT = None
            return {"ok": True, "message": "vLLM is already stopped"}
        append_log("Stopping vLLM")
        try:
            PROCESS.send_signal(signal.SIGTERM)
            PROCESS.wait(timeout=20)
        except subprocess.TimeoutExpired:
            append_log("vLLM did not stop after SIGTERM; killing")
            PROCESS.kill()
            PROCESS.wait(timeout=10)
        finally:
            PROCESS = None
            STARTED_AT = None
        return {"ok": True, "message": "vLLM stopped"}


def scan_models() -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for root in model_roots():
        if not root.exists():
            continue
        for config_file in root.rglob("config.json"):
            model_dir = config_file.parent
            path = str(model_dir)
            if path in seen:
                continue
            seen.add(path)
            rel = path
            try:
                rel = str(model_dir.relative_to(root))
            except ValueError:
                pass
            results.append({"name": rel if rel != "." else model_dir.name, "path": path})
            if len(results) >= 500:
                break
    return sorted(results, key=lambda item: item["name"].lower())


def hf_search(query: str, limit: int = 20) -> list[dict[str, Any]]:
    query = query.strip()
    if not query:
        return []
    if HfApi is None:
        raise RuntimeError("huggingface_hub is not installed in this image")
    api = HfApi(token=hf_token())
    models = api.list_models(search=query, limit=min(max(limit, 1), 50), full=False)
    results: list[dict[str, Any]] = []
    for model in models:
        model_id = getattr(model, "modelId", None) or getattr(model, "id", None)
        if not model_id:
            continue
        results.append(
            {
                "id": model_id,
                "author": getattr(model, "author", "") or "",
                "downloads": getattr(model, "downloads", None),
                "likes": getattr(model, "likes", None),
                "tags": list(getattr(model, "tags", []) or [])[:12],
                "pipeline_tag": getattr(model, "pipeline_tag", None),
            }
        )
    return results


def download_worker(download_id: str, repo_id: str, target: Path, revision: str | None) -> None:
    with DOWNLOAD_LOCK:
        DOWNLOADS[download_id].update({"state": "running", "message": "Downloading", "started_at": now()})
    append_log(f"Downloading Hugging Face model {repo_id} to {target}")
    try:
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub is not installed in this image")
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision or None,
            local_dir=str(target),
            token=hf_token(),
            resume_download=True,
        )
        with DOWNLOAD_LOCK:
            DOWNLOADS[download_id].update(
                {
                    "state": "completed",
                    "message": "Download completed",
                    "completed_at": now(),
                    "path": str(target),
                }
            )
        append_log(f"Completed download for {repo_id}")
    except Exception as exc:  # noqa: BLE001
        with DOWNLOAD_LOCK:
            DOWNLOADS[download_id].update({"state": "failed", "message": str(exc), "completed_at": now()})
        append_log(f"Download failed for {repo_id}: {exc}")


def start_download(payload: dict[str, Any]) -> dict[str, Any]:
    repo_id = str(payload.get("repo_id") or payload.get("id") or "").strip()
    if not repo_id or "/" not in repo_id:
        raise ValueError("A Hugging Face model repo_id like organization/model is required")
    revision = str(payload.get("revision") or "").strip() or None
    dir_name = safe_model_dir_name(repo_id, str(payload.get("local_dir") or "").strip() or None)
    target = primary_model_root() / dir_name
    try:
        target.resolve().relative_to(primary_model_root().resolve())
    except ValueError as exc:
        raise ValueError("Download target must stay inside the configured /models root") from exc

    download_id = f"{safe_model_dir_name(repo_id)}-{int(time.time())}"
    with DOWNLOAD_LOCK:
        DOWNLOADS[download_id] = {
            "id": download_id,
            "repo_id": repo_id,
            "revision": revision or "main",
            "target": str(target),
            "state": "queued",
            "message": "Queued",
            "created_at": now(),
        }
    threading.Thread(target=download_worker, args=(download_id, repo_id, target, revision), daemon=True).start()
    return DOWNLOADS[download_id]


def downloads_payload() -> list[dict[str, Any]]:
    with DOWNLOAD_LOCK:
        return sorted(DOWNLOADS.values(), key=lambda item: item.get("created_at", ""), reverse=True)


def status_payload() -> dict[str, Any]:
    running = is_running()
    exit_code = None
    pid = None
    uptime = None
    with PROCESS_LOCK:
        if PROCESS is not None:
            pid = PROCESS.pid
            exit_code = PROCESS.poll()
        if running and STARTED_AT:
            uptime = int(time.time() - STARTED_AT)
    cfg = load_config()
    command: list[str] | None = None
    command_error = None
    try:
        command = build_command(cfg)
    except Exception as exc:  # noqa: BLE001
        command_error = str(exc)
    return {
        "running": running,
        "pid": pid,
        "exit_code": exit_code,
        "uptime_seconds": uptime,
        "config_path": str(CONFIG_PATH),
        "api_base": f"http://{cfg.get('host', '0.0.0.0')}:{cfg.get('port', 8000)}/v1",
        "command": command,
        "command_error": command_error,
        "downloads": downloads_payload(),
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        append_log("manager: " + (fmt % args))

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        data = self.rfile.read(length)
        parsed = json.loads(data.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("JSON object expected")
        return parsed

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/status":
            self.send_json(status_payload())
            return
        if parsed.path == "/api/config":
            self.send_json(load_config())
            return
        if parsed.path == "/api/models":
            self.send_json({"models": scan_models()})
            return
        if parsed.path == "/api/hf/search":
            params = urllib.parse.parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            limit = int(params.get("limit", ["20"])[0])
            self.send_json({"models": hf_search(query, limit=limit)})
            return
        if parsed.path == "/api/downloads":
            self.send_json({"downloads": downloads_payload()})
            return
        if parsed.path == "/api/logs":
            self.send_json({"logs": list(LOGS)})
            return
        if parsed.path == "/healthz":
            self.send_json({"ok": True, "running": is_running()})
            return
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/api/config":
                cfg = default_config()
                cfg.update(self.read_json())
                build_command(cfg)
                save_config(cfg)
                self.send_json({"ok": True, "config": cfg, "command": build_command(cfg)})
                return
            if parsed.path == "/api/start":
                cfg = default_config()
                cfg.update(self.read_json() or load_config())
                self.send_json(start_vllm(cfg))
                return
            if parsed.path == "/api/stop":
                self.send_json(stop_vllm())
                return
            if parsed.path == "/api/restart":
                cfg = default_config()
                cfg.update(self.read_json() or load_config())
                stop_vllm()
                self.send_json(start_vllm(cfg))
                return
            if parsed.path == "/api/hf/download":
                self.send_json({"ok": True, "download": start_download(self.read_json())})
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:  # noqa: BLE001
            append_log(f"API error: {exc}")
            self.send_json({"ok": False, "message": str(exc)}, status=400)


def main() -> None:
    cfg = load_config()
    host = os.environ.get("VLLM_MANAGER_HOST", "0.0.0.0")
    port = int(os.environ.get("VLLM_MANAGER_PORT", "8080"))
    append_log(f"vLLM ROCm Manager starting on {host}:{port}")
    append_log(f"Config path: {CONFIG_PATH}")
    append_log(f"Primary model download root: {primary_model_root()}")
    if cfg.get("auto_start"):
        try:
            start_vllm(cfg)
        except Exception as exc:  # noqa: BLE001
            append_log(f"Auto-start failed: {exc}")
    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        stop_vllm()


if __name__ == "__main__":
    main()
