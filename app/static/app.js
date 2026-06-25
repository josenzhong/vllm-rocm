const fields = [
  "model",
  "served_model_name",
  "dtype",
  "gpu_memory_utilization",
  "max_model_len",
  "max_num_batched_tokens",
  "max_num_seqs",
  "kv_cache_dtype",
  "quantization",
  "cpu_offload_gb",
  "swap_space",
  "tensor_parallel_size",
  "pipeline_parallel_size",
  "speculative_config",
  "extra_args",
];

const checkboxes = [
  "enable_prefix_caching",
  "trust_remote_code",
  "disable_log_requests",
  "auto_start",
];

const $ = (id) => document.getElementById(id);
let currentConfig = {};

function setToast(message, kind = "") {
  const toast = $("toast");
  toast.textContent = message;
  toast.className = `toast ${kind}`;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await res.json();
  if (!res.ok || payload.ok === false) {
    throw new Error(payload.message || `Request failed: ${res.status}`);
  }
  return payload;
}

function collectConfig() {
  const cfg = { ...currentConfig };
  for (const id of fields) {
    const el = $(id);
    cfg[id] = el.value;
  }
  for (const id of checkboxes) {
    cfg[id] = $(id).checked;
  }

  for (const key of [
    "port",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "cpu_offload_gb",
    "swap_space",
  ]) {
    if (cfg[key] !== "" && cfg[key] !== undefined) {
      const n = Number(cfg[key]);
      cfg[key] = Number.isFinite(n) ? n : cfg[key];
    }
  }

  if (cfg.gpu_memory_utilization !== "" && cfg.gpu_memory_utilization !== undefined) {
    const n = Number(cfg.gpu_memory_utilization);
    cfg.gpu_memory_utilization = Number.isFinite(n) ? n : cfg.gpu_memory_utilization;
  }
  return cfg;
}

function applyConfig(cfg) {
  currentConfig = cfg || {};
  for (const id of fields) {
    const el = $(id);
    if (el) el.value = currentConfig[id] ?? "";
  }
  for (const id of checkboxes) {
    const el = $(id);
    if (el) el.checked = Boolean(currentConfig[id]);
  }
  if (!$("gpu_memory_utilization").value) $("gpu_memory_utilization").value = "0.9";
  if (!$("tensor_parallel_size").value) $("tensor_parallel_size").value = "1";
  if (!$("pipeline_parallel_size").value) $("pipeline_parallel_size").value = "1";
}

function formatCommand(command, error) {
  if (error) return `Error: ${error}`;
  if (!Array.isArray(command)) return "Command will appear after settings are valid.";
  return command
    .map((part) => (/^[A-Za-z0-9_./:=+-]+$/.test(String(part)) ? String(part) : JSON.stringify(String(part))))
    .join(" ");
}

async function refreshStatus() {
  const status = await api("/api/status");
  $("runningStatus").textContent = status.running ? "Running" : "Stopped";
  $("pidStatus").textContent = `PID: ${status.pid ?? "-"}`;
  $("uptimeStatus").textContent = status.uptime_seconds ? `Uptime: ${status.uptime_seconds}s` : "Uptime: -";
  $("apiStatus").textContent = status.api_base || "-";
  $("modelStatus").textContent = $("model").value || "-";
  $("commandPreview").textContent = formatCommand(status.command, status.command_error);
}

async function refreshLogs() {
  const payload = await api("/api/logs");
  $("logs").textContent = payload.logs?.join("\n") || "No logs yet.";
  $("logs").scrollTop = $("logs").scrollHeight;
}

async function loadConfig() {
  const cfg = await api("/api/config");
  applyConfig(cfg);
  await refreshStatus();
}

async function saveConfig() {
  const cfg = collectConfig();
  const payload = await api("/api/config", { method: "POST", body: JSON.stringify(cfg) });
  currentConfig = payload.config;
  $("commandPreview").textContent = formatCommand(payload.command);
  setToast("Settings saved.", "good");
  await refreshStatus();
}

async function start() {
  const cfg = collectConfig();
  const payload = await api("/api/start", { method: "POST", body: JSON.stringify(cfg) });
  setToast(payload.message, "good");
  await refreshStatus();
  await refreshLogs();
}

async function stop() {
  const payload = await api("/api/stop", { method: "POST", body: "{}" });
  setToast(payload.message, "good");
  await refreshStatus();
  await refreshLogs();
}

async function restart() {
  const cfg = collectConfig();
  const payload = await api("/api/restart", { method: "POST", body: JSON.stringify(cfg) });
  setToast(payload.message, "good");
  await refreshStatus();
  await refreshLogs();
}

async function scanModels() {
  const select = $("modelSelect");
  select.innerHTML = "<option value=''>Scanning...</option>";
  const payload = await api("/api/models");
  const models = payload.models || [];
  select.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = models.length ? "Choose a local model..." : "No local models found";
  select.appendChild(placeholder);
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.path;
    option.textContent = `${model.name} — ${model.path}`;
    select.appendChild(option);
  }
  setToast(`Found ${models.length} local model folder(s).`, "good");
}

function initTheme() {
  const saved = localStorage.getItem("theme");
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = saved || (prefersDark ? "dark" : "light");
  $("themeToggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("theme", next);
  });
}

function wireEvents() {
  $("saveBtn").addEventListener("click", () => saveConfig().catch((e) => setToast(e.message, "bad")));
  $("startBtn").addEventListener("click", () => start().catch((e) => setToast(e.message, "bad")));
  $("stopBtn").addEventListener("click", () => stop().catch((e) => setToast(e.message, "bad")));
  $("restartBtn").addEventListener("click", () => restart().catch((e) => setToast(e.message, "bad")));
  $("refreshLogs").addEventListener("click", () => refreshLogs().catch((e) => setToast(e.message, "bad")));
  $("refreshModels").addEventListener("click", () => scanModels().catch((e) => setToast(e.message, "bad")));
  $("modelSelect").addEventListener("change", (event) => {
    if (event.target.value) {
      $("model").value = event.target.value;
      refreshStatus().catch(() => {});
    }
  });
  $("copyCommand").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("commandPreview").textContent);
      setToast("Command copied.", "good");
    } catch {
      setToast("Copy failed; select the command manually.", "bad");
    }
  });
  for (const id of [...fields, ...checkboxes]) {
    const el = $(id);
    if (el) el.addEventListener("input", () => refreshStatus().catch(() => {}));
  }
}

async function boot() {
  initTheme();
  wireEvents();
  await loadConfig();
  await scanModels().catch(() => {});
  await refreshLogs();
  setInterval(() => refreshStatus().catch(() => {}), 4000);
  setInterval(() => refreshLogs().catch(() => {}), 5000);
}

boot().catch((error) => setToast(error.message, "bad"));
