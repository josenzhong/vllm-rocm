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
let configMode = "simple";

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

function setConfigMode(mode) {
  configMode = mode === "advanced" ? "advanced" : "simple";
  const form = $("settingsForm");
  const simpleBtn = $("simpleModeBtn");
  const advancedBtn = $("advancedModeBtn");
  const isAdvanced = configMode === "advanced";

  if (form) {
    form.classList.toggle("simple-mode", !isAdvanced);
    form.classList.toggle("advanced-mode", isAdvanced);
  }
  for (const el of document.querySelectorAll(".advanced-config")) {
    el.hidden = !isAdvanced;
  }
  for (const el of document.querySelectorAll(".simple-config")) {
    el.hidden = false;
  }
  if (simpleBtn && advancedBtn) {
    simpleBtn.classList.toggle("active", !isAdvanced);
    advancedBtn.classList.toggle("active", isAdvanced);
    simpleBtn.setAttribute("aria-selected", String(!isAdvanced));
    advancedBtn.setAttribute("aria-selected", String(isAdvanced));
  }
  const help = $("modeHelp");
  if (help) {
    help.textContent = isAdvanced
      ? "Advanced mode shows every vLLM tuning option, including KV cache, quantization, offload, batching, and MTP/speculative JSON."
      : "Simple mode shows the common model, memory, dtype, download, and startup controls.";
  }
  localStorage.setItem("configMode", configMode);
}

function initConfigMode() {
  const saved = localStorage.getItem("configMode") || "simple";
  setConfigMode(saved);
}

async function refreshStatus() {
  const status = await api("/api/status");
  $("runningStatus").textContent = status.running ? "Running" : "Stopped";
  $("pidStatus").textContent = `PID: ${status.pid ?? "-"}`;
  $("uptimeStatus").textContent = status.uptime_seconds ? `Uptime: ${status.uptime_seconds}s` : "Uptime: -";
  $("apiStatus").textContent = status.api_base || "-";
  $("modelStatus").textContent = $("model").value || "-";
  $("commandPreview").textContent = formatCommand(status.command, status.command_error);
  renderDownloads(status.downloads || []);
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

function renderHfResults(models) {
  const container = $("hfResults");
  container.innerHTML = "";
  if (!models.length) {
    container.innerHTML = "<div class='hub-result'>No matching models found.</div>";
    return;
  }
  for (const model of models) {
    const card = document.createElement("article");
    card.className = "hub-result";
    const tags = (model.tags || []).slice(0, 5).map((tag) => `<span class='chip'>${escapeHtml(tag)}</span>`).join("");
    card.innerHTML = `
      <div class="hub-result-main">
        <div>
          <strong>${escapeHtml(model.id)}</strong>
          <div class="hub-meta">
            <span>${model.downloads ?? 0} downloads</span>
            <span>${model.likes ?? 0} likes</span>
            ${model.pipeline_tag ? `<span>${escapeHtml(model.pipeline_tag)}</span>` : ""}
          </div>
        </div>
        <div class="button-row">
          <button class="secondary use-hf" type="button">Use ID</button>
          <button class="primary download-hf" type="button">Download</button>
        </div>
      </div>
      <div class="hub-meta">${tags}</div>
    `;
    card.querySelector(".use-hf").addEventListener("click", () => {
      $("model").value = model.id;
      $("hfRepoId").value = model.id;
      refreshStatus().catch(() => {});
    });
    card.querySelector(".download-hf").addEventListener("click", () => {
      $("hfRepoId").value = model.id;
      downloadHfModel().catch((error) => setToast(error.message, "bad"));
    });
    container.appendChild(card);
  }
}

async function searchHfModels() {
  const query = $("hfQuery").value.trim();
  if (!query) throw new Error("Enter a search query first.");
  setToast("Searching Hugging Face...", "");
  const payload = await api(`/api/hf/search?q=${encodeURIComponent(query)}&limit=20`);
  renderHfResults(payload.models || []);
  setToast(`Found ${(payload.models || []).length} model(s).`, "good");
}

async function downloadHfModel() {
  const repoId = $("hfRepoId").value.trim();
  if (!repoId) throw new Error("Enter a Hugging Face repository ID first.");
  const payload = await api("/api/hf/download", {
    method: "POST",
    body: JSON.stringify({
      repo_id: repoId,
      revision: $("hfRevision").value.trim(),
      local_dir: $("hfLocalDir").value.trim(),
    }),
  });
  setToast(`Download queued for ${payload.download.repo_id}.`, "good");
  await refreshDownloads();
}

async function refreshDownloads() {
  const payload = await api("/api/downloads");
  renderDownloads(payload.downloads || []);
}

function renderDownloads(downloads) {
  const container = $("downloadStatus");
  if (!container) return;
  container.innerHTML = "";
  if (!downloads.length) return;
  for (const item of downloads.slice(0, 6)) {
    const card = document.createElement("article");
    card.className = "download-item";
    card.innerHTML = `
      <strong>${escapeHtml(item.repo_id)}</strong>
      <div class="hub-meta">
        <span class="chip">${escapeHtml(item.state)}</span>
        <span>${escapeHtml(item.message || "")}</span>
      </div>
      <small>${escapeHtml(item.target || item.path || "")}</small>
    `;
    if (item.state === "completed" && item.path) {
      const button = document.createElement("button");
      button.className = "secondary";
      button.type = "button";
      button.textContent = "Use downloaded path";
      button.addEventListener("click", () => {
        $("model").value = item.path;
        refreshStatus().catch(() => {});
      });
      card.appendChild(button);
    }
    container.appendChild(card);
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
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
  $("hfSearchBtn").addEventListener("click", () => searchHfModels().catch((e) => setToast(e.message, "bad")));
  $("hfDownloadBtn").addEventListener("click", () => downloadHfModel().catch((e) => setToast(e.message, "bad")));
  $("simpleModeBtn")?.addEventListener("click", () => setConfigMode("simple"));
  $("advancedModeBtn")?.addEventListener("click", () => setConfigMode("advanced"));
  $("hfQuery").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      searchHfModels().catch((e) => setToast(e.message, "bad"));
    }
  });
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
  initConfigMode();
  await loadConfig();
  $("hfQuery").value = "Gemma";
  $("hfRepoId").value = $("model").value.includes("/") ? $("model").value : "";
  await scanModels().catch(() => {});
  await refreshDownloads().catch(() => {});
  await refreshLogs();
  setInterval(() => refreshStatus().catch(() => {}), 4000);
  setInterval(() => refreshDownloads().catch(() => {}), 4000);
  setInterval(() => refreshLogs().catch(() => {}), 5000);
}

boot().catch((error) => setToast(error.message, "bad"));
