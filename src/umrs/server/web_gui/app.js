const api = "/umrs/web/dashboard";
const jobsBody = document.querySelector("#jobs");
const modelsBody = document.querySelector("#models");
const detail = document.querySelector("#job-detail");

const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[char]));

async function getJson(path) {
  const response = await fetch(`${api}${path}`, {headers: {Accept: "application/json"}});
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function loadHealth() {
  const node = document.querySelector("#health");
  try {
    const data = await getJson("/health");
    node.textContent = `${data.status.toUpperCase()} · uptime ${Math.round(data.uptime_seconds)}s`;
  } catch (error) {
    node.textContent = `UNAVAILABLE · ${error.message}`;
    node.className = "error";
  }
}

async function loadJobs() {
  const status = document.querySelector("#job-status").value;
  try {
    const data = await getJson(`/jobs${status ? `?status=${encodeURIComponent(status)}` : ""}`);
    jobsBody.innerHTML = data.items.length ? data.items.map((job) => `<tr data-id="${escapeHtml(job.job_id)}"><td>${escapeHtml(job.job_id)}</td><td>${escapeHtml(job.job_type)}</td><td class="status">${escapeHtml(job.status)}</td><td>${job.ttl_seconds ?? "immediate"}</td><td>${escapeHtml(job.updated_at ?? "")}</td></tr>`).join("") : '<tr><td colspan="5">No jobs found</td></tr>';
  } catch (error) {
    jobsBody.innerHTML = `<tr><td class="error" colspan="5">${escapeHtml(error.message)}</td></tr>`;
  }
}

async function loadModels() {
  try {
    const data = await getJson("/models");
    modelsBody.innerHTML = data.items.length ? data.items.map((model) => `<tr><td>${escapeHtml(model.name)}</td><td>${escapeHtml(model.model_type)}</td><td class="status">${escapeHtml(model.status)}</td><td>${escapeHtml(model.latest_version ?? model.version ?? "-")}</td></tr>`).join("") : '<tr><td colspan="4">No models found</td></tr>';
  } catch (error) {
    modelsBody.innerHTML = `<tr><td class="error" colspan="4">${escapeHtml(error.message)}</td></tr>`;
  }
}

document.querySelector("#job-status").addEventListener("change", loadJobs);
jobsBody.addEventListener("click", async (event) => {
  const row = event.target.closest("tr[data-id]");
  if (!row) return;
  try {
    detail.textContent = JSON.stringify(await getJson(`/jobs/${encodeURIComponent(row.dataset.id)}`), null, 2);
    detail.hidden = false;
  } catch (error) {
    detail.textContent = error.message;
    detail.hidden = false;
  }
});

Promise.all([loadHealth(), loadJobs(), loadModels()]);
setInterval(() => { loadHealth(); loadJobs(); }, 15000);
