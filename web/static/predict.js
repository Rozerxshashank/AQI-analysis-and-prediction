/* AQI Predictor app (/predict): manual, city, real-day and CSV modes against /api/assess and /api/batch. */
"use strict";
const $ = (s) => document.querySelector(s);
const fmt = (v, d = 0) => (v == null || isNaN(v) ? "–" : Number(v).toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d }));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const TIER = { Good: "#34C759", Satisfactory: "#A3D13C", Moderate: "#F5B82E", Poor: "#FF7A2F", "Very Poor": "#FF3B5C", Severe: "#B23BD6" };
const ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"];
const MODELS = [["sy_rf", "Random Forest · sylearn"], ["sy_dt", "Decision Tree · sylearn"], ["sy_lr", "Linear Regression · sylearn"],
  ["sk_rf", "Random Forest · scikit-learn"], ["sk_dt", "Decision Tree · scikit-learn"], ["sk_lr", "Linear Regression · scikit-learn"], ["xgb", "XGBoost"]];
const ALIAS = { pm25: "PM2.5", "pm2.5": "PM2.5", pm2_5: "PM2.5", pm10: "PM10", no2: "NO2", so2: "SO2", co: "CO", o3: "O3", city: "City", date: "Date" };
async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const d = await r.json().catch(() => ({})); if (!r.ok) throw new Error(typeof d.detail === "string" ? d.detail : "Something went wrong. Check the values and try again."); return d;
}
function toast(m) { const t = $("#toast"); t.textContent = m; t.classList.add("show"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 1700); }
function store(k, v) { try { if (v === undefined) return JSON.parse(localStorage.getItem(k) || "null"); localStorage.setItem(k, JSON.stringify(v)); } catch { return null; } }
function download(name, text, type = "text/csv") { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }

let M = null, mode = "manual", last = null, realDay = null;
const INTRO = {
  manual: "Type the <b>24-hour averages</b> your sensor or report shows. Leave anything you don't have empty; the model fills it with a typical value and tells you.",
  city: "Pick a city to load its <b>typical day</b> (median of 2015–2020), then adjust any value.",
  real: "Load an actual day from one of the <b>6 cities the model never trained on</b>, and see how close it gets to the real AQI.",
};

// ------------------------------------------------------------------ inputs
function buildFields() {
  const box = $("#fields");
  for (const f of M.features) {
    const d = document.createElement("div"); d.className = "num"; d.dataset.f = f.key;
    d.innerHTML = `<label for="in-${f.key}"><b>${f.key}</b><small>safe ≤ ${f.safe}</small></label>
      <div class="wrap"><input id="in-${f.key}" inputmode="decimal" type="number" min="0" step="any" placeholder="–"><span class="unit">${f.unit}</span></div><div class="hint"></div>`;
    const inp = d.querySelector("input");
    inp.addEventListener("input", () => { realDay = null; hint(f.key); });
    inp.addEventListener("keydown", (e) => e.key === "Enter" && check());
    box.appendChild(d);
  }
}
function hint(k) {
  const f = M.features.find((x) => x.key === k), d = document.querySelector(`.num[data-f="${CSS.escape(k)}"]`), raw = d.querySelector("input").value, v = raw === "" ? null : +raw;
  d.classList.remove("over", "over2");
  if (v == null) return (d.querySelector(".hint").textContent = "not measured");
  const r = v / f.safe;
  d.classList.toggle("over", r > 1 && r <= 2); d.classList.toggle("over2", r > 2);
  d.querySelector(".hint").textContent = r <= 1 ? "within the safe limit" : `${fmt(r, 1)}× the safe limit`;
}
const readValues = () => Object.fromEntries(M.features.map((f) => { const v = $(`#in-${CSS.escape(f.key)}`).value; return [f.key, v === "" ? null : +v]; }));
function setValues(vals) { for (const f of M.features) { const v = vals[f.key]; $(`#in-${CSS.escape(f.key)}`).value = v == null ? "" : +(+v).toFixed(2); hint(f.key); } }

function setMode(m) {
  mode = m; document.querySelectorAll("#modes button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
  $("#single").hidden = m === "batch"; $("#batch").hidden = m !== "batch";
  if (m === "batch") return;
  $("#modeIntro").innerHTML = INTRO[m]; $("#cityRow").hidden = m !== "city"; $("#realRow").hidden = m !== "real";
  if (m === "city") loadCity();
}
function loadCity() { const c = M.cities.find((x) => x.name === $("#citySel").value); realDay = null; setValues(c.medians); $("#placeIn").value = `${c.name} · typical day`; check(); }
async function loadReal() {
  const s = await api("/api/sample"); realDay = s; setValues(s.values);
  $("#placeIn").value = `${s.city} · ${new Date(s.date).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}`;
  $("#realInfo").textContent = `${s.city}, ${new Date(s.date).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" })}. Compare the model with the real AQI below.`;
  check(true);
}

// ------------------------------------------------------------------ assess
async function check(keepReal = false) {
  if (!keepReal && mode !== "real") realDay = null;
  $("#err").textContent = "";
  const btn = $("#checkBtn"); btn.disabled = true; btn.textContent = "Checking…";
  try { last = await api("/api/assess", { values: readValues(), model: $("#modelSel").value }); last.place = $("#placeIn").value.trim(); render(last); remember(last); }
  catch (e) { $("#err").textContent = e.message; }
  finally { btn.disabled = false; btn.textContent = "Check air quality"; }
}
let shown = 0, tw = 0;
function render(r) {
  const c = TIER[r.tier.label], card = $("#result");
  card.style.setProperty("--band", c); card.style.setProperty("--glow", 1); document.documentElement.style.setProperty("--band", c);
  $("#empty").hidden = true; $("#full").hidden = false;
  $("#resPlace").textContent = r.place || "Your reading";
  const id = ++tw, from = shown, start = performance.now();
  const step = (now) => { if (id !== tw) return; const t = Math.min(1, (now - start) / 550), e = 1 - Math.pow(1 - t, 4); shown = from + (r.aqi - from) * e; $("#resAqi").textContent = Math.round(shown); if (t < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
  $("#resTier").textContent = r.tier.label;
  $("#resRange").textContent = `${fmt(r.range[0])}–${fmt(r.range[1])}`;
  $("#resModel").textContent = `80% of past predictions by ${r.model_name} fell in a range this wide`;
  $("#resMark").style.left = `${clamp(r.aqi / 500, 0, 1) * 100}%`;
  const st = $("#resStatus");
  if (r.review.length) { st.className = "status warn"; st.innerHTML = `<b>Worth a second look</b><ul>${r.review.map((x) => `<li>${x}</li>`).join("")}</ul>`; }
  else { st.className = "status ok"; st.innerHTML = `<b>Looks reliable</b>All six pollutants given, within the range the model learned from, and not near a tier boundary.`; }
  const act = $("#resActual");
  if (realDay) { act.hidden = false; act.innerHTML = `The real AQI that day was <b>${fmt(realDay.actual)}</b> (${tierOf(realDay.actual)}). The model was off by <b>${fmt(Math.abs(r.aqi - realDay.actual))}</b>, in a city it never trained on.`; }
  else act.hidden = true;
  $("#advHealth").textContent = r.tier.health; $("#advPublic").textContent = r.tier.public; $("#advSensitive").textContent = r.tier.sensitive;
  const items = Object.entries(r.contributions).sort((a, b) => b[1] - a[1]), max = Math.max(...items.map((i) => Math.abs(i[1])), 1);
  $("#drivers").innerHTML = items.map(([k, v]) => `<div class="drv"><b>${k}</b><span class="dbar"><i style="width:${(Math.max(0, v) / max) * 100}%"></i></span><span>${v >= 0 ? "+" : ""}${fmt(v)}</span></div>`).join("") +
    `<p class="drv-note">AQI points each pollutant adds in sylearn's Linear Regression (base ${fmt(r.intercept, 1)}). The biggest bar is the main culprit.</p>`;
  $("#allModels").innerHTML = r.all.map((m) => `<div class="am ${m.key === r.model ? "sel" : ""}"><span>${m.name}</span><b>${fmt(m.aqi)}</b><span class="chip" style="background:${TIER[m.tier]}">${m.tier}</span></div>`).join("");
  if (innerWidth < 960) $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
}
function tierOf(v) { return v <= 50 ? "Good" : v <= 100 ? "Satisfactory" : v <= 200 ? "Moderate" : v <= 300 ? "Poor" : v <= 400 ? "Very Poor" : "Severe"; }
function summary(r) {
  return `${r.place || "Air quality check"}: AQI ${fmt(r.aqi)} (${r.tier.label}), likely ${fmt(r.range[0])}–${fmt(r.range[1])}. ${r.tier.health} Everyone: ${r.tier.public} Sensitive groups: ${r.tier.sensitive} (sylearn · ${r.model_name})`;
}

// ------------------------------------------------------------------ history (this browser only)
function remember(r) {
  const h = (store("aqi-history") || []).filter((x) => JSON.stringify(x.values) !== JSON.stringify(readValues()));
  h.unshift({ place: r.place || "Your reading", aqi: r.aqi, tier: r.tier.label, values: readValues(), at: Date.now() });
  store("aqi-history", h.slice(0, 6)); drawHistory();
}
function drawHistory() {
  const h = store("aqi-history") || []; $("#historyBox").hidden = !h.length;
  $("#history").innerHTML = h.map((x, i) => `<li data-i="${i}"><span>${x.place}</span><b style="color:${TIER[x.tier]}">${fmt(x.aqi)} · ${x.tier}</b></li>`).join("");
  $("#history").querySelectorAll("li").forEach((li) => (li.onclick = () => { const x = h[+li.dataset.i]; setMode("manual"); setValues(x.values); $("#placeIn").value = x.place; check(); }));
}

// ------------------------------------------------------------------ batch CSV
function parseCSV(text) {
  const rows = []; let row = [], cell = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) { if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (ch === '"') q = false; else cell += ch; }
    else if (ch === '"') q = true; else if (ch === ",") { row.push(cell); cell = ""; }
    else if (ch === "\n" || ch === "\r") { if (ch === "\r" && text[i + 1] === "\n") i++; row.push(cell); if (row.some((c) => c.trim() !== "")) rows.push(row); row = []; cell = ""; }
    else cell += ch;
  }
  row.push(cell); if (row.some((c) => c.trim() !== "")) rows.push(row);
  return rows;
}
let batchRows = null, batchRes = null;
async function handleFile(file) {
  $("#batchErr").textContent = ""; $("#batchOut").hidden = true;
  if (!file) return;
  if (!/\.csv$/i.test(file.name)) return ($("#batchErr").textContent = "Please choose a .csv file.");
  if (file.size > 5 * 1024 * 1024) return ($("#batchErr").textContent = "That file is over 5 MB. Split it into smaller files.");
  const grid = parseCSV(await file.text());
  if (grid.length < 2) return ($("#batchErr").textContent = "The file has no data rows.");
  const head = grid[0].map((h) => ALIAS[h.trim().toLowerCase()] || h.trim());
  const keys = M.features.map((f) => f.key);
  if (!head.some((h) => keys.includes(h))) return ($("#batchErr").textContent = `No pollutant columns found. Expected any of: ${keys.join(", ")}.`);
  batchRows = grid.slice(1).map((r) => Object.fromEntries(head.map((h, i) => [h, keys.includes(h) ? (r[i] === undefined || r[i].trim() === "" ? null : r[i].trim()) : (r[i] ?? "").trim()])));
  $("#drop .drop-title").textContent = `Scoring ${fmt(batchRows.length)} rows from ${file.name}…`;
  try { batchRes = await api("/api/batch", { rows: batchRows, model: $("#batchModel").value }); drawBatch(file.name); }
  catch (e) { $("#batchErr").textContent = e.message; }
  finally { $("#drop .drop-title").textContent = "Drop another CSV, or click to choose"; }
}
function drawBatch(name) {
  const r = batchRes, total = r.scored || 1;
  $("#batchOut").hidden = false;
  $("#batchStats").innerHTML = `<div><b>${fmt(r.scored)}</b><span>readings scored</span></div><div><b>${fmt(r.review)}</b><span>worth a second look</span></div><div><b>${fmt(r.skipped)}</b><span>skipped (empty or impossible values)</span></div>`;
  $("#tierBar").innerHTML = ORDER.filter((t) => r.tiers[t]).map((t) => `<div style="flex:${r.tiers[t]};background:${TIER[t]}" title="${t}: ${r.tiers[t]}"></div>`).join("");
  $("#tierLegend").innerHTML = ORDER.filter((t) => r.tiers[t]).map((t) => `<span><i style="background:${TIER[t]}"></i>${t} · ${fmt(r.tiers[t])} (${fmt((r.tiers[t] / total) * 100)}%)</span>`).join("");
  const keys = M.features.map((f) => f.key), extra = ["City", "Date"].filter((k) => batchRows.some((x) => x[k]));
  const show = batchRows.slice(0, 200);
  $("#batchTable").innerHTML = `<thead><tr>${extra.map((k) => `<th>${k}</th>`).join("")}${keys.map((k) => `<th>${k}</th>`).join("")}<th>AQI</th><th>Tier</th><th>Review</th></tr></thead><tbody>` +
    show.map((row, i) => { const o = r.results[i];
      return `<tr>${extra.map((k) => `<td>${row[k] ?? ""}</td>`).join("")}${keys.map((k) => `<td class="n">${row[k] ?? "–"}</td>`).join("")}` +
        (o ? `<td class="n"><b>${fmt(o.aqi)}</b></td><td><span class="chip" style="background:${TIER[o.tier]}">${o.tier}</span></td><td>${o.review ? "⚠︎" : ""}</td>` : `<td class="skip" colspan="3">skipped</td>`) + "</tr>"; }).join("") + "</tbody>";
  $("#batchNote").textContent = `${r.model_name} · showing ${fmt(show.length)} of ${fmt(batchRows.length)} rows · download for all of them.`;
  $("#downloadBtn").onclick = () => {
    const cols = [...extra, ...keys];
    const lines = [[...cols, "predicted_aqi", "tier", "needs_review", "not_measured"].join(",")].concat(batchRows.map((row, i) => { const o = r.results[i];
      return [...cols.map((k) => `"${String(row[k] ?? "").replace(/"/g, '""')}"`), o ? o.aqi : "", o ? o.tier : "skipped", o ? (o.review ? "yes" : "no") : "", o ? `"${o.missing.join(" ")}"` : ""].join(","); }));
    download(`aqi_results_${name.replace(/\.csv$/i, "")}.csv`, lines.join("\n"));
  };
}

// ------------------------------------------------------------------ boot
(async function boot() {
  M = await api("/api/meta");
  buildFields();
  for (const sel of [$("#modelSel"), $("#batchModel")]) sel.innerHTML = MODELS.map(([k, n]) => `<option value="${k}">${n}</option>`).join("");
  $("#citySel").innerHTML = [...M.cities].sort((a, b) => a.name.localeCompare(b.name)).map((c) => `<option>${c.name}</option>`).join("");
  $("#citySel").value = "Delhi"; $("#citySel").onchange = loadCity;
  document.querySelectorAll("#modes button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  $("#checkBtn").onclick = () => check(); $("#realBtn").onclick = loadReal; $("#modelSel").onchange = () => last && check(true);
  $("#clearBtn").onclick = () => { setValues({}); $("#placeIn").value = ""; realDay = null; $("#full").hidden = true; $("#empty").hidden = false; $("#result").style.setProperty("--glow", 0); };
  $("#printBtn").onclick = () => window.print();
  $("#copyBtn").onclick = async () => { try { await navigator.clipboard.writeText(summary(last)); toast("Summary copied"); } catch { toast("Couldn't copy"); } };
  const drop = $("#drop"), fileIn = $("#fileIn");
  drop.onclick = () => fileIn.click(); drop.onkeydown = (e) => (e.key === "Enter" || e.key === " ") && fileIn.click();
  fileIn.onchange = () => handleFile(fileIn.files[0]);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); }; drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); handleFile(e.dataTransfer.files[0]); };
  $("#templateBtn").onclick = () => download("aqi_template.csv", "City,Date,PM2.5,PM10,NO2,SO2,CO,O3\nDelhi,2020-01-15,180.5,290,55.2,15.1,2.1,30.4\nBengaluru,2020-01-15,42,88,28.7,6.3,0.9,35.2\n");
  setValues({}); drawHistory();
  const hash = location.hash.slice(1); if (["manual", "city", "real", "batch"].includes(hash)) setMode(hash); else setMode("manual");
  addEventListener("hashchange", () => { const h = location.hash.slice(1); if (["manual", "city", "real", "batch"].includes(h)) setMode(h); });
})().catch((e) => { $("#err").textContent = `Couldn't reach the model server (${e.message}). Start it with .venv/bin/python web/server.py`; });
