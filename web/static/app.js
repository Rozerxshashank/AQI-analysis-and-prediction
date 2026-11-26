/* sylearn launch page: scroll-driven chapters + a live predictor. Every number comes from /api (the real models). */
"use strict";
const $ = (s) => document.querySelector(s);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const fmt = (v, d = 0) => (v == null || isNaN(v) ? "–" : Number(v).toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d }));
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";
const sci = (v) => { const [m, e] = v.toExponential(1).split("e"); return `${m} × 10${+e < 0 ? "⁻" : ""}${[...String(Math.abs(+e))].map((c) => SUP[c]).join("")}`; };
async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const d = await r.json().catch(() => ({})); if (!r.ok) throw new Error(d.detail || r.status); return d;
}
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 1600); }
async function copyPip() { try { await navigator.clipboard.writeText("pip install sylearn"); toast("Copied: pip install sylearn"); } catch { toast("pip install sylearn"); } }

let M = null;                                   // /api/meta
const band = (v) => M.bands.find((b) => b.max == null || v <= b.max);
const TIER_COLORS = { Good: "#34C759", Satisfactory: "#A3D13C", Moderate: "#F5B82E", Poor: "#FF7A2F", "Very Poor": "#FF3B5C", Severe: "#B23BD6" };
const tierColor = (v) => TIER_COLORS[band(v).name];

// ------------------------------------------------------------------ scroll engine
const scenes = [];
function scene(el, update) { scenes.push({ el, update }); }
let ticking = false;
function onScroll() {
  if (ticking) return; ticking = true;
  requestAnimationFrame(() => {
    ticking = false;
    for (const s of scenes) {
      const r = s.el.getBoundingClientRect(), span = Math.max(1, r.height - innerHeight);
      if (r.bottom < -innerHeight || r.top > innerHeight * 2) continue;
      s.update(REDUCED ? 1 : clamp(-r.top / span, 0, 1));
    }
  });
}
addEventListener("scroll", onScroll, { passive: true });
addEventListener("resize", () => { sizeCanvas(); onScroll(); });

// ------------------------------------------------------------------ 1 · hero device
function heroScene(el) {
  scene(el, (p) => {
    const t = smooth(0, 0.55, p);
    $("#device").style.setProperty("--rx", `${lerp(28, 0, t)}deg`);
    $("#device").style.setProperty("--sc", lerp(0.9, 1.04, t));
  });
}
function fillDevice(pred, values) {
  const v = pred.predictions.sy_rf, b = band(v), c = tierColor(v);
  document.documentElement.style.setProperty("--band", c);
  $("#devAqi").textContent = fmt(v); $("#devTier").textContent = b.name;
  $("#devMark").style.left = `${clamp(v / 500, 0, 1) * 100}%`;
  $("#devRows").innerHTML = M.features.map((f) => `<div><b>${fmt(values[f.key], values[f.key] < 10 ? 2 : 0)}</b>${f.key} · ${f.unit}</div>`).join("");
}

// ------------------------------------------------------------------ 2 · the problem
function countScene(el, node, to, dec = 0) {
  scene(el, (p) => { node.textContent = fmt(to * smooth(0.05, 0.55, p), dec); node.style.color = p > 0.5 ? tierColor(to) : ""; });
}

// ------------------------------------------------------------------ 3 · learns every day
function learnScene(el) {
  const st = M.stream, svg = $("#learnChart");
  $("#usNum").textContent = `${fmt(st.sylearn_us_per_day)} µs`;
  scene(el, (p) => {
    const t = smooth(0.04, 0.7, p), i = Math.round(t * (st.points.length - 1)), pt = st.points[i];
    $("#dayNum").textContent = fmt(pt.day);
    drawLearn(svg, i);
    const punch = smooth(0.74, 0.86, p);
    $("#learnPunch").style.opacity = punch;
    $(".learn-stage").style.opacity = 1 - punch; $("#learnTitle").style.opacity = 1 - punch;
  });
}
function drawLearn(svg, upto) {
  const st = M.stream, w = svg.clientWidth || 900, h = svg.clientHeight || 260, pad = 6, yLo = 0.4, yHi = 0.92;
  const x = (d) => pad + (d / st.days) * (w - 2 * pad), y = (v) => h - pad - ((clamp(v, yLo, yHi) - yLo) / (yHi - yLo)) * (h - 2 * pad);
  const pts = st.points.slice(0, upto + 1), line = (k) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.day).toFixed(1)},${y(p[k]).toFixed(1)}`).join(" ");
  const last = pts[pts.length - 1];
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.innerHTML = `
    <line x1="${pad}" x2="${w - pad}" y1="${y(st.full.r2)}" y2="${y(st.full.r2)}" stroke="rgba(255,255,255,.35)" stroke-dasharray="4 6"/>
    <text x="${w - pad}" y="${y(st.full.r2) - 10}" text-anchor="end" fill="#86868B" font-size="13" font-family="Inter">full retraining · R² ${fmt(st.full.r2, 3)}</text>
    <path d="${line("r2_sgd")}" fill="none" stroke="#48484A" stroke-width="2.5"/>
    <path d="${line("r2_sylearn")}" fill="none" stroke="#2997FF" stroke-width="4" stroke-linecap="round" style="filter:drop-shadow(0 0 10px #2997FF)"/>
    <circle cx="${x(last.day)}" cy="${y(last.r2_sylearn)}" r="7" fill="#fff"/>`;
}

// ------------------------------------------------------------------ 4 · 56 numbers
const DOTS = { n: 1898, from: [], to: [], col: [], cv: null, ctx: null, w: 0, h: 0, t: 0 };
function sizeCanvas() {
  if (!DOTS.cv) return; const dpr = Math.min(devicePixelRatio || 1, 2);
  DOTS.w = DOTS.cv.clientWidth; DOTS.h = DOTS.cv.clientHeight; DOTS.cv.width = DOTS.w * dpr; DOTS.cv.height = DOTS.h * dpr; DOTS.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  layoutDots(); drawDots();
}
function layoutDots() {
  let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const { w, h } = DOTS, cell = Math.min(w / 14, h / 9), gx = w / 2 - cell * 4.3, gy = h / 2 - cell * 3.5;
  DOTS.cell = cell; DOTS.cells = [];
  for (let r = 0; r < 7; r++) for (let c = 0; c < 8; c++) DOTS.cells.push([gx + c * cell + (c === 7 ? cell * 0.6 : 0), gy + r * cell]);
  const palette = Object.values(TIER_COLORS);
  DOTS.from = []; DOTS.to = []; DOTS.col = [];
  for (let i = 0; i < DOTS.n; i++) {
    const a = rnd() * Math.PI * 2, rr = Math.sqrt(rnd());
    DOTS.from.push([w / 2 + Math.cos(a) * rr * w * 0.48, h / 2 + Math.sin(a) * rr * h * 0.46]);
    const [cx, cy] = DOTS.cells[i % 56];
    DOTS.to.push([cx + cell * 0.5 + (rnd() - 0.5) * cell * 0.5, cy + cell * 0.5 + (rnd() - 0.5) * cell * 0.5]);
    DOTS.col.push(palette[Math.floor(rnd() * palette.length)]);
  }
}
function drawDots() {
  const c = DOTS.ctx; if (!c) return; const t = DOTS.t, e = t * t * (3 - 2 * t);
  c.clearRect(0, 0, DOTS.w, DOTS.h);
  const squares = smooth(0.82, 1, t);
  for (let i = 0; i < DOTS.n; i++) {
    const [fx, fy] = DOTS.from[i], [tx, ty] = DOTS.to[i];
    c.globalAlpha = 0.75 * (1 - squares * 0.9); c.fillStyle = DOTS.col[i];
    c.beginPath(); c.arc(lerp(fx, tx, e), lerp(fy, ty, e), lerp(2.6, 1.4, e), 0, 6.283); c.fill();
  }
  if (squares > 0) {
    c.globalAlpha = squares;
    DOTS.cells.forEach(([x, y], i) => { c.fillStyle = i % 8 === 7 ? "#B23BD6" : "#2997FF"; const s = DOTS.cell * 0.82; c.beginPath(); c.roundRect(x + (DOTS.cell - s) / 2, y + (DOTS.cell - s) / 2, s, s, 6); c.fill(); });
  }
  c.globalAlpha = 1;
}
function memoryScene(el) {
  DOTS.cv = $("#dots"); DOTS.ctx = DOTS.cv.getContext("2d"); sizeCanvas();
  scene(el, (p) => {
    DOTS.t = smooth(0.08, 0.78, p); drawDots();
    const n = Math.round(lerp(M.stream.rows, M.stream.stored_numbers, Math.pow(DOTS.t, 0.35)));
    $("#memNum").textContent = fmt(n);
    $("#memWord").textContent = DOTS.t > 0.97 ? "numbers." : "readings.";
    $("#memNum").style.color = DOTS.t > 0.97 ? "#2997FF" : "";
  });
}

// ------------------------------------------------------------------ 5 · merge
function mergeScene(el, merged) {
  const orbit = $("#orbit"), cities = M.merge.cities.map((c) => c.name);
  const spans = cities.map((c, i) => { const s = document.createElement("span"); s.textContent = c; orbit.appendChild(s); return s; });
  const core = document.createElement("div"); core.className = "core"; orbit.appendChild(core);
  $("#mergeGap").textContent = sci(merged.max_diff);
  scene(el, (p) => {
    const t = smooth(0.12, 0.7, p), W = orbit.clientWidth, H = orbit.clientHeight;
    spans.forEach((s, i) => {
      const a = (i / spans.length) * Math.PI * 2 + p * 1.2, rx = W * 0.44, ry = H * 0.42;
      const x = Math.cos(a) * rx * (1 - t), y = Math.sin(a) * ry * (1 - t);
      s.style.transform = `translate(calc(-50% + ${x}px), calc(-50% + ${y}px)) scale(${lerp(1, 0.4, t)})`;
      s.style.opacity = 1 - smooth(0.55, 0.72, p);
    });
    const coreT = smooth(0.5, 0.78, p);
    core.style.transform = `scale(${lerp(0.25, 1.15, coreT)})`; core.style.opacity = coreT;
    core.innerHTML = coreT > 0.6 ? "1<br>model" : "";
    $("#mergeOut").style.opacity = smooth(0.78, 0.9, p);
  });
}

// ------------------------------------------------------------------ 6 · stats count-up, 7 · city roll
function countUp(node) {
  const to = +node.dataset.to, dec = +(node.dataset.dec || 0), start = performance.now(), dur = 1400;
  if (REDUCED) { node.textContent = fmt(to, dec); return; }
  const step = (now) => { const t = Math.min(1, (now - start) / dur), e = 1 - Math.pow(1 - t, 4); node.textContent = fmt(to * e, dec); if (t < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}
function statsScene() {
  const sy = M.bench.find((b) => b.mine), sk = M.bench.find((b) => b.name.includes("default"));
  const nodes = [...document.querySelectorAll(".statscene .count")];
  const vals = [M.models.sy_rf.R2, sy.size_kb, sy.predict_ms, M.models.sy_rf.MAE];
  nodes.forEach((n, i) => (n.dataset.to = vals[i]));
  $("#skR2").textContent = fmt(M.models.sk_rf.R2, 3); $("#skSize").textContent = `${fmt(sk.size_kb / 1000)} MB`;
  $("#skMs").textContent = `${fmt(sk.predict_ms, 0)} ms`; $("#skMae").textContent = fmt(sk.mae, 1);
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { nodes.forEach(countUp); io.disconnect(); } }), { threshold: 0.4 });
  io.observe($(".statscene .stats"));
}
function cityScene(el) {
  const roll = $("#cityRoll"), spans = M.dataset.test_cities.map((c) => { const s = document.createElement("span"); s.textContent = c; roll.appendChild(s); return s; });
  scene(el, (p) => spans.forEach((s, i) => s.classList.toggle("lit", p > 0.08 + i * 0.09)));
}

// ------------------------------------------------------------------ 8 · try it
const T = { values: {}, fam: "rf", lib: "sy", pred: null, real: null };
const key = () => (T.fam === "xgb" ? "xgb" : `${T.lib}_${T.fam}`);
function buildTry() {
  const fam = $("#famSeg");
  [["rf", "Forest"], ["dt", "Tree"], ["lr", "Linear"], ["xgb", "XGBoost"]].forEach(([k, l]) => {
    const b = document.createElement("button"); b.textContent = l; b.dataset.k = k; b.onclick = () => { T.fam = k; syncSeg(); renderTry(); }; fam.appendChild(b);
  });
  document.querySelectorAll("#libSeg button").forEach((b) => (b.onclick = () => { T.lib = b.dataset.b; syncSeg(); renderTry(); }));
  const box = $("#sliders");
  for (const f of M.features) {
    const row = document.createElement("div"); row.className = "sl"; row.dataset.f = f.key;
    const lp = (f.safe / f.max) * 100;
    row.innerHTML = `<div class="sl-head"><label class="sl-name" for="s-${CSS.escape(f.key)}">${f.key}<small>${f.name}</small></label><span class="sl-val"></span></div>
      <div class="sl-track"><input type="range" id="s-${f.key}" min="0" max="${f.max}" step="${f.step}"><i class="sl-limit" style="left:calc(${lp}% + ${14 - lp * 0.28}px)"></i></div><div class="sl-note"></div>`;
    row.querySelector("input").addEventListener("input", (e) => { T.values[f.key] = +e.target.value; T.real = null; syncSlider(f.key); schedule(); });
    box.appendChild(row);
  }
  const sel = $("#cityPreset");
  [...M.cities].sort((a, b) => a.name.localeCompare(b.name)).forEach((c) => { const o = document.createElement("option"); o.value = c.name; o.textContent = c.name; sel.appendChild(o); });
  sel.onchange = () => { const c = M.cities.find((x) => x.name === sel.value); if (c) { T.real = null; setAllTry(c.medians); } };
  $("#sampleBtn").onclick = async () => { const s = await api("/api/sample"); sel.value = ""; T.real = s; setAllTry(s.values); };
  syncSeg();
}
function syncSeg() {
  document.querySelectorAll("#famSeg button").forEach((b) => b.classList.toggle("on", b.dataset.k === T.fam));
  document.querySelectorAll("#libSeg button").forEach((b) => { b.classList.toggle("on", T.fam !== "xgb" && b.dataset.b === T.lib); b.disabled = T.fam === "xgb"; });
}
function syncSlider(k) {
  const f = M.features.find((x) => x.key === k), v = T.values[k], s = v == null ? f.median : v, ratio = s / f.safe;
  const row = document.querySelector(`.sl[data-f="${CSS.escape(k)}"]`), inp = row.querySelector("input");
  inp.value = Math.min(s, f.max);
  const col = v == null ? "#48484A" : ratio <= 1 ? "#fff" : ratio <= 2 ? "#FF7A2F" : "#FF3B5C";
  inp.style.setProperty("--pct", `${(Math.min(s, f.max) / f.max) * 100}%`); inp.style.setProperty("--fill", col);
  row.querySelector(".sl-val").innerHTML = v == null ? `–<small>not measured</small>` : `${fmt(s, s < 10 ? 2 : 0)}<small>${f.unit}</small>`;
  row.querySelector(".sl-note").textContent = v == null ? `Median ${fmt(f.median, 1)} used` : ratio <= 1 ? `Within the safe limit (${f.safe})` : `${fmt(ratio, 1)}× the safe limit`;
}
function setAllTry(vals) { for (const f of M.features) { T.values[f.key] = vals[f.key] ?? null; syncSlider(f.key); } schedule(0); }
let timer, seq = 0;
function schedule(ms = 40) { clearTimeout(timer); timer = setTimeout(runTry, ms); }
async function runTry() {
  const my = ++seq;
  try { const r = await api("/api/predict", { values: T.values }); if (my !== seq) return; $("#predictError").textContent = ""; T.pred = r; renderTry(); renderWork(); }
  catch (e) { $("#predictError").textContent = e.message; }
}
let shown = 0, tw = 0;
function renderTry() {
  if (!T.pred) return;
  const k = key(), v = T.pred.predictions[k], b = band(v), c = tierColor(v);
  $("#readout").style.setProperty("--band", c);
  $("#tierName").textContent = b.name; $("#healthText").textContent = b.health;
  $("#scaleMark").style.left = `${clamp(v / 500, 0, 1) * 100}%`;
  const id = ++tw, from = shown, start = performance.now();
  const step = (now) => { if (id !== tw) return; const t = Math.min(1, (now - start) / 500), e = 1 - Math.pow(1 - t, 4); shown = lerp(from, v, e); $("#aqiValue").textContent = Math.round(shown); if (t < 1) requestAnimationFrame(step); };
  REDUCED ? ($("#aqiValue").textContent = Math.round((shown = v))) : requestAnimationFrame(step);
  const r = T.real;
  $("#realDay").innerHTML = r ? `${r.city}, ${new Date(r.date).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}: the real AQI was <b>${fmt(r.actual)}</b>. Off by <b>${fmt(Math.abs(v - r.actual))}</b>, in a city the model never saw.` : "";
}

// ------------------------------------------------------------------ 9 · shows its work
function renderWork() {
  const lin = T.pred.explain.linear, items = M.features.map((f) => [f.key, lin.contributions[f.key]]).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...items.map((i) => Math.abs(i[1])), 1), total = T.pred.predictions.sy_lr;
  $("#contrib").innerHTML = `<p class="dim" style="margin:-8px 0 14px;font-size:14px">sylearn Linear Regression · starts at ${fmt(lin.intercept, 1)}</p>` +
    items.map(([k, v]) => `<div class="cb"><span class="cb-name">${k}</span><span class="cb-bar"><i style="width:${(Math.max(0, v) / max) * 100}%"></i></span><span class="cb-val">${v >= 0 ? "+" : ""}${fmt(v, 0)}</span></div>`).join("") +
    `<div class="cb total"><span class="cb-name">AQI</span><span></span><span class="cb-val">${fmt(total)}</span></div>`;
  const nodes = M.tree, mi = T.pred.model_input;
  $("#treePath").innerHTML = T.pred.explain.tree_path.map((id) => {
    const n = nodes[id];
    if (n.leaf) return `<li class="leaf" style="--band:${tierColor(n.value)}">AQI ${fmt(n.value)} · ${band(n.value).name}</li>`;
    const val = mi[n.feature], yes = val <= n.threshold;
    return `<li><span>Is <b>${n.feature}</b> ≤ ${fmt(n.threshold, n.feature === "CO" ? 2 : 1)}?</span><span class="ans" style="color:${yes ? "#34C759" : "#FF7A2F"}">${yes ? "Yes" : "No"} · ${fmt(val, val < 10 ? 2 : 0)}</span></li>`;
  }).join("");
}

// ------------------------------------------------------------------ 2b · AQI tiers
function buildTiers() {
  const box = $("#tiers"); let lo = 0;
  M.bands.forEach((b, i) => {
    const hi = b.max ?? 500, row = document.createElement("div");
    row.className = "tier"; row.style.setProperty("--c", TIER_COLORS[b.name]); row.style.transitionDelay = `${i * 0.08}s`;
    row.innerHTML = `<span class="tier-name">${b.name}</span><span class="tier-range">${lo === 0 ? 0 : lo + 1}–${hi}</span><span class="tier-text">${b.health}</span>`;
    box.appendChild(row); lo = hi;
  });
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { box.querySelectorAll(".tier").forEach((t) => t.classList.add("in")); io.disconnect(); } }), { threshold: 0.2 });
  io.observe(box);
}

// ------------------------------------------------------------------ 2c · map
const MAP = { map: null, layer: null, metric: "AQI" };
const SAFE = [[0.5, "#34C759", "Under half the safe limit"], [1, "#A3D13C", "Within the safe limit"], [2, "#FF7A2F", "Up to 2× the limit"], [Infinity, "#FF3B5C", "Over 2× the limit"]];
const featOf = (k) => M.features.find((f) => f.key === k);
const mColor = (v) => (MAP.metric === "AQI" ? tierColor(v) : SAFE.find(([r]) => v / featOf(MAP.metric).safe <= r)[1]);
const mLabel = (v) => (MAP.metric === "AQI" ? band(v).name : `${fmt(v / featOf(MAP.metric).safe, 1)}× the safe limit`);
function buildMap() {
  if (!window.L) { $("#mapBox").innerHTML = `<p style="padding:30px;color:#86868B">The map needs an internet connection.</p>`; return; }
  const seg = $("#metricSeg");
  ["AQI", ...M.features.map((f) => f.key)].forEach((m) => {
    const b = document.createElement("button"); b.textContent = m; b.classList.toggle("on", m === MAP.metric);
    b.onclick = () => { MAP.metric = m; seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); drawMap(); };
    seg.appendChild(b);
  });
  MAP.map = L.map("mapBox", { scrollWheelZoom: false, zoomSnap: 0.25, zoomControl: true });
  MAP.map.fitBounds(L.latLngBounds(M.cities.map((c) => c.coord)), { padding: [50, 50] });
  L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", { maxZoom: 10, attribution: "Tiles © Esri" }).addTo(MAP.map);
  new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && MAP.map.invalidateSize()), { threshold: 0.1 }).observe($("#mapBox"));
  drawMap();
}
function drawMap() {
  if (MAP.layer) MAP.map.removeLayer(MAP.layer); MAP.layer = L.layerGroup().addTo(MAP.map);
  const unit = MAP.metric === "AQI" ? "" : ` ${featOf(MAP.metric).unit}`;
  const rows = M.cities.filter((c) => c.means[MAP.metric] != null).map((c) => [c, c.means[MAP.metric]]).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...rows.map((r) => r[1]));
  for (const [c, v] of [...rows].reverse()) {
    const col = mColor(v), r = 7 + 22 * Math.sqrt(v / max);
    L.circleMarker(c.coord, { radius: r * 2.6, stroke: false, fillColor: col, fillOpacity: 0.12, interactive: false }).addTo(MAP.layer);
    L.circleMarker(c.coord, { radius: r * 1.5, stroke: false, fillColor: col, fillOpacity: 0.18, interactive: false }).addTo(MAP.layer);
    const m = L.circleMarker(c.coord, { radius: r, color: "#fff", weight: c.test ? 2 : 0, dashArray: c.test ? "3 3" : null, fillColor: col, fillOpacity: 0.95 }).addTo(MAP.layer);
    m.bindTooltip(`<b>${c.name}</b><br>${MAP.metric} ${fmt(v, v < 10 ? 2 : 0)}${unit} · ${mLabel(v)}${c.test ? "<br><span style='color:#86868B'>never seen in training</span>" : ""}`, { direction: "top", offset: [0, -r] });
    m.on("click", () => openCity(c.name));
  }
  $("#mpTitle").textContent = MAP.metric === "AQI" ? "Worst average AQI" : `Most ${MAP.metric}`;
  $("#mpList").innerHTML = rows.slice(0, 5).map(([c, v]) => `<li data-c="${c.name}"><span>${c.name}</span><b style="color:${mColor(v)}">${fmt(v, v < 10 ? 2 : 0)}</b></li>`).join("");
  $("#mpList").querySelectorAll("li").forEach((li) => (li.onclick = () => { const c = M.cities.find((x) => x.name === li.dataset.c); MAP.map.flyTo(c.coord, 6.5, { duration: 0.9 }); }));
  $("#mapLegend").innerHTML = (MAP.metric === "AQI" ? M.bands.map((b) => [TIER_COLORS[b.name], b.name]) : SAFE.map(([, c, l]) => [c, l]))
    .map(([c, l]) => `<span><i style="background:${c};box-shadow:0 0 10px ${c}"></i>${l}</span>`).join("") + `<span><i style="border:2px dashed #fff;width:12px;height:12px"></i>Test city</span>`;
}

// ------------------------------------------------------------------ 2d · city story
const tipEl = document.createElement("div"); tipEl.className = "hovertip"; document.body.appendChild(tipEl);
const showTip = (e, html) => { tipEl.innerHTML = html; tipEl.classList.add("show"); tipEl.style.left = `${Math.min(e.clientX + 14, innerWidth - 220)}px`; tipEl.style.top = `${e.clientY + 14}px`; };
function buildCity() {
  const sel = $("#citySelect");
  [...M.cities].sort((a, b) => a.name.localeCompare(b.name)).forEach((c) => { const o = document.createElement("option"); o.value = c.name; o.textContent = c.name; sel.appendChild(o); });
  sel.onchange = () => loadCityStory(sel.value);
  sel.value = "Delhi"; loadCityStory("Delhi");
}
function openCity(name) { $("#citySelect").value = name; loadCityStory(name); document.getElementById("city").scrollIntoView(); }
async function loadCityStory(name) {
  const d = await api(`/api/city/${encodeURIComponent(name)}`), c = M.cities.find((x) => x.name === name);
  const total = Object.values(d.band_days).reduce((a, b) => a + b, 0) || 1, bad = (d.band_days["Poor"] + d.band_days["Very Poor"] + d.band_days["Severe"]) / total;
  $("#cityTitle").innerHTML = `${name}<span style="color:${tierColor(c.aqi)}">.</span>`;
  $("#cityKpis").innerHTML = [
    [fmt(c.aqi), "average AQI", band(c.aqi).name, tierColor(c.aqi)],
    [`${fmt(bad * 100)}%`, "of days Poor or worse", `${fmt(total)} days recorded`],
    [d.worst ? fmt(d.worst.aqi) : "–", "worst day", d.worst ? new Date(d.worst.date).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }) : ""],
    [c.test ? "Unseen" : "Trained", "by the model", c.test ? "held out for testing" : "used in training"],
  ].map(([b, s, sm, col]) => `<div><b style="${col ? `color:${col}` : ""}">${b}</b><span>${s}</span><small>${sm}</small></div>`).join("");
  drawTrend(d); drawHeat(d);
}
function drawTrend(d) {
  const svg = $("#cityTrend"), w = svg.clientWidth || 600, h = svg.clientHeight || 300, pad = { l: 34, r: 8, t: 10, b: 24 };
  const pts = d.monthly.dates.map((t, i) => [i, d.monthly.aqi[i], t]).filter((p) => p[1] != null), n = d.monthly.dates.length;
  if (!pts.length) { svg.innerHTML = `<text x="10" y="30" fill="#86868B">No AQI readings.</text>`; return; }
  const max = Math.max(120, ...pts.map((p) => p[1])) * 1.1, x = (i) => pad.l + (i / Math.max(1, n - 1)) * (w - pad.l - pad.r), y = (v) => h - pad.b - (v / max) * (h - pad.t - pad.b);
  let path = "", prev = -2; pts.forEach(([i, v]) => { path += `${i === prev + 1 ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)} `; prev = i; });
  const stops = M.bands.map((b) => { const top = Math.min(b.max ?? 500, max); return `<stop offset="${(1 - top / max) * 100}%" stop-color="${TIER_COLORS[b.name]}"/>`; }).reverse().join("");
  const ticks = [0, 100, 200, 300, 400].filter((t) => t < max).map((t) => `<line x1="${pad.l}" x2="${w - pad.r}" y1="${y(t)}" y2="${y(t)}" stroke="#E5E5EA"/><text x="${pad.l - 8}" y="${y(t) + 4}" text-anchor="end" font-size="11" fill="#86868B">${t}</text>`).join("");
  const years = d.monthly.dates.map((t, i) => [t, i]).filter(([t]) => t.endsWith("-01")).map(([t, i]) => `<text x="${x(i)}" y="${h - 4}" text-anchor="middle" font-size="11" fill="#86868B">${t.slice(0, 4)}</text>`).join("");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.innerHTML = `<defs><linearGradient id="tg" x1="0" y1="${pad.t}" x2="0" y2="${h - pad.b}" gradientUnits="userSpaceOnUse">${stops}</linearGradient></defs>${ticks}${years}
    <path d="${path}" fill="none" stroke="url(#tg)" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/>
    <rect class="hit" x="${pad.l}" y="${pad.t}" width="${w - pad.l - pad.r}" height="${h - pad.t - pad.b}" fill="transparent"/>`;
  svg.querySelector(".hit").addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect(), mx = ((e.clientX - r.left) / r.width) * w; let best = pts[0];
    for (const p of pts) if (Math.abs(x(p[0]) - mx) < Math.abs(x(best[0]) - mx)) best = p;
    showTip(e, `${new Date(best[2] + "-01").toLocaleDateString("en-GB", { month: "short", year: "numeric" })} · <b>${fmt(best[1])}</b> ${band(best[1]).name}`);
  });
  svg.querySelector(".hit").addEventListener("mouseleave", () => tipEl.classList.remove("show"));
}
function drawHeat(d) {
  const box = $("#cityHeat"), months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  box.style.gridTemplateColumns = "44px repeat(12, minmax(0, 1fr))";
  box.innerHTML = `<span></span>${months.map((m) => `<span class="lbl">${m}</span>`).join("")}` + d.heatmap.years.map((yr, yi) => `<span class="lbl">${yr}</span>` +
    d.heatmap.values[yi].map((v, mi) => (v == null ? `<span class="cell empty"></span>` : `<span class="cell" data-t="${names[mi]} ${yr} · ${fmt(v)} ${band(v).name}" style="background:${tierColor(v)}"></span>`)).join("")).join("");
  box.querySelectorAll(".cell[data-t]").forEach((c) => { c.onmousemove = (e) => showTip(e, c.dataset.t); c.onmouseleave = () => tipEl.classList.remove("show"); });
}

// ------------------------------------------------------------------ boot
(async function boot() {
  M = await api("/api/meta");
  const delhi = M.cities.find((c) => c.name === "Delhi");
  const [heroPred, merged] = await Promise.all([api("/api/predict", { values: delhi.medians }), api("/api/merge", { cities: M.merge.cities.map((c) => c.name) })]);
  fillDevice(heroPred, delhi.medians);
  const sc = [...document.querySelectorAll(".scene")];
  heroScene(sc[0]);
  countScene(sc[1], $("#delhiNum"), delhi.aqi);
  learnScene(sc[2]);
  memoryScene(sc[3]);
  mergeScene(sc[4], merged);
  statsScene();
  cityScene(sc[6]);
  buildTiers(); buildMap(); buildCity();
  buildTry(); $("#cityPreset").value = "Delhi"; setAllTry(delhi.medians);
  $("#pipHero").onclick = copyPip; $("#pipClose").onclick = copyPip;
  onScroll();
})().catch((e) => { document.body.insertAdjacentHTML("afterbegin", `<p style="position:fixed;top:60px;left:0;right:0;text-align:center;z-index:99;color:#FF453A">Start the server with <code>.venv/bin/python web/server.py</code> and reload. (${e.message})</p>`); });
