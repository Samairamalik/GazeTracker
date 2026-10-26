// Stage 1 Biometric HUD, Stage 2 Interactive Precision Calibration with Spring Physics,
// Blink-Aware Progress Ring, and Instant Vector Error Field Report Card.
const $ = (id) => document.getElementById(id);
const ws = new WebSocket(`ws://${location.host}/ws`);
let calib = null, stats = { fps: 0, latency_ms: 0 }, started = false, hudOn = true;
let latestTelemetry = null;
const openHistR = [], openHistL = [];
const MAX_WAVE_PTS = 110;

// Spring-physics state for smooth target reticle travel
const reticle = {
  active: false,
  x: innerWidth / 2, y: innerHeight / 2,
  vx: 0, vy: 0,
  tx: innerWidth / 2, ty: innerHeight / 2,
};

function stepReticleSpring() {
  if (reticle.active) {
    const dt = 1 / 60;
    const stiffness = 170, damping = 22;
    const ax = stiffness * (reticle.tx - reticle.x) - damping * reticle.vx;
    const ay = stiffness * (reticle.ty - reticle.y) - damping * reticle.vy;
    reticle.vx += ax * dt;
    reticle.vy += ay * dt;
    reticle.x += reticle.vx * dt;
    reticle.y += reticle.vy * dt;
    const d = $("dot");
    if (d) d.style.transform = `translate3d(${reticle.x.toFixed(1)}px, ${reticle.y.toFixed(1)}px, 0)`;
  }
  requestAnimationFrame(stepReticleSpring);
}
requestAnimationFrame(stepReticleSpring);

function send(msg) {
  if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

function pushConfigFromUI() {
  const passSel = $("cfg-passage");
  const isCustom = passSel && passSel.value === "__custom__";
  const customTxt = $("cfg-custom-text") ? $("cfg-custom-text").value.trim() : "";
  send({
    type: "configure",
    text: isCustom ? undefined : (passSel ? passSel.value : "alice_excerpts"),
    custom_text: isCustom ? customTxt : "",
    n_points: $("cfg-npoints") ? Number($("cfg-npoints").value) : 9,
    features: $("cfg-features") ? $("cfg-features").value : "iris_pose",
    drift_lock: $("cfg-drift-lock") ? $("cfg-drift-lock").checked : true,
    show_cursor: $("cfg-show-cursor") ? $("cfg-show-cursor").checked : false,
    skim: $("cfg-skim") ? $("cfg-skim").checked : false,
  });
}

function applyTheme(theme) {
  document.body.classList.remove("theme-sepia", "theme-dark", "theme-light");
  document.body.classList.add(`theme-${theme}`);
  if ($("cfg-theme") && $("cfg-theme").value !== theme) $("cfg-theme").value = theme;
}

function hello() {
  calib = null;
  Reader.reset();
  Heatmap.clear();
  if ($("calib-report")) $("calib-report").style.display = "none";
  $("intro").style.display = "none";
  $("gaze").style.display = "none";
  $("msg").style.display = "none";
  document.body.classList.add("calibrating");
  reticle.x = innerWidth / 2;
  reticle.y = innerHeight / 2;
  reticle.vx = 0;
  reticle.vy = 0;
  pushConfigFromUI();
  send({ type: "hello", w: innerWidth, h: innerHeight, dpr: devicePixelRatio });
  renderHud("calibrating…");
}

function start() {
  started = true;
  $("intro").style.display = "none";
  if ($("calib-report")) $("calib-report").style.display = "none";
  let done = false;
  const go = (delay) => { if (!done) { done = true; setTimeout(hello, delay); } };
  const el = document.documentElement;
  if (el.requestFullscreen && !document.fullscreenElement) {
    el.requestFullscreen().then(() => go(350), () => go(0));
    setTimeout(() => go(0), 900);
  } else go(0);
}

function showMsg(text) {
  const m = $("msg");
  m.textContent = text;
  m.style.display = "block";
}

function showDot(m) {
  const d = $("dot");
  d.className = m.phase === "validate" ? "validate" : (m.phase === "retry" ? "retry" : "");
  d.style.display = "block";
  reticle.active = true;
  reticle.tx = m.x;
  reticle.ty = m.y;
  const ring = $("dot-ring");
  if (ring) ring.style.strokeDashoffset = "138.2";
  const lbl = $("dot-label");
  const pName = m.phase === "fit" ? "CALIB" : (m.phase === "validate" ? "VALID" : "RETRY");
  if (lbl) lbl.textContent = `${pName} ${m.i + 1}/${m.n}`;
  renderHud(`${m.phase === "fit" ? "calibrating" : (m.phase === "validate" ? "validating" : "retrying")} ${m.i + 1}/${m.n}`);
}

function updateDotProgress(m) {
  const d = $("dot");
  const ring = $("dot-ring");
  const lbl = $("dot-label");
  if (!d || !ring) return;
  const circumference = 138.2;
  const offset = circumference * (1 - Math.min(1, Math.max(0, m.progress || 0)));
  ring.style.strokeDashoffset = offset.toFixed(1);
  d.classList.toggle("paused", !m.locked);
  if (lbl) {
    const pName = m.phase === "fit" ? "CALIB" : (m.phase === "validate" ? "VALID" : "RETRY");
    lbl.textContent = m.locked
      ? `${pName} ${m.i + 1}/${m.n} · ${Math.round((m.progress || 0) * 100)}%`
      : `BLINK PAUSED · HOLD GAZE`;
  }
}

function fmt(v, digits = 0) { return v == null ? "–" : Number(v).toFixed(digits); }

function renderHud(status) {
  if (!hudOn) { $("hud").style.display = "none"; return; }
  $("hud").style.display = "block";
  let s = status ? status + "\n" : "";
  if (calib) {
    const model = calib.degree === 1 ? "affine" : "poly2";
    s += `σ ${fmt(calib.sigma_px)} px  (x ${fmt(calib.rms_x)}, y ${fmt(calib.rms_y)})\n`;
    s += `${model} α=${Number(calib.alpha).toPrecision(2)}  LOO ${fmt(calib.loo_rms_px)} px\n`;
    s += `${calib.n_points} dots (${calib.n_fit_dots} used)  ${calib.variant} · features ${calib.features}\n`;
    for (const c of calib.compare || [])
      s += `  ${c.variant.padEnd(6)} σ ${fmt(c.sigma_px)} px (x ${fmt(c.rms_x)}, y ${fmt(c.rms_y)})  LOO ${fmt(c.loo_rms_px)}\n`;
  }
  s += `${fmt(stats.fps, 1)} fps  pipeline ${fmt(stats.latency_ms, 1)} ms\n`;
  s += "C recalibrate · H hide HUD";
  $("hud").textContent = s;
  if (calib && calib.source === "synthetic") {
    const warn = document.createElement("div");
    warn.className = "synthetic";
    warn.textContent = "SYNTHETIC SOURCE: not a measurement";
    $("hud").prepend(warn);
  }
}

// ==================== STAGE 1 BIOMETRIC HUD RENDERERS ====================
function updateStage1Hud(t) {
  latestTelemetry = t;
  stats.fps = t.fps ?? stats.fps;
  stats.latency_ms = t.latency_ms ?? stats.latency_ms;

  const srcBadge = $("source-badge");
  if (srcBadge) {
    srcBadge.textContent = t.source === "synthetic" ? "SYNTHETIC SIMULATOR" : "LIVE WEBCAM STREAM";
    srcBadge.className = "badge " + (t.source === "synthetic" ? "warn" : "ok");
  }

  const lockBadge = $("mesh-lock-badge");
  if (lockBadge) {
    lockBadge.textContent = t.valid ? "BIOMETRIC LOCK ACTIVE" : "BLINK / POSE UNLOCKED";
    lockBadge.className = "badge " + (t.valid ? "ok" : "warn");
  }

  if ($("tele-iris-r")) $("tele-iris-r").textContent = `${fmt(t.uR, 2)}, ${fmt(t.vR, 2)}`;
  if ($("tele-iris-l")) $("tele-iris-l").textContent = `${fmt(t.uL, 2)}, ${fmt(t.vL, 2)}`;
  if ($("tele-fps")) $("tele-fps").textContent = `${fmt(t.fps, 1)} FPS`;
  if ($("tele-lat")) $("tele-lat").textContent = `${fmt(t.latency_ms, 1)} ms`;
  if ($("tele-pose-text")) {
    const dist = t.tz != null ? Math.abs(t.tz) : 60;
    $("tele-pose-text").textContent = `Yaw ${fmt(t.yaw, 1)}° · Pitch ${fmt(t.pitch, 1)}° · ${fmt(dist, 0)} cm`;
  }
  if ($("tele-open-text")) {
    $("tele-open-text").textContent = `R ${fmt(t.openR, 2)} · L ${fmt(t.openL, 2)}`;
  }

  const poseOk = t.valid && Math.abs(t.yaw || 0) <= 20 && Math.abs(t.pitch || 0) <= 18;
  const prBadge = $("pose-ready-badge");
  if (prBadge) {
    prBadge.textContent = poseOk ? "POSE & DISTANCE OPTIMAL" : "CENTER HEAD (~60 CM)";
    prBadge.className = "badge " + (poseOk ? "ok" : "warn");
  }

  openHistR.push(t.openR ?? 0);
  openHistL.push(t.openL ?? 0);
  if (openHistR.length > MAX_WAVE_PTS) openHistR.shift();
  if (openHistL.length > MAX_WAVE_PTS) openHistL.shift();

  // Only draw Stage 1 canvases when #intro is visible
  if ($("intro") && $("intro").style.display !== "none") {
    drawCyberMesh(t);
    drawPoseRadar(t);
    drawEyeWaveforms();
  }
}

let previewImg = new Image();
let hasPreviewLoaded = false;
previewImg.onload = () => { hasPreviewLoaded = true; };

function drawCyberMesh(t) {
  const cv = $("mesh-canvas");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const w = cv.width, h = cv.height;
  ctx.clearRect(0, 0, w, h);

  if (t.preview) {
    previewImg.src = "data:image/jpeg;base64," + t.preview;
  }
  if (t.preview && hasPreviewLoaded) {
    ctx.save();
    ctx.globalAlpha = 0.45;
    ctx.drawImage(previewImg, 0, 0, w, h);
    ctx.fillStyle = "rgba(5, 12, 28, 0.35)";
    ctx.fillRect(0, 0, w, h);
    ctx.restore();
  }

  // Subtle radar crosshair
  ctx.strokeStyle = "rgba(56, 189, 248, 0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
  ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
  ctx.stroke();

  if (!t.mesh || !t.mesh.contours) {
    ctx.fillStyle = "#64748b";
    ctx.font = "600 13px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No face detected — position face in front of camera", w / 2, h / 2);
    return;
  }

  const c = t.mesh.contours;
  // Mirror X so moving head right moves mesh right like a mirror
  const mx = (pt) => (1 - pt[0]) * w;
  const my = (pt) => pt[1] * h;

  const drawChain = (pts, stroke, width = 1.4, close = false) => {
    if (!pts || !pts.length) return;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.beginPath();
    pts.forEach((p, idx) => {
      if (idx === 0) ctx.moveTo(mx(p), my(p));
      else ctx.lineTo(mx(p), my(p));
    });
    if (close) ctx.closePath();
    ctx.stroke();
  };

  drawChain(c.oval, "rgba(0, 240, 255, 0.42)", 1.5, true);
  drawChain(c.right_brow, "rgba(56, 189, 248, 0.55)", 1.5);
  drawChain(c.left_brow, "rgba(56, 189, 248, 0.55)", 1.5);
  drawChain(c.nose, "rgba(148, 163, 184, 0.4)", 1.2);
  drawChain(c.lips, "rgba(16, 185, 129, 0.45)", 1.3, true);
  drawChain(c.right_eye, t.valid ? "#00f0ff" : "#f59e0b", 1.8, true);
  drawChain(c.left_eye, t.valid ? "#00f0ff" : "#f59e0b", 1.8, true);

  // Draw glowing 5-point irises & gaze vector arrows
  const drawIris = (irisPts, u, v, signU) => {
    if (!irisPts || !irisPts.length) return;
    const center = irisPts[0];
    const cx = mx(center), cy = my(center);
    ctx.fillStyle = "#10b981";
    ctx.beginPath();
    ctx.arc(cx, cy, 4.2, 0, Math.PI * 2);
    ctx.fill();

    // Ring points
    ctx.strokeStyle = "rgba(16, 185, 129, 0.7)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let i = 1; i < irisPts.length; i++) {
      const rx = mx(irisPts[i]), ry = my(irisPts[i]);
      ctx.moveTo(rx + 1.5, ry);
      ctx.arc(rx, ry, 1.5, 0, Math.PI * 2);
    }
    ctx.stroke();

    // Gaze direction vector ray
    const vx = (signU * (u || 0)) * 110;
    const vy = (v || 0) * 110;
    ctx.strokeStyle = "#00f0ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + vx, cy + vy);
    ctx.stroke();
  };

  drawIris(t.mesh.right_iris, t.uR, t.vR, 1);
  drawIris(t.mesh.left_iris, t.uL, t.vL, -1);
}

function drawPoseRadar(t) {
  const cv = $("pose-canvas");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const w = cv.width, h = cv.height;
  ctx.clearRect(0, 0, w, h);

  // Left side: XY Orientation Target
  const cx = 105, cy = h / 2, r = 52;
  ctx.strokeStyle = "rgba(148, 163, 184, 0.22)";
  ctx.lineWidth = 1.2;
  [r, r * 0.6].forEach((rad) => {
    ctx.beginPath(); ctx.arc(cx, cy, rad, 0, Math.PI * 2); ctx.stroke();
  });
  ctx.beginPath();
  ctx.moveTo(cx - r - 6, cy); ctx.lineTo(cx + r + 6, cy);
  ctx.moveTo(cx, cy - r - 6); ctx.lineTo(cx, cy + r + 6);
  ctx.stroke();

  const px = cx - ((t.yaw || 0) / 30) * r;
  const py = cy + ((t.pitch || 0) / 25) * r;
  const ok = Math.hypot(px - cx, py - cy) <= r * 0.75;
  ctx.fillStyle = ok ? "#10b981" : "#f59e0b";
  ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.fill();

  // Right side: Distance bar (40..85 cm, optimal 50..70 cm) + Roll readout
  const bx = 215, by = 28, bw = 160, bh = 22;
  ctx.fillStyle = "rgba(30, 41, 59, 0.85)";
  ctx.fillRect(bx, by, bw, bh);
  // Sweet spot 50..70 cm
  const s0 = ((50 - 35) / 55) * bw, s1 = ((70 - 35) / 55) * bw;
  ctx.fillStyle = "rgba(16, 185, 129, 0.25)";
  ctx.fillRect(bx + s0, by, s1 - s0, bh);

  const dist = Math.abs(t.tz || 60);
  const dNorm = Math.min(1, Math.max(0, (dist - 35) / 55));
  ctx.fillStyle = (dist >= 48 && dist <= 72) ? "#00f0ff" : "#f59e0b";
  ctx.fillRect(bx + dNorm * bw - 3, by - 3, 6, bh + 6);

  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText(`Distance: ${dist.toFixed(0)} cm (Sweet spot: 50–70 cm)`, bx, by + bh + 18);
  ctx.fillText(`Head Roll: ${(t.roll || 0).toFixed(1)}° · Pos (${(t.tx || 0).toFixed(1)}, ${(t.ty || 0).toFixed(1)}) cm`, bx, by + bh + 38);
}

function drawEyeWaveforms() {
  const cv = $("wave-canvas");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const w = cv.width, h = cv.height;
  ctx.clearRect(0, 0, w, h);

  const yMap = (val) => h - 10 - Math.min(1, Math.max(0, val / 0.45)) * (h - 20);
  // Blink threshold line at 0.15
  const yThr = yMap(0.15);
  ctx.strokeStyle = "rgba(244, 63, 94, 0.45)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(0, yThr); ctx.lineTo(w, yThr); ctx.stroke();
  ctx.setLineDash([]);

  const plotLine = (arr, color) => {
    if (arr.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    arr.forEach((v, i) => {
      const x = (i / (MAX_WAVE_PTS - 1)) * w;
      const y = yMap(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };

  plotLine(openHistR, "#10b981");
  plotLine(openHistL, "#00f0ff");
}

// ==================== STAGE 2 CALIBRATION REPORT CARD ====================
function showCalibReport(m) {
  const modal = $("calib-report");
  if (!modal) return;
  modal.style.display = "flex";

  const pitch = m.line_pitch || 66;
  const isLine = m.rms_y <= pitch / 2;
  const badge = $("report-grade-badge");
  if (badge) {
    badge.textContent = isLine ? "✓ LINE-LEVEL PRECISION UNLOCKED" : "PARAGRAPH-LEVEL PRECISION";
    badge.className = "badge " + (isLine ? "ok" : "warn");
  }

  if ($("rep-sigma")) $("rep-sigma").textContent = `±${fmt(m.sigma_px, 0)} px`;
  if ($("rep-xy")) $("rep-xy").textContent = `Left/Right ±${fmt(m.rms_x, 0)} px · Up/Down ±${fmt(m.rms_y, 0)} px`;
  if ($("rep-ratio")) $("rep-ratio").textContent = isLine ? "Line-by-Line" : "Paragraph";
  if ($("rep-gran-note")) {
    $("rep-gran-note").textContent = isLine
      ? `Clear enough to track single lines (±${fmt(m.rms_y, 0)} px vs ${fmt(pitch, 0)} px)`
      : `Tracks whole paragraphs smoothly (±${fmt(m.rms_y, 0)} px vs ${fmt(pitch, 0)} px)`;
  }
  if ($("rep-model")) $("rep-model").textContent = m.sigma_px < 55 ? "High Precision" : (m.sigma_px < 95 ? "Good Fit" : "Standard Fit");
  if ($("rep-loo")) $("rep-loo").textContent = `${m.degree === 1 ? "Linear" : "Curved"} map · Hold error ±${fmt(m.loo_rms_px, 0)} px`;
  if ($("rep-dots")) $("rep-dots").textContent = `${m.n_fit_dots} / ${m.n_points} Dots`;
  if ($("rep-feat")) $("rep-feat").textContent = `All ${m.n_fit_dots} screen points locked`;

  drawErrorVectorMap(m);
}

function drawErrorVectorMap(m) {
  const cv = $("error-map-canvas");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const w = cv.width, h = cv.height;
  ctx.clearRect(0, 0, w, h);

  const sx = w / Math.max(innerWidth, 1);
  const sy = h / Math.max(innerHeight, 1);

  // Subtle screen grid
  ctx.strokeStyle = "rgba(56, 189, 248, 0.1)";
  ctx.lineWidth = 1;
  for (let gx = 0.2; gx < 1; gx += 0.2) {
    ctx.beginPath(); ctx.moveTo(gx * w, 0); ctx.lineTo(gx * w, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, gx * h); ctx.lineTo(w, gx * h); ctx.stroke();
  }

  // Draw fit points and LOO vectors
  for (const fp of m.fit_points || []) {
    const tx = fp.x * sx, ty = fp.y * sy;
    ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
    ctx.beginPath();
    ctx.moveTo(tx - 4, ty); ctx.lineTo(tx + 4, ty);
    ctx.moveTo(tx, ty - 4); ctx.lineTo(tx, ty + 4);
    ctx.stroke();
    if (fp.pred_x != null && fp.pred_y != null) {
      ctx.strokeStyle = "rgba(0, 240, 255, 0.35)";
      ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(fp.pred_x * sx, fp.pred_y * sy); ctx.stroke();
    }
  }

  // Draw 5 held-out validation points, sample clouds, and error vectors
  for (const vp of m.validation_points || []) {
    const tx = vp.x * sx, ty = vp.y * sy;
    // Sample cloud
    ctx.fillStyle = "rgba(16, 185, 129, 0.35)";
    for (const s of vp.samples || []) {
      ctx.beginPath(); ctx.arc(s[0] * sx, s[1] * sy, 2.2, 0, Math.PI * 2); ctx.fill();
    }
    // Target ring
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.arc(tx, ty, 6, 0, Math.PI * 2); ctx.stroke();

    // Error vector to median prediction
    if (vp.pred_x != null && vp.pred_y != null) {
      const px = vp.pred_x * sx, py = vp.pred_y * sy;
      ctx.strokeStyle = "#f43f5e";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(px, py); ctx.stroke();
      ctx.fillStyle = "#f43f5e";
      ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2); ctx.fill();
    }
  }
}

// ==================== WEBSOCKET HANDLERS & CONTROLS ====================
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.type === "config") {
    if (m.passages && $("cfg-passage")) {
      const sel = $("cfg-passage");
      const curVal = sel.value;
      sel.innerHTML = m.passages.map((p) =>
        `<option value="${p.id}">${p.title} (${p.para_count} paras · ${p.word_count} words)</option>`
      ).join("") + `<option value="__custom__">✍️ Paste Custom Text / Article…</option>`;
      if (curVal === "__custom__") sel.value = "__custom__";
      else if (m.reading && m.reading.text) sel.value = m.reading.text;
    }
  }
  Reader.handle(m);
  Heatmap.handle(m);

  if (m.type === "telemetry") {
    updateStage1Hud(m);
  } else if (m.type === "dot") {
    showDot(m);
  } else if (m.type === "dot_progress") {
    updateDotProgress(m);
  } else if (m.type === "calib_done") {
    calib = m;
    reticle.active = false;
    $("dot").style.display = "none";
    document.body.classList.remove("calibrating");
    $("gaze").style.display = "block";
    showCalibReport(m);
    renderHud();
  } else if (m.type === "calib_failed") {
    reticle.active = false;
    $("dot").style.display = "none";
    document.body.classList.remove("calibrating");
    showMsg(`Calibration failed: ${m.reason}. Press C to retry.`);
    renderHud();
  } else if (m.type === "gaze") {
    const g = $("gaze");
    if (m.valid && m.x != null && m.y != null) {
      g.style.transform = `translate3d(${m.x.toFixed(1)}px, ${m.y.toFixed(1)}px, 0)`;
      g.style.opacity = "1";
    } else {
      g.style.opacity = "0.2";
    }
    Reader.onGaze(m);
  } else if (m.type === "drift_locked") {
    const pill = $("drift-status-pill");
    if (pill) pill.textContent = `⚓ Drift Lock (${m.dx >= 0 ? "+" : ""}${m.dx}, ${m.dy >= 0 ? "+" : ""}${m.dy} px)`;
  } else if (m.type === "stats") {
    stats = m;
    renderHud();
  }
};

ws.onopen = () => renderHud("connected · press Enter to calibrate");
ws.onclose = () => showMsg("Server disconnected. Restart `make app` and reload.");

// UI Event Listeners
window.addEventListener("DOMContentLoaded", () => {
  if ($("btn-start-calib")) $("btn-start-calib").addEventListener("click", () => start());
  if ($("btn-recalib")) $("btn-recalib").addEventListener("click", () => hello());
  if ($("btn-begin-reading")) {
    $("btn-begin-reading").addEventListener("click", () => {
      if ($("calib-report")) $("calib-report").style.display = "none";
      if (Reader.state === "ready") Reader.start();
    });
  }
  if ($("cfg-passage")) {
    $("cfg-passage").addEventListener("change", (e) => {
      const isCustom = e.target.value === "__custom__";
      if ($("cfg-custom-text")) $("cfg-custom-text").style.display = isCustom ? "block" : "none";
      pushConfigFromUI();
    });
  }
  if ($("cfg-custom-text")) {
    $("cfg-custom-text").addEventListener("input", () => pushConfigFromUI());
  }
  ["cfg-npoints", "cfg-features", "cfg-drift-lock", "cfg-show-cursor", "cfg-skim"].forEach((id) => {
    if ($(id)) $(id).addEventListener("change", () => pushConfigFromUI());
  });
  if ($("cfg-theme")) {
    $("cfg-theme").addEventListener("change", (e) => applyTheme(e.target.value));
  }
  if ($("btn-reader-theme")) {
    $("btn-reader-theme").addEventListener("click", () => {
      const order = ["sepia", "dark", "light"];
      const cur = $("cfg-theme") ? $("cfg-theme").value : "sepia";
      const next = order[(order.indexOf(cur) + 1) % order.length];
      applyTheme(next);
    });
  }
  if ($("btn-next-page")) {
    $("btn-next-page").addEventListener("click", () => {
      if (Reader.state === "reading") Reader.next();
    });
  }
});

addEventListener("keydown", (e) => {
  if (document.activeElement && document.activeElement.tagName === "TEXTAREA") return;
  if (e.key === "Enter" && !started) {
    e.preventDefault();
    start();
  } else if ((e.key === "c" || e.key === "C") && started) {
    hello();
  } else if (e.key === "h" || e.key === "H") {
    hudOn = !hudOn;
    renderHud();
  }
});

addEventListener("resize", () => {
  if (calib) showMsg("Window size changed: press C to recalibrate.");
});
