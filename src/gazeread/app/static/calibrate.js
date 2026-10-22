// Calibration dots, validation dots, live gaze cursor and HUD.
// All coordinates are viewport CSS px; the server drives the timing.
const $ = (id) => document.getElementById(id);
const ws = new WebSocket(`ws://${location.host}/ws`);
let calib = null, stats = { fps: 0, latency_ms: 0 }, started = false, hudOn = true;

function send(msg) { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg)); }

function hello() {
  calib = null;
  Reader.reset();
  Heatmap.clear();
  $("gaze").style.display = "none";
  $("msg").style.display = "none";
  document.body.classList.add("calibrating");
  send({ type: "hello", w: innerWidth, h: innerHeight, dpr: devicePixelRatio });
  renderHud("calibrating…");
}

function start() {
  started = true;
  $("intro").style.display = "none";
  // Start once fullscreen settles, or after 1 s if the browser never answers the request.
  let done = false;
  const go = (delay) => { if (!done) { done = true; setTimeout(hello, delay); } };
  const el = document.documentElement;
  if (el.requestFullscreen && !document.fullscreenElement) {
    el.requestFullscreen().then(() => go(400), () => go(0));
    setTimeout(() => go(0), 1000);
  } else go(0);
}

function showMsg(text) { $("msg").textContent = text; $("msg").style.display = "block"; }

function showDot(m) {
  const d = $("dot");
  d.className = m.phase === "validate" ? "validate" : "";
  d.style.setProperty("--dur", `${m.dur}s`);
  d.style.transform = "";
  d.style.left = `${m.x}px`;
  d.style.top = `${m.y}px`;
  d.style.display = "block";
  void d.offsetWidth;                                   // restart the shrink animation
  d.classList.add("shrink");
  renderHud(`${m.phase === "fit" ? "calibrating" : "validating"} ${m.i + 1}/${m.n}`);
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
    s += `${calib.n_points} dots (${calib.n_fit_dots} used)  features ${calib.features}\n`;
  }
  s += `${fmt(stats.fps, 1)} fps  pipeline ${fmt(stats.latency_ms, 1)} ms\n`;
  s += "C recalibrate · H hide";
  $("hud").textContent = s;
  if (calib && calib.source === "synthetic") {
    const warn = document.createElement("div");
    warn.className = "synthetic";
    warn.textContent = "SYNTHETIC SOURCE: not a measurement";
    $("hud").prepend(warn);
  }
}

ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  Reader.handle(m);
  Heatmap.handle(m);
  if (m.type === "dot") showDot(m);
  else if (m.type === "calib_done") {
    calib = m;
    $("dot").style.display = "none";
    document.body.classList.remove("calibrating");
    $("gaze").style.display = "block";
    renderHud();
  } else if (m.type === "calib_failed") {
    $("dot").style.display = "none";
    document.body.classList.remove("calibrating");
    showMsg(`Calibration failed: ${m.reason}. Press C to retry.`);
    renderHud();
  } else if (m.type === "gaze") {
    const g = $("gaze");
    if (m.valid) {
      g.style.transform = `translate(${m.x}px, ${m.y}px)`;
      g.style.opacity = "1";
    } else g.style.opacity = "0.25";
  } else if (m.type === "stats") {
    stats = m;
    renderHud();
  }
};
ws.onopen = () => renderHud("connected, press Enter");
ws.onclose = () => showMsg("Server disconnected. Restart `make app` and reload.");

addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !started) start();
  else if ((e.key === "c" || e.key === "C") && started) hello();
  else if (e.key === "h" || e.key === "H") { hudOn = !hudOn; renderHud(); }
});
addEventListener("resize", () => {
  if (calib) showMsg("Window size changed: press C to recalibrate.");
});
