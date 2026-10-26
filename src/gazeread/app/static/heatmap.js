// Stage 4: Human-Friendly Reading Intelligence Dashboard with 4 Clear Views:
// 1. Watch Your Eyes Read (1x / 2x Animated Gaze Replay)
// 2. Attention Heatmap & Paragraph Breakdown
// 3. Plain-English Verdict, Smart Review & Reading Timeline
// 4. Download / Share Report

const Heatmap = {
  result: null,
  dir: "",
  sessionId: "",
  page: 0,
  tab: "telemetry",     // Open directly to the Plain-English Verdict & Summary first, or switch freely!
  view: "gaussian",     // "gaussian" | "para" | "line" | "scan"
  source: "fixations",  // "fixations" | "samples"

  // Replay animation state
  playing: false,
  speed: 1.0,
  replayTime: 0.0,
  replayDuration: 1.0,
  lastFrameTs: null,
  rafId: null,

  handle(m) {
    if (m.type === "analysis") {
      this.stopReplay();
      this.result = m.result;
      this.dir = m.dir || "";
      this.sessionId = m.session_id || (this.dir ? this.dir.split(/[\\/]/).pop() : "");
      this.source = m.result.primary_source || "fixations";
      this.view = "gaussian";
      this.tab = "heatmap";
      this.page = 0;
      $("msg").style.display = "none";
      document.body.classList.remove("reading", "hide-cursor");
      document.body.classList.add("results");
      this.initPageReplayBounds();
      this.draw();
    } else if (m.type === "analysis_failed") {
      showMsg(`Analysis failed: ${m.reason}`);
    }
  },

  initPageReplayBounds() {
    if (!this.result || !this.result.pages[this.page]) return;
    const pg = this.result.pages[this.page];
    this.replayDuration = Math.max(0.5, (pg.t_off || 1) - (pg.t_on || 0));
    this.replayTime = 0;
  },

  statusColor(status) {
    if (status === "Deep Read") return { bg: "rgba(16, 185, 129, 0.16)", border: "rgba(16, 185, 129, 0.45)", fg: "#34d399" };
    if (status === "Struggled / Re-read") return { bg: "rgba(244, 63, 94, 0.16)", border: "rgba(244, 63, 94, 0.45)", fg: "#fb7185" };
    if (status === "Skimmed") return { bg: "rgba(245, 158, 11, 0.16)", border: "rgba(245, 158, 11, 0.45)", fg: "#fbbf24" };
    return { bg: "rgba(148, 163, 184, 0.15)", border: "rgba(148, 163, 184, 0.35)", fg: "#94a3b8" };
  },

  friendlyBadge(pm) {
    if (pm.friendly_status) return pm.friendly_status;
    if (pm.status === "Deep Read") return "✅ Read Carefully";
    if (pm.status === "Struggled / Re-read") return "🔄 Re-read / Tricky";
    if (pm.status === "Skimmed") return "⚡ Skimmed Quickly";
    return "⏭️ Skipped Over";
  },

  heat(f) {
    const v = Math.min(1, Math.max(0, f));
    return `rgba(255, 90, 0, ${(0.08 + 0.58 * v).toFixed(3)})`;
  },

  colormapRGBA(val) {
    const v = Math.min(1, Math.max(0, val));
    if (v < 0.03) return [0, 0, 0, 0];
    let r = 0, g = 0, b = 0;
    if (v < 0.33) {
      const t = v / 0.33;
      r = Math.round(0 * (1 - t) + 16 * t);
      g = Math.round(210 * (1 - t) + 185 * t);
      b = Math.round(255 * (1 - t) + 129 * t);
    } else if (v < 0.66) {
      const t = (v - 0.33) / 0.33;
      r = Math.round(16 * (1 - t) + 245 * t);
      g = Math.round(185 * (1 - t) + 158 * t);
      b = Math.round(129 * (1 - t) + 11 * t);
    } else {
      const t = (v - 0.66) / 0.34;
      r = Math.round(245 * (1 - t) + 244 * t);
      g = Math.round(158 * (1 - t) + 63 * t);
      b = Math.round(11 * (1 - t) + 94 * t);
    }
    const alpha = Math.round(Math.min(0.72, 0.12 + 0.62 * Math.pow(v, 0.8)) * 255);
    return [r, g, b, alpha];
  },

  draw() {
    if (!this.result) return;
    const r = this.result;
    const root = $("results");
    root.innerHTML = "";

    const header = document.createElement("div");
    header.className = "res-header";
    header.innerHTML = `
      <div style="display:flex; align-items:center; gap:12px;">
        <strong style="color:#fff; font-size:15px;">Reading Report</strong>
        <div class="res-tabs">
          <button class="res-tab ${this.tab === "replay" ? "active" : ""}" data-tab="replay">1. ▶ Watch Your Eyes Read</button>
          <button class="res-tab ${this.tab === "heatmap" ? "active" : ""}" data-tab="heatmap">2. 🔥 Attention Heatmap</button>
          <button class="res-tab ${this.tab === "telemetry" ? "active" : ""}" data-tab="telemetry">3. 📋 Plain-English Verdict &amp; Review</button>
          <button class="res-tab ${this.tab === "export" ? "active" : ""}" data-tab="export">4. ⬇ Save Report</button>
        </div>
      </div>
      <div class="res-controls">
        ${(this.tab === "heatmap" || this.tab === "replay") ? `
          <button class="ctrl-btn" id="btn-prev-pg" ${this.page === 0 ? "disabled" : ""}>← Prev Page</button>
          <span style="font:600 12px ui-monospace,monospace; color:#cbd5e1;">Page ${this.page + 1} of ${r.pages.length}</span>
          <button class="ctrl-btn" id="btn-next-pg" ${this.page >= r.pages.length - 1 ? "disabled" : ""}>Next Page →</button>
        ` : ""}
        ${this.tab === "heatmap" ? `
          <button class="ctrl-btn ${this.view === "gaussian" ? "active" : ""}" data-view="gaussian">Smooth Glow (G)</button>
          <button class="ctrl-btn ${this.view === "para" ? "active" : ""}" data-view="para">Paragraphs (P)</button>
          <button class="ctrl-btn ${this.view === "line" ? "active" : ""}" data-view="line">Lines (L)</button>
          <button class="ctrl-btn ${this.view === "scan" ? "active" : ""}" data-view="scan">Eye Path (S)</button>
        ` : ""}
        <button class="ctrl-btn" id="btn-new-session" style="border-color:var(--cyan); color:var(--cyan);">Read Another Passage (C)</button>
      </div>
    `;
    root.appendChild(header);

    header.querySelectorAll(".res-tab").forEach((b) => {
      b.addEventListener("click", () => {
        this.stopReplay();
        this.tab = b.dataset.tab;
        this.draw();
        if (this.tab === "replay") this.startReplay();
      });
    });
    header.querySelectorAll("[data-view]").forEach((b) => {
      b.addEventListener("click", () => { this.view = b.dataset.view; this.draw(); });
    });
    const prevBtn = header.querySelector("#btn-prev-pg");
    if (prevBtn) prevBtn.addEventListener("click", () => this.changePage(-1));
    const nextBtn = header.querySelector("#btn-next-pg");
    if (nextBtn) nextBtn.addEventListener("click", () => this.changePage(1));
    const newBtn = header.querySelector("#btn-new-session");
    if (newBtn) newBtn.addEventListener("click", () => hello());

    const body = document.createElement("div");
    body.className = "res-body";
    root.appendChild(body);

    if (this.tab === "heatmap" || this.tab === "replay") {
      this.renderStageAndSidebar(body);
    } else if (this.tab === "telemetry") {
      this.renderTelemetryDashboard(body);
    } else if (this.tab === "export") {
      this.renderExportView(body);
    }

    this.updateLegend();
  },

  changePage(delta) {
    if (!this.result) return;
    const next = Math.min(this.result.pages.length - 1, Math.max(0, this.page + delta));
    if (next !== this.page) {
      this.page = next;
      this.initPageReplayBounds();
      this.draw();
    }
  },

  renderStageAndSidebar(container) {
    const r = this.result;
    const sum = r.summary || {};
    const pg = r.pages[this.page];
    const d = pg.dwell[this.source];
    const [vw, vh] = r.viewport || [innerWidth, innerHeight];
    const pitch = r.line_pitch;

    const stageWrap = document.createElement("div");
    stageWrap.className = "res-stage-wrap";

    const vpContainer = document.createElement("div");
    vpContainer.className = "res-viewport-container";

    const pageWrap = document.createElement("div");
    pageWrap.className = "res-page-canvas-wrap";
    pageWrap.id = "res-page-stage";
    const availW = Math.max(400, innerWidth - 380);
    const availH = Math.max(300, innerHeight - (this.tab === "replay" ? 115 : 65));
    const scale = Math.min(1, availW / vw, availH / vh);
    pageWrap.style.width = `${vw}px`;
    pageWrap.style.height = `${vh}px`;
    if (scale < 0.99) {
      pageWrap.style.transform = `scale(${scale.toFixed(3)})`;
      pageWrap.style.transformOrigin = "top center";
    }

    const lmax = Math.max(...d.line_frac, 1e-9);
    const pmax = Math.max(...d.para_frac, 1e-9);
    const pfrac = Object.fromEntries(d.para_ids.map((id, i) => [id, d.para_frac[i]]));

    if (this.tab === "heatmap" && this.view === "para") {
      for (const pid of d.para_ids) {
        const ls = pg.lines.filter((l) => l.para_id === pid);
        const b = document.createElement("div");
        b.className = "box";
        const x0 = Math.min(...ls.map((l) => l.x0)), y0 = Math.min(...ls.map((l) => l.y0));
        Object.assign(b.style, {
          left: `${x0 - 12}px`, top: `${y0}px`,
          width: `${Math.max(...ls.map((l) => l.x1)) - x0 + 24}px`,
          height: `${Math.max(...ls.map((l) => l.y1)) - y0}px`,
          background: this.heat(pfrac[pid] / pmax),
        });
        pageWrap.appendChild(b);
      }
    }

    if (this.tab === "heatmap" && this.view === "gaussian") {
      const gCanvas = document.createElement("canvas");
      gCanvas.id = "gaussian-canvas";
      gCanvas.width = vw;
      gCanvas.height = vh;
      Object.assign(gCanvas.style, { position: "absolute", inset: "0", pointerEvents: "none", zIndex: "1" });
      pageWrap.appendChild(gCanvas);
      setTimeout(() => this.drawGaussianKDE(gCanvas, pg, r), 0);
    }

    pg.lines.forEach((l, i) => {
      if (this.tab === "heatmap" && this.view === "line") {
        const b = document.createElement("div");
        b.className = "box";
        Object.assign(b.style, {
          left: `${l.x0}px`, top: `${l.y0}px`,
          width: `${l.x1 - l.x0}px`, height: `${l.y1 - l.y0}px`,
          background: this.heat(d.line_frac[i] / lmax),
        });
        pageWrap.appendChild(b);
      }
      const t = document.createElement("div");
      t.className = "rline";
      Object.assign(t.style, {
        left: `${l.x0}px`, top: `${l.y0}px`,
        height: `${pitch}px`, lineHeight: `${pitch}px`,
        font: `${r.font_px}px Georgia, "Times New Roman", serif`,
      });
      t.textContent = l.text;
      pageWrap.appendChild(t);
    });

    if (this.tab === "heatmap" && this.view === "scan") {
      const ns = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(ns, "svg");
      svg.setAttribute("class", "scan");
      const fixs = pg.fixation_details || pg.fixations || [];
      const pts = fixs.map((f) => `${f.x},${f.y}`).join(" ");
      svg.innerHTML =
        `<polyline points="${pts}" fill="none" stroke="#0284c7" stroke-width="2.2" opacity=".65"/>` +
        fixs.map((f, idx) => {
          const col = f.regression ? "#f43f5e" : "#0284c7";
          const rad = (6 + 28 * Math.sqrt(f.dur)).toFixed(1);
          return `<circle cx="${f.x}" cy="${f.y}" r="${rad}" fill="${col}" opacity=".42"/>` +
                 `<text x="${f.x}" y="${f.y + 3}" text-anchor="middle" fill="#fff" font-size="10" font-family="monospace">${idx + 1}</text>`;
        }).join("");
      pageWrap.appendChild(svg);
    }

    if (this.tab === "replay") {
      const rCanvas = document.createElement("canvas");
      rCanvas.id = "replay-canvas";
      rCanvas.width = vw;
      rCanvas.height = vh;
      Object.assign(rCanvas.style, { position: "absolute", inset: "0", pointerEvents: "none", zIndex: "4" });
      pageWrap.appendChild(rCanvas);
    }

    vpContainer.appendChild(pageWrap);
    stageWrap.appendChild(vpContainer);

    // Right Sidebar: Plain-English Verdict + Paragraph Cards
    const sidebar = document.createElement("div");
    sidebar.className = "res-sidebar";
    sidebar.innerHTML = `
      <div style="background:linear-gradient(135deg, rgba(0,240,255,0.12), rgba(16,185,129,0.12)); border:1px solid rgba(0,240,255,0.35); border-radius:10px; padding:11px 13px;">
        <div style="font-size:11px; font-weight:700; text-transform:uppercase; color:var(--cyan); letter-spacing:0.6px; margin-bottom:4px;">
          📖 Overall Reading Verdict
        </div>
        <div style="font-size:12.5px; color:#f8fafc; line-height:1.4;">
          ${sum.verdict_headline || "Here is how your attention was distributed across this page."}
        </div>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px;">
        <span style="font-size:12px; font-weight:700; text-transform:uppercase; color:#cbd5e1;">
          Page ${this.page + 1} Paragraphs
        </span>
        <span class="badge">${r.granularity === "line" ? "Line-by-Line Clarity" : "Paragraph Clarity"}</span>
      </div>
    `;

    const pMetrics = pg.para_metrics || d.para_ids.map((pid, idx) => ({
      para_id: pid,
      dwell_s: d.para_s[idx],
      dwell_frac: d.para_frac[idx],
      wpm: 0,
      fixation_count: 0,
      regressions: 0,
      status: "Deep Read",
      insight: "",
    }));

    pMetrics.forEach((pm) => {
      const sc = this.statusColor(pm.status);
      const pct = (100 * (pm.dwell_frac || 0)).toFixed(0);
      const card = document.createElement("div");
      card.className = "para-card";
      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <strong style="color:#fff; font-size:13.5px;">Paragraph ${pm.para_id + 1}</strong>
          <span style="padding:2px 8px; border-radius:6px; font-size:11px; font-weight:600; background:${sc.bg}; color:${sc.fg}; border:1px solid ${sc.border};">
            ${this.friendlyBadge(pm)}
          </span>
        </div>
        <div class="para-bar-track">
          <div class="para-bar-fill" id="pbar-${pm.para_id}" style="width:${pct}%; background:${sc.fg};"></div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:6px; font:11.5px ui-monospace,monospace; color:#cbd5e1; margin-top:4px;">
          <div>Time: <b>${Number(pm.dwell_s).toFixed(1)}s</b> (${pct}%)</div>
          <div>Pace: <b>${pm.wpm > 0 ? Math.round(pm.wpm) + " WPM" : "Skipped"}</b></div>
          <div>Re-reads: <b>${pm.regressions || 0}</b></div>
        </div>
        ${pm.insight ? `<div style="margin-top:5px; font-size:11.5px; color:var(--text-muted); line-height:1.35;">${pm.insight}</div>` : ""}
      `;
      sidebar.appendChild(card);
    });

    stageWrap.appendChild(sidebar);
    container.appendChild(stageWrap);

    if (this.tab === "replay") {
      const rbar = document.createElement("div");
      rbar.className = "replay-bar";
      rbar.innerHTML = `
        <button class="ctrl-btn active" id="btn-replay-play" style="min-width:85px;">${this.playing ? "⏸ Pause" : "▶ Play"}</button>
        <button class="ctrl-btn ${this.speed === 0.5 ? "active" : ""}" data-spd="0.5">0.5×</button>
        <button class="ctrl-btn ${this.speed === 1.0 ? "active" : ""}" data-spd="1.0">1×</button>
        <button class="ctrl-btn ${this.speed === 2.0 ? "active" : ""}" data-spd="2.0">2×</button>
        <input type="range" class="replay-slider" id="replay-scrubber" min="0" max="${this.replayDuration.toFixed(2)}" step="0.02" value="${this.replayTime.toFixed(2)}">
        <span id="replay-time-label" style="font:600 12px ui-monospace,monospace; color:var(--cyan); min-width:110px; text-align:right;">
          ${this.replayTime.toFixed(1)}s / ${this.replayDuration.toFixed(1)}s
        </span>
      `;
      container.appendChild(rbar);

      rbar.querySelector("#btn-replay-play").addEventListener("click", () => {
        if (this.playing) this.stopReplay();
        else this.startReplay();
      });
      rbar.querySelectorAll("[data-spd]").forEach((b) => {
        b.addEventListener("click", () => {
          this.speed = Number(b.dataset.spd);
          rbar.querySelectorAll("[data-spd]").forEach((x) => x.classList.toggle("active", Number(x.dataset.spd) === this.speed));
        });
      });
      const scrub = rbar.querySelector("#replay-scrubber");
      scrub.addEventListener("input", (e) => {
        this.replayTime = Number(e.target.value);
        this.drawReplayFrame();
      });
      this.drawReplayFrame();
    }
  },

  drawGaussianKDE(canvas, pg, r) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const scale = 8;
    const gw = Math.ceil(w / scale), gh = Math.ceil(h / scale);
    const grid = new Float32Array(gw * gh);

    const sigX = Math.max(28, r.rms_x || 38) / scale;
    const sigY = Math.max(22, r.rms_y || 32) / scale;
    const inv2Sx2 = 0.5 / (sigX * sigX);
    const inv2Sy2 = 0.5 / (sigY * sigY);
    const radX = Math.ceil(sigX * 3), radY = Math.ceil(sigY * 3);

    const pts = [];
    if (this.source === "fixations" && pg.fixations && pg.fixations.length) {
      pg.fixations.forEach((f) => pts.push({ x: f.x / scale, y: f.y / scale, w: f.dur }));
    } else if (pg.samples_trace && pg.samples_trace.length) {
      pg.samples_trace.forEach((s) => {
        if (s.valid && s.x != null && s.y != null) pts.push({ x: s.x / scale, y: s.y / scale, w: 0.033 });
      });
    }

    let maxVal = 1e-9;
    for (const p of pts) {
      const gx0 = Math.max(0, Math.floor(p.x - radX)), gx1 = Math.min(gw - 1, Math.ceil(p.x + radX));
      const gy0 = Math.max(0, Math.floor(p.y - radY)), gy1 = Math.min(gh - 1, Math.ceil(p.y + radY));
      for (let gy = gy0; gy <= gy1; gy++) {
        const dy2 = (gy - p.y) * (gy - p.y) * inv2Sy2;
        const rowOff = gy * gw;
        for (let gx = gx0; gx <= gx1; gx++) {
          const dx2 = (gx - p.x) * (gx - p.x) * inv2Sx2;
          const v = p.w * Math.exp(-(dx2 + dy2));
          grid[rowOff + gx] += v;
          if (grid[rowOff + gx] > maxVal) maxVal = grid[rowOff + gx];
        }
      }
    }

    const offCanvas = document.createElement("canvas");
    offCanvas.width = gw;
    offCanvas.height = gh;
    const offCtx = offCanvas.getContext("2d");
    const imgData = offCtx.createImageData(gw, gh);
    const data = imgData.data;

    for (let i = 0; i < grid.length; i++) {
      const norm = grid[i] / maxVal;
      const [cr, cg, cb, ca] = this.colormapRGBA(norm);
      const idx = i * 4;
      data[idx] = cr;
      data[idx + 1] = cg;
      data[idx + 2] = cb;
      data[idx + 3] = ca;
    }
    offCtx.putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(offCanvas, 0, 0, w, h);
  },

  startReplay() {
    if (!this.result) return;
    if (this.tab !== "replay") {
      this.tab = "replay";
      this.draw();
    }
    if (this.replayTime >= this.replayDuration - 0.05) this.replayTime = 0;
    this.playing = true;
    this.lastFrameTs = performance.now();
    const playBtn = $("btn-replay-play");
    if (playBtn) playBtn.textContent = "⏸ Pause";

    const tick = (now) => {
      if (!this.playing) return;
      const dt = (now - (this.lastFrameTs || now)) / 1000;
      this.lastFrameTs = now;
      this.replayTime += dt * this.speed;
      if (this.replayTime >= this.replayDuration) {
        this.replayTime = this.replayDuration;
        this.drawReplayFrame();
        this.stopReplay();
        return;
      }
      this.drawReplayFrame();
      this.rafId = requestAnimationFrame(tick);
    };
    this.rafId = requestAnimationFrame(tick);
  },

  stopReplay() {
    this.playing = false;
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = null;
    const playBtn = $("btn-replay-play");
    if (playBtn) playBtn.textContent = "▶ Play";
  },

  drawReplayFrame() {
    const cv = $("replay-canvas");
    if (!cv || !this.result) return;
    const ctx = cv.getContext("2d");
    const w = cv.width, h = cv.height;
    ctx.clearRect(0, 0, w, h);

    const pg = this.result.pages[this.page];
    const tCur = this.replayTime;
    const tAbs = (pg.t_on || 0) + tCur;

    const scrub = $("replay-scrubber");
    if (scrub) scrub.value = tCur.toFixed(2);
    const lbl = $("replay-time-label");
    if (lbl) lbl.textContent = `${tCur.toFixed(1)}s / ${this.replayDuration.toFixed(1)}s`;

    const fixs = pg.fixation_details || pg.fixations || [];
    ctx.lineWidth = 2;
    for (let i = 0; i < fixs.length; i++) {
      const f = fixs[i];
      if (f.t0 > tAbs) break;
      const activeDur = Math.min(f.dur, Math.max(0.02, tAbs - f.t0));
      const rad = 6 + 26 * Math.sqrt(activeDur);
      const isReg = Boolean(f.regression);
      ctx.fillStyle = isReg ? "rgba(244, 63, 94, 0.35)" : "rgba(0, 240, 255, 0.28)";
      ctx.strokeStyle = isReg ? "#f43f5e" : "#0284c7";
      ctx.beginPath();
      ctx.arc(f.x, f.y, rad, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    const trace = (pg.samples_trace || []).filter((s) => s.valid && s.x != null && s.t <= tCur && s.t >= tCur - 1.2);
    if (trace.length >= 2) {
      for (let i = 1; i < trace.length; i++) {
        const p0 = trace[i - 1], p1 = trace[i];
        const age = Math.max(0, 1 - (tCur - p1.t) / 1.2);
        ctx.strokeStyle = `rgba(0, 240, 255, ${(age * 0.75).toFixed(2)})`;
        ctx.lineWidth = 1.5 + age * 3;
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();
      }
    }

    const curPt = trace.length ? trace[trace.length - 1] : null;
    if (curPt) {
      const grad = ctx.createRadialGradient(curPt.x, curPt.y, 4, curPt.x, curPt.y, 38);
      grad.addColorStop(0, "rgba(0, 240, 255, 0.75)");
      grad.addColorStop(0.5, "rgba(16, 185, 129, 0.35)");
      grad.addColorStop(1, "rgba(0, 240, 255, 0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(curPt.x, curPt.y, 38, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(curPt.x, curPt.y, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  },

  // ==================== VIEW 3: PLAIN-ENGLISH VERDICT, SMART REVIEW & TIMELINE ====================
  renderTelemetryDashboard(container) {
    const r = this.result;
    const sum = r.summary || {
      reading_duration_s: 0, overall_wpm: 0, total_words: 0, est_words_read: 0,
      focus_score: 0, valid_gaze_ratio: 0, regression_rate: 0,
      total_regressions: 0, paragraphs: [],
    };

    const reviewParas = (sum.paragraphs || []).filter(
      (p) => p.status !== "Deep Read"
    );

    const wrap = document.createElement("div");
    wrap.className = "telemetry-dashboard";
    wrap.innerHTML = `
      <!-- Top Plain-English Verdict Banner -->
      <div style="background:linear-gradient(135deg, rgba(0,240,255,0.14), rgba(16,185,129,0.14)); border:1px solid var(--cyan); border-radius:14px; padding:18px 22px; display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;">
        <div>
          <div style="font-size:12px; font-weight:700; text-transform:uppercase; color:var(--cyan); letter-spacing:0.9px;">
            📖 Plain-English Reading Verdict
          </div>
          <div style="font-size:18px; font-weight:700; color:#fff; margin-top:4px;">
            ${sum.verdict_headline || "Session complete — see your reading breakdown below."}
          </div>
        </div>
        <div style="display:flex; gap:8px;">
          <span class="badge ok">✅ ${sum.counts ? sum.counts.deep : 0} Read Carefully</span>
          <span class="badge err">🔄 ${sum.counts ? sum.counts.tricky : 0} Tricky / Re-read</span>
          <span class="badge warn">⚡ ${sum.counts ? (sum.counts.skimmed + sum.counts.skipped) : 0} Skimmed / Skipped</span>
        </div>
      </div>

      <!-- 5 Plain-English Metric Cards -->
      <div class="summary-kpi-row">
        <div class="kpi-box">
          <div class="k-title">Reading Pace</div>
          <div class="k-num" style="color:var(--cyan);">${sum.overall_wpm > 0 ? Math.round(sum.overall_wpm) + " WPM" : "Skimmed"}</div>
          <div class="k-sub">Read ~${sum.est_words_read ?? sum.total_words} of ${sum.total_words} words in ${(sum.reading_duration_s || 0).toFixed(0)}s (Normal: 180–250 WPM)</div>
        </div>
        <div class="kpi-box">
          <div class="k-title">Focus &amp; Attention</div>
          <div class="k-num" style="color:var(--emerald);">${(sum.focus_score || 0).toFixed(0)}%</div>
          <div class="k-sub">Eyes stayed on the page ${((sum.valid_gaze_ratio || 0) * 100).toFixed(0)}% of the time</div>
        </div>
        <div class="kpi-box">
          <div class="k-title">Backtracking (Re-reads)</div>
          <div class="k-num" style="color:var(--amber);">${sum.total_regressions || 0} times</div>
          <div class="k-sub">How often your eyes jumped back to re-read a sentence</div>
        </div>
        <div class="kpi-box">
          <div class="k-title">Webcam Tracking Clarity</div>
          <div class="k-num">${r.granularity === "line" ? "Line-by-Line" : "Paragraph"}</div>
          <div class="k-sub">Vertical error ±${r.rms_y.toFixed(0)} px vs ${r.line_pitch.toFixed(0)} px line height</div>
        </div>
      </div>

      <div class="viz-grid">
        <div class="hud-card viz-card">
          <div class="card-header">
            <span class="card-title">Attention by Paragraph</span>
            <span class="badge">DWELL TIME</span>
          </div>
          <canvas id="paragraph-chart" width="620" height="220"></canvas>
        </div>
        <div class="hud-card viz-card">
          <div class="card-header">
            <span class="card-title">Reading Mode Mix</span>
            <span class="badge">SESSION PROFILE</span>
          </div>
          <canvas id="status-chart" width="360" height="220"></canvas>
        </div>
      </div>

      ${reviewParas.length ? `
        <div class="hud-card" style="padding:16px 20px; border-color:rgba(245, 158, 11, 0.45);">
          <div class="card-header">
            <span class="card-title" style="color:var(--amber);">🎯 Smart Study &amp; Review Helper — Sections You Skipped, Skimmed, or Found Tricky</span>
            <span class="badge warn">${reviewParas.length} Paragraph(s) Flagged for Quick Review</span>
          </div>
          <div style="display:flex; flex-direction:column; gap:10px;">
            ${reviewParas.map((p) => `
              <div style="background:rgba(15,23,42,0.78); border:1px solid rgba(148,163,184,0.2); border-radius:10px; padding:12px 14px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                  <strong style="color:#fff;">Paragraph ${p.para_id + 1} — ${this.friendlyBadge(p)}</strong>
                  <span style="font-size:12px; color:var(--text-muted);">Spent ${p.dwell_s.toFixed(1)}s · ${p.regressions} re-reads</span>
                </div>
                <div style="font:14px/1.5 Georgia,serif; color:#e2e8f0; padding:8px 12px; background:rgba(2,6,23,0.6); border-radius:8px; margin-top:6px;">
                  "${p.text || ""}"
                </div>
              </div>
            `).join("")}
          </div>
        </div>
      ` : ""}

      <div>
        <h3 style="margin:0 0 10px; font-size:14px; color:var(--cyan); text-transform:uppercase; letter-spacing:0.8px;">
          Paragraph-by-Paragraph Breakdown
        </h3>
        <div class="cog-cards-grid" id="cog-cards"></div>
      </div>

      <div class="hud-card" style="padding:16px 20px;">
        <div class="card-header">
          <span class="card-title">Second-by-Second Reading Journey (0s → ${(sum.reading_duration_s || 0).toFixed(0)}s)</span>
          <span class="badge">GREEN = MOVING FORWARD · RED DOT = JUMPED BACK TO RE-READ</span>
        </div>
        <div style="height:175px; background:#040711; border-radius:10px; border:1px solid rgba(56,189,248,0.2); overflow:hidden;">
          <canvas id="timeline-canvas" width="1200" height="175" style="width:100%; height:100%; display:block;"></canvas>
        </div>
      </div>
    `;
    container.appendChild(wrap);

    const grid = wrap.querySelector("#cog-cards");
    (sum.paragraphs || []).forEach((p) => {
      const sc = this.statusColor(p.status);
      const card = document.createElement("div");
      card.className = "para-card";
      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <strong style="font-size:14.5px; color:#fff;">Paragraph ${p.para_id + 1} (Page ${(p.page || 0) + 1})</strong>
          <span style="padding:3px 9px; border-radius:6px; font-size:11.5px; font-weight:700; background:${sc.bg}; color:${sc.fg}; border:1px solid ${sc.border};">
            ${this.friendlyBadge(p)}
          </span>
        </div>
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:6px; font:12px ui-monospace,monospace; color:#e2e8f0; margin:8px 0;">
          <div>Time: <b>${p.dwell_s.toFixed(1)}s</b></div>
          <div>Share: <b>${(100 * p.dwell_frac).toFixed(0)}%</b></div>
          <div>Pace: <b>${p.wpm > 0 ? Math.round(p.wpm) + " WPM" : "Skipped"}</b></div>
          <div>Re-reads: <b>${p.regressions}</b></div>
        </div>
        <div style="font-size:12.5px; color:#cbd5e1; background:rgba(15,23,42,0.65); padding:8px 10px; border-radius:8px; border-left:3px solid ${sc.fg};">
          ${p.insight}
        </div>
      `;
      grid.appendChild(card);
    });

    setTimeout(() => this.drawTimelineChart(), 0);
    setTimeout(() => {
      this.drawParagraphChart();
      this.drawStatusChart();
    }, 0);
  },

  drawParagraphChart() {
    const cv = $("paragraph-chart");
    if (!cv || !this.result) return;
    const ctx = cv.getContext("2d"), ps = this.result.summary?.paragraphs || [];
    const w = cv.width, h = cv.height, left = 42, right = 18, top = 18, bottom = 34;
    ctx.clearRect(0, 0, w, h);
    if (!ps.length) return;
    const max = Math.max(...ps.map((p) => Number(p.dwell_s) || 0), 1);
    const plotW = w - left - right, plotH = h - top - bottom;
    ctx.font = "11px ui-monospace, monospace";
    ctx.textAlign = "center";
    ps.forEach((p, i) => {
      const value = Number(p.dwell_s) || 0;
      const bw = Math.max(12, plotW / ps.length - 10);
      const x = left + (i + 0.5) * plotW / ps.length;
      const bh = value / max * plotH;
      const sc = this.statusColor(p.status);
      ctx.fillStyle = sc.fg;
      ctx.fillRect(x - bw / 2, top + plotH - bh, bw, bh);
      ctx.fillStyle = "#cbd5e1";
      ctx.fillText(`P${i + 1}`, x, h - 12);
      ctx.fillStyle = "#f8fafc";
      ctx.fillText(`${value.toFixed(1)}s`, x, Math.max(12, top + plotH - bh - 6));
    });
    ctx.textAlign = "left";
    ctx.fillStyle = "#64748b";
    ctx.fillText("More dwell time", 8, 12);
  },

  drawStatusChart() {
    const cv = $("status-chart");
    if (!cv || !this.result) return;
    const ctx = cv.getContext("2d"), counts = this.result.summary?.counts || {};
    const items = [
      ["Read carefully", Number(counts.deep) || 0, "#34d399"],
      ["Tricky / re-read", Number(counts.tricky) || 0, "#fb7185"],
      ["Skimmed", Number(counts.skimmed) || 0, "#fbbf24"],
      ["Skipped", Number(counts.skipped) || 0, "#94a3b8"],
    ].filter((x) => x[1] > 0);
    const total = items.reduce((sum, item) => sum + item[1], 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (!total) return;
    let angle = -Math.PI / 2;
    const cx = 100, cy = 108, radius = 70;
    items.forEach(([, value, color]) => {
      const next = angle + (value / total) * Math.PI * 2;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, radius, angle, next); ctx.closePath();
      ctx.fillStyle = color; ctx.fill();
      angle = next;
    });
    ctx.beginPath(); ctx.arc(cx, cy, 38, 0, Math.PI * 2);
    ctx.fillStyle = "#0f172a"; ctx.fill();
    ctx.fillStyle = "#f8fafc"; ctx.font = "700 20px system-ui"; ctx.textAlign = "center";
    ctx.fillText(total, cx, cy + 7);
    ctx.font = "11px system-ui"; ctx.fillStyle = "#94a3b8"; ctx.fillText("paragraphs", cx, cy + 23);
    ctx.textAlign = "left";
    items.forEach(([label, value, color], i) => {
      const y = 40 + i * 36;
      ctx.fillStyle = color; ctx.fillRect(190, y - 9, 10, 10);
      ctx.fillStyle = "#e2e8f0"; ctx.font = "12px system-ui";
      ctx.fillText(`${label} · ${value}`, 210, y);
    });
  },

  drawTimelineChart() {
    const cv = $("timeline-canvas");
    if (!cv || !this.result) return;
    const ctx = cv.getContext("2d");
    const w = cv.width, h = cv.height;
    ctx.clearRect(0, 0, w, h);

    const tl = (this.result.timeline || []).filter((pt) => pt.para_id != null);
    if (tl.length < 2) {
      ctx.fillStyle = "#64748b";
      ctx.font = "13px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Not enough gaze samples recorded to plot timeline", w / 2, h / 2);
      return;
    }

    const maxT = Math.max(...tl.map((p) => p.t), 1);
    const maxPara = Math.max(...tl.map((p) => p.para_id), 1);
    const padL = 70, padR = 25, padT = 22, padB = 26;
    const pw = w - padL - padR, ph = h - padT - padB;

    ctx.font = "11px ui-monospace, monospace";
    ctx.fillStyle = "#94a3b8";
    for (let p = 0; p <= maxPara; p++) {
      const y = padT + (p / Math.max(1, maxPara)) * ph;
      ctx.strokeStyle = "rgba(148, 163, 184, 0.14)";
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.fillText(`Para ${p + 1}`, 12, y + 4);
    }

    ctx.lineWidth = 2.4;
    ctx.strokeStyle = "#00f0ff";
    ctx.beginPath();
    tl.forEach((pt, i) => {
      const x = padL + (pt.t / maxT) * pw;
      const y = padT + (pt.para_id / Math.max(1, maxPara)) * ph;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    tl.forEach((pt, i) => {
      const x = padL + (pt.t / maxT) * pw;
      const y = padT + (pt.para_id / Math.max(1, maxPara)) * ph;
      const isJumpBack = i > 0 && pt.para_id < tl[i - 1].para_id;
      ctx.fillStyle = isJumpBack ? "#f43f5e" : "#10b981";
      ctx.beginPath();
      ctx.arc(x, y, isJumpBack ? 4.5 : 2.5, 0, Math.PI * 2);
      ctx.fill();
    });
  },

  renderExportView(container) {
    const r = this.result;
    const wrap = document.createElement("div");
    wrap.className = "telemetry-dashboard";
    const baseUrl = this.sessionId ? `/sessions/${this.sessionId}` : "";

    const pageLinks = r.pages.map((_, idx) => `
      <a class="ctrl-btn" href="${baseUrl}/heatmap_p${idx + 1}.png" download="heatmap_p${idx + 1}.png" style="text-decoration:none;">
        ⬇ Page ${idx + 1} Heatmap Image (PNG)
      </a>
      <a class="ctrl-btn" href="${baseUrl}/scanpath_p${idx + 1}.png" download="scanpath_p${idx + 1}.png" style="text-decoration:none;">
        ⬇ Page ${idx + 1} Eye-Path Image (PNG)
      </a>
    `).join("");

    wrap.innerHTML = `
      <div class="hud-card" style="padding:24px;">
        <h2 style="margin:0 0 6px; color:#fff;">Save &amp; Share Your Reading Report</h2>
        <p style="margin:0 0 18px; color:var(--text-muted);">
          Your session files are saved on your computer in <code>${this.dir || "sessions/"}</code> and can also be downloaded below.
        </p>
        <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px;">
          <button class="btn-primary" id="btn-dl-html-report" type="button">⬇ Download Readable HTML Summary Report</button>
          <button class="btn-secondary" id="btn-dl-json" type="button">⬇ Download Raw Data (heatmap.json)</button>
          ${baseUrl ? pageLinks : ""}
        </div>
        <div style="background:#040711; border:1px solid rgba(56,189,248,0.2); border-radius:10px; padding:14px; font:12px/1.45 ui-monospace,monospace; color:#94a3b8; max-height:360px; overflow:auto;">
          <pre style="margin:0;">${JSON.stringify(r.summary || r, null, 2)}</pre>
        </div>
      </div>
    `;
    container.appendChild(wrap);

    wrap.querySelector("#btn-dl-json").addEventListener("click", () => {
      const blob = new Blob([JSON.stringify(this.result, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `gazeread_${this.sessionId || "session"}_heatmap.json`;
      a.click();
    });

    wrap.querySelector("#btn-dl-html-report").addEventListener("click", () => {
      const sum = r.summary || {};
      const html = `<!doctype html><html><head><meta charset="utf-8"><title>GazeRead Reading Report</title>
      <style>body{font:15px/1.5 system-ui,sans-serif;background:#0b0f19;color:#f1f5f9;padding:32px;max-width:960px;margin:0 auto}
      .card{background:#1e293b;border-radius:10px;padding:16px;margin:12px 0;border:1px solid #334155}
      h1,h2{color:#38bdf8}</style></head><body>
      <h1>GazeRead Reading Summary Report</h1>
      <div class="card"><b>Verdict:</b> ${sum.verdict_headline || ""}</div>
      <p>Session: <b>${this.sessionId}</b> · Tracking Clarity: <b>${r.granularity}</b> · Reading Pace: <b>${sum.overall_wpm || 0} WPM</b> · Focus: <b>${sum.focus_score || 0}%</b></p>
      <h2>Paragraph Breakdown</h2>
      ${(sum.paragraphs || []).map((p) => `<div class="card"><b>Paragraph ${p.para_id + 1} — ${this.friendlyBadge(p)}</b><br>Time Spent: ${p.dwell_s}s (${(100 * p.dwell_frac).toFixed(0)}%) · Pace: ${p.wpm ? Math.round(p.wpm) + " WPM" : "Skipped"} · Re-reads: ${p.regressions}<br><em>${p.insight}</em></div>`).join("")}
      </body></html>`;
      const blob = new Blob([html], { type: "text/html" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `gazeread_${this.sessionId || "session"}_report.html`;
      a.click();
    });
  },

  updateLegend() {
    if (!this.result) { $("legend").textContent = ""; return; }
    const r = this.result;
    const pitch = r.line_pitch;
    const viewName = {
      gaussian: "Smooth Attention Glow",
      line: "Line-by-Line Highlight",
      para: "Paragraph Highlight",
      scan: "Eye Movement Path",
    }[this.view] || this.view;
    let s = `Page ${this.page + 1} of ${r.pages.length}  ·  ${viewName}  ·  Tracking Clarity: ${r.granularity === "line" ? "Line-by-Line" : "Paragraph"}\n`;
    s += `Keys: 1-4 switch tabs · G smooth glow · P paragraphs · L lines · S eye path · ←/→ change page`;
    $("legend").textContent = s;
  },

  key(e) {
    if (!this.result) return;
    if (document.activeElement && document.activeElement.tagName === "TEXTAREA") return;
    const k = e.key.toLowerCase();
    if (k === "1") { this.tab = "replay"; this.startReplay(); return; }
    if (k === "2") { this.stopReplay(); this.tab = "heatmap"; this.draw(); return; }
    if (k === "3") { this.stopReplay(); this.tab = "telemetry"; this.draw(); return; }
    if (k === "4") { this.stopReplay(); this.tab = "export"; this.draw(); return; }
    if (k === "g") { this.tab = "heatmap"; this.view = "gaussian"; }
    else if (k === "l") { this.tab = "heatmap"; this.view = "line"; }
    else if (k === "p") { this.tab = "heatmap"; this.view = "para"; }
    else if (k === "s") { this.tab = "heatmap"; this.view = "scan"; }
    else if (k === "f") { this.source = this.source === "fixations" ? "samples" : "fixations"; }
    else if (e.key === "ArrowRight") { this.changePage(1); return; }
    else if (e.key === "ArrowLeft") { this.changePage(-1); return; }
    else if (e.key === " " && this.tab === "replay") {
      e.preventDefault();
      if (this.playing) this.stopReplay();
      else this.startReplay();
      return;
    } else return;
    this.draw();
  },

  clear() {
    this.stopReplay();
    this.result = null;
    $("results").innerHTML = "";
    $("legend").textContent = "";
    document.body.classList.remove("results");
  },
};

addEventListener("keydown", (e) => Heatmap.key(e));
