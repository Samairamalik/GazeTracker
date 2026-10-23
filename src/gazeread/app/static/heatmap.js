// Results view: redraws each page from the line boxes that were logged while reading,
// coloured by dwell. Keys: L line, P paragraph, S scanpath (coarse), F fixations/samples, ←/→ pages.
const Heatmap = {
  result: null, page: 0, view: "para", source: "fixations",

  handle(m) {
    if (m.type === "analysis" && Quiz.active) { Quiz.pending = m; return; }   // after the questions
    if (m.type === "analysis") {
      this.result = m.result;
      this.dir = m.dir;
      this.source = m.result.primary_source;
      this.view = m.result.granularity === "line" ? "line" : "para";
      this.page = 0;
      $("msg").style.display = "none";
      document.body.classList.remove("reading", "hide-cursor");
      document.body.classList.add("results");
      this.draw();
    } else if (m.type === "analysis_failed") {
      showMsg(`Analysis failed: ${m.reason}`);
    }
  },

  heat(f) { return `rgba(255, 90, 0, ${(0.06 + 0.6 * f).toFixed(3)})`; },

  draw() {
    const r = this.result, pg = r.pages[this.page], d = pg.dwell[this.source];
    const root = $("results");
    root.innerHTML = "";
    const pitch = r.line_pitch;
    const lmax = Math.max(...d.line_frac, 1e-9), pmax = Math.max(...d.para_frac, 1e-9);
    const pfrac = Object.fromEntries(d.para_ids.map((id, i) => [id, d.para_frac[i]]));
    if (this.view === "para") {
      for (const pid of d.para_ids) {
        const ls = pg.lines.filter((l) => l.para_id === pid);
        const b = document.createElement("div");
        b.className = "box";
        const x0 = Math.min(...ls.map((l) => l.x0)), y0 = Math.min(...ls.map((l) => l.y0));
        Object.assign(b.style, { left: `${x0 - 10}px`, top: `${y0}px`,
          width: `${Math.max(...ls.map((l) => l.x1)) - x0 + 20}px`,
          height: `${Math.max(...ls.map((l) => l.y1)) - y0}px`, background: this.heat(pfrac[pid] / pmax) });
        b.title = `paragraph ${pid}: ${(100 * pfrac[pid]).toFixed(0)}% of page time`;
        root.appendChild(b);
      }
    }
    pg.lines.forEach((l, i) => {
      if (this.view === "line") {
        const b = document.createElement("div");
        b.className = "box";
        Object.assign(b.style, { left: `${l.x0}px`, top: `${l.y0}px`, width: `${l.x1 - l.x0}px`,
          height: `${l.y1 - l.y0}px`, background: this.heat(d.line_frac[i] / lmax) });
        root.appendChild(b);
      }
      const t = document.createElement("div");
      t.className = "rline";
      Object.assign(t.style, { left: `${l.x0}px`, top: `${l.y0}px`, height: `${pitch}px`,
        lineHeight: `${pitch}px`, font: `${r.font_px}px Georgia, "Times New Roman", serif` });
      t.textContent = l.text;
      root.appendChild(t);
    });
    if (this.view === "scan") {
      const ns = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(ns, "svg");
      svg.setAttribute("class", "scan");
      const pts = pg.fixations.map((f) => `${f.x},${f.y}`).join(" ");
      svg.innerHTML = `<polyline points="${pts}" fill="none" stroke="#1f5fbf" stroke-width="2" opacity=".6"/>` +
        pg.fixations.map((f) => `<circle cx="${f.x}" cy="${f.y}" r="${6 + 30 * Math.sqrt(f.dur)}" fill="#1f5fbf" opacity=".4"/>`).join("");
      root.appendChild(svg);
    }
    const viewName = { line: "line heatmap", para: "paragraph heatmap", scan: "scanpath (coarse)" }[this.view];
    const lineNote = r.granularity === "line" ? "" : "  ·  line-level: low confidence";
    let s = `${r.source === "synthetic" ? "SYNTHETIC (not a measurement)  ·  " : ""}`;
    s += `page ${this.page + 1}/${r.pages.length}  ·  ${viewName}  ·  source ${this.source}\n`;
    s += `granularity ${r.granularity} (σy ${r.rms_y.toFixed(0)} px vs line pitch ${pitch.toFixed(0)} px)${lineNote}\n`;
    s += `L line · P paragraph · S scanpath · F fixations/samples · ←/→ page · saved in ${this.dir}`;
    $("legend").textContent = s;
  },

  key(e) {
    if (!this.result) return;
    const k = e.key.toLowerCase();
    if (k === "l") this.view = "line";
    else if (k === "p") this.view = "para";
    else if (k === "s") this.view = "scan";
    else if (k === "f") this.source = this.source === "fixations" ? "samples" : "fixations";
    else if (e.key === "ArrowRight") this.page = Math.min(this.page + 1, this.result.pages.length - 1);
    else if (e.key === "ArrowLeft") this.page = Math.max(this.page - 1, 0);
    else return;
    this.draw();
  },

  clear() {
    this.result = null;
    $("results").innerHTML = "";
    $("legend").textContent = "";
    document.body.classList.remove("results");
  },
};

addEventListener("keydown", (e) => Heatmap.key(e));
