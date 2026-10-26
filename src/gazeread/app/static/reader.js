// Stage 3: Modern Editorial Reader + Zero-Effort Micro-Drift Lock + Auto-Resume Gaze Bookmarking.
// The whole text is measured once in a hidden column, split into lines, then into pages of whole
// paragraphs. Each shown page reports its line boxes (viewport CSS px) to the server as a `page` event.
const Reader = {
  cfg: null,
  paras: [],
  pages: [],
  cur: -1,
  state: "idle", // idle | ready | reading | done
  currentBoxes: [],
  lastAttendedLineIdx: 0,
  invalidSince: null,
  bookmarkTimer: null,

  async handle(m) {
    if (m.type === "config") {
      this.cfg = m.reading;
      if (m.reading.custom_text && m.reading.custom_text.trim()) {
        this.paras = m.reading.custom_text
          .split(/\n\s*\n/)
          .map((p) => p.replace(/\s+/g, " ").trim())
          .filter(Boolean);
      } else {
        const txt = await (await fetch(`texts/${m.reading.text}.txt`)).text();
        this.paras = txt
          .split(/\n\s*\n/)
          .map((p) => p.replace(/\s+/g, " ").trim())
          .filter(Boolean);
      }
    } else if (m.type === "calib_done") {
      this.state = "ready";
    }
  },

  pitch() { return this.cfg.font_px * this.cfg.line_height; },

  column() {
    const col = $("column");
    col.style.font = `${this.cfg.font_px}px Georgia, "Times New Roman", serif`;
    col.style.lineHeight = `${this.pitch()}px`;
    col.style.width = `${this.cfg.column_ch}ch`;
    col.style.top = `${this.cfg.margin_y_px}px`;
    col.style.left = `${Math.max(0, (innerWidth - col.offsetWidth) / 2)}px`;
    return col;
  },

  // Measure: wrap every word in a span, group spans into lines by their top offset.
  measure() {
    const col = this.column();
    col.style.visibility = "hidden";
    col.innerHTML = "";
    const paraLines = this.paras.map((p) => {
      const div = document.createElement("div");
      div.innerHTML = p
        .split(" ")
        .map((w) => `<span>${w.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</span>`)
        .join(" ");
      col.appendChild(div);
      const lines = [];
      let top = null;
      for (const s of div.children) {
        if (s.offsetTop !== top) { lines.push([]); top = s.offsetTop; }
        lines[lines.length - 1].push(s.textContent);
      }
      return lines.map((ws) => ws.join(" "));
    });
    col.innerHTML = "";
    col.style.visibility = "visible";
    return paraLines;
  },

  // Pages of whole paragraphs that fit the viewport height.
  paginate(paraLines) {
    const pitch = this.pitch(), gap = this.cfg.para_gap_lines * pitch;
    const avail = innerHeight - 2 * this.cfg.margin_y_px;
    const pages = [];
    let page = [], h = 0, lineId = 0;
    paraLines.forEach((lines, pid) => {
      const ph = lines.length * pitch;
      if (page.length && h + gap + ph > avail) { pages.push(page); page = []; h = 0; }
      h += (page.length ? gap : 0) + ph;
      page.push({ pid, lines: lines.map((text) => ({ id: lineId++, text })) });
    });
    if (page.length) pages.push(page);
    return pages;
  },

  render(k) {
    const col = this.column(), pitch = this.pitch();
    col.innerHTML = "";
    this.pages[k].forEach((para, i) => {
      const pd = document.createElement("div");
      pd.className = "para-block";
      pd.dataset.paraId = para.pid;
      if (i > 0) pd.style.marginTop = `${this.cfg.para_gap_lines * pitch}px`;
      for (const ln of para.lines) {
        const ld = document.createElement("div");
        ld.className = "line";
        ld.dataset.lineId = ln.id;
        ld.style.height = `${pitch}px`;
        const span = document.createElement("span");
        span.textContent = ln.text;
        ld.appendChild(span);
        pd.appendChild(ld);
      }
      col.appendChild(pd);
    });
    const boxes = [];
    this.pages[k].forEach((para, i) => {
      const pEl = col.children[i];
      pEl.querySelectorAll(".line").forEach((ld, j) => {
        const sr = ld.firstChild.getBoundingClientRect(), lr = ld.getBoundingClientRect();
        boxes.push({
          line_id: para.lines[j].id,
          para_id: para.pid,
          x0: sr.left,
          y0: lr.top,
          x1: sr.right,
          y1: lr.bottom,
          text: para.lines[j].text,
          el: ld,
          pEl,
        });
      });
    });
    this.currentBoxes = boxes;
    return boxes.map(({ el, pEl, ...rest }) => rest);
  },

  pulseDriftAnchor(lines) {
    const anchor = $("drift-anchor");
    if (!anchor || !lines || !lines.length) return;
    const first = lines[0];
    anchor.style.left = `${first.x0 + 20}px`;
    anchor.style.top = `${0.5 * (first.y0 + first.y1)}px`;
    anchor.style.display = "none";
    void anchor.offsetWidth;
    anchor.style.display = "block";
    setTimeout(() => { anchor.style.display = "none"; }, 400);
  },

  onGaze(m) {
    if (this.state !== "reading" || !this.currentBoxes.length) return;
    const now = performance.now();
    const inBounds = m.valid && m.x != null && m.y != null &&
      m.x >= 0 && m.x <= innerWidth && m.y >= 0 && m.y <= innerHeight;

    if (!inBounds) {
      if (this.invalidSince === null) this.invalidSince = now;
      return;
    }

    // Check if returning from a look-away (>= 2.5 seconds) -> Auto-Resume Gaze Bookmark!
    if (this.invalidSince !== null && now - this.invalidSince >= 2500) {
      this.triggerAutoResume();
    }
    this.invalidSince = null;

    // Update last attended line index
    let bestIdx = this.lastAttendedLineIdx;
    let bestDist = Infinity;
    this.currentBoxes.forEach((b, idx) => {
      const cy = 0.5 * (b.y0 + b.y1);
      const d = Math.abs(m.y - cy);
      if (d < bestDist && d <= this.pitch() * 1.5) {
        bestDist = d;
        bestIdx = idx;
      }
    });
    this.lastAttendedLineIdx = bestIdx;
    const activeBox = this.currentBoxes[bestIdx];
    if (activeBox && activeBox.pEl) {
      document.querySelectorAll(".para-block").forEach((el) => {
        el.classList.toggle("active-para", el === activeBox.pEl);
      });
    }
  },

  triggerAutoResume() {
    const box = this.currentBoxes[this.lastAttendedLineIdx];
    if (!box || !box.el) return;
    this.currentBoxes.forEach((b) => b.el && b.el.classList.remove("bookmark-glow"));
    box.el.classList.add("bookmark-glow");
    const banner = $("auto-resume-banner");
    if (banner) {
      banner.style.display = "block";
      clearTimeout(this.bookmarkTimer);
      this.bookmarkTimer = setTimeout(() => {
        banner.style.display = "none";
        if (box.el) box.el.classList.remove("bookmark-glow");
      }, 2400);
    }
  },

  start() {
    if ($("calib-report")) $("calib-report").style.display = "none";
    $("msg").style.display = "none";
    document.body.classList.add("reading");
    document.body.classList.toggle("hide-cursor", !this.cfg.show_cursor);
    this.pages = this.paginate(this.measure());
    this.cur = -1;
    this.state = "reading";
    this.invalidSince = null;
    this.next();
  },

  next() {
    this.cur++;
    if (this.cur >= this.pages.length) {
      this.state = "done";
      $("column").innerHTML = "";
      $("pageno").textContent = "";
      if ($("reader-progress")) $("reader-progress").style.width = "100%";
      send({ type: "reading_end" });
      showMsg("Computing Reading & Attention Analytics…");
      return;
    }
    const lines = this.render(this.cur);
    this.lastAttendedLineIdx = 0;
    this.pulseDriftAnchor(lines);
    const last = this.cur === this.pages.length - 1;
    $("pageno").textContent = `Page ${this.cur + 1} of ${this.pages.length} · Space = ${last ? "finish & analyze" : "next page"}`;
    if ($("btn-next-page")) {
      $("btn-next-page").textContent = last ? "Finish Reading (Space) ✓" : "Next Page (Space) →";
    }
    if ($("reader-progress")) {
      $("reader-progress").style.width = `${((this.cur + 1) / this.pages.length) * 100}%`;
    }
    send({
      type: "page",
      page: this.cur,
      n_pages: this.pages.length,
      lines,
      line_pitch: this.pitch(),
      font_px: this.cfg.font_px,
    });
  },

  reset() {
    this.state = "idle";
    this.currentBoxes = [];
    $("column").innerHTML = "";
    if ($("reader-progress")) $("reader-progress").style.width = "0%";
    document.body.classList.remove("reading", "hide-cursor");
  },
};

let lastPageAt = 0; // ignore Space for 0.8 s after a page appears so a double press can't skip a page
addEventListener("keydown", (e) => {
  if (e.key !== " ") return;
  if (document.activeElement && document.activeElement.tagName === "TEXTAREA") return;
  if (document.body.classList.contains("results")) return;
  e.preventDefault();
  if (performance.now() - lastPageAt < 800) return;
  lastPageAt = performance.now();
  if (Reader.state === "ready") Reader.start();
  else if (Reader.state === "reading") Reader.next();
});
