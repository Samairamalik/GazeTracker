// Paged reading view. The whole text is measured once in a hidden column, split into lines,
// then into pages of whole paragraphs. Each shown page reports its line boxes (viewport CSS px)
// to the server as a `page` event; Space advances, and after the last page `reading_end` is sent.
const Reader = {
  cfg: null, paras: [], pages: [], cur: -1, state: "idle",   // idle | ready | reading | done

  async handle(m) {
    if (m.type === "config") {
      this.cfg = m.reading;
      const txt = await (await fetch(`texts/${m.reading.text}.txt`)).text();
      this.paras = txt.split(/\n\s*\n/).map((p) => p.replace(/\s+/g, " ").trim()).filter(Boolean);
    } else if (m.type === "calib_done") {
      this.state = "ready";
      showMsg("Press Space to start reading. Space = next page.");
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
      div.innerHTML = p.split(" ").map((w) => `<span>${w.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</span>`).join(" ");
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
      if (i > 0) pd.style.marginTop = `${this.cfg.para_gap_lines * pitch}px`;
      for (const ln of para.lines) {
        const ld = document.createElement("div");
        ld.className = "line";
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
      col.children[i].querySelectorAll(".line").forEach((ld, j) => {
        const sr = ld.firstChild.getBoundingClientRect(), lr = ld.getBoundingClientRect();
        boxes.push({ line_id: para.lines[j].id, para_id: para.pid, x0: sr.left, y0: lr.top,
                     x1: sr.right, y1: lr.bottom, text: para.lines[j].text });
      });
    });
    return boxes;
  },

  start() {
    $("msg").style.display = "none";
    document.body.classList.add("reading");      // must be visible before measuring the layout
    document.body.classList.toggle("hide-cursor", !this.cfg.show_cursor);
    this.pages = this.paginate(this.measure());
    this.cur = -1;
    this.state = "reading";
    this.next();
  },

  next() {
    this.cur++;
    if (this.cur >= this.pages.length) {
      this.state = "done";
      $("column").innerHTML = "";
      $("pageno").textContent = "";
      send({ type: "reading_end" });
      if (!Quiz.start()) showMsg("Analysing…");
      return;
    }
    const lines = this.render(this.cur);
    const last = this.cur === this.pages.length - 1;
    $("pageno").textContent = `Page ${this.cur + 1} of ${this.pages.length} · Space = ${last ? "finish" : "next page"}`;
    send({ type: "page", page: this.cur, n_pages: this.pages.length, lines,
           line_pitch: this.pitch(), font_px: this.cfg.font_px });
  },

  reset() {
    this.state = "idle";
    $("column").innerHTML = "";
    document.body.classList.remove("reading", "hide-cursor");
  },
};

let lastPageAt = 0;   // ignore Space for 1 s after a page appears, so a double press can't skip a page
addEventListener("keydown", (e) => {
  if (e.key !== " ") return;
  e.preventDefault();
  if (performance.now() - lastPageAt < 1000) return;
  lastPageAt = performance.now();
  if (Reader.state === "ready") Reader.start();
  else if (Reader.state === "reading") Reader.next();
});
