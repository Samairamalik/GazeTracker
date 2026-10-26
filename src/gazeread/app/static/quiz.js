// Comprehension questions after the last page (human check). Keys 1-4 or mouse clicks answer;
// the page has no answer key: every answer goes to the server, which scores and logs it.
const Quiz = {
  qs: [], i: 0, active: false, pending: null,

  handle(m) {
    if (m.type === "config") this.qs = m.questions || [];
  },

  start() {
    if (!this.qs.length) return false;
    this.active = true;
    this.i = 0;
    $("quiz").style.display = "flex";
    this.render();
    return true;
  },

  render() {
    const q = this.qs[this.i];
    $("quiz").innerHTML =
      `<div class="qbox"><div class="qnum">Comprehension Check · Question ${this.i + 1} of ${this.qs.length}</div>` +
      `<div class="qtext"></div>` +
      q.options.map((_, k) => `<div class="qopt" data-idx="${k}"><b>${k + 1}</b> <span></span></div>`).join("") +
      `<div class="qhint">Press <b>1–4</b> or click an option</div></div>`;
    $("quiz").querySelector(".qtext").textContent = q.text;
    $("quiz").querySelectorAll(".qopt").forEach((el, k) => {
      el.querySelector("span").textContent = q.options[k];
      el.addEventListener("click", () => this.answer(k));
    });
  },

  answer(k) {
    if (!this.active || k < 0 || k >= this.qs[this.i].options.length) return;
    send({ type: "answer", question_id: this.qs[this.i].id, response: k });
    this.i++;
    if (this.i < this.qs.length) this.render();
    else this.finish();
  },

  key(e) {
    if (!this.active) return false;
    const k = "1234".indexOf(e.key);
    if (k < 0 || k >= this.qs[this.i].options.length) return true;
    this.answer(k);
    return true;
  },

  finish() {
    this.active = false;
    $("quiz").style.display = "none";
    if (this.pending) { const m = this.pending; this.pending = null; Heatmap.handle(m); }
    else showMsg("Analysing…");
  },
};

addEventListener("keydown", (e) => { if (Quiz.active) { e.stopImmediatePropagation(); Quiz.key(e); } }, true);
