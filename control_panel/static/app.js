"use strict";
// Rachel control panel — single page, hash-routed, no build step.

// ─── helpers ─────────────────────────────────────────────
// append/replaceChildren take the same kids as h(): nested arrays flattened, null/false skipped.
for (const name of ["append", "replaceChildren"]) {
  const native = Element.prototype[name];
  Element.prototype[name] = function (...kids) { return native.apply(this, kids.flat(Infinity).filter(k => k != null && k !== false)); };
}
const $ = (s, r = document) => r.querySelector(s);
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else if (k in el && typeof v !== "string") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
async function api(path, opts = {}) {
  const init = { method: opts.method || (opts.body !== undefined ? "POST" : "GET"), headers: {} };
  if (opts.body !== undefined) { init.body = JSON.stringify(opts.body); init.headers["Content-Type"] = "application/json"; }
  const r = await fetch(path, init);
  let data = null;
  try { data = await r.json(); } catch { /* not json */ }
  if (!r.ok) throw new Error((data && (data.detail || data.error)) || `${r.status} ${r.statusText}`);
  return data;
}
function toast(msg, kind = "info", ms = 4500) {
  const t = h("div", { class: `toast ${kind}` }, msg);
  $("#toasts").append(t);
  setTimeout(() => t.remove(), ms);
}
const fmtTime = ts => ts ? new Date(ts * 1000).toLocaleTimeString() : "";
const fmtDur = s => s == null ? "" : s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${Math.floor(s / 3600)}h ${Math.floor(s % 3600 / 60)}m`;
const kv = (cls, ...pairs) => h("div", { class: `kv ${cls}` }, pairs.map(p => h("span", {}, p ?? "—")));
const qs = o => Object.entries(o).filter(([, v]) => v !== "" && v != null).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join("&");
const imgUrl = (p, base = "repo") => `/api/data/img?${qs({ p, base })}`;
const pretty = o => JSON.stringify(o, null, 2);
async function guard(fn, okMsg) {
  try { const r = await fn(); if (okMsg) toast(okMsg, "ok"); return r; }
  catch (e) { toast(e.message, "error", 8000); return null; }
}
function openDrawer(title, ...content) {
  $("#drawer-title").textContent = title;
  $("#drawer-body").replaceChildren(...content.flat());
  $("#drawer").classList.remove("hidden");
}
$("#drawer-close").onclick = () => $("#drawer").classList.add("hidden");
document.addEventListener("keydown", e => { if (e.key === "Escape") { $("#drawer").classList.add("hidden"); $(".lightbox")?.remove(); } });
function lightbox(src) { document.body.append(h("div", { class: "lightbox", onclick: e => e.currentTarget.remove() }, h("img", { src }))); }
function frames(paths, base = "repo", big = false) {
  if (!paths || !paths.length) return null;
  return h("div", { class: "frames" + (big ? " big" : "") }, paths.map(p => h("img", { src: imgUrl(p, base), loading: "lazy", title: p, onclick: e => { e.stopPropagation(); lightbox(imgUrl(p, base)); } })));
}
function pager(total, offset, limit, go) {
  return h("div", { class: "pager" },
    h("button", { class: "small", disabled: offset <= 0, onclick: () => go(Math.max(0, offset - limit)) }, "‹ prev"),
    h("span", { class: "muted small" }, total ? `${offset + 1}–${Math.min(offset + limit, total)} of ${total}` : "0 items"),
    h("button", { class: "small", disabled: offset + limit >= total, onclick: () => go(offset + limit) }, "next ›"));
}

// ─── shared state + websocket ────────────────────────────
const S = { services: {}, logs: [], jobs: {}, statusLines: {}, dream: null, page: null, timers: [] };
const LOG_CAP = 6000;
let ws;
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => $("#ws-state").className = "dot on";
  ws.onclose = () => { $("#ws-state").className = "dot crashed"; setTimeout(connect, 2000); };
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.type === "backfill") { S.logs = m.entries; S.statusLines = m.status_lines || {}; pageHook("onBackfill"); }
    else if (m.type === "log") { S.logs.push(m); if (S.logs.length > LOG_CAP) S.logs.splice(0, S.logs.length - LOG_CAP); pageHook("onLog", m); }
    else if (m.type === "status_line") { S.statusLines[m.source] = m.text; pageHook("onStatusLine", m); }
    else if (m.type === "service") { S.services[m.service.name] = m.service; renderMini(); pageHook("onService", m.service); }
    else if (m.type === "job") { S.jobs[m.job.id] = m.job; pageHook("onJob", m.job); }
    else if (m.type === "dream") { S.dream = m.chain; pageHook("onDream", m.chain); }
  };
}
function pageHook(name, arg) { const p = PAGES[S.page]; if (p && p[name]) p[name](arg); }
async function loadServices() {
  const d = await api("/api/services");
  for (const s of d.services) S.services[s.name] = s;
  renderMini();
}
function renderMini() {
  $("#svc-mini").replaceChildren(...Object.values(S.services).map(s =>
    h("div", { title: `${s.title}: ${s.state}` }, h("span", { class: `dot ${s.state}` }), s.name)));
}
function every(ms, fn) { fn(); S.timers.push(setInterval(fn, ms)); }

// ─── Stream page ─────────────────────────────────────────
const Stream = {
  render(main) {
    const preflight = h("input", { type: "checkbox", checked: true });
    main.append(
      h("div", { class: "row section" },
        h("h2", { class: "grow", style: { margin: 0 } }, "Stream"),
        h("label", {}, preflight, " preflight first"),
        h("button", { class: "primary", onclick: async () => {
          const r = await guard(() => api("/api/stack/start", { body: { preflight: preflight.checked } }));
          if (r && r.error) toast(r.error, "error", 8000); else if (r) toast("starting stack…", "ok");
        } }, "▶ Go live"),
        h("button", { onclick: () => guard(() => api("/api/stack/stop", { body: {} }), "stack stopped") }, "■ Stop stack"),
        h("button", { class: "danger", title: "also stops services adopted as external (e.g. a hand-started llama-server)",
          onclick: () => confirm("Stop every service, including external ones?") && guard(() => api("/api/stack/stop", { body: { include_external: true } }), "all stopped") }, "Stop all")),
      h("div", { id: "strip", class: "strip section" }),
      h("div", { id: "svc-grid", class: "grid g3 section" }),
      h("div", { class: "grid", style: { gridTemplateColumns: "minmax(300px, 380px) 1fr" } },
        Stream.controls(), h("div", { class: "card" }, h("h3", {}, "Live reactions (today)"), h("div", { id: "feed", class: "col" }))));
    this.cards = {};
    Object.values(S.services).forEach(s => this.onService(s));
    every(2000, () => this.status());
    every(4000, () => this.feed());
  },
  onService(s) {
    const grid = $("#svc-grid"); if (!grid) return;
    let c = this.cards[s.name];
    if (!c) { c = this.cards[s.name] = this.card(s); grid.append(c.el); }
    c.update(s);
  },
  onStatusLine(m) { const c = this.cards[m.source]; if (c) c.line.textContent = m.text; },
  card(s) {
    const dot = h("span", { class: "dot" }), badge = h("span", { class: "badge" }), meta = h("div", { class: "svc-meta" });
    const line = h("div", { class: "svc-line" }), err = h("div", { class: "small", style: { color: "var(--err)" } });
    const rr = h("span", { class: "badge warn hidden" }, "restart required");
    const inStack = h("input", { type: "checkbox", onchange: () => api(`/api/services/${s.name}/stack`, { body: { in_stack: inStack.checked } }) });
    const act = a => guard(() => api(`/api/services/${s.name}/${a}`, { body: {} }));
    const stop = async () => {
      const r = await act("stop");
      if (r && r.needs_confirm && confirm(`${s.name} was not started by the panel (external, pid ${r.pid}). Kill it?`))
        guard(() => api(`/api/services/${s.name}/stop`, { body: { force_external: true } }));
    };
    const inputs = {};
    const flags = h("div", { class: "flags" }, s.flag_schema.map(f => {
      let inp;
      if (f.type === "bool") inp = h("input", { type: "checkbox" });
      else if (f.type === "choice") inp = h("select", {}, f.choices.map(c => h("option", { value: c }, c)));
      else inp = h("input", { type: f.type === "str" ? "text" : "number", step: "any", placeholder: f.default ?? "" });
      inputs[f.name] = [inp, f];
      return [h("label", { title: f.help }, f.name), inp];
    }));
    const setFlagInputs = v => { for (const [k, [inp, f]] of Object.entries(inputs)) {
      const val = v[k];
      if (f.type === "bool") inp.checked = !!val; else inp.value = val ?? "";
    } };
    setFlagInputs(s.flags);
    const saveFlags = async () => {
      const out = {};
      for (const [k, [inp, f]] of Object.entries(inputs)) {
        out[k] = f.type === "bool" ? inp.checked : f.type === "int" ? (inp.value === "" ? null : parseInt(inp.value))
          : f.type === "float" ? (inp.value === "" ? null : parseFloat(inp.value)) : inp.value;
      }
      await guard(() => api(`/api/services/${s.name}/flags`, { body: { flags: out } }), "flags saved — restart to apply");
    };
    const el = h("div", { class: "card" },
      h("div", { class: "svc-head" }, dot, h("b", {}, s.title), badge),
      meta, rr, err, line,
      h("div", { class: "row", style: { marginTop: "8px" } },
        h("button", { class: "small", onclick: () => act("start") }, "Start"),
        h("button", { class: "small", onclick: stop }, "Stop"),
        h("button", { class: "small", onclick: () => {
          const cur = S.services[s.name];
          if (cur.state === "external" && !confirm(`${s.name} was not started by the panel (pid ${cur.pid}). Kill it and start the panel's own copy?`)) return;
          act("restart");
        } }, "Restart"),
        h("button", { class: "small ghost", onclick: () => { location.hash = "#console"; Console.select = s.name; } }, "logs ›"),
        h("label", { class: "grow", style: { textAlign: "right" } }, inStack, " in stack")),
      s.flag_schema.length ? h("details", {}, h("summary", {}, "flags"), flags,
        h("div", { class: "row", style: { marginTop: "6px" } }, h("button", { class: "small", onclick: saveFlags }, "Save flags"))) : null);
    return { el, line, update(s) {
      dot.className = `dot ${s.state}`; badge.className = `badge ${s.state}`; badge.textContent = s.state;
      meta.replaceChildren(...[s.port && `:${s.port}`, s.pid && `pid ${s.pid}`, s.uptime_s != null && `up ${fmtDur(s.uptime_s)}`,
        s.restarts && `${s.restarts} restarts`, s.next_restart_in != null && `restart in ${s.next_restart_in}s`,
        s.exit_code != null && `exit ${s.exit_code}`, s.gpu && "GPU"].filter(Boolean).map(t => h("span", {}, t)));
      rr.classList.toggle("hidden", !s.restart_required);
      err.textContent = s.error || "";
      line.textContent = S.statusLines[s.name] || s.last_line || "";
      inStack.checked = s.in_stack;
    } };
  },
  async status() {
    const d = await api("/api/stream/status").catch(() => null);
    const strip = $("#strip"); if (!d || !strip) return;
    const st = d.bridge?.stream || {}, o = d.orchestrator || {};
    const lora = (d.lora || []).map(a => `${a.path.split(/[\\/]/).pop()}×${a.scale}`).join(", ");
    const stat = (k, v, title) => h("div", { class: "stat", title }, h("div", { class: "k" }, k), h("div", { class: "v" }, v ?? "—"));
    strip.replaceChildren(
      stat("Stream", d.bridge ? (st.active ? "🔴 live" : "offline") : "bridge down", st.duration),
      stat("Game / scene", d.bridge ? `${st.game || "?"} · ${st.scene || "?"}` : "—"),
      stat("Viewers", st.viewers),
      stat("Speaking", d.speaking ? "🗣 yes (mute lock)" : "no"),
      stat("Orchestrator", d.orchestrator ? (o.busy ? "⏳ reacting" : o.engine ? "idle" : "no engine") : "down",
        o.transport && `transport=${o.transport} prompt=${o.prompt_version}`),
      stat("Window", d.orchestrator && o.engine ? `${o.window_age_s}s · ${o.speech_n} speech · ${o.chat_n} chat${o.vad_idle ? "" : " · VAD"}` : "—"),
      stat("Last reaction", o.last_reaction ? `${o.last_reaction.reason} · ${o.last_reaction.latency_ms} ms` : "—", o.last_reaction?.id),
      stat("Adapter", o.adapter || d.registry_current || "base_core", `registry current: ${d.registry_current || "none"}\nloaded: ${lora || "none"}`),
      stat("VRAM", d.vram ? `${(d.vram.used_mb / 1024).toFixed(1)} / ${(d.vram.total_mb / 1024).toFixed(1)} GB · ${d.vram.util}%` : "—"));
  },
  controls() {
    const user = h("input", { type: "text", value: "ben", style: { width: "90px" } });
    const chat = h("input", { type: "text", class: "grow", placeholder: "say something to Rachel…" });
    const speak = h("input", { type: "text", class: "grow", placeholder: "text for TTS (/speak)" });
    const expr = h("select", {}, h("option", { value: "" }, "expression…"));
    const anim = h("select", {}, h("option", { value: "" }, "animation…"));
    const dry = h("input", { type: "text", class: "grow", placeholder: "dry reaction: Ben says…" });
    api("/api/stream/tags").then(t => {
      expr.append(...t.expressions.map(x => h("option", { value: x }, x)));
      anim.append(...t.animations.map(x => h("option", { value: x }, x)));
    });
    const sendChat = () => chat.value && guard(() => api("/api/stream/chat", { body: { user: user.value, text: chat.value } }), "sent").then(() => chat.value = "");
    return h("div", { class: "card col" }, h("h3", {}, "Controls"),
      h("label", {}, "Chat → orchestrator /chat-input (direct trigger)"),
      h("div", { class: "row" }, user, chat, h("button", { onclick: sendChat }, "Send")),
      h("label", {}, "TTS test → server.py /speak"),
      h("div", { class: "row" }, speak, h("button", { onclick: () => speak.value && guard(() => api("/api/stream/speak", { body: { text: speak.value } }), "spoken") }, "Speak")),
      h("label", {}, "Avatar → server.py /avatar"),
      h("div", { class: "row" }, expr, anim, h("button", { onclick: () => guard(() => api("/api/stream/avatar", { body: { expression: expr.value, animation: anim.value } }), "sent") }, "Send")),
      h("label", {}, "Dry reaction (orchestrator --text, printed not spoken, not logged)"),
      h("div", { class: "row" }, dry, h("button", { onclick: async () => {
        if (!dry.value) return;
        const j = await guard(() => api("/api/stream/dry-reaction", { body: { text: dry.value } }));
        if (j) { toast("running — output in Console › job:dry-reaction"); }
      } }, "Run")));
  },
  lastFeed: null,
  async feed() {
    const today = new Date().toISOString().slice(0, 10);
    const d = await api(`/api/data/rows?${qs({ date: today, limit: 12 })}`).catch(() => null);
    const box = $("#feed"); if (!d || !box) return;
    const key = d.items.map(r => r.id + (r.human || "")).join(",");
    if (key === this.lastFeed) return;
    this.lastFeed = key;
    box.replaceChildren(...(d.items.length ? d.items.map(r => reactionCard(r)) : [h("div", { class: "empty" }, "No reaction rows today yet — rows appear here once the orchestrator reacts in frames mode.")]));
  },
};

function planChips(plan) {
  if (!plan || !plan.steps) return null;
  return h("div", { class: "plan" }, plan.steps.map(s =>
    s.type === "speak" ? h("span", { class: "chip" }, s.text)
      : s.type === "vocal" ? h("span", { class: "chip vocal" }, `[${s.tag}]`)
        : h("span", { class: "chip act" }, [s.expr && `expr:${s.expr}`, s.anim && `anim:${s.anim}`].filter(Boolean).join(" "))),
    (plan.flags || []).map(f => h("span", { class: "badge warn" }, f)));
}
function traceBar(trace) {
  const es = trace?.entries; if (!es || !es.length) return null;
  const end = Math.max(trace.complete_t || 0, ...es.map(e => (e.actual_t ?? e.scheduled_t ?? 0) + (e.clip_duration || 0)), 0.1);
  return h("div", { class: "trace", title: `playback ${end.toFixed(1)}s` }, es.map(e => {
    const t = e.actual_t ?? e.scheduled_t ?? 0, step = e.step || {};
    return h("span", { class: step.type || "", style: { left: `${t / end * 100}%`, width: `${Math.max((e.clip_duration || 0) / end * 100, 0.6)}%` }, title: `${step.type} @${t.toFixed(2)}s ${e.clip_duration ? e.clip_duration.toFixed(2) + "s" : ""}` });
  }));
}
function lines(speech, chat, t0) {
  const rel = t => t0 && t ? `+${(t - t0).toFixed(1)}s ` : "";
  return h("div", { class: "lines" },
    (speech || []).map(s => h("div", {}, rel(s.t), h("b", {}, "ben: "), s.text)),
    (chat || []).map(c => h("div", {}, rel(c.t), h("b", {}, `${c.user}: `), c.text)));
}
function markButtons(r, after) {
  const m = (human, fix_text) => guard(() => api("/api/data/mark", { body: { id: r.id, human, fix_text } }), `marked ${human}`).then(x => x && after && after());
  return h("div", { class: "row" },
    h("button", { class: "small", onclick: e => { e.stopPropagation(); m("good"); } }, "👍 good"),
    h("button", { class: "small", onclick: e => { e.stopPropagation(); m("bad"); } }, "👎 bad"),
    h("button", { class: "small", onclick: e => { e.stopPropagation(); const t = prompt("Fixed reply (same tag grammar):", r.raw_output); if (t != null) m("fix", t); } }, "✎ fix"));
}
function reactionCard(r, after) {
  return h("div", { class: "card reaction", style: { cursor: "pointer" }, onclick: () => rowDrawer(r.id) },
    h("div", { class: "row small muted" }, h("span", {}, fmtTime(r.ts)), h("span", { class: "badge" }, r.input_kind), h("span", {}, r.flush_reason),
      h("span", {}, `${r.latency_ms} ms`), h("span", {}, r.adapter_version), h("span", { class: `badge ${r.status}` }, r.status),
      r.human && h("span", { class: `badge ${r.human}` }, `human: ${r.human}`), r.uncertainty != null && h("span", {}, `u=${r.uncertainty}`)),
    frames(r.frame_paths), lines(r.speech, r.chat, r.window_start),
    h("div", { class: "out" }, r.human_fix || r.raw_output), planChips(r.plan), traceBar(r.playback_trace),
    markButtons(r, after));
}
async function rowDrawer(id) {
  const r = await guard(() => api(`/api/data/row/${encodeURIComponent(id)}`)); if (!r) return;
  openDrawer(id,
    kv("section", "status", r.status, "human", r.human || "—", "input", r.input_kind, "flush", r.flush_reason,
      "adapter", r.adapter_version, "prompt", r.prompt_version, "latency", `${r.latency_ms} ms`, "uncertainty", r.uncertainty),
    frames(r.frame_paths, "repo", true), lines(r.speech, r.chat, r.window_start), hrr(),
    h("h3", {}, "Output"), h("pre", {}, r.raw_output), planChips(r.plan), traceBar(r.playback_trace),
    r.human_fix && [h("h3", {}, "Human fix"), h("pre", {}, r.human_fix)],
    markButtons(r, () => rowDrawer(id)), hrr(),
    r.review && [h("h3", {}, "Review"), h("pre", {}, pretty(r.review)), hrr()],
    h("details", {}, h("summary", {}, "exact prompt (prompt_format.render_inference)"), h("pre", {}, r.exact_prompt)),
    h("details", {}, h("summary", {}, "raw row"), h("pre", {}, pretty(r))));
}
const hrr = () => h("hr");

// ─── Console page ────────────────────────────────────────
const Console = {
  select: null,
  render(main) {
    this.source = this.select || "all"; this.select = null;
    this.filter = ""; this.level = ""; this.paused = false;
    this.tabs = h("div", { class: "tabs" });
    this.view = h("div", { class: "logview" });
    const filter = h("input", { type: "text", placeholder: "filter (regex)", oninput: e => { this.filter = e.target.value; this.redraw(); } });
    const level = h("select", { onchange: e => { this.level = e.target.value; this.redraw(); } },
      ["", "info", "warn", "error"].map(l => h("option", { value: l }, l || "all levels")));
    const pause = h("input", { type: "checkbox", onchange: e => { this.paused = e.target.checked; if (!this.paused) this.redraw(); } });
    this.dl = h("a", { class: "small", target: "_blank" }, "download today");
    this.jobsBox = h("div", { class: "col" });
    main.append(
      h("div", { class: "row section" }, h("h2", { class: "grow", style: { margin: 0 } }, "Console"),
        filter, level, h("label", {}, pause, " pause"),
        h("button", { class: "small", onclick: () => { this.cleared = S.logs.length ? S.logs[S.logs.length - 1].seq : 0; this.redraw(); } }, "clear view"), this.dl),
      this.tabs, this.view,
      h("div", { class: "grid g2", style: { marginTop: "14px" } },
        h("div", { class: "card" }, h("h3", {}, "Commands"), h("div", { id: "palette", class: "row" }), h("hr"), h("h3", {}, "Jobs"), this.jobsBox),
        this.inspector()));
    this.cleared = 0;
    this.loadPalette();
    this.redraw();
  },
  sources() {
    const set = new Set(Object.keys(S.services));
    S.logs.forEach(l => set.add(l.source));
    return ["all", ...[...set].sort()];
  },
  drawTabs() {
    this.tabs.replaceChildren(...this.sources().map(s => h("button", { class: s === this.source ? "active" : "",
      onclick: () => { this.source = s; this.redraw(); } }, s)));
    this.dl.href = this.source === "all" ? "#" : `/api/logs/download?source=${encodeURIComponent(this.source)}`;
    this.dl.classList.toggle("hidden", this.source === "all");
  },
  match(l) {
    if (l.seq <= this.cleared) return false;
    if (this.source !== "all" && l.source !== this.source) return false;
    if (this.level && l.level !== this.level) return false;
    if (this.filter) { try { if (!new RegExp(this.filter, "i").test(l.text)) return false; } catch { if (!l.text.includes(this.filter)) return false; } }
    return true;
  },
  lineEl(l) {
    return h("div", { class: `logline ${l.level}` }, h("span", { class: "t" }, fmtTime(l.ts)),
      this.source === "all" ? h("span", { class: "s", style: { color: colorFor(l.source) } }, l.source) : null, h("span", { class: "x" }, l.text));
  },
  redraw() {
    if (!this.view) return;
    this.drawTabs();
    const ls = S.logs.filter(l => this.match(l)).slice(-2500);
    this.view.replaceChildren(...ls.map(l => this.lineEl(l)));
    this.view.scrollTop = this.view.scrollHeight;
  },
  onBackfill() { this.redraw(); },
  onLog(l) {
    if (this.paused || !this.view || !this.match(l)) { if (!this.tabs.textContent.includes(l.source)) this.drawTabs(); return; }
    const atBottom = this.view.scrollTop + this.view.clientHeight >= this.view.scrollHeight - 30;
    this.view.append(this.lineEl(l));
    if (this.view.childElementCount > 3000) this.view.firstChild.remove();
    if (atBottom) this.view.scrollTop = this.view.scrollHeight;
  },
  onJob() { this.drawJobs(); },
  async loadPalette() {
    const d = await api("/api/jobs");
    d.jobs.forEach(j => S.jobs[j.id] = j);
    $("#palette").replaceChildren(...Object.entries(d.palette).map(([k, v]) =>
      h("button", { class: "small", title: `${v.help}\n${v.argv}`, onclick: async () => {
        const j = await guard(() => api("/api/jobs/run", { body: { name: k } }));
        if (j) { this.source = `job:${k}`; this.redraw(); }
      } }, k)));
    this.drawJobs();
  },
  drawJobs() {
    if (!this.jobsBox) return;
    const js = Object.values(S.jobs).sort((a, b) => b.id - a.id).slice(0, 12);
    this.jobsBox.replaceChildren(...(js.length ? js.map(j => h("div", { class: "row small" },
      h("span", { class: `dot ${j.state}` }), h("a", { href: "#console", onclick: () => { this.source = j.source; this.redraw(); } }, `#${j.id} ${j.name}`),
      h("span", { class: "muted" }, j.state + (j.rc != null ? ` (${j.rc})` : "")), h("span", { class: "muted" }, fmtTime(j.started)),
      j.state === "running" ? h("button", { class: "small", onclick: () => api(`/api/jobs/${j.id}/cancel`, { body: {} }) }, "cancel") : null,
      j.result ? h("button", { class: "small ghost", onclick: () => openDrawer(`#${j.id} ${j.name} result`, h("pre", {}, pretty(j.result))) }, "result") : null))
      : [h("div", { class: "muted small" }, "no jobs yet")]));
  },
  inspector() {
    const port = h("select", {}, [8052, 8055, 8051, 8053, 8080, 8081, 8050].map(p => h("option", { value: p }, p)));
    const method = h("select", {}, h("option", {}, "GET"), h("option", {}, "POST"));
    const path = h("input", { type: "text", value: "/status", class: "grow" });
    const body = h("textarea", { rows: 3, placeholder: '{"json": "body for POST"}' });
    const out = h("pre", { style: { maxHeight: "320px", overflow: "auto" } });
    const presets = [["8052", "GET", "/status"], ["8055", "GET", "/status"], ["8051", "GET", "/memory/stats"], ["8080", "GET", "/health"],
      ["8080", "GET", "/lora-adapters"], ["8080", "GET", "/props"], ["8053", "GET", "/audio-context"]];
    return h("div", { class: "card col" }, h("h3", {}, "Request inspector (local services only)"),
      h("div", { class: "row" }, presets.map(([p, m, pa]) => h("button", { class: "small ghost", onclick: () => { port.value = p; method.value = m; path.value = pa; } }, `:${p}${pa}`))),
      h("div", { class: "row" }, method, port, path, h("button", { onclick: async () => {
        out.textContent = "…";
        const r = await guard(() => api("/api/request", { body: { port: +port.value, method: method.value, path: path.value, body: body.value } }));
        out.textContent = r ? pretty(r) : "";
      } }, "Send")), body, out);
  },
};
const _colors = {};
function colorFor(s) {
  if (!_colors[s]) { let x = 0; for (const c of s) x = (x * 31 + c.charCodeAt(0)) >>> 0; _colors[s] = `hsl(${x % 360} 65% 68%)`; }
  return _colors[s];
}

// ─── Data page ───────────────────────────────────────────
const Data = {
  tab: "reactions",
  render(main) {
    const tabs = [["reactions", "Reactions"], ["queue", "Review queue"], ["reviews", "Reviews"], ["curated", "Curated"], ["dpo", "DPO"],
      ["datasets", "Datasets"], ["adapters", "Adapters"], ["memory", "Memory"], ["session_log", "Session logs"],
      ["legacy_training", "Legacy training"], ["trainingsets", "Training sets"], ["ledger", "Ledger"]];
    this.body = h("div");
    this.tabsEl = h("div", { class: "tabs" });
    const draw = () => this.tabsEl.replaceChildren(...tabs.map(([k, label]) => h("button", { class: k === this.tab ? "active" : "", onclick: () => { this.tab = k; draw(); this.show(); } }, label)));
    draw();
    main.append(h("h2", {}, "Data explorer"), this.tabsEl, this.body);
    api("/api/data/overview").then(o => { this.ov = o; this.show(); });
  },
  show() {
    if (!this.ov) return;
    const fn = { reactions: this.reactions, queue: this.queue, reviews: () => this.store("reviews"), curated: () => this.store("curated"),
      dpo: () => this.store("dpo"), datasets: this.datasets, adapters: this.adapters, memory: this.memory,
      session_log: () => this.store("session_log"), legacy_training: () => this.store("legacy_training"),
      trainingsets: this.trainingsets, ledger: this.ledger }[this.tab];
    this.body.replaceChildren();
    fn.call(this, this.body);
  },
  reactions(box) {
    if (!this.ov.rows.length) { box.append(h("div", { class: "empty" }, "No reaction rows yet (rows/ is empty). They are written by the orchestrator in frames mode.")); return; }
    const f = { date: this.ov.rows[0], status: "", kind: "", q: "", offset: 0, limit: 20 };
    const list = h("div", { class: "col" });
    const sel = (key, opts) => h("select", { onchange: e => { f[key] = e.target.value; f.offset = 0; load(); } }, opts.map(o => h("option", { value: o }, o || `any ${key}`)));
    const load = async () => {
      const d = await guard(() => api(`/api/data/rows?${qs(f)}`)); if (!d) return;
      list.replaceChildren(pager(d.total, f.offset, f.limit, o => { f.offset = o; load(); }), ...d.items.map(r => reactionCard(r, load)));
    };
    const dateSel = sel("date", ["", ...this.ov.rows]); dateSel.value = f.date;
    box.append(h("div", { class: "row section" }, dateSel,
      sel("status", ["", "raw", "reviewed", "accepted", "rejected", "excluded"]),
      sel("kind", ["", "frames", "frames+speech", "frames+chat", "frames+chat+speech"]),
      h("input", { type: "text", placeholder: "search output / speech / chat", onchange: e => { f.q = e.target.value; f.offset = 0; load(); } })), list);
    load();
  },
  async queue(box) {
    const d = await guard(() => api("/api/data/queue?n=40")); if (!d) return;
    box.append(h("p", { class: "muted small" }, `${d.total} unmarked rows, most uncertain first (training_logger.uncertainty).`),
      ...(d.items.length ? d.items.map(r => reactionCard(r, () => this.show())) : [h("div", { class: "empty" }, "Queue is empty.")]));
  },
  async store(store) {
    const box = this.body;
    const d = await guard(() => api(`/api/data/store/${store}`)); if (!d) return;
    const opts = d.files.flatMap(g => g.files.map(f => ({ f, dir: g.dir })));
    if (!opts.length) { box.append(h("div", { class: "empty" }, `No ${store} files yet.`)); return; }
    const st = { name: opts[0].f, offset: 0, limit: 20, q: "" };
    const list = h("div", { class: "col" });
    const load = async () => {
      const r = await guard(() => api(`/api/data/store/${store}/${encodeURIComponent(st.name)}?${qs({ offset: st.offset, limit: st.limit, q: st.q })}`)); if (!r) return;
      list.replaceChildren(pager(r.total, st.offset, st.limit, o => { st.offset = o; load(); }), ...r.items.map(x => this.storeRow(store, x)));
    };
    box.append(h("div", { class: "row section" },
      h("select", { onchange: e => { st.name = e.target.value; st.offset = 0; load(); } }, opts.map(o => h("option", { value: o.f, title: o.dir }, `${o.f}  (${o.dir.split(/[\\/]/).slice(-2).join("/")})`))),
      h("input", { type: "text", placeholder: "search", onchange: e => { st.q = e.target.value; st.offset = 0; load(); } })), list);
    load();
  },
  storeRow(store, x) {
    const open = () => openDrawer(x.id || x.time || store, h("pre", {}, pretty(x)));
    if (store === "session_log") {
      return h("div", { class: "card reaction", onclick: open, style: { cursor: "pointer" } },
        h("div", { class: "row small muted" }, h("span", {}, x.time || fmtTime(x.ts)), h("span", { class: "badge" }, x.trigger),
          x.emotion && h("span", { class: "badge" }, `${x.emotion}/${x.intensity}`), x.game && h("span", {}, x.game), x.scene && h("span", {}, x.scene)),
        frames(x.images, "imagelog"),
        x.heard && h("div", { class: "lines" }, h("b", {}, "heard: "), x.heard),
        x.thought && h("div", { class: "lines" }, h("b", {}, "thought: "), x.thought),
        h("div", { class: "out" }, x.spoke || ""));
    }
    if (store === "legacy_training") {
      const msgs = x.messages || [];
      const user = msgs.filter(m => m.role === "user").map(m => typeof m.content === "string" ? m.content : JSON.stringify(m.content)).join("\n");
      const model = msgs.filter(m => m.role !== "system" && m.role !== "user").map(m => m.content).join("\n");
      return h("div", { class: "card reaction", onclick: open, style: { cursor: "pointer" } },
        h("div", { class: "row small muted" }, x.emotion && h("span", { class: "badge" }, `${x.emotion}/${x.intensity}`)),
        frames(x.metadata?.images, "imagelog"),
        user && h("div", { class: "lines" }, h("b", {}, "user: "), user.slice(0, 400)),
        h("div", { class: "out" }, model));
    }
    if (store === "reviews") {
      const rv = x.review || {};
      return h("div", { class: "card", onclick: open, style: { cursor: "pointer" } },
        h("div", { class: "row small" }, h("span", { class: "mono" }, x.id), h("span", { class: `badge ${x.status}` }, x.status),
          rv.stage && h("span", { class: "badge" }, `stage ${rv.stage}`), rv.verdict && h("span", {}, rv.verdict), x.reason && h("span", { class: "muted" }, x.reason)),
        rv.scores && h("div", { class: "small muted" }, Object.entries(rv.scores).map(([k, v]) => `${k} ${v}`).join(" · ")),
        rv.edited_output && h("div", { class: "out" }, "✎ ", rv.edited_output));
    }
    const text = x.completion || x.chosen || x.reply || x.spoke || "";
    return h("div", { class: "card", onclick: open, style: { cursor: "pointer" } },
      h("div", { class: "row small muted" }, x.id && h("span", { class: "mono" }, x.id), x.input_kind && h("span", { class: "badge" }, x.input_kind),
        x.source && h("span", {}, x.source), x.human && h("span", { class: "badge good" }, "human"), x.score != null && h("span", {}, `score ${x.score}`)),
      frames(x.images), text && h("div", { class: "out" }, text.replace(/<turn\|>$/, "")),
      x.rejected && h("div", { class: "lines" }, h("b", {}, "rejected: "), x.rejected));
  },
  async datasets(box) {
    const d = await guard(() => api("/api/data/datasets")); if (!d) return;
    if (!d.items.length) { box.append(h("div", { class: "empty" }, "No datasets built yet (datasets/). Build one on the Dream page.")); return; }
    box.append(h("table", {}, h("tr", {}, ["run", "created", "nights", "new", "replay", "train (weighted)", "holdout", "parent", ""].map(t => h("th", {}, t))),
      d.items.map(m => h("tr", {}, h("td", { class: "mono" }, m.run_id), h("td", {}, m.created), h("td", {}, (m.data_range?.nights || []).join(", ")),
        h("td", {}, m.counts?.new), h("td", {}, m.counts?.replay), h("td", {}, m.counts?.train_weighted), h("td", {}, m.counts?.holdout), h("td", {}, m.parent || "—"),
        h("td", {}, ["train", "holdout"].map(sp => h("button", { class: "small", onclick: async () => {
          const r = await guard(() => api(`/api/data/datasets/${m.run_id}/${sp}?limit=50`));
          if (r) openDrawer(`${m.run_id} / ${sp} (${r.total})`, r.items.map(x => this.storeRow("curated", x)));
        } }, sp)))))));
  },
  async adapters(box) {
    const reg = await guard(() => api("/api/data/adapters")); if (!reg) return;
    box.append(h("div", { class: "row section" }, h("span", {}, "current: ", h("b", {}, reg.current || "base_core (none)")),
      h("button", { class: "small", disabled: !reg.current, onclick: () => confirm("Roll back current to its parent?") &&
        guard(() => api("/api/data/adapters/rollback", { body: {} }), "rolled back").then(() => this.show()) }, "Rollback")));
    if (!reg.adapters.length) { box.append(h("div", { class: "empty" }, "No adapters registered yet (adapters/registry.json is empty).")); return; }
    box.append(h("table", {}, h("tr", {}, ["version", "status", "parent", "nights", "loss", "gguf", "created", ""].map(t => h("th", {}, t))),
      reg.adapters.slice().reverse().map(a => h("tr", {},
        h("td", { class: "mono" }, a.version, a.version === reg.current ? " ★" : ""), h("td", {}, h("span", { class: `badge ${a.status === "promoted" ? "healthy" : a.status === "rejected" ? "failed" : "running"}` }, a.status)),
        h("td", {}, a.parent || "—"), h("td", {}, (a.data_range?.nights || []).join(", ")), h("td", {}, a.train?.loss ?? "—"),
        h("td", {}, a.gguf ? (a.gguf_exists ? "✓" : "missing") : "—"), h("td", {}, fmtTime(a.created)),
        h("td", { class: "row" },
          h("button", { class: "small", disabled: a.version === reg.current, onclick: () => confirm(`Promote ${a.version} to current?`) &&
            guard(() => api("/api/data/adapters/promote", { body: { version: a.version } }), "promoted").then(() => this.show()) }, "Promote"),
          a.eval_report && h("button", { class: "small", onclick: () => evalDrawer(a.eval_report) }, "eval"))))));
  },
  async memory(box) {
    const st = { q: "", emotion: "", offset: 0, limit: 40 };
    const out = h("div");
    const load = async () => {
      const d = await guard(() => api(`/api/data/memory?${qs(st)}`)); if (!d) return;
      const head = h("div", { class: "row small muted section" }, d.server ? "server.py up — search uses embeddings (/memory/recall)" : "server.py down — reading the Chroma store directly, text search only",
        d.stats && h("span", {}, JSON.stringify(d.stats)));
      if (d.error) { out.replaceChildren(head, h("div", { class: "empty" }, d.error)); return; }
      const items = d.recall ? d.recall.memories || [] : d.items || [];
      out.replaceChildren(head, d.recall ? null : pager(d.total, st.offset, st.limit, o => { st.offset = o; load(); }),
        h("table", {}, h("tr", {}, ["text", "emotion", "int", "salience", "type", "created / score"].map(t => h("th", {}, t))),
          items.map(m => h("tr", {}, h("td", {}, m.text), h("td", {}, m.emotion), h("td", {}, m.intensity), h("td", {}, m.salience != null ? (+m.salience).toFixed(2) : ""),
            h("td", {}, m.memory_type || ""), h("td", { class: "small muted" }, m.score != null ? `score ${(+m.score).toFixed(3)}` : (m.created_at || ""))))));
    };
    box.append(h("div", { class: "row section" },
      h("input", { type: "text", placeholder: "search memories", onchange: e => { st.q = e.target.value; st.offset = 0; load(); } }),
      h("select", { onchange: e => { st.emotion = e.target.value; st.offset = 0; load(); } }, ["", "joy", "sadness", "anger", "fear", "surprise", "disgust", "love", "neutral"].map(x => h("option", { value: x }, x || "any emotion"))),
      h("button", { class: "small", title: "server.py POST /memory/dream — merge near-duplicates, decay, prune (no LLM)",
        onclick: () => confirm("Run memory merge/decay/prune now?") && guard(() => api("/api/data/memory/dream", { body: {} })).then(r => r && openDrawer("memory dream", h("pre", {}, pretty(r)))) }, "Merge / decay now")), out);
    load();
  },
  async trainingsets(box) {
    if (!this.ov.trainingsets.length) { box.append(h("div", { class: "empty" }, "No TrainingSets/*/ready found.")); return; }
    const st = { name: this.ov.trainingsets[this.ov.trainingsets.length - 1], split: "holdout", offset: 0, limit: 20, q: "" };
    const info = h("div"), list = h("div", { class: "col" });
    const loadInfo = async () => {
      const d = await guard(() => api(`/api/data/trainingset/${encodeURIComponent(st.name)}`)); if (!d) return;
      info.replaceChildren(h("div", { class: "row small muted section" }, Object.entries(d.splits).map(([k, v]) => h("span", { class: "badge" }, `${k}: ${v}`)),
        h("button", { class: "small ghost", onclick: () => openDrawer(`${st.name} report.md`, h("pre", {}, d.report), h("hr"), h("pre", {}, pretty(d.meta))) }, "report / meta")));
    };
    const load = async () => {
      const d = await guard(() => api(`/api/data/trainingset/${encodeURIComponent(st.name)}/${st.split}?${qs({ offset: st.offset, limit: st.limit, q: st.q })}`)); if (!d) return;
      list.replaceChildren(pager(d.total, st.offset, st.limit, o => { st.offset = o; load(); }), ...d.items.map(x => {
        const lastUser = (x.prompt || "").split("<|turn>user\n").pop().split("<turn|>")[0];
        return h("div", { class: "card", style: { cursor: "pointer" }, onclick: () => openDrawer(x.id, h("pre", {}, (x.prompt || "") + (x.completion || "")), h("hr"), h("pre", {}, pretty({ ...x, prompt: "(above)" }))) },
          h("div", { class: "row small muted" }, h("span", { class: "mono" }, x.id), x.source && h("span", { class: "badge" }, x.source), x.reason && h("span", { class: "badge failed" }, x.reason),
            (x.edits || []).length ? h("span", {}, `${x.edits.length} edits`) : null),
          lastUser && h("div", { class: "lines" }, lastUser.slice(-300)),
          h("div", { class: "out" }, x.reply || x.raw_reply), planChips(x.plan));
      }));
    };
    box.append(h("div", { class: "row section" },
      h("select", { onchange: e => { st.name = e.target.value; st.offset = 0; loadInfo(); load(); } }, this.ov.trainingsets.map(n => h("option", { value: n, selected: n === st.name }, n))),
      h("select", { onchange: e => { st.split = e.target.value; st.offset = 0; load(); } }, ["holdout", "train", "rejects"].map(n => h("option", { value: n }, n))),
      h("input", { type: "text", placeholder: "search reply / prompt tail", onchange: e => { st.q = e.target.value; st.offset = 0; load(); } })), info, list);
    loadInfo(); load();
  },
  async ledger(box) {
    const d = await guard(() => api("/api/data/ledger")); if (!d) return;
    box.append(h("p", { class: "muted small" }, "Opinion proposals from the reviewer. Merging into persona/STANCES.md stays manual (guardrail)."),
      d.items.length ? h("table", {}, h("tr", {}, ["topic", "stance", "row", "chat", "status"].map(t => h("th", {}, t))),
        d.items.map(x => h("tr", {}, h("td", {}, x.topic), h("td", {}, x.stance), h("td", { class: "mono" }, x.row_id), h("td", {}, x.chat_present ? "yes" : ""), h("td", {}, x.status))))
        : h("div", { class: "empty" }, "No pending opinions."));
  },
};
function evalDrawer(rep) {
  const keys = [...new Set([...Object.keys(rep.candidate_metrics || {}), ...Object.keys(rep.baseline_metrics || {})])];
  openDrawer(`eval: ${rep.candidate} vs ${rep.baseline || "base_core"}`,
    h("div", { class: "row section" }, h("span", { class: `badge ${rep.promote ? "healthy" : "failed"}` }, rep.promote ? "promoted" : "rejected"), (rep.reasons || []).join("; ")),
    h("table", {}, h("tr", {}, ["metric", "candidate", "baseline"].map(t => h("th", {}, t))),
      keys.map(k => h("tr", {}, h("td", {}, k), h("td", {}, fmtNum(rep.candidate_metrics?.[k])), h("td", {}, fmtNum(rep.baseline_metrics?.[k]))))),
    h("h3", { style: { marginTop: "14px" } }, "Candidate samples"), (rep.candidate_samples || []).map(s => h("div", { class: "card", style: { marginBottom: "6px" } }, h("div", { class: "small muted mono" }, s.id), s.output)));
}
const fmtNum = v => typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : v == null ? "—" : JSON.stringify(v);

// ─── Dream page (training pipeline) ──────────────────────
const Dream = {
  async render(main) {
    this.main = main;
    const d = await guard(() => api("/api/dream")); if (!d) return;
    this.st = d; S.dream = d.chain || S.dream;
    const p = this.params = {
      review: { dates: d.defaults.dates.join(","), backend: d.anthropic_key ? "api" : "cli", limit: "", stage_a_only: false, force: false, dry_run: false },
      dataset: { run_id: d.defaults.run_id, k: 3, seed: 0 },
      train: { run_id: d.defaults.run_id, version: d.defaults.version, lr: 3e-5, epochs: 1, grad_accum: 8, r: 16, trainer: "sft", check: false, export_gguf: true },
      eval: { version: d.defaults.version, no_judge: !d.anthropic_key, dry_run: false },
    };
    this.include = { review: true, dataset: true, train: true, eval: true };
    const field = (step, key, type = "text", opts) => {
      const v = p[step][key];
      if (type === "bool") return h("label", {}, h("input", { type: "checkbox", checked: v, onchange: e => p[step][key] = e.target.checked }), ` ${key.replace(/_/g, " ")}`);
      if (type === "select") return h("label", {}, `${key} `, h("select", { onchange: e => p[step][key] = e.target.value }, opts.map(o => h("option", { value: o, selected: o === v }, o))));
      return h("label", {}, `${key.replace(/_/g, " ")} `, h("input", { type, value: v, step: "any", style: { width: type === "number" ? "90px" : "170px" },
        oninput: e => { p[step][key] = type === "number" ? +e.target.value : e.target.value; if (key === "version" || key === "run_id") this.sync(step, key, e.target.value); } }));
    };
    const warn = (cond, text) => cond ? h("div", { class: "small", style: { color: "var(--warn)" } }, "⚠ ", text) : null;
    const stepCard = (step, title, desc, fields, warnings) => h("div", { class: "card step col", id: `step-${step}` },
      h("div", { class: "row" }, h("label", {}, h("input", { type: "checkbox", checked: true, onchange: e => this.include[step] = e.target.checked })),
        h("b", { class: "grow" }, title), h("span", { class: "badge", id: `step-state-${step}` }, "—"),
        h("button", { class: "small", onclick: () => this.run([step]) }, "Run this step")),
      h("div", { class: "small muted" }, desc), h("div", { class: "row" }, fields), warnings, h("div", { id: `step-result-${step}` }));
    main.replaceChildren(
      h("div", { class: "row section" }, h("h2", { class: "grow", style: { margin: 0 } }, "Dream — nightly training loop"),
        h("button", { class: "primary", onclick: () => this.run(Object.keys(this.include).filter(k => this.include[k])) }, "▶ Run selected"),
        h("button", { class: "danger", onclick: () => guard(() => api("/api/dream/cancel", { body: {} }), "cancelling") }, "Cancel")),
      kv("card section",
        "rows", d.row_dates.length ? `${d.row_dates.length} days (latest ${d.row_dates[d.row_dates.length - 1]})` : "none yet — stream in frames mode first",
        "curated nights", d.curated_nights.length ? d.curated_nights.join(", ") : "none — run Review",
        "next dataset", d.next_nights.length ? `new: ${d.next_nights.join(", ")}  ·  replay: ${d.replay_nights.join(", ") || "—"}` : "—",
        "eval set", d.eval_set ? "training/data/eval_set.jsonl ✓" : "missing — built automatically before eval",
        "registry current", d.registry.current || "base_core",
        "GPU", d.gpu_busy.length ? `in use by ${d.gpu_busy.join(", ")}` : "free",
        "Claude review/judge", d.anthropic_key ? "ANTHROPIC_API_KEY set" : d.claude_cli ? "no API key — `claude` CLI available (--backend cli)" : "no API key and no claude CLI — Stage-A only / --no-judge",
        "BASE_CORE_HF", `${d.base_core_hf} ${d.base_core_hf_exists ? "✓" : "✗ missing"}`,
        "BASE_CORE_GGUF", d.base_core_gguf || "✗ not set (eval llama-server needs it)"),
      h("div", { class: "col section" },
        stepCard("review", "1 · Review", "training.reviewer — Stage A deterministic filter, Stage B Claude; writes curated/ + dpo/ + rows/_review/.",
          [field("review", "dates"), field("review", "backend", "select", ["api", "cli"]), field("review", "limit", "number"),
            field("review", "stage_a_only", "bool"), field("review", "force", "bool"), field("review", "dry_run", "bool")],
          warn(!d.row_dates.length, "no rows to review")),
        stepCard("dataset", "2 · Dataset", "training.dataset_builder — new curated nights + ~50% replay, split by session → datasets/<run_id>/.",
          [field("dataset", "run_id"), field("dataset", "k", "number"), field("dataset", "seed", "number")],
          warn(!d.curated_nights.length, "no curated nights yet")),
        stepCard("train", "3 · Train", "training.train_nightly — QLoRA E4B on the GPU, registers a candidate. Tick 'check' for a config dry run.",
          [field("train", "run_id"), field("train", "version"), field("train", "lr", "number"), field("train", "epochs", "number"),
            field("train", "grad_accum", "number"), field("train", "r", "number"), field("train", "trainer", "select", ["sft", "hf"]),
            field("train", "check", "bool"), field("train", "export_gguf", "bool")],
          [warn(d.gpu_busy.length, `GPU in use by ${d.gpu_busy.join(", ")} — you'll be asked to stop the live stack`),
            warn(!d.base_core_hf_exists, "BASE_CORE_HF path not found")],
          ),
        stepCard("eval", "4 · Eval gate", "training.eval_gate run — temporary llama-server on :8081 with every registered GGUF; promotes or rejects.",
          [field("eval", "version"), field("eval", "no_judge", "bool"), field("eval", "dry_run", "bool")],
          warn(!d.base_core_gguf, "BASE_CORE_GGUF not set — eval llama-server can't start"))),
      h("div", { class: "card section" }, h("h3", {}, "Training loss"), h("div", { id: "loss" })),
      h("div", { class: "card section" }, h("h3", {}, "5 · Registry"), h("div", { id: "dream-registry" })));
    this.onDream(S.dream);
    this.registry();
  },
  sync(step, key, val) {
    if (key === "run_id") { this.params.dataset.run_id = val; this.params.train.run_id = val; }
    if (key === "version") { this.params.train.version = val; this.params.eval.version = val; }
  },
  async run(steps, stop_live = false) {
    const p = JSON.parse(JSON.stringify(this.params));
    p.review.dates = String(p.review.dates).split(",").map(s => s.trim()).filter(Boolean);
    const r = await guard(() => api("/api/dream/run", { body: { steps, params: p, stop_live } })); if (!r) return;
    if (r.needs_stop_live) { if (confirm(`${r.error}\n\nStop the live stack (including external llama-server) and continue?`)) this.run(steps, true); return; }
    if (r.error) { toast(r.error, "error"); return; }
    toast(`dream: running ${steps.join(" → ")}`, "ok");
  },
  onDream(chain) {
    if (!chain || !this.main?.isConnected) return;
    for (const s of chain.steps) {
      const badge = $(`#step-state-${s.step}`), card = $(`#step-${s.step}`), res = $(`#step-result-${s.step}`);
      if (!badge) continue;
      badge.textContent = s.state; badge.className = `badge ${s.state}`;
      card.className = `card step col ${s.state}`;
      res.replaceChildren(...[s.job && h("a", { href: "#console", onclick: () => { Console.select = `job:dream-${s.step}`; } }, `logs › job:dream-${s.step}`),
        s.result && h("details", { open: s.step !== "train" }, h("summary", {}, "result"), h("pre", {}, pretty(s.result)))].filter(Boolean));
    }
    this.loss(chain.loss || []);
    if (chain.state !== "running") this.registry();
  },
  loss(pts) {
    const box = $("#loss"); if (!box) return;
    if (!pts.length) { box.replaceChildren(h("div", { class: "muted small" }, "no loss points yet (logged every 5 steps while training)")); return; }
    const w = 600, ht = 120, ys = pts.map(p => p.loss), lo = Math.min(...ys), hi = Math.max(...ys), span = hi - lo || 1;
    const path = ys.map((y, i) => `${i ? "L" : "M"}${(i / Math.max(ys.length - 1, 1) * w).toFixed(1)},${(ht - (y - lo) / span * (ht - 10) - 5).toFixed(1)}`).join("");
    box.innerHTML = `<svg viewBox="0 0 ${w} ${ht}" style="width:100%;max-width:${w}px;height:${ht}px"><path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2"/></svg>
      <div class="small muted">${pts.length} points · last ${ys[ys.length - 1]} · min ${lo}</div>`;
  },
  async registry() {
    const box = $("#dream-registry"); if (!box) return;
    const reg = await api("/api/data/adapters").catch(() => null); if (!reg) return;
    const cands = reg.adapters.filter(a => a.status === "candidate");
    box.replaceChildren(h("div", { class: "row" }, "current: ", h("b", {}, reg.current || "base_core"),
      h("button", { class: "small", disabled: !reg.current, onclick: () => confirm("Roll back current to its parent?") && guard(() => api("/api/data/adapters/rollback", { body: {} }), "rolled back").then(() => this.registry()) }, "Rollback"),
      h("button", { class: "small", title: "a new GGUF is only loaded when llama-server restarts", onclick: () => confirm("Restart llama-server with serve_cmd (base_core + every registered GGUF)? A hand-started llama-server will be killed.") &&
        guard(() => api("/api/services/llama/restart", { body: {} }), "restarting llama-server") }, "Restart llama-server")),
      cands.length ? h("div", { class: "col", style: { marginTop: "8px" } }, cands.map(a => h("div", { class: "row" }, h("span", { class: "mono" }, a.version), h("span", { class: "badge running" }, "candidate"),
        h("button", { class: "small", onclick: () => confirm(`Promote ${a.version}?`) && guard(() => api("/api/data/adapters/promote", { body: { version: a.version } }), "promoted").then(() => this.registry()) }, "Promote"),
        a.eval_report && h("button", { class: "small", onclick: () => evalDrawer(a.eval_report) }, "eval")))) : null);
  },
};

// ─── Settings page ───────────────────────────────────────
const PATH_KEYS = new Set(["BASE_CORE_GGUF", "BASE_CORE_MMPROJ", "BASE_CORE_HF", "LLAMACPP_DIR", "VOICE_REF_PATH"]);
const Settings = {
  async render(main) {
    const d = await guard(() => api("/api/settings")); if (!d) return;
    this.dirty = {};
    const save = h("button", { class: "primary", disabled: true, onclick: () => this.save() }, "Save .env");
    this.saveBtn = save;
    const groups = {};
    for (const s of d.settings) (groups[s.group] ||= []).push(s);
    main.replaceChildren(
      h("div", { class: "row section", style: { position: "sticky", top: "-18px", background: "var(--bg)", padding: "10px 0", zIndex: 5 } },
        h("h2", { class: "grow", style: { margin: 0 } }, "Settings"), h("span", { class: "muted small mono" }, d.env_path), save),
      h("p", { class: "muted small" }, "Values are written to .env. Blank = use the config.py default (shown as placeholder). Secrets are write-only. Services that read a changed key get a 'restart required' badge."),
      this.devicesCard(),
      d.groups.filter(g => groups[g]).map(g => h("div", { class: "card section" }, h("h3", {}, g), groups[g].map(s => this.row(s)))),
      this.personaCard());
  },
  row(s) {
    let inp;
    const mark = () => { this.dirty[s.key] = inp.value; el.classList.add("dirty"); this.saveBtn.disabled = false; };
    if (s.type === "bool") {
      inp = h("select", { onchange: mark }, h("option", { value: "" }, `default (${s.default})`), h("option", { value: "true" }, "true"), h("option", { value: "false" }, "false"));
      inp.value = (s.value || "").toLowerCase() === "true" ? "true" : (s.value || "").toLowerCase() === "false" ? "false" : "";
    } else if (s.secret) {
      inp = h("input", { type: "password", placeholder: s.set ? "•••••• (set — type to replace)" : "unset", oninput: mark, autocomplete: "off" });
    } else {
      inp = h("input", { type: s.type === "str" ? "text" : "number", step: "any", value: s.value ?? "", placeholder: s.default, oninput: mark });
    }
    const exists = PATH_KEYS.has(s.key) ? h("span", { class: "small" }) : null;
    const check = async () => {
      if (!exists) return;
      const p = inp.value || s.default;
      if (!p) { exists.textContent = "✗ not set"; exists.style.color = "var(--err)"; return; }
      const r = await api(`/api/settings/file-exists?path=${encodeURIComponent(p)}`).catch(() => null);
      exists.textContent = r?.exists ? "✓ exists" : "✗ not found"; exists.style.color = r?.exists ? "var(--ok)" : "var(--err)";
    };
    if (exists) { inp.addEventListener("change", check); check(); }
    const el = h("div", { class: "setting" },
      h("div", {}, h("div", { class: "key" }, s.key), h("div", { class: "help" }, [s.comment, s.restart.length && `restart: ${s.restart.join(", ")}`].filter(Boolean).join(" · "))),
      inp,
      h("div", { class: "row" }, exists, s.secret && s.set ? h("button", { class: "small ghost", title: "remove from .env", onclick: () => { inp.value = ""; this.dirty[s.key] = ""; el.classList.add("dirty"); this.saveBtn.disabled = false; inp.placeholder = "will be removed"; } }, "clear") : null));
    el.dataset.key = s.key;
    return el;
  },
  async save() {
    const r = await guard(() => api("/api/settings", { body: { changes: this.dirty } })); if (!r) return;
    const errs = Object.entries(r.errors);
    if (errs.length) toast("not saved: " + errs.map(([k, v]) => `${k}: ${v}`).join("; "), "error", 9000);
    if (r.changed.length) toast(`saved ${r.changed.join(", ")}${r.restart.length ? ` — restart: ${r.restart.join(", ")}` : ""}`, "ok", 7000);
    this.render($("#main"));
  },
  devicesCard() {
    const box = h("div", { class: "col" });
    const scan = async () => {
      box.replaceChildren(h("span", { class: "muted small" }, "scanning…"));
      const d = await guard(() => api("/api/settings/devices")); if (!d) return;
      const rank = new Map(d.ranked.map((r, i) => [r.index, i + 1]));
      const levels = {};
      box.replaceChildren(
        h("div", { class: "small" }, "MIC_KEYWORD = ", h("b", {}, d.keyword || "(unset)"), " → stt-server tries: ",
          d.ranked.length ? d.ranked.map((r, i) => `${i + 1}. [${r.index}] ${r.name} @${r.rate}`).join("   ") : h("span", { style: { color: "var(--err)" } }, "no match — system default input")),
        h("table", {}, h("tr", {}, ["#", "input device", "host API", "ch", "rate", "", ""].map(t => h("th", {}, t))),
          d.inputs.map(x => {
            const lv = levels[x.index] = h("td", { class: "small mono" });
            return h("tr", {}, h("td", {}, x.index), h("td", {}, x.name, rank.has(x.index) ? h("span", { class: "badge healthy", style: { marginLeft: "6px" } }, `pick #${rank.get(x.index)}`) : null),
              h("td", { class: "muted" }, x.host_api), h("td", {}, x.channels), h("td", {}, x.rate),
              h("td", { class: "row" },
                h("button", { class: "small", title: "set MIC_KEYWORD to this device name", onclick: () => {
                  const row = document.querySelector('.setting[data-key="MIC_KEYWORD"] input');
                  if (row) { row.value = x.name.replace(/\s*\(.*$/, "").trim() || x.name; row.dispatchEvent(new Event("input")); toast("MIC_KEYWORD updated — Save .env to apply"); }
                } }, "use"),
                h("button", { class: "small", onclick: async () => {
                  lv.textContent = "recording 3s…";
                  const r = await guard(() => api("/api/settings/level-test", { body: { index: x.index, seconds: 3 } }));
                  lv.textContent = r ? `RMS ${r.rms_dbfs.map((v, i) => `ch${i + 1} ${v}`).join(" · ")} dBFS · mono mix ${r.mono_mix_dbfs}` : "failed";
                } }, "test level")), lv);
          })),
        h("details", {}, h("summary", {}, `output devices (${d.outputs.length}) — for CABLE_DEVICE_NAME / SPEAKER_DEVICE_NAME`),
          h("table", {}, d.outputs.map(x => h("tr", {}, h("td", {}, x.index), h("td", {}, x.name), h("td", { class: "muted" }, x.host_api), h("td", {}, x.channels), h("td", {}, x.rate))))));
    };
    return h("div", { class: "card section" }, h("div", { class: "row" }, h("h3", { class: "grow", style: { margin: 0 } }, "Audio devices"), h("button", { class: "small", onclick: scan }, "Scan devices")), box);
  },
  personaCard() {
    const box = h("div", { class: "row" });
    api("/api/settings/persona").then(d => box.replaceChildren(...d.files.map(f => h("button", { class: "small", onclick: () =>
      openDrawer(f.name, h("div", { class: "row section" }, h("a", { href: `vscode://file/${f.path.replace(/\\/g, "/")}` }, "open in VS Code"), h("span", { class: "muted small" }, "read-only here — persona edits stay manual")), h("pre", {}, f.text)) }, f.name))));
    return h("div", { class: "card section" }, h("h3", {}, "Persona (read-only)"), box);
  },
};

// ─── router ──────────────────────────────────────────────
const PAGES = { stream: Stream, console: Console, data: Data, dream: Dream, settings: Settings };
function route() {
  const page = (location.hash || "#stream").slice(1).split("?")[0];
  const p = PAGES[page] ? page : "stream";
  S.timers.forEach(clearInterval); S.timers = [];
  S.page = p;
  document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("active", a.dataset.page === p));
  const main = $("#main");
  main.replaceChildren();
  PAGES[p].render(main);
}
window.addEventListener("hashchange", route);
loadServices().then(() => { connect(); route(); });
