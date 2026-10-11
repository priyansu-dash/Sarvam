/* =========================================================
   sarvam/paper-lab — frontend logic
   Talks ONLY to our backend (/api/*). No Sarvam key here.
   ========================================================= */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const MAX_REC_SECONDS = 30;
  const MIN_REC_SECONDS = 0.5;
  const RING_LEN = 339.3;

  const state = {
    paperId: null,
    paperReady: false,
    pollTimer: null,
    summary: "",
    summaryAudioKey: null,
    rec: null,          // { recorder, chunks, startedAt, raf }
    busy: false,        // voice pipeline in progress
    stream: null,       // MediaStream (cached)
    audioCtx: null,
    analyser: null,
  };

  const voice = new Audio();
  voice.preload = "auto";

  // ---------------------------------------------------------------- console / toast
  const logBody = $("console-body");
  function log(msg, level = "") {
    const t = new Date().toLocaleTimeString([], { hour12: false });
    const line = document.createElement("div");
    line.className = `log-line ${level}`;
    line.innerHTML = `<span class="t">${t}</span><span class="m"></span>`;
    line.querySelector(".m").textContent = msg;
    logBody.appendChild(line);
    logBody.scrollTop = logBody.scrollHeight;
  }
  let toastTimer;
  function toast(msg) {
    const el = $("toast");
    el.textContent = msg;
    el.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add("hidden"), 6000);
  }
  $("btn-console").addEventListener("click", () => $("console").classList.toggle("collapsed"));

  // ---------------------------------------------------------------- API helper
  class ApiError extends Error {
    constructor(code, message, status) { super(message); this.code = code; this.status = status; }
  }
  async function api(path, { method = "GET", json, form, raw = false } = {}) {
    const opts = { method, headers: {} };
    if (json !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(json); }
    if (form) opts.body = form;
    let res;
    try { res = await fetch(path, opts); }
    catch { throw new ApiError("network", "Cannot reach the backend. Is the server running?", 0); }
    if (!res.ok) {
      let code = "http_" + res.status, message = `Request failed (${res.status})`;
      try { const b = await res.json(); if (b.error) ({ code, message } = b.error); } catch { /* not json */ }
      throw new ApiError(code, message, res.status);
    }
    return raw ? res : res.json();
  }

  function setBtnLoading(btn, loading, label) {
    btn.disabled = loading;
    btn.classList.toggle("loading", loading);
    if (label) btn.querySelector(".btn-label").textContent = label;
  }

  // ---------------------------------------------------------------- health
  (async function health() {
    try {
      const h = await api("/api/health");
      $("dot-key").className = "dot " + (h.key_configured ? "ok" : "bad");
      $("sys-key").textContent = h.key_configured ? "api key: configured" : "api key: MISSING (.env)";
      $("sys-models").textContent = `${h.models.chat} · ${h.models.stt} · ${h.models.tts}`;
      log(`backend online — ${h.models.chat}, ${h.models.stt}, ${h.models.tts}`, h.key_configured ? "ok" : "warn");
      if (!h.key_configured) toast("SARVAM_API_KEY is not set on the server. Add it to .env and restart.");
    } catch (e) {
      $("dot-key").className = "dot bad";
      $("sys-key").textContent = "backend offline";
      log(e.message, "err");
    }
  })();

  // ---------------------------------------------------------------- upload
  const dz = $("dropzone");
  const fileInput = $("file-input");
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) uploadFile(f); });
  fileInput.addEventListener("change", () => { const f = fileInput.files[0]; if (f) uploadFile(f); fileInput.value = ""; });

  function setPaperChip(text, cls = "") {
    const c = $("paper-state-chip");
    c.textContent = text;
    c.className = "chip mono " + cls;
  }

  async function uploadFile(file) {
    if (!/\.pdf$/i.test(file.name) && file.type !== "application/pdf") return toast("Please choose a PDF file.");
    if (file.size > 50 * 1024 * 1024) return toast("PDF exceeds the 50 MB limit.");

    resetPaperUI();
    dz.classList.add("disabled");
    $("ocr-box").classList.remove("hidden");
    $("ocr-file").textContent = file.name;
    $("ocr-stage").innerHTML = `<span class="spinner"></span> uploading ${(file.size / 1024 / 1024).toFixed(1)} MB…`;
    setPaperChip("uploading", "busy");
    log(`upload: ${file.name} (${(file.size / 1024).toFixed(0)} KB)`, "info");

    const form = new FormData();
    form.append("file", file);
    try {
      const p = await api("/api/upload", { method: "POST", form });
      state.paperId = p.paper_id;
      localStorage.setItem("paperId", p.paper_id);
      log(`paper ${p.paper_id}: ${p.page_count} pages → ${p.chunk_count} OCR chunk(s) of ≤10 pages`, "info");
      renderOcr(p);
      startPolling();
    } catch (e) {
      ocrFailed(e.message);
    }
  }

  function startPolling() {
    clearInterval(state.pollTimer);
    state.pollTimer = setInterval(pollPaper, 2000);
    pollPaper();
  }

  let lastStage = "";
  async function pollPaper() {
    if (!state.paperId) return;
    try {
      const p = await api(`/api/papers/${state.paperId}`);
      renderOcr(p);
      if (p.stage !== lastStage) { lastStage = p.stage; if (p.state === "processing") log(p.stage); }
      if (p.state === "ready") { clearInterval(state.pollTimer); paperReady(p); }
      else if (p.state === "failed") { clearInterval(state.pollTimer); ocrFailed(p.error?.message || "OCR failed."); }
    } catch (e) {
      if (e.status === 404) { clearInterval(state.pollTimer); ocrFailed("Paper not found on server (it may have restarted). Upload again."); }
      else log(`poll: ${e.message}`, "warn");
    }
  }

  function renderOcr(p) {
    $("ocr-box").classList.remove("hidden");
    $("ocr-file").textContent = p.filename;
    $("ocr-count").textContent = `${p.chunks_done}/${p.chunk_count} chunks · ${p.page_count} pages`;
    const bar = $("ocr-segbar");
    if (bar.children.length !== p.chunk_count) {
      bar.innerHTML = "";
      for (let i = 0; i < p.chunk_count; i++) bar.appendChild(Object.assign(document.createElement("div"), { className: "seg" }));
    }
    [...bar.children].forEach((seg, i) => {
      seg.className = "seg" + (i < p.chunks_done ? " done" : (i === p.chunks_done && p.state === "processing" ? " active" : ""));
      if (p.state === "failed" && i === p.chunks_done) seg.classList.add("fail");
    });
    if (p.state !== "ready" && p.state !== "failed") {
      $("ocr-stage").innerHTML = `<span class="spinner"></span> <span></span>`;
      $("ocr-stage").lastChild.textContent = p.stage;
      setPaperChip(p.state === "queued" ? "queued" : "ocr…", "busy");
    }
  }

  function ocrFailed(msg) {
    dz.classList.remove("disabled");
    setPaperChip("failed", "err");
    $("ocr-stage").textContent = "✕ " + msg;
    log(`ocr failed: ${msg}`, "err");
    toast(msg);
    localStorage.removeItem("paperId");
  }

  function paperReady(p) {
    state.paperReady = true;
    dz.classList.remove("disabled");
    $("ocr-box").classList.add("hidden");
    $("meta-box").classList.remove("hidden");
    $("m-pages").textContent = p.page_count;
    $("m-chunks").textContent = p.chunk_count;
    $("m-chars").textContent = p.char_count.toLocaleString();
    $("m-id").textContent = p.paper_id;
    $("m-preview").textContent = (p.preview || "") + (p.char_count > 1200 ? "\n\n…" : "");
    const w = $("m-warnings");
    w.innerHTML = "";
    (p.warnings || []).forEach((txt) => { const li = document.createElement("li"); li.textContent = txt; w.appendChild(li); log(txt, "warn"); });
    w.classList.toggle("hidden", !(p.warnings || []).length);
    setPaperChip("ready", "ok");
    log(`paper ready — ${p.char_count.toLocaleString()} chars extracted in page order`, "ok");
    enableQA(true);
    if (p.has_summary) summarize(false);
  }

  function resetPaperUI() {
    clearInterval(state.pollTimer);
    Object.assign(state, { paperId: null, paperReady: false, summary: "", summaryAudioKey: null });
    lastStage = "";
    $("meta-box").classList.add("hidden");
    $("summary-box").classList.add("hidden");
    $("ocr-box").classList.add("hidden");
    $("ocr-segbar").innerHTML = "";
    const a = $("summary-audio"); a.pause(); a.removeAttribute("src"); a.classList.add("hidden");
    $("chat").querySelectorAll(".msg").forEach((m) => m.remove());
    $("chat-empty").classList.remove("hidden");
    enableQA(false);
    setPaperChip("idle");
  }

  $("btn-new").addEventListener("click", () => { localStorage.removeItem("paperId"); resetPaperUI(); fileInput.click(); });

  // Resume the last paper after a reload (backend caches extracted text on disk).
  (function resume() {
    const id = localStorage.getItem("paperId");
    if (!id) return;
    state.paperId = id;
    log(`resuming paper ${id}…`);
    startPolling();
  })();

  // ---------------------------------------------------------------- summary
  $("btn-summarize").addEventListener("click", () => summarize(!!state.summary));

  async function summarize(force) {
    if (!state.paperReady) return;
    const btn = $("btn-summarize");
    setBtnLoading(btn, true, "Summarizing");
    $("summary-box").classList.remove("hidden");
    $("summary-content").innerHTML = `<div class="skeleton">${[92, 78, 85, 60, 88, 70, 40].map((w) => `<i style="width:${w}%"></i>`).join("")}</div>`;
    $("summary-meta").textContent = "sarvam-105b · reading the paper…";
    log(force ? "regenerating summary…" : "summarizing with sarvam-105b…", "info");
    const t0 = performance.now();
    try {
      const r = await api("/api/summarize", { method: "POST", json: { paper_id: state.paperId, force } });
      state.summary = r.summary;
      state.summaryAudioKey = null;
      const a = $("summary-audio"); a.pause(); a.removeAttribute("src"); a.classList.add("hidden");
      $("btn-read").querySelector(".btn-label").textContent = "▶ Read aloud";
      $("summary-content").innerHTML = renderMarkdown(r.summary);
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      $("summary-meta").textContent = r.cached ? "summary · cached" : `summary · ${r.strategy} · ${secs}s`;
      log(`summary ready (${r.cached ? "cached" : r.strategy + ", " + secs + "s"})`, "ok");
      setBtnLoading(btn, false, "Regenerate summary");
    } catch (e) {
      $("summary-content").innerHTML = "";
      $("summary-box").classList.add("hidden");
      log(`summary failed: ${e.message}`, "err");
      toast(e.message);
      setBtnLoading(btn, false, state.summary ? "Regenerate summary" : "Generate summary");
    }
  }

  // Read aloud (REST TTS, chunked server-side, MP3)
  $("btn-read").addEventListener("click", async () => {
    const audio = $("summary-audio");
    const btn = $("btn-read");
    const speaker = $("voice-select").value;
    const pace = parseFloat($("pace-select").value);
    const key = `${speaker}|${pace}`;

    if (state.summaryAudioKey === key && audio.src) {
      if (audio.paused) audio.play(); else audio.pause();
      return;
    }
    voice.pause();
    setBtnLoading(btn, true, "Synthesizing");
    log(`tts: synthesizing summary with ${speaker} @ ${pace}×…`, "info");
    try {
      const res = await api("/api/tts", { method: "POST", json: { text: state.summary, speaker, pace, language_code: "en-IN" }, raw: true });
      const blob = await res.blob();
      if (audio.src) URL.revokeObjectURL(audio.src);
      audio.src = URL.createObjectURL(blob);
      audio.classList.remove("hidden");
      state.summaryAudioKey = key;
      await audio.play().catch(() => {});
      log(`tts: ${(blob.size / 1024).toFixed(0)} KB audio · ${res.headers.get("X-TTS-Segments") || 1} segment(s)`, "ok");
    } catch (e) {
      log(`tts failed: ${e.message}`, "err");
      toast(e.message);
    } finally {
      setBtnLoading(btn, false, "⏯ Play / pause");
    }
  });
  ["voice-select", "pace-select"].forEach((id) => $(id).addEventListener("change", () => {
    if (state.summaryAudioKey) $("btn-read").querySelector(".btn-label").textContent = "▶ Read aloud";
  }));

  // ---------------------------------------------------------------- markdown (tiny, safe)
  function esc(s) { return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  function inline(s) {
    return esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*(?!\s)(.+?)\*(?!\*)/g, "$1<em>$2</em>");
  }
  function renderMarkdown(md) {
    const lines = md.replace(/\r/g, "").split("\n");
    const out = [];
    let list = null, para = [], table = null;
    const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; } };
    const flushList = () => { if (list) { out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; } };
    const flushTable = () => {
      if (table) {
        const [head, ...rows] = table;
        out.push(`<table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
        table = null;
      }
    };
    const flushAll = () => { flushPara(); flushList(); flushTable(); };
    for (const raw of lines) {
      const line = raw.trimEnd();
      let m;
      if (!line.trim()) { flushAll(); continue; }
      if ((m = line.match(/^\s*(#{1,6})\s+(.*)$/))) { flushAll(); const lvl = Math.min(Math.max(m[1].length, 2), 4); out.push(`<h${lvl}>${inline(m[2].replace(/#+\s*$/, ""))}</h${lvl}>`); continue; }
      if (/^\s*\|.*\|\s*$/.test(line)) {
        flushPara(); flushList();
        if (/^\s*\|[\s:\-|]+\|\s*$/.test(line)) continue;
        (table ||= []).push(line.trim().slice(1, -1).split("|").map((c) => c.trim()));
        continue;
      }
      if ((m = line.match(/^\s*[-*+]\s+(.*)$/))) { flushPara(); flushTable(); if (!list || list.tag !== "ul") { flushList(); list = { tag: "ul", items: [] }; } list.items.push(m[1]); continue; }
      if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) { flushPara(); flushTable(); if (!list || list.tag !== "ol") { flushList(); list = { tag: "ol", items: [] }; } list.items.push(m[1]); continue; }
      if (list && /^\s{2,}\S/.test(raw)) { list.items[list.items.length - 1] += " " + line.trim(); continue; }
      flushList(); flushTable();
      para.push(line.trim());
    }
    flushAll();
    return out.join("\n");
  }

  // ---------------------------------------------------------------- Q&A UI
  const chat = $("chat");
  const pipeline = $("pipeline");
  function setStep(step, status) {
    const li = pipeline.querySelector(`[data-step="${step}"]`);
    if (li) li.className = status || "";
  }
  function resetPipeline() { pipeline.querySelectorAll("li").forEach((li) => (li.className = "")); }

  function enableQA(on) {
    $("btn-mic").disabled = !on;
    $("ask-input").disabled = !on;
    $("btn-ask").disabled = !on;
    $("mic-label").innerHTML = on ? "hold to talk <kbd>space</kbd>" : "upload a paper first";
  }

  function addMsg(role, text, extra = {}) {
    $("chat-empty").classList.add("hidden");
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    const label = role === "user" ? "you" : role === "bot" ? "sarvam-105b" : "error";
    el.innerHTML = `<div class="who"><span>${label}</span></div><div class="bubble"></div>`;
    const bubble = el.querySelector(".bubble");
    if (extra.typing) bubble.innerHTML = `<span class="typing"><span></span><span></span><span></span></span>`;
    else bubble.textContent = text;
    if (extra.meta) el.querySelector(".who").insertAdjacentHTML("beforeend", `<span>${esc(extra.meta)}</span>`);
    chat.appendChild(el);
    chat.scrollTop = chat.scrollHeight;
    return el;
  }

  $("btn-reset").addEventListener("click", async () => {
    voice.pause();
    chat.querySelectorAll(".msg").forEach((m) => m.remove());
    $("chat-empty").classList.remove("hidden");
    resetPipeline();
    if (state.paperId && state.paperReady) {
      try { await api(`/api/papers/${state.paperId}/reset`, { method: "POST" }); log("conversation memory cleared", "ok"); } catch (e) { log(e.message, "warn"); }
    }
  });

  $("ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("ask-input").value.trim();
    if (!q || state.busy) return;
    $("ask-input").value = "";
    resetPipeline();
    setStep("listen", "done");
    setStep("transcribe", "done");
    askQuestion(q);
  });

  async function askQuestion(question) {
    state.busy = true;
    $("btn-mic").classList.add("busy");
    addMsg("user", question);
    const pending = addMsg("bot", "", { typing: true });
    setStep("think", "active");
    log(`ask: "${question}"`, "info");
    const t0 = performance.now();
    try {
      const r = await api("/api/ask", { method: "POST", json: { paper_id: state.paperId, question } });
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      setStep("think", "done");
      pending.querySelector(".bubble").textContent = r.answer;
      pending.querySelector(".who").insertAdjacentHTML("beforeend", `<span>${secs}s${r.used_excerpts ? " · excerpts" : ""}</span>`);
      const replay = Object.assign(document.createElement("button"), { className: "replay", type: "button", textContent: "▶ replay" });
      pending.querySelector(".who").appendChild(replay);
      chat.scrollTop = chat.scrollHeight;
      log(`answer in ${secs}s`, "ok");
      let streamUrl = null;
      const speak = async () => {
        try {
          if (!streamUrl) {
            const s = await api("/api/tts/stream", { method: "POST", json: { text: r.answer, speaker: $("voice-select").value, pace: parseFloat($("pace-select").value) } });
            streamUrl = s.stream_url;
          }
          await playVoice(streamUrl);
        } catch (e) { streamUrl = null; throw e; }
      };
      replay.addEventListener("click", () => { resetPipeline(); speak().catch((e) => { setStep("speak", "fail"); toast(e.message); }); });
      await speak();
    } catch (e) {
      pending.remove();
      addMsg("err", e.message);
      pipeline.querySelector("li.active")?.classList.replace("active", "fail");
      log(`q&a failed: ${e.message}`, "err");
    } finally {
      state.busy = false;
      $("btn-mic").classList.remove("busy");
    }
  }

  function playVoice(url) {
    return new Promise((resolve, reject) => {
      $("summary-audio").pause();
      setStep("speak", "active");
      voice.onended = () => { setStep("speak", "done"); resolve(); };
      voice.onerror = () => { setStep("speak", "fail"); reject(new Error("Could not play the spoken answer.")); };
      voice.src = url;
      voice.play().catch((err) => {
        // Autoplay blocked: answer is still visible; user can hit replay.
        setStep("speak", "");
        log("autoplay blocked — press ▶ replay to hear the answer", "warn");
        resolve(err);
      });
    });
  }

  // ---------------------------------------------------------------- recording (push-to-talk)
  const mic = $("btn-mic");
  const ringFg = $("ring-fg");
  const level = $("level");
  const lctx = level.getContext("2d");

  async function ensureMic() {
    if (state.stream && state.stream.active) return state.stream;
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser does not support microphone capture.");
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    state.audioCtx = state.audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const src = state.audioCtx.createMediaStreamSource(state.stream);
    state.analyser = state.audioCtx.createAnalyser();
    state.analyser.fftSize = 512;
    src.connect(state.analyser);
    return state.stream;
  }

  async function startRec() {
    if (state.rec || state.busy || mic.disabled) return;
    voice.pause(); // barge-in: stop any answer playback
    $("summary-audio").pause();
    let stream;
    try { stream = await ensureMic(); }
    catch (e) { toast(e.name === "NotAllowedError" ? "Microphone permission denied." : e.message); log(`mic: ${e.message}`, "err"); return; }
    if (state.audioCtx.state === "suspended") await state.audioCtx.resume();

    const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"].find((m) => window.MediaRecorder?.isTypeSupported?.(m)) || "";
    const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    const rec = { recorder, chunks: [], startedAt: performance.now(), raf: 0, cancelled: false };
    recorder.ondataavailable = (e) => e.data.size && rec.chunks.push(e.data);
    recorder.onstop = () => onRecStop(rec);
    recorder.start(250);
    state.rec = rec;

    resetPipeline();
    setStep("listen", "active");
    mic.classList.add("rec");
    $("mic-label").textContent = "listening… release to send";
    tick();
  }

  function tick() {
    const rec = state.rec;
    if (!rec) return;
    const secs = (performance.now() - rec.startedAt) / 1000;
    ringFg.style.strokeDashoffset = String(RING_LEN * (1 - Math.min(secs / MAX_REC_SECONDS, 1)));
    mic.classList.toggle("warn", secs > MAX_REC_SECONDS - 5);
    $("mic-timer").textContent = `${secs.toFixed(1).padStart(4, "0")}s / ${MAX_REC_SECONDS}s`;
    drawLevel();
    if (secs >= MAX_REC_SECONDS) { log("30s limit reached — sending", "warn"); stopRec(); return; }
    rec.raf = requestAnimationFrame(tick);
  }

  function drawLevel() {
    const w = level.width, h = level.height;
    lctx.clearRect(0, 0, w, h);
    if (!state.analyser) return;
    const data = new Uint8Array(state.analyser.frequencyBinCount);
    state.analyser.getByteTimeDomainData(data);
    const bars = 36, bw = w / bars;
    const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
    lctx.fillStyle = `hsl(${accent})`.replace("hsl(hsl(", "hsl(").replace("))", ")");
    for (let i = 0; i < bars; i++) {
      let peak = 0;
      const start = Math.floor((i * data.length) / bars), end = Math.floor(((i + 1) * data.length) / bars);
      for (let j = start; j < end; j++) peak = Math.max(peak, Math.abs(data[j] - 128));
      const bh = Math.max(2, (peak / 128) * h * 1.6);
      lctx.fillRect(i * bw + 1, (h - Math.min(bh, h)) / 2, bw - 2, Math.min(bh, h));
    }
  }

  function stopRec(cancel = false) {
    const rec = state.rec;
    if (!rec) return;
    rec.cancelled = cancel;
    cancelAnimationFrame(rec.raf);
    if (rec.recorder.state !== "inactive") rec.recorder.stop();
    state.rec = null;
    mic.classList.remove("rec", "warn");
    ringFg.style.strokeDashoffset = String(RING_LEN);
    lctx.clearRect(0, 0, level.width, level.height);
    $("mic-label").innerHTML = "hold to talk <kbd>space</kbd>";
  }

  async function onRecStop(rec) {
    const secs = (performance.now() - rec.startedAt) / 1000;
    $("mic-timer").textContent = `00.0s / ${MAX_REC_SECONDS}s`;
    if (rec.cancelled) { resetPipeline(); return; }
    if (secs < MIN_REC_SECONDS) { resetPipeline(); log("too short — hold the button while you speak", "warn"); return; }
    setStep("listen", "done");
    setStep("transcribe", "active");
    state.busy = true;
    mic.classList.add("busy");
    try {
      const blob = new Blob(rec.chunks, { type: rec.recorder.mimeType || "audio/webm" });
      let upload = blob, name = "question.webm";
      try { upload = await toWav16k(blob, MAX_REC_SECONDS); name = "question.wav"; }
      catch (e) { log(`wav conversion skipped (${e.message}); sending ${blob.type}`, "warn"); }
      log(`stt: sending ${(upload.size / 1024).toFixed(0)} KB (${secs.toFixed(1)}s) to saaras:v4`, "info");
      const form = new FormData();
      form.append("file", upload, name);
      if (state.paperId) form.append("paper_id", state.paperId);
      const t = await api("/api/transcribe", { method: "POST", form });
      setStep("transcribe", "done");
      log(`transcript: "${t.transcript}" (${t.language_code || "?"})`, "ok");
      state.busy = false;
      await askQuestion(t.transcript);
    } catch (e) {
      setStep("transcribe", "fail");
      addMsg("err", e.message);
      log(`stt failed: ${e.message}`, "err");
    } finally {
      state.busy = false;
      mic.classList.remove("busy");
    }
  }

  // Decode the recording and re-encode as 16 kHz mono 16-bit PCM WAV (best STT accuracy).
  async function toWav16k(blob, maxSeconds) {
    const buf = await blob.arrayBuffer();
    const ctx = state.audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await ctx.decodeAudioData(buf.slice(0));
    const rate = 16000;
    const dur = Math.min(decoded.duration, maxSeconds);
    const off = new OfflineAudioContext(1, Math.ceil(dur * rate), rate);
    const src = off.createBufferSource();
    src.buffer = decoded;
    src.connect(off.destination);
    src.start(0);
    const rendered = await off.startRendering();
    const pcm = rendered.getChannelData(0);
    const out = new DataView(new ArrayBuffer(44 + pcm.length * 2));
    const wstr = (o, s) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
    wstr(0, "RIFF"); out.setUint32(4, 36 + pcm.length * 2, true); wstr(8, "WAVE");
    wstr(12, "fmt "); out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
    out.setUint32(24, rate, true); out.setUint32(28, rate * 2, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true);
    wstr(36, "data"); out.setUint32(40, pcm.length * 2, true);
    for (let i = 0; i < pcm.length; i++) {
      const s = Math.max(-1, Math.min(1, pcm[i]));
      out.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([out], { type: "audio/wav" });
  }

  // Pointer (mouse/touch/pen) push-to-talk
  mic.addEventListener("pointerdown", (e) => { e.preventDefault(); mic.setPointerCapture(e.pointerId); startRec(); });
  mic.addEventListener("pointerup", () => stopRec());
  mic.addEventListener("pointercancel", () => stopRec(true));
  mic.addEventListener("contextmenu", (e) => e.preventDefault());

  // Keyboard: hold Space to talk, S to summarize (when not typing)
  const typing = () => ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
  window.addEventListener("keydown", (e) => {
    if (typing()) return;
    if (e.code === "Space") { e.preventDefault(); if (!e.repeat) startRec(); }
    else if (e.key.toLowerCase() === "s" && !e.metaKey && !e.ctrlKey && state.paperReady && !$("btn-summarize").disabled) summarize(!!state.summary);
  });
  window.addEventListener("keyup", (e) => { if (e.code === "Space" && !typing()) { e.preventDefault(); stopRec(); } });
  window.addEventListener("blur", () => stopRec());

  log("ready — drop a PDF to begin");
})();
