/**
 * ResolveIQ — Drop-in customer support chat widget
 *
 * Usage (any website):
 *   <script src="https://your-server.com/widget.js"
 *           data-api="https://your-server.com/api"
 *           data-company="Your Company"
 *           data-color="#2563EB"
 *           async></script>
 */
(function () {
  "use strict";

  // ── Read config from the script tag that loaded this file ──────────────
  const currentScript =
    document.currentScript ||
    (function () {
      const scripts = document.getElementsByTagName("script");
      return scripts[scripts.length - 1];
    })();

  const API_BASE    = currentScript.getAttribute("data-api")     || "http://localhost:8000/api";
  const COMPANY     = currentScript.getAttribute("data-company") || "Support";
  const COLOR       = currentScript.getAttribute("data-color")   || "#2563EB";

  // Derive a lighter shade for hover states
  const COLOR_DARK  = COLOR; // keep same; hover handled by opacity

  // ── Inject styles ───────────────────────────────────────────────────────
  const style = document.createElement("style");
  style.textContent = `
    #riq-root *, #riq-root *::before, #riq-root *::after { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    #riq-btn {
      position: fixed; bottom: 24px; right: 24px; z-index: 2147483646;
      width: 56px; height: 56px; border-radius: 50%; border: none; cursor: pointer;
      background: ${COLOR}; color: #fff;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 4px 14px rgba(0,0,0,.25);
      transition: transform .2s, opacity .2s;
    }
    #riq-btn:hover { opacity: .9; transform: scale(1.06); }
    #riq-panel {
      position: fixed; bottom: 92px; right: 24px; z-index: 2147483645;
      width: 360px; max-width: calc(100vw - 32px);
      height: 520px; max-height: calc(100vh - 112px);
      background: #fff; border-radius: 16px; overflow: hidden;
      display: flex; flex-direction: column;
      box-shadow: 0 20px 60px rgba(0,0,0,.18), 0 4px 16px rgba(0,0,0,.1);
      transition: opacity .2s, transform .2s;
    }
    #riq-panel[hidden] { opacity: 0; transform: translateY(10px) scale(.97); pointer-events: none; display: flex !important; }
    #riq-panel.riq-open { opacity: 1; transform: none; pointer-events: auto; }
    .riq-hdr { background: ${COLOR}; padding: 14px 16px; display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
    .riq-hdr-av { width: 34px; height: 34px; border-radius: 50%; background: rgba(255,255,255,.2); display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; }
    .riq-hdr-info { flex: 1; min-width: 0; }
    .riq-hdr-name { font-weight: 700; font-size: 14px; color: #fff; }
    .riq-hdr-sub { font-size: 11.5px; color: rgba(255,255,255,.75); display: flex; align-items: center; gap: 5px; margin-top: 1px; }
    .riq-live-dot { width: 6px; height: 6px; background: #4ADE80; border-radius: 50%; }
    .riq-close { background: none; border: none; color: rgba(255,255,255,.8); cursor: pointer; font-size: 18px; line-height: 1; padding: 4px; }
    .riq-msgs { flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 10px; }
    .riq-msgs::-webkit-scrollbar { width: 3px; }
    .riq-msgs::-webkit-scrollbar-thumb { background: #ddd; border-radius: 2px; }
    .riq-row { display: flex; gap: 7px; align-items: flex-end; }
    .riq-row.riq-user { flex-direction: row-reverse; }
    .riq-av { width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 700; flex-shrink: 0; }
    .riq-row.riq-bot .riq-av { background: #EFF6FF; color: ${COLOR}; }
    .riq-row.riq-user .riq-av { background: #F1F5F9; color: #64748B; }
    .riq-bub { max-width: 78%; padding: 9px 13px; border-radius: 14px; font-size: 13.5px; line-height: 1.5; }
    .riq-row.riq-bot .riq-bub { background: #F1F5F9; color: #0F172A; border-radius: 2px 14px 14px 14px; }
    .riq-row.riq-user .riq-bub { background: ${COLOR}; color: #fff; border-radius: 14px 2px 14px 14px; }
    .riq-ts { font-size: 10px; color: #94A3B8; margin-top: 3px; }
    .riq-row.riq-user .riq-ts { text-align: right; }
    .riq-chip { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 99px; font-size: 11px; font-weight: 700; margin-top: 6px; }
    .riq-chip-ok { background: #DCFCE7; color: #166534; }
    .riq-chip-warn { background: #FEF9C3; color: #854D0E; }
    .riq-chip-bad { background: #FEE2E2; color: #991B1B; }
    .riq-typing { display: flex; align-items: center; gap: 4px; padding: 4px 0; }
    .riq-typing span { width: 7px; height: 7px; background: #94A3B8; border-radius: 50%; animation: riq-bounce .8s infinite ease-in-out; }
    .riq-typing span:nth-child(2) { animation-delay: .16s; }
    .riq-typing span:nth-child(3) { animation-delay: .32s; }
    @keyframes riq-bounce { 0%,80%,100% { transform: translateY(0); } 40% { transform: translateY(-6px); } }
    .riq-inp-row { padding: 10px 12px; border-top: 1px solid #E2E8F0; display: flex; gap: 8px; flex-shrink: 0; }
    .riq-inp { flex: 1; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 9px; padding: 8px 12px; font-size: 13.5px; color: #0F172A; resize: none; outline: none; max-height: 80px; overflow-y: auto; transition: border-color .15s; }
    .riq-inp:focus { border-color: ${COLOR}; }
    .riq-send { width: 34px; height: 34px; background: ${COLOR}; border: none; border-radius: 8px; cursor: pointer; display: flex; align-items: center; justify-content: center; flex-shrink: 0; transition: opacity .15s; }
    .riq-send:hover { opacity: .88; }
    .riq-send:disabled { opacity: .4; cursor: default; }
    .riq-footer { text-align: center; font-size: 10.5px; color: #CBD5E1; padding: 4px 0 8px; flex-shrink: 0; }
    @media (max-width: 420px) {
      #riq-panel { width: calc(100vw - 16px); right: 8px; bottom: 76px; height: calc(100vh - 96px); }
      #riq-btn { bottom: 16px; right: 16px; }
    }
  `;
  document.head.appendChild(style);

  // ── Build DOM ────────────────────────────────────────────────────────────
  const root = document.createElement("div");
  root.id = "riq-root";

  root.innerHTML = `
    <button id="riq-btn" aria-label="Open support chat">
      <svg id="riq-icon-chat" width="22" height="22" fill="none" stroke="#fff" stroke-width="2" viewBox="0 0 24 24"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
      <svg id="riq-icon-x" style="display:none" width="18" height="18" fill="none" stroke="#fff" stroke-width="2.5" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
    <div id="riq-panel" hidden>
      <div class="riq-hdr">
        <div class="riq-hdr-av">🤖</div>
        <div class="riq-hdr-info">
          <div class="riq-hdr-name" id="riq-company-name">${COMPANY}</div>
          <div class="riq-hdr-sub"><span class="riq-live-dot"></span> AI-powered · instant replies</div>
        </div>
        <button class="riq-close" id="riq-close-btn" aria-label="Close">✕</button>
      </div>
      <div class="riq-msgs" id="riq-msgs"></div>
      <div class="riq-inp-row">
        <textarea class="riq-inp" id="riq-inp" rows="1" placeholder="Describe your issue…" aria-label="Message"></textarea>
        <button class="riq-send" id="riq-send" aria-label="Send">
          <svg width="14" height="14" fill="none" stroke="#fff" stroke-width="2.5" viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
        </button>
      </div>
      <div class="riq-footer">Powered by ResolveIQ</div>
    </div>
  `;
  document.body.appendChild(root);

  // ── State ─────────────────────────────────────────────────────────────────
  let isOpen = false;
  let sessionId = null;
  let typingRow = null;

  const panel   = document.getElementById("riq-panel");
  const btn     = document.getElementById("riq-btn");
  const msgs    = document.getElementById("riq-msgs");
  const inp     = document.getElementById("riq-inp");
  const sendBtn = document.getElementById("riq-send");
  const closeBtn = document.getElementById("riq-close-btn");
  const iconChat = document.getElementById("riq-icon-chat");
  const iconX    = document.getElementById("riq-icon-x");

  // ── Toggle ────────────────────────────────────────────────────────────────
  function toggle() {
    isOpen = !isOpen;
    panel.hidden = !isOpen;
    if (isOpen) panel.classList.add("riq-open");
    else panel.classList.remove("riq-open");
    iconChat.style.display = isOpen ? "none"  : "";
    iconX.style.display    = isOpen ? ""      : "none";
    if (isOpen) {
      if (!msgs.children.length) greeting();
      setTimeout(() => inp.focus(), 200);
    }
  }

  btn.addEventListener("click", toggle);
  closeBtn.addEventListener("click", toggle);
  document.addEventListener("keydown", e => { if (e.key === "Escape" && isOpen) toggle(); });

  // ── Greeting ──────────────────────────────────────────────────────────────
  function greeting() {
    sessionId = "riq-" + Math.random().toString(36).slice(2, 10);
    addMsg("bot", `Hi! I'm the ${COMPANY} support assistant. I can help with missing items, quality issues, wrong orders, and late deliveries. What's your order number and what happened?`);
  }

  // ── Render message ────────────────────────────────────────────────────────
  function ts() { return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }

  function addMsg(role, text, decision) {
    const row = document.createElement("div");
    row.className = "riq-row riq-" + role;

    const av = document.createElement("div");
    av.className = "riq-av";
    av.textContent = role === "user" ? "You" : "AI";

    const body = document.createElement("div");
    const bub = document.createElement("div");
    bub.className = "riq-bub";
    bub.textContent = text;

    if (decision && decision !== "need_order_id") {
      const chip = document.createElement("div");
      const map = {
        full_refund:    ["riq-chip-ok",   "✓ Full refund approved"],
        partial_refund: ["riq-chip-ok",   "✓ Partial refund approved"],
        deny:           ["riq-chip-bad",  "✗ Claim not approved"],
        escalate:       ["riq-chip-warn", "⚑ Escalated to team"],
      };
      const [cls, lbl] = map[decision] || ["", decision];
      chip.className = "riq-chip " + cls;
      chip.textContent = lbl;
      bub.appendChild(document.createElement("br"));
      bub.appendChild(chip);
    }

    const time = document.createElement("div");
    time.className = "riq-ts";
    time.textContent = ts();

    body.appendChild(bub);
    body.appendChild(time);
    row.appendChild(av);
    row.appendChild(body);
    if (role === "user") row.insertBefore(body, av);

    msgs.appendChild(row);
    msgs.scrollTop = msgs.scrollHeight;
  }

  // ── Typing indicator ──────────────────────────────────────────────────────
  function showTyping() {
    if (typingRow) return;
    typingRow = document.createElement("div");
    typingRow.className = "riq-row riq-bot";
    typingRow.innerHTML = '<div class="riq-av">AI</div><div class="riq-bub"><div class="riq-typing"><span></span><span></span><span></span></div></div>';
    msgs.appendChild(typingRow);
    msgs.scrollTop = msgs.scrollHeight;
  }
  function hideTyping() {
    if (typingRow) { typingRow.remove(); typingRow = null; }
  }

  // ── Send ──────────────────────────────────────────────────────────────────
  async function send() {
    const text = inp.value.trim();
    if (!text) return;
    inp.value = "";
    inp.style.height = "auto";
    sendBtn.disabled = true;
    addMsg("user", text);
    showTyping();

    try {
      const res = await fetch(API_BASE + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });
      const data = await res.json();
      sessionId = data.session_id || sessionId;
      hideTyping();
      addMsg("bot", data.response, data.decision);
    } catch (_) {
      hideTyping();
      addMsg("bot", "I'm having trouble connecting to the server. Please try again in a moment.");
    } finally {
      sendBtn.disabled = false;
      inp.focus();
    }
  }

  sendBtn.addEventListener("click", send);
  inp.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  inp.addEventListener("input", () => {
    inp.style.height = "auto";
    inp.style.height = Math.min(inp.scrollHeight, 80) + "px";
  });
})();
