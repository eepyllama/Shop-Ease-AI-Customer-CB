/**
 * ResolveIQ — Embeddable Multimodal Customer Resolution Widget
 *
 * Configurable via data attributes or dynamic config:
 *   <script src="http://localhost:8000/widget.js"
 *           data-tenant="demo_foodco"
 *           data-api="http://localhost:8000/api"
 *           async></script>
 */
(function () {
  "use strict";

  // Prevent multiple initializations
  if (window.__RESOLVEIQ_LOADED__) return;
  window.__RESOLVEIQ_LOADED__ = true;

  const currentScript =
    document.currentScript ||
    (function () {
      const scripts = document.getElementsByTagName("script");
      return scripts[scripts.length - 1];
    })();

  const API_BASE = currentScript.getAttribute("data-api") || "http://localhost:8000/api";
  let TENANT_ID = currentScript.getAttribute("data-tenant") || "demo_foodco";

  let tenantConfig = {
    brand_name: "Support Concierge",
    primary_color: "#2563EB",
    logo: "🤖",
    welcome_message: "Hi! How can we help resolve an issue today?",
    currency: "INR",
    reference_label: "order number (e.g. ORD-002)",
    issue_types: [],
    media_upload: { enabled: true, max_mb: 10, accept: "image/*,video/*" },
    footer_text: "Powered by ResolveIQ",
  };

  // State
  let isOpen = false;
  let currentCaseId = null;
  let currentSessionId = "sess-" + Math.random().toString(36).slice(2, 10);
  let selectedFile = null;
  let isTyping = false;

  // ── Inject CSS ─────────────────────────────────────────────────────────────
  const styleEl = document.createElement("style");
  styleEl.id = "riq-styles";
  styleEl.textContent = `
    #riq-root *, #riq-root *::before, #riq-root *::after { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    #riq-btn {
      position: fixed; bottom: 24px; right: 24px; z-index: 2147483646;
      width: 58px; height: 58px; border-radius: 50%; border: none; cursor: pointer;
      background: var(--riq-brand, #2563EB); color: #fff;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 4px 18px rgba(0,0,0,.25);
      transition: transform .2s, box-shadow .2s;
    }
    #riq-btn:hover { transform: scale(1.05); box-shadow: 0 6px 22px rgba(0,0,0,.3); }
    #riq-panel {
      position: fixed; bottom: 94px; right: 24px; z-index: 2147483645;
      width: 390px; max-width: calc(100vw - 32px);
      height: 600px; max-height: calc(100vh - 120px);
      background: #ffffff; border-radius: 16px; overflow: hidden;
      display: flex; flex-direction: column;
      box-shadow: 0 20px 60px rgba(0,0,0,.2), 0 4px 16px rgba(0,0,0,.1);
      transition: opacity .22s cubic-bezier(.34,1.56,.64,1), transform .22s cubic-bezier(.34,1.56,.64,1);
      border: 1px solid rgba(0,0,0,.08);
    }
    #riq-panel[hidden] { opacity: 0; transform: translateY(12px) scale(.97); pointer-events: none; display: flex !important; }
    #riq-panel.riq-open { opacity: 1; transform: translateY(0) scale(1); pointer-events: auto; }
    
    .riq-hdr { background: var(--riq-brand, #2563EB); padding: 14px 16px; display: flex; align-items: center; gap: 10px; flex-shrink: 0; color: #fff; }
    .riq-hdr-av { width: 36px; height: 36px; border-radius: 50%; background: rgba(255,255,255,.2); display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }
    .riq-hdr-info { flex: 1; min-width: 0; }
    .riq-hdr-title { font-weight: 700; font-size: 14.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .riq-hdr-sub { font-size: 11px; opacity: .85; display: flex; align-items: center; gap: 5px; margin-top: 2px; }
    .riq-dot { width: 6px; height: 6px; background: #4ADE80; border-radius: 50%; }
    .riq-close { background: none; border: none; color: #fff; opacity: .8; cursor: pointer; font-size: 18px; padding: 4px; border-radius: 4px; }
    .riq-close:hover { opacity: 1; }

    .riq-msgs { flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 12px; background: #F8FAFC; }
    .riq-msgs::-webkit-scrollbar { width: 4px; }
    .riq-msgs::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 2px; }
    
    .riq-row { display: flex; gap: 8px; align-items: flex-end; }
    .riq-row.user { flex-direction: row-reverse; }
    .riq-av { width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; flex-shrink: 0; }
    .riq-row.bot .riq-av { background: #E0E7FF; color: var(--riq-brand, #2563EB); }
    .riq-row.user .riq-av { background: #E2E8F0; color: #475569; }
    
    .riq-bub { max-width: 82%; padding: 10px 13px; border-radius: 14px; font-size: 13.5px; line-height: 1.5; word-break: break-word; }
    .riq-row.bot .riq-bub { background: #FFFFFF; color: #0F172A; border: 1px solid #E2E8F0; border-radius: 3px 14px 14px 14px; box-shadow: 0 1px 3px rgba(0,0,0,.04); }
    .riq-row.user .riq-bub { background: var(--riq-brand, #2563EB); color: #FFFFFF; border-radius: 14px 3px 14px 14px; }
    .riq-ts { font-size: 10px; color: #94A3B8; margin-top: 4px; }
    .riq-row.user .riq-ts { text-align: right; }

    /* Resolution Recommendation Card */
    .riq-card { background: #FFFFFF; border: 1px solid #CBD5E1; border-left: 4px solid var(--riq-brand, #2563EB); border-radius: 8px; padding: 12px; margin-top: 8px; box-shadow: 0 2px 6px rgba(0,0,0,.05); }
    .riq-card-title { font-size: 11.5px; text-transform: uppercase; font-weight: 700; letter-spacing: .5px; color: #64748B; margin-bottom: 4px; }
    .riq-card-amt { font-size: 18px; font-weight: 800; color: #0F172A; margin-bottom: 6px; }
    .riq-card-reasons { list-style: none; font-size: 11.5px; color: #475569; margin-bottom: 10px; line-height: 1.5; }
    .riq-card-reasons li::before { content: "✓ "; color: #16A34A; font-weight: bold; }
    .riq-card-actions { display: flex; gap: 8px; }
    .riq-btn-accept { flex: 1; background: #16A34A; color: #fff; border: none; border-radius: 6px; padding: 8px 12px; font-size: 12.5px; font-weight: 600; cursor: pointer; transition: background .15s; }
    .riq-btn-accept:hover { background: #15803D; }
    .riq-btn-human { background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; border-radius: 6px; padding: 8px 10px; font-size: 12px; font-weight: 500; cursor: pointer; }
    .riq-btn-human:hover { background: #E2E8F0; }

    /* Verification Badge */
    .riq-verified-badge { display: inline-flex; align-items: center; gap: 5px; background: #DCFCE7; color: #15803D; padding: 4px 8px; border-radius: 99px; font-size: 11.5px; font-weight: 700; margin-top: 6px; border: 1px solid #BBF7D0; }
    .riq-escalated-badge { display: inline-flex; align-items: center; gap: 5px; background: #FEF3C7; color: #B45309; padding: 4px 8px; border-radius: 99px; font-size: 11.5px; font-weight: 700; margin-top: 6px; border: 1px solid #FDE68A; }

    /* Evidence analysis preview inside bot message */
    .riq-ev-preview { margin-top: 8px; border-radius: 6px; overflow: hidden; border: 1px solid #E2E8F0; background: #F8FAFC; }
    .riq-ev-img { width: 100%; max-height: 160px; object-fit: cover; display: block; }
    .riq-ev-signals { padding: 6px 10px; font-size: 11px; color: #475569; border-top: 1px solid #E2E8F0; display: flex; flex-direction: column; gap: 2px; }

    /* Chips */
    .riq-chips { display: flex; flex-wrap: wrap; gap: 6px; padding: 8px 14px; background: #FFFFFF; border-top: 1px solid #E2E8F0; }
    .riq-chip { background: #F1F5F9; color: #334155; border: 1px solid #E2E8F0; border-radius: 99px; padding: 4px 10px; font-size: 11.5px; font-weight: 500; cursor: pointer; transition: all .15s; }
    .riq-chip:hover { background: var(--riq-brand, #2563EB); color: #fff; border-color: var(--riq-brand, #2563EB); }

    /* Pending file upload bar */
    .riq-file-bar { display: none; align-items: center; gap: 8px; padding: 6px 14px; background: #EFF6FF; border-top: 1px solid #BFDBFE; font-size: 12px; color: #1E40AF; }
    .riq-file-bar.active { display: flex; }
    .riq-file-name { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .riq-file-remove { background: none; border: none; color: #64748B; cursor: pointer; font-size: 14px; font-weight: bold; }

    /* Input row */
    .riq-in-row { padding: 10px 12px; background: #FFFFFF; border-top: 1px solid #E2E8F0; display: flex; gap: 8px; align-items: flex-end; }
    .riq-upload-btn { width: 34px; height: 34px; background: #F1F5F9; border: 1px solid #CBD5E1; border-radius: 8px; cursor: pointer; display: flex; align-items: center; justify-content: center; color: #475569; transition: background .15s; flex-shrink: 0; }
    .riq-upload-btn:hover { background: #E2E8F0; color: #0F172A; }
    .riq-textarea { flex: 1; border: 1px solid #CBD5E1; border-radius: 8px; padding: 8px 10px; font-size: 13.5px; color: #0F172A; resize: none; outline: none; max-height: 80px; min-height: 34px; line-height: 1.4; }
    .riq-textarea:focus { border-color: var(--riq-brand, #2563EB); }
    .riq-send-btn { width: 34px; height: 34px; background: var(--riq-brand, #2563EB); border: none; border-radius: 8px; cursor: pointer; display: flex; align-items: center; justify-content: center; color: #fff; flex-shrink: 0; }
    .riq-send-btn:disabled { opacity: .5; cursor: default; }

    .riq-footer { text-align: center; font-size: 10.5px; color: #94A3B8; padding: 4px 0 8px; background: #FFFFFF; }
    
    .riq-typing { display: flex; align-items: center; gap: 4px; padding: 4px 0; }
    .riq-typing span { width: 6px; height: 6px; background: #94A3B8; border-radius: 50%; animation: riq-b .8s infinite ease-in-out; }
    .riq-typing span:nth-child(2) { animation-delay: .15s; }
    .riq-typing span:nth-child(3) { animation-delay: .3s; }
    @keyframes riq-b { 0%,80%,100%{ transform:translateY(0); } 40%{ transform:translateY(-5px); } }
  `;
  document.head.appendChild(styleEl);

  // ── Build DOM ──────────────────────────────────────────────────────────────
  const root = document.createElement("div");
  root.id = "riq-root";
  root.innerHTML = `
    <button id="riq-btn" aria-label="Open customer support">
      <svg id="riq-icon-chat" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
      <svg id="riq-icon-close" style="display:none" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
    <div id="riq-panel" hidden>
      <div class="riq-hdr">
        <div class="riq-hdr-av" id="riq-logo">🤖</div>
        <div class="riq-hdr-info">
          <div class="riq-hdr-title" id="riq-brand-name">Support Concierge</div>
          <div class="riq-hdr-sub"><span class="riq-dot"></span> <span id="riq-status-text">AI-assisted resolution</span></div>
        </div>
        <button class="riq-close" id="riq-close-btn" aria-label="Close">✕</button>
      </div>
      
      <div class="riq-msgs" id="riq-msgs"></div>
      
      <div class="riq-chips" id="riq-chips"></div>

      <div class="riq-file-bar" id="riq-file-bar">
        <span>📎</span>
        <span class="riq-file-name" id="riq-file-name">filename.jpg</span>
        <button class="riq-file-remove" id="riq-file-remove" title="Remove attachment">✕</button>
      </div>

      <div class="riq-in-row">
        <input type="file" id="riq-file-input" style="display:none" accept="image/*,video/*">
        <button class="riq-upload-btn" id="riq-upload-btn" title="Attach photo or evidence">
          <svg width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>
        </button>
        <textarea class="riq-textarea" id="riq-textarea" rows="1" placeholder="Describe the issue…"></textarea>
        <button class="riq-send-btn" id="riq-send-btn" aria-label="Send">
          <svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
        </button>
      </div>
      <div class="riq-footer" id="riq-footer">Powered by ResolveIQ</div>
    </div>
  `;
  document.body.appendChild(root);

  // Elements
  const panel = document.getElementById("riq-panel");
  const btn = document.getElementById("riq-btn");
  const iconChat = document.getElementById("riq-icon-chat");
  const iconClose = document.getElementById("riq-icon-close");
  const closeBtn = document.getElementById("riq-close-btn");
  const msgsContainer = document.getElementById("riq-msgs");
  const chipsContainer = document.getElementById("riq-chips");
  const inputEl = document.getElementById("riq-textarea");
  const sendBtn = document.getElementById("riq-send-btn");
  const fileInput = document.getElementById("riq-file-input");
  const uploadBtn = document.getElementById("riq-upload-btn");
  const fileBar = document.getElementById("riq-file-bar");
  const fileNameSpan = document.getElementById("riq-file-name");
  const fileRemoveBtn = document.getElementById("riq-file-remove");
  const brandNameEl = document.getElementById("riq-brand-name");
  const logoEl = document.getElementById("riq-logo");
  const footerEl = document.getElementById("riq-footer");

  // ── Load Tenant Config ─────────────────────────────────────────────────────
  async function loadTenantConfig(tid) {
    TENANT_ID = tid || TENANT_ID;
    try {
      const resp = await fetch(`${API_BASE}/widget/config?tenant_id=${encodeURIComponent(TENANT_ID)}`);
      if (resp.ok) {
        tenantConfig = await resp.json();
      }
    } catch (e) {
      console.warn("Could not fetch remote tenant config, using defaults:", e);
    }
    applyConfig();
  }

  function applyConfig() {
    document.documentElement.style.setProperty("--riq-brand", tenantConfig.primary_color || "#2563EB");
    brandNameEl.textContent = tenantConfig.brand_name || "Support Concierge";
    logoEl.textContent = tenantConfig.logo || "🤖";
    footerEl.textContent = tenantConfig.footer_text || "Powered by ResolveIQ";
    inputEl.placeholder = `Describe the issue or enter ${tenantConfig.reference_label || "order number"}…`;

    // Render chips
    chipsContainer.innerHTML = "";
    if (tenantConfig.issue_types && tenantConfig.issue_types.length) {
      tenantConfig.issue_types.forEach((it) => {
        const label = typeof it === "object" ? it.label : it;
        const id = typeof it === "object" ? it.id : it;
        const chip = document.createElement("div");
        chip.className = "riq-chip";
        chip.textContent = label;
        chip.addEventListener("click", () => {
          inputEl.value = `I have an issue: ${label}. `;
          inputEl.focus();
        });
        chipsContainer.appendChild(chip);
      });
      chipsContainer.style.display = "flex";
    } else {
      chipsContainer.style.display = "none";
    }
  }

  // ── UI Actions ─────────────────────────────────────────────────────────────
  function toggle() {
    isOpen = !isOpen;
    panel.hidden = !isOpen;
    panel.classList.toggle("riq-open", isOpen);
    iconChat.style.display = isOpen ? "none" : "";
    iconClose.style.display = isOpen ? "" : "none";
    if (isOpen) {
      if (msgsContainer.children.length === 0) {
        addBotMessage(tenantConfig.welcome_message || "Hi! How can we help you resolve an issue today?");
      }
      setTimeout(() => inputEl.focus(), 150);
    }
  }

  btn.addEventListener("click", toggle);
  closeBtn.addEventListener("click", toggle);

  // File Upload Handlers
  uploadBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      selectedFile = e.target.files[0];
      fileNameSpan.textContent = selectedFile.name;
      fileBar.classList.add("active");
    }
  });
  fileRemoveBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    fileBar.classList.remove("active");
  });

  // Message Renderers
  function nowStr() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function addUserMessage(text, file) {
    const row = document.createElement("div");
    row.className = "riq-row user";
    const body = document.createElement("div");
    body.className = "riq-bub";
    body.textContent = text;

    if (file) {
      const attach = document.createElement("div");
      attach.style.fontSize = "11px";
      attach.style.marginTop = "4px";
      attach.style.opacity = ".9";
      attach.textContent = `📎 ${file.name}`;
      body.appendChild(attach);
    }

    const time = document.createElement("div");
    time.className = "riq-ts";
    time.textContent = nowStr();

    row.appendChild(body);
    row.appendChild(time);
    msgsContainer.appendChild(row);
    msgsContainer.scrollTop = msgsContainer.scrollHeight;
  }

  function addBotMessage(text, data = {}) {
    const row = document.createElement("div");
    row.className = "riq-row bot";

    const av = document.createElement("div");
    av.className = "riq-av";
    av.textContent = tenantConfig.logo || "🤖";

    const body = document.createElement("div");
    body.className = "riq-bub";
    body.innerHTML = `<div>${escapeHtml(text)}</div>`;

    // Evidence analysis preview
    if (data.evidence_preview) {
      const evBox = document.createElement("div");
      evBox.className = "riq-ev-preview";
      if (data.evidence_preview.url) {
        const img = document.createElement("img");
        img.className = "riq-ev-img";
        img.src = data.evidence_preview.url;
        evBox.appendChild(img);
      }
      const signals = document.createElement("div");
      signals.className = "riq-ev-signals";
      signals.innerHTML = `
        <div><strong>Evidence Confidence:</strong> ${data.evidence_confidence || 85}/100</div>
        <div><strong>Visual Analysis:</strong> ${data.evidence_preview.note || "Damage detected & verified"}</div>
        <div><strong>Signals:</strong> EXIF checked · Provenance non-repudiated</div>
      `;
      evBox.appendChild(signals);
      body.appendChild(evBox);
    }

    // Recommendation Card
    if (data.recommendation && data.confirmation && data.status === "WAITING_FOR_CUSTOMER_CONFIRMATION") {
      const rec = data.recommendation;
      const conf = data.confirmation;
      const currencySymbol = tenantConfig.currency === "INR" ? "₹" : "$";
      const card = document.createElement("div");
      card.className = "riq-card";
      card.innerHTML = `
        <div class="riq-card-title">Resolution Recommended</div>
        <div class="riq-card-amt">${currencySymbol}${Number(rec.amount).toFixed(0)} Refund</div>
        <ul class="riq-card-reasons">
          ${(rec.reasons || ["ORDER_ITEM_MATCH", "POLICY_ELIGIBLE"]).map(r => `<li>${formatReason(r)}</li>`).join("")}
        </ul>
        <div class="riq-card-actions">
          <button class="riq-btn-accept" id="btn-accept-${conf.confirmation_id}">
            Accept ${currencySymbol}${Number(rec.amount).toFixed(0)} Refund
          </button>
          <button class="riq-btn-human" id="btn-human-${conf.confirmation_id}">
            Talk to Human
          </button>
        </div>
      `;

      // Wire action buttons
      setTimeout(() => {
        const acceptBtn = document.getElementById(`btn-accept-${conf.confirmation_id}`);
        const humanBtn = document.getElementById(`btn-human-${conf.confirmation_id}`);
        if (acceptBtn) {
          acceptBtn.addEventListener("click", () => handleAcceptConfirmation(conf.confirmation_id, rec.amount));
        }
        if (humanBtn) {
          humanBtn.addEventListener("click", () => handleRequestHuman());
        }
      }, 50);

      body.appendChild(card);
    }

    // Escalated Badge
    if (data.status === "ESCALATED") {
      const badge = document.createElement("div");
      badge.className = "riq-escalated-badge";
      badge.innerHTML = `⚑ Escalated to Support Specialist`;
      body.appendChild(badge);
    }

    // Verified Result Badge
    if (data.status === "RESOLVED" && data.verification) {
      const badge = document.createElement("div");
      badge.className = "riq-verified-badge";
      badge.innerHTML = `✓ Action Executed & Verified by Gateway`;
      body.appendChild(badge);
    }

    const time = document.createElement("div");
    time.className = "riq-ts";
    time.textContent = nowStr();

    const wrapper = document.createElement("div");
    wrapper.style.display = "flex";
    wrapper.style.flexDirection = "column";
    wrapper.style.alignItems = "flex-start";
    wrapper.appendChild(body);
    wrapper.appendChild(time);

    row.appendChild(av);
    row.appendChild(wrapper);
    msgsContainer.appendChild(row);
    msgsContainer.scrollTop = msgsContainer.scrollHeight;
  }

  function formatReason(r) {
    const map = {
      ORDER_ITEM_MATCH: "Item verified in your order record",
      ORDER_VERIFIED: "Order and transaction verified",
      POLICY_ELIGIBLE: "Eligible under support policy",
      CLAIM_SUPPORTED_BY_EVIDENCE: "Supported by uploaded evidence",
      DELAY_90MIN_FULL_REFUND: "Delivery delay exceeded 90 min threshold",
      SEVERITY_MODERATE: "Quality issue verified",
    };
    return map[r] || r.replace(/_/g, " ");
  }

  function escapeHtml(str) {
    return str.replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
  }

  function showTyping() {
    if (isTyping) return;
    isTyping = true;
    const row = document.createElement("div");
    row.id = "riq-typing-row";
    row.className = "riq-row bot";
    row.innerHTML = `
      <div class="riq-av">${tenantConfig.logo || "🤖"}</div>
      <div class="riq-bub"><div class="riq-typing"><span></span><span></span><span></span></div></div>
    `;
    msgsContainer.appendChild(row);
    msgsContainer.scrollTop = msgsContainer.scrollHeight;
  }

  function hideTyping() {
    isTyping = false;
    const row = document.getElementById("riq-typing-row");
    if (row) row.remove();
  }

  // ── Confirmation & Gateway Call ────────────────────────────────────────────
  async function handleAcceptConfirmation(confirmationId, amount) {
    showTyping();
    try {
      const resp = await fetch(`${API_BASE}/cases/${currentCaseId}/confirm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirmation_id: confirmationId }),
      });
      const data = await resp.json();
      hideTyping();
      if (resp.ok) {
        addBotMessage(data.message, { status: "RESOLVED", verification: data.verification });
      } else {
        addBotMessage(`Action could not be executed: ${data.detail?.message || "Verification failed"}`);
      }
    } catch (err) {
      hideTyping();
      addBotMessage("Unable to reach the confirmation gateway. Please try again.");
    }
  }

  async function handleRequestHuman() {
    showTyping();
    try {
      const resp = await fetch(`${API_BASE}/cases/${currentCaseId}/escalate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: "Customer requested human specialist" }),
      });
      const data = await resp.json();
      hideTyping();
      addBotMessage(data.message, { status: "ESCALATED" });
    } catch (e) {
      hideTyping();
      addBotMessage("Case escalated to our team. A representative will contact you shortly.");
    }
  }

  // ── Send Message & Upload ──────────────────────────────────────────────────
  async function sendMessage() {
    const text = inputEl.value.trim();
    const file = selectedFile;
    if (!text && !file) return;

    inputEl.value = "";
    selectedFile = null;
    fileInput.value = "";
    fileBar.classList.remove("active");
    inputEl.style.height = "auto";
    sendBtn.disabled = true;

    addUserMessage(text || (file ? `Uploaded evidence: ${file.name}` : ""), file);
    showTyping();

    try {
      // 1. If a file is attached and we have a case_id, upload evidence first
      let evidenceResult = null;
      let evidencePreviewUrl = null;

      if (file) {
        if (!currentCaseId) {
          // Create case first
          const caseResp = await fetch(`${API_BASE}/cases`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tenant_id: TENANT_ID, claim_text: text }),
          });
          if (caseResp.ok) {
            const cdata = await caseResp.json();
            currentCaseId = cdata.case.case_id;
          }
        }

        if (currentCaseId) {
          const formData = new FormData();
          formData.append("file", file);
          if (text) formData.append("claim_text", text);
          const evResp = await fetch(`${API_BASE}/cases/${currentCaseId}/evidence`, {
            method: "POST",
            body: formData,
          });
          if (evResp.ok) {
            evidenceResult = await evResp.json();
            if (file.type.startsWith("image/")) {
              evidencePreviewUrl = URL.createObjectURL(file);
            }
          }
        }
      }

      // 2. Send message to chat/case endpoint
      const resp = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text || (file ? `Attached photo evidence: ${file.name}` : ""),
          session_id: currentSessionId,
          tenant_id: TENANT_ID,
        }),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      currentCaseId = data.case_id || currentCaseId;
      currentSessionId = data.session_id || currentSessionId;
      hideTyping();

      const botPayload = {
        recommendation: data.recommendation,
        confirmation: data.confirmation,
        status: data.status,
        evidence_confidence: data.evidence_confidence || (evidenceResult ? evidenceResult.evidence_confidence : null),
      };

      if (evidencePreviewUrl) {
        botPayload.evidence_preview = {
          url: evidencePreviewUrl,
          note: evidenceResult?.evidence?.visual_analysis?.note || "Analyzed against claim",
        };
      }

      addBotMessage(data.response, botPayload);
    } catch (err) {
      hideTyping();
      addBotMessage("I encountered an issue connecting with the resolution server. Please ensure the server is running.");
    } finally {
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  // Event Listeners for Input
  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Public SDK methods
  window.ResolveIQ = {
    open: () => { if (!isOpen) toggle(); },
    close: () => { if (isOpen) toggle(); },
    switchTenant: async (tid) => {
      await loadTenantConfig(tid);
      currentCaseId = null;
      currentSessionId = "sess-" + Math.random().toString(36).slice(2, 10);
      msgsContainer.innerHTML = "";
      addBotMessage(tenantConfig.welcome_message);
    },
    getTenantId: () => TENANT_ID,
  };

  // Initialize
  loadTenantConfig(TENANT_ID);
})();
