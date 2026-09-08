"""Primus Gradio UI — the whole presentation layer.

Default view is the conversation and nothing else: a thin top bar (wordmark + one status dot
+ the active chat name), the transcript in a centred column, and a single composer pinned to
the bottom. Everything else — chats/projects, mode, status & setup, knowledge, scheduled
agents, voice & window, connections, activity/background agents — lives in one slide-over
drawer behind the Menu button. The only exception is the command-approval bar, which appears
directly above the composer whenever something risky is queued, drawer open or not.

The drawer is CSS-only (``body.primus-drawer-open`` toggled by the load-JS in
``_keyboard_js``), so every component inside stays mounted and every Gradio event stays wired
whether it is on screen or not. This module also holds the stylesheet (``PRIMUS_CSS``), voice
input (faster-whisper), the typing effect, every Gradio event handler/closure, ``build_ui``
itself, and the no-Gradio fallback setup server.

The ONLY mechanical change vs. the original single-file code is that references to admin_assistant
module globals are reached through the live host module as ``_host.<name>`` (no shadowing exists),
which keeps the import-cycle safe and preserves runtime-rebound config/model globals
(``_host.CFG`` / ``_host.DEFAULT_MODEL`` / ``_host.FORGE_MODEL``). ``gr`` (the gradio module, or
``None`` when gradio is absent) is bound from the host. Nothing here is locked — edit the layout,
theme, handlers, and wiring exactly as before.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta
from functools import partial
from html import escape as _esc

import sys as _sys

# Resolve the host module without importing it by name (admin_assistant runs as "__main__").
_host = _sys.modules.get("admin_assistant") or _sys.modules["__main__"]

# Operator identity (generic "the operator" unless ~/.config/primus/operator.json personalizes it).
from primus.core.identity import OPERATOR as _OPERATOR, render as _render_op  # noqa: E402

# gradio module (or None when gradio is unavailable) — bound from the host so the UI uses the exact
# same object. build_ui is only ever called when gradio is present; importing this module is safe
# either way (module-level code here references no host globals).
gr = _host.gr


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

PRIMUS_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ===========================================================================
   Primus — private terminal for an operator.
   Near-black surfaces, hairline borders, one restrained platinum accent.
   Default view is the conversation only; everything else lives in the drawer.
   =========================================================================== */

:root, .gradio-container {
  --p-bg:#07080B;
  --p-surface:#0C0E13;
  --p-elev:#141821;
  --p-elev-2:#1C222C;
  --p-line:rgba(255,255,255,.06);
  --p-line-2:rgba(255,255,255,.10);
  --p-text:#E8EAED;
  --p-muted:#8B909A;
  --p-faint:#5C616B;
  --p-accent:#9AA7B2;
  --p-accent-2:#7A8B9A;
  --p-ok:#3D8B6E;
  --p-err:#C45C4A;
  --p-warn:#B08A4A;
  --p-font:'Inter','IBM Plex Sans',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --p-mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  --p-col:760px;
  --p-drawer:352px;
  --p-topbar:46px;
}

/* ---- page shell ---- */
html, body, .gradio-container, .gradio-container .main, gradio-app {
  background:var(--p-bg) !important; }
body { margin:0 !important; }
.gradio-container {
  font-family:var(--p-font) !important; font-size:15px; color:var(--p-text) !important;
  max-width:100% !important; padding:0 !important; }
.gradio-container .contain, .gradio-container .wrap, .gradio-container .fillable { background:transparent !important; }
/* "Built with Gradio" / settings footer, and any tab strip a future component drags in —
   the drawer is the only navigation. */
footer, .gradio-container footer { display:none !important; }
.tab-nav, .gradio-container .tab-nav { display:none !important; }
.gradio-container .block, .gradio-container .form, .gradio-container .panel {
  background:transparent !important; border:none !important; box-shadow:none !important;
  padding:0 !important; }
.gradio-container .block { min-width:0 !important; }
.gradio-container .row, .gradio-container .column { gap:8px !important; }

/* ---- type ---- */
.gradio-container, .gradio-container p, .gradio-container span, .gradio-container li,
.gradio-container td, .gradio-container th, .gradio-container .prose,
.gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container h4 {
  color:var(--p-text) !important; }
.gradio-container .prose h1, .gradio-container .prose h2 { font-size:15px !important; font-weight:600 !important; }
.gradio-container .prose h3, .gradio-container .prose h4 {
  font-size:11px !important; font-weight:500 !important; letter-spacing:.10em;
  text-transform:uppercase; color:var(--p-muted) !important; margin:2px 0 4px !important; }
.gradio-container label, .gradio-container .block-info, .gradio-container span[data-testid="block-info"] {
  color:var(--p-muted) !important; font-size:11.5px !important; font-weight:500 !important;
  letter-spacing:.015em; }
.gradio-container .prose a { color:var(--p-accent) !important; text-decoration:none;
  border-bottom:1px solid rgba(154,167,178,.28); }
.gradio-container code, .gradio-container .prose code {
  font-family:var(--p-mono) !important; font-size:.86em !important;
  color:#C6CDD4 !important; background:rgba(255,255,255,.045) !important;
  border:1px solid var(--p-line) !important; border-radius:4px; padding:1px 4px; }
.gradio-container pre, .gradio-container .prose pre {
  background:var(--p-surface) !important; border:1px solid var(--p-line) !important;
  border-radius:8px !important; }
.gradio-container pre code { border:none !important; background:transparent !important; }
.gradio-container hr { border-color:var(--p-line) !important; }

/* ---- controls ---- */
.gradio-container textarea, .gradio-container input[type=text], .gradio-container input[type=number],
.gradio-container input[type=password], .gradio-container select, .gradio-container .wrap-inner {
  background:var(--p-surface) !important; color:var(--p-text) !important;
  border:1px solid var(--p-line) !important; border-radius:8px !important;
  font-family:var(--p-font) !important; font-size:13.5px !important; box-shadow:none !important; }
.gradio-container textarea:focus, .gradio-container input:focus, .gradio-container .wrap-inner:focus-within {
  border-color:rgba(154,167,178,.42) !important;
  box-shadow:0 0 0 1px rgba(154,167,178,.14) !important; outline:none !important; }
.gradio-container ::placeholder { color:var(--p-faint) !important; }
.gradio-container input[type=checkbox], .gradio-container input[type=radio] {
  accent-color:var(--p-accent-2) !important; }
.gradio-container .options, .gradio-container ul.options {
  background:var(--p-elev) !important; border:1px solid var(--p-line-2) !important;
  border-radius:8px !important; }
.gradio-container .options li.item:hover, .gradio-container .options .active {
  background:rgba(255,255,255,.06) !important; }
.gradio-container input[type=range] { accent-color:var(--p-accent-2) !important; }

.gradio-container button {
  font-family:var(--p-font) !important; font-weight:500 !important; font-size:12.5px !important;
  border-radius:8px !important; letter-spacing:.01em; box-shadow:none !important;
  transition:background .15s ease, border-color .15s ease, color .15s ease; }
.gradio-container button.primary, .gradio-container .primary {
  background:var(--p-elev-2) !important; color:var(--p-text) !important;
  border:1px solid var(--p-line-2) !important; }
.gradio-container button.primary:hover, .gradio-container .primary:hover {
  background:#242C38 !important; border-color:rgba(255,255,255,.18) !important; }
.gradio-container button.secondary, .gradio-container .secondary {
  background:#0F1218 !important; color:var(--p-text) !important;
  border:1px solid var(--p-line) !important; }
.gradio-container button.secondary:hover { background:#151A22 !important; border-color:var(--p-line-2) !important; }
.gradio-container button.stop, .gradio-container .stop {
  background:transparent !important; color:var(--p-err) !important;
  border:1px solid rgba(196,92,74,.38) !important; }
.gradio-container button.stop:hover { background:rgba(196,92,74,.10) !important;
  border-color:rgba(196,92,74,.55) !important; }
.gradio-container .icon-button-wrapper, .gradio-container .icon-button-wrapper button {
  background:transparent !important; border-color:var(--p-line) !important; color:var(--p-muted) !important; }
/* Gradio flashes an animated accent border on every component a streaming handler touches.
   The quiet activity line under the composer already reports state — no pulsing. */
.gradio-container .generating, .gradio-container .block.generating {
  border:none !important; box-shadow:none !important; animation:none !important; }
.gradio-container .progress-bar, .gradio-container .progress-level-inner {
  background:var(--p-accent-2) !important; color:var(--p-text) !important; }
.gradio-container .progress-text, .gradio-container .meta-text {
  background:var(--p-elev) !important; color:var(--p-muted) !important;
  border:1px solid var(--p-line) !important; }

/* ===================== top bar ===================== */
#primus-topbar {
  position:fixed !important; top:0; left:0; right:0; height:var(--p-topbar);
  z-index:40; display:flex !important; flex-wrap:nowrap !important; align-items:center;
  gap:12px !important; padding:0 14px !important;
  background:rgba(7,8,11,.86) !important; backdrop-filter:blur(14px);
  border-bottom:1px solid var(--p-line) !important; }
#primus-topbar > * { min-width:0 !important; }
#menu-btn {
  flex:0 0 auto !important; width:32px !important; min-width:32px !important; height:28px !important;
  padding:0 !important; font-size:15px !important; line-height:1 !important;
  background:transparent !important; border:1px solid var(--p-line) !important;
  color:var(--p-muted) !important; }
#menu-btn:hover { color:var(--p-text) !important; background:#11151C !important;
  border-color:var(--p-line-2) !important; }
#primus-brand { flex:1 1 auto !important; overflow:hidden; }
.pr-brand { display:flex; align-items:center; gap:12px; min-width:0; }
.pr-mark { font-size:12px; font-weight:500; letter-spacing:.14em; color:var(--p-text);
  white-space:nowrap; }
.pr-status { display:inline-flex; align-items:center; gap:6px; font-size:11px;
  color:var(--p-muted); font-family:var(--p-mono); white-space:nowrap; }
.pr-chat { font-size:11.5px; color:var(--p-faint); white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; }
.pr-dot, .health-dot { display:inline-block; width:6px; height:6px; border-radius:50%;
  background:var(--p-faint); flex:0 0 auto; }
.pr-dot.ok, .health-dot.ok { background:var(--p-ok); }
.pr-dot.warn, .health-dot.warn { background:var(--p-warn); }
.pr-dot.bad, .health-dot.bad { background:var(--p-err); }

/* ===================== conversation ===================== */
#primus-main { padding:var(--p-topbar) 16px 0 !important; }
#primus-stage { width:100%; max-width:var(--p-col); margin:0 auto !important; }
/* The height itself is passed to gr.Chatbot as a CSS string so Gradio sizes its own scroll
   container (and keeps auto-scrolling to the newest message). */
#primus-chat { background:transparent !important; border:none !important; }
#primus-chat .bubble-wrap, #primus-chat .message-wrap { background:transparent !important;
  padding:8px 0 28px !important; }
#primus-chat .message-row { margin:14px 0 !important; padding:0 !important; }
#primus-chat .icon-button-wrapper { opacity:.3; transition:opacity .15s ease; }
#primus-chat .icon-button-wrapper:hover { opacity:1; }
/* Gradio nests two `.message` elements per turn; flatten both, then dress only the outer
   one so user/assistant turns never end up with a bubble inside a bubble. */
#primus-chat .message, #primus-chat .message-content,
#primus-chat [data-testid="bot"], #primus-chat [data-testid="user"] {
  font-family:var(--p-font) !important; font-size:15px !important; line-height:1.55 !important;
  color:var(--p-text) !important; background:transparent !important; border:none !important;
  border-radius:0 !important; box-shadow:none !important; padding:0 !important; }
/* assistant: no bubble — a whisper-quiet left rule and type on the page */
#primus-chat .bot.message {
  border-left:1px solid rgba(255,255,255,.09) !important; padding:0 0 0 14px !important; }
/* user: slightly elevated surface, hairline border */
#primus-chat .user.message {
  background:var(--p-elev) !important; border:1px solid var(--p-line-2) !important;
  border-radius:12px !important; padding:9px 13px !important; }
#primus-chat .placeholder-content, #primus-chat .placeholder, #primus-chat .empty {
  color:var(--p-faint) !important; }
#primus-chat .placeholder-content p, #primus-chat .placeholder p {
  font-size:13.5px !important; color:var(--p-faint) !important; }
#primus-chat .thought, #primus-chat .metadata { color:var(--p-muted) !important; font-size:12px !important; }
.readiness-banner {
  font-size:11.5px; color:var(--p-muted); padding:7px 12px; margin:10px 0 0;
  border:1px solid var(--p-line); border-left:2px solid var(--p-warn);
  border-radius:8px; background:var(--p-surface); }
.readiness-banner.ready { border-left-color:var(--p-ok); }

/* ===================== dock: approval bar + composer ===================== */
#primus-dock {
  position:fixed !important; left:0; right:0; bottom:0; z-index:30;
  display:flex !important; flex-direction:column !important; gap:8px !important;
  padding:14px 16px 16px !important;
  background:linear-gradient(to top, var(--p-bg) 58%, rgba(7,8,11,0)) !important; }
#primus-dock > * { width:100%; max-width:var(--p-col); margin:0 auto !important; }
/* :not(.hide) so Gradio's own visibility toggle still wins when nothing is queued.
   Wraps to two lines (summary, then actions) instead of crushing the summary. */
#approval-bar:not(.hide) {
  display:flex !important; flex-wrap:wrap !important; align-items:center;
  justify-content:flex-end; gap:6px !important;
  background:rgba(196,92,74,.06) !important; border:1px solid rgba(196,92,74,.26) !important;
  border-radius:10px !important; padding:8px 10px !important; }
#cmd-preview { flex:1 1 100% !important; min-width:0 !important;
  max-height:40px; overflow:hidden; }
#cmd-preview *, #cmd-preview p, #cmd-preview li {
  font-size:11.5px !important; color:var(--p-muted) !important; margin:0 !important;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
#cmd-preview strong { color:var(--p-text) !important; }
#cmd-preview code { font-size:10.5px !important; }
#approval-bar button {
  flex:0 0 auto !important; min-width:auto !important; padding:5px 12px !important; }
#primus-composer {
  display:flex !important; flex-wrap:nowrap !important; align-items:flex-end; gap:6px !important;
  background:var(--p-surface) !important; border:1px solid var(--p-line) !important;
  border-radius:12px !important; padding:5px 6px 5px 12px !important; }
#primus-composer:focus-within { border-color:rgba(154,167,178,.30) !important; }
#primus-input { flex:1 1 auto !important; min-width:0 !important; }
#primus-input textarea {
  background:transparent !important; border:none !important; border-radius:0 !important;
  resize:none !important; font-size:15px !important; line-height:1.5 !important;
  padding:9px 2px !important; }
#primus-input textarea:focus { border:none !important; box-shadow:none !important; }
#send-btn, #halt-btn { flex:0 0 auto !important; height:34px !important;
  padding:0 14px !important; min-width:auto !important; }
/* Chat attachment button — same quiet chrome as Send/Halt, icon-only. */
#attach-btn { flex:0 0 auto !important; height:34px !important; min-width:34px !important;
  width:34px !important; padding:0 !important; background:transparent !important;
  border:1px solid var(--p-line) !important; border-radius:8px !important;
  color:var(--p-muted) !important; font-size:15px !important; line-height:1 !important; }
#attach-btn:hover { border-color:rgba(154,167,178,.42) !important; color:var(--p-text) !important;
  background:rgba(154,167,178,.06) !important; }
#log-line { padding:0 4px !important; }
#log-line input, #log-line textarea {
  background:transparent !important; border:none !important; box-shadow:none !important;
  padding:0 !important; text-align:right; font-size:10.5px !important;
  font-family:var(--p-mono) !important; color:var(--p-faint) !important; cursor:default; }
#send-btn { background:#1E2933 !important; border:1px solid rgba(154,167,178,.26) !important;
  color:var(--p-text) !important; }
#send-btn:hover { background:#26333F !important; border-color:rgba(154,167,178,.42) !important; }

/* ===================== drawer ===================== */
#primus-scrim {
  position:fixed; inset:0; z-index:50; background:rgba(4,5,7,.52);
  backdrop-filter:blur(2px); opacity:0; pointer-events:none;
  transition:opacity 200ms ease; }
body.primus-drawer-open #primus-scrim { opacity:1; pointer-events:auto; }
#primus-drawer {
  position:fixed !important; top:0; bottom:0; left:0; z-index:60;
  width:var(--p-drawer); max-width:88vw;
  display:flex !important; flex-direction:column !important; gap:0 !important;
  padding:0 0 32px !important; overflow-x:hidden; overflow-y:auto;
  background:rgba(9,11,15,.97) !important; backdrop-filter:blur(18px);
  border-right:1px solid var(--p-line) !important;
  transform:translateX(-100%); transition:transform 200ms ease; will-change:transform; }
/* Gradio 5 re-emits every custom rule a second time, prefixed with
   `.gradio-container.gradio-container-5-xx .contain`. That copy of a `body.…` selector can
   never match (body is outside `.contain`) but the prefixed copy of the *closed* rule above
   outranks a single-id open rule — so the drawer would stay shut. Doubling the id (two ids
   beats three classes) is what makes the open state win. Keep it doubled, keep !important. */
body.primus-drawer-open #primus-drawer#primus-drawer {
  transform:translateX(0) !important; box-shadow:24px 0 60px rgba(0,0,0,.55) !important; }
#primus-drawer::-webkit-scrollbar, #primus-chat ::-webkit-scrollbar,
#cmd-preview::-webkit-scrollbar { width:8px; height:8px; }
#primus-drawer::-webkit-scrollbar-thumb, #primus-chat ::-webkit-scrollbar-thumb,
#cmd-preview::-webkit-scrollbar-thumb { background:rgba(255,255,255,.08); border-radius:4px; }
#drawer-head {
  position:sticky; top:0; z-index:2; display:flex !important; flex-wrap:nowrap !important;
  align-items:center; justify-content:space-between; gap:8px !important;
  height:var(--p-topbar); padding:0 12px 0 16px !important;
  background:rgba(9,11,15,.97) !important; border-bottom:1px solid var(--p-line) !important; }
.drawer-title { font-size:11px; font-weight:500; letter-spacing:.14em;
  text-transform:uppercase; color:var(--p-muted); }
#drawer-close { flex:0 0 auto !important; width:28px !important; min-width:28px !important;
  height:26px !important; padding:0 !important; background:transparent !important;
  border:1px solid var(--p-line) !important; color:var(--p-muted) !important;
  font-size:12px !important; }
#drawer-close:hover { color:var(--p-text) !important; background:#11151C !important; }

.drawer-sec { border:none !important; border-bottom:1px solid var(--p-line) !important;
  border-radius:0 !important; background:transparent !important; overflow:visible !important; }
.drawer-sec > button, .drawer-sec > .label-wrap, .drawer-sec .label-wrap {
  padding:12px 16px !important; background:transparent !important; border:none !important; }
.drawer-sec > button span, .drawer-sec .label-wrap span, .drawer-sec .label-wrap > span > span {
  font-size:11px !important; font-weight:500 !important; letter-spacing:.11em;
  text-transform:uppercase; color:var(--p-muted) !important; }
.drawer-sec > button:hover span, .drawer-sec .label-wrap:hover span { color:var(--p-text) !important; }
.drawer-body { padding:0 16px 16px !important; gap:9px !important; }
.drawer-body > * { min-width:0 !important; }
.drawer-note { font-size:11px !important; color:var(--p-faint) !important; line-height:1.45; }
.drawer-note p, .drawer-note em { font-size:11px !important; color:var(--p-faint) !important; }
.drawer-sub { border:1px solid var(--p-line) !important; border-radius:8px !important;
  background:var(--p-surface) !important; }
.drawer-sub > button, .drawer-sub .label-wrap { padding:8px 10px !important; }
.drawer-sub > button span, .drawer-sub .label-wrap span {
  letter-spacing:.06em !important; text-transform:none !important; font-size:11.5px !important; }
.drawer-sub .drawer-body { padding:0 10px 10px !important; }
#primus-drawer button { padding:6px 10px !important; }
/* let button rows sit side by side in a 352px column instead of stacking */
#primus-drawer .row { flex-wrap:wrap !important; gap:6px !important; }
#primus-drawer .row button:not(#drawer-close) {
  min-width:auto !important; flex:1 1 auto !important; }
#primus-drawer .prose p, #primus-drawer .prose li { font-size:12px !important; }
#primus-drawer textarea, #primus-drawer input[type=text], #primus-drawer input[type=number],
#primus-drawer input[type=password] { font-size:12.5px !important; }

/* chats list — Gradio lays a Radio out as a wrapping row, which packs short chat names
   two-to-a-line; force one chat per line so the list reads as a list. */
#chat-list .wrap { display:flex !important; flex-direction:column !important;
  flex-wrap:nowrap !important; align-items:stretch !important; gap:2px !important; }
#chat-list label { display:flex; width:100%; align-items:center; gap:8px;
  border:1px solid transparent !important; border-radius:8px !important;
  padding:7px 9px !important; background:transparent !important;
  font-size:13px !important; color:var(--p-muted) !important; transition:background .12s ease; }
#chat-list label:hover { background:#11151C !important; color:var(--p-text) !important; }
#chat-list label:has(input:checked) { background:var(--p-elev) !important;
  border-color:var(--p-line-2) !important; color:var(--p-text) !important; }
#chat-list input { width:0 !important; height:0 !important; opacity:0 !important;
  position:absolute !important; }
#proj-status, #bg-submit-status, #tasks-status, #code-submit-status, #kb-status,
#web-status, #voice-status, #typing-status {
  font-size:11px !important; color:var(--p-muted) !important; }
#proj-status p, #kb-status p, #web-status p, #voice-status p { font-size:11px !important;
  color:var(--p-muted) !important; }

/* mode radio → segmented control */
#mode-radio .wrap { display:flex !important; gap:4px !important; }
#mode-radio label { flex:1 1 0; justify-content:center; border:1px solid var(--p-line) !important;
  border-radius:8px !important; padding:7px 8px !important; font-size:12px !important;
  color:var(--p-muted) !important; background:var(--p-surface) !important; }
#mode-radio label:has(input:checked) { background:var(--p-elev-2) !important;
  border-color:var(--p-line-2) !important; color:var(--p-text) !important; }
#mode-radio input { width:0 !important; height:0 !important; opacity:0 !important;
  position:absolute !important; }

/* ===================== drawer panels (host-rendered HTML) ===================== */
.status-bar { font-family:var(--p-mono); font-size:10.5px; line-height:1.5;
  color:var(--p-muted); letter-spacing:.01em; padding:8px 10px;
  border:1px solid var(--p-line); border-radius:8px; background:var(--p-surface); }
.think-panel { font-family:var(--p-mono); font-size:10.5px; max-height:180px; overflow-y:auto;
  border:1px solid var(--p-line); border-radius:8px; padding:8px 10px; background:var(--p-surface); }
.think-panel.empty { color:var(--p-faint); }
.think-step { display:flex; gap:8px; align-items:baseline; padding:3px 0;
  border-bottom:1px solid rgba(255,255,255,.035); }
.think-step:last-child { border-bottom:none; }
.think-step .think-title { color:var(--p-muted); }
.think-step.running .think-title { color:var(--p-text); }
.think-step.done .think-title { color:var(--p-accent); }
.think-step.error .think-title { color:var(--p-err); }
.think-icon { width:12px; flex:0 0 auto; color:var(--p-faint); }
.think-ts { margin-left:auto; color:var(--p-faint); font-size:9.5px; }
.think-detail { width:100%; padding-left:20px; font-size:9.5px; color:var(--p-faint); }

.bg-panel { font-family:var(--p-mono); font-size:10.5px; display:flex; flex-direction:column;
  gap:6px; max-height:240px; overflow-y:auto; }
.bg-panel.empty { color:var(--p-faint); padding:6px 2px; }
.bg-task { border:1px solid var(--p-line); border-left:2px solid var(--p-line-2);
  border-radius:8px; padding:7px 9px; background:var(--p-surface); }
.bg-task.running { border-left-color:var(--p-accent); }
.bg-task.completed { border-left-color:var(--p-ok); }
.bg-task.failed { border-left-color:var(--p-err); }
.bg-task.cancelled { opacity:.55; }
.bg-status { color:var(--p-muted); text-transform:uppercase; letter-spacing:.08em; font-size:9.5px; }
.bg-task.running .bg-status { color:var(--p-text); }
.bg-task.completed .bg-status { color:var(--p-ok); }
.bg-task.failed .bg-status { color:var(--p-err); }
.bg-age { float:right; color:var(--p-faint); }
.bg-title { color:var(--p-text); margin-top:3px; font-family:var(--p-font); font-size:11.5px; }
.bg-last { color:var(--p-faint); margin-top:2px; }

.setup-dashboard { display:flex; flex-direction:column; gap:8px; }
.setup-banner { display:flex; align-items:center; gap:8px; padding:8px 10px; font-size:11.5px;
  border:1px solid var(--p-line); border-left:2px solid var(--p-line-2);
  border-radius:8px; background:var(--p-surface); color:var(--p-muted); }
.setup-banner.ready { border-left-color:var(--p-ok); }
.setup-banner.partial { border-left-color:var(--p-warn); }
.setup-banner.needs-setup { border-left-color:var(--p-err); }
.setup-badge { font-family:var(--p-mono); font-size:9.5px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--p-text); }
.setup-metrics { display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:11px; }
.setup-metrics .metric { padding:7px 9px; border:1px solid var(--p-line); border-radius:8px;
  background:var(--p-surface); color:var(--p-muted); }
.setup-metrics .metric span { display:block; font-size:9.5px; letter-spacing:.10em;
  text-transform:uppercase; color:var(--p-faint); margin-bottom:2px; }
.setup-metrics code { font-size:10.5px !important; }
.setup-metrics em { font-size:9.5px; margin-left:5px; font-style:normal; }
.setup-metrics .forge-online { color:var(--p-ok); }
.setup-metrics .forge-offline { color:var(--p-err); }
.setup-grid { display:flex; flex-direction:column; gap:3px; max-height:220px; overflow-y:auto; }
.setup-chip { display:flex; flex-wrap:wrap; align-items:center; gap:7px; padding:5px 8px;
  font-size:11px; border:1px solid var(--p-line); border-radius:7px; color:var(--p-muted); }
.setup-chip.ok { background:rgba(61,139,110,.06); }
.setup-chip.bad { background:rgba(196,92,74,.07); border-color:rgba(196,92,74,.22); }
.setup-chip.opt { opacity:.7; }
.chip-icon { width:14px; flex:0 0 auto; filter:saturate(.5); }
.chip-name { flex:1; min-width:110px; color:var(--p-text); }
.fix-cmd { font-size:9.5px !important; color:var(--p-warn) !important;
  background:transparent !important; border:none !important; word-break:break-all; }

/* knowledge dropzone */
#kb-dropzone { border:1px dashed var(--p-line-2) !important; border-radius:10px !important;
  background:var(--p-surface) !important; transition:border-color .15s ease, background .15s ease; }
#kb-dropzone:hover, #kb-dropzone.drag-active {
  border-color:rgba(154,167,178,.42) !important; background:#0F1218 !important; }
#kb-dropzone .wrap, #kb-dropzone label { color:var(--p-muted) !important; font-weight:500; }

/* inbox: list + read pane scroll INSIDE the drawer (never explode the 520px drawer layout).
   Doubled ids beat Gradio 5's prefixed copy — same reason as the drawer rules above. */
#inbox-list#inbox-list { max-height:240px !important; overflow-y:auto !important; }
#inbox-read#inbox-read { max-height:260px !important; overflow-y:auto !important;
  border:1px solid var(--p-line); border-radius:10px; padding:10px 12px;
  background:var(--p-surface); font-size:13px; }

/* voice */
#primus-mic { border:1px solid var(--p-line) !important; border-radius:10px !important;
  background:var(--p-surface) !important; }
#primus-mic.recording, #primus-mic:has(button[aria-label*="Stop"]) {
  border-color:rgba(196,92,74,.45) !important; }
#primus-mic .wrap, #primus-mic .controls { background:transparent !important; }

/* narrow / compact
   Same doubled-id trick as the drawer open state: Gradio re-emits each rule under
   `.gradio-container.gradio-container-5-xx .contain`, which outranks a plain `#id`, so the
   compact overrides need two ids to land. Do not "simplify" these to a single id. */
.gradio-container.compact-mode { font-size:14px; }
.gradio-container.compact-mode #primus-main#primus-main {
  padding-left:10px !important; padding-right:10px !important; }
.gradio-container.compact-mode #primus-dock#primus-dock { padding:10px 10px 12px !important; }
.gradio-container.compact-mode #primus-chat#primus-chat .message-row { margin:10px 0 !important; }
@media (max-width:760px) {
  :root, .gradio-container { --p-drawer:320px; }
  #primus-main { padding-left:10px !important; padding-right:10px !important; }
  #primus-dock { padding:10px 10px 12px !important; }
  .pr-chat { display:none; }
}
@media (prefers-reduced-motion: reduce) {
  #primus-drawer, #primus-scrim { transition:none !important; }
}
"""

# Kept as an alias: build_ui returns this to admin_assistant.main() (theme/css are applied on
# gr.Blocks, not launch()). The cyberpunk sheet it used to hold is gone, not stacked.
CYBER_CSS = PRIMUS_CSS


def _keyboard_js() -> str:
    """Load-time JS: keyboard shortcuts + the drawer (open/close, scrim, Esc, click-outside).

    The drawer is a CSS-only slide-over: this toggles `body.primus-drawer-open`, so no Gradio
    state or re-render is involved and every component inside stays mounted and wired.
    """
    compact = "true" if _host.CFG.get("compact_mode") else "false"
    return f"""
() => {{
  const root = document.querySelector('.gradio-container');
  if (root && {compact}) root.classList.add('compact-mode');
  const body = document.body;
  if (!document.getElementById('primus-scrim')) {{
    const scrim = document.createElement('div');
    scrim.id = 'primus-scrim';
    body.appendChild(scrim);
  }}
  const isOpen = () => body.classList.contains('primus-drawer-open');
  const setDrawer = (open) => {{
    body.classList.toggle('primus-drawer-open', !!open);
    const btn = document.getElementById('menu-btn');
    if (btn) btn.setAttribute('aria-expanded', open ? 'true' : 'false');
  }};
  const focusInput = () => document.querySelector('#primus-input textarea')?.focus();

  // Keep the conversation pinned to the newest turn: Gradio only sticks to the bottom when you
  // are already there, so a long transcript otherwise opens at the top and a stream you started
  // scrolls away. `stick` disarms the moment you scroll up to read and re-arms when you come
  // back to the bottom — or send a new message — so a long stream never yanks the view.
  const wrap = () => document.querySelector('#primus-chat .bubble-wrap');
  let stick = true;
  const pin = () => {{ const w = wrap(); if (w) w.scrollTop = w.scrollHeight; }};
  const arm = () => {{ stick = true; pin(); }};
  // scroll doesn't bubble — capture phase is the only way to see the transcript's own scrolling.
  document.addEventListener('scroll', (e) => {{
    const t = e.target;
    if (!(t instanceof Element) || !t.matches('#primus-chat .bubble-wrap')) return;
    stick = (t.scrollHeight - t.scrollTop - t.clientHeight) < 120;
  }}, true);
  const chat = document.querySelector('#primus-chat');
  if (chat && 'MutationObserver' in window) {{
    let queued = false;
    new MutationObserver(() => {{
      if (!stick || queued) return;
      queued = true;
      requestAnimationFrame(() => {{ queued = false; if (stick) pin(); }});
    }}).observe(chat, {{ childList: true, subtree: true, characterData: true }});
  }}
  setTimeout(arm, 250);
  setTimeout(arm, 1200);

  document.addEventListener('click', (e) => {{
    const t = e.target;
    if (!(t instanceof Element)) return;
    if (t.closest('#menu-btn')) {{ setDrawer(!isOpen()); return; }}
    if (t.closest('#drawer-close')) {{ setDrawer(false); return; }}
    if (t.id === 'primus-scrim') {{ setDrawer(false); return; }}
    if (t.closest('#send-btn')) {{ arm(); return; }}
    // Compact mode is saved server-side by its own handler; flip the class now so the
    // density change is visible without a reload.
    if (t.closest('#compact-cb')) {{
      const cb = document.querySelector('#compact-cb input[type=checkbox]');
      if (cb && root) root.classList.toggle('compact-mode', cb.checked);
    }}
  }});

  document.addEventListener('keydown', (e) => {{
    if (!(e.ctrlKey || e.metaKey)) {{
      if (e.key === 'ArrowUp' && e.target?.matches('#primus-input textarea') && !e.target.value) {{
        e.preventDefault(); document.getElementById('hist-prev')?.click();
      }}
      if (e.key === 'Enter' && !e.shiftKey && e.target?.matches('#primus-input textarea')) {{
        arm();
      }}
      if (e.key === 'Escape') {{
        if (isOpen()) {{ e.preventDefault(); setDrawer(false); }}
        else {{ document.getElementById('hide-tray-btn')?.click(); }}
      }}
      return;
    }}
    const k = e.key.toLowerCase();
    if (k==='k') {{ e.preventDefault(); setDrawer(false); focusInput(); }}
    if (k==='m') {{ e.preventDefault(); setDrawer(!isOpen()); }}
    if (k==='l') {{ e.preventDefault(); document.getElementById('clear-btn')?.click(); }}
    if (k==='s') {{ e.preventDefault(); document.getElementById('mode-suggest')?.click(); }}
    if (k==='e') {{ e.preventDefault(); document.getElementById('mode-exec')?.click(); }}
    if (k==='h') {{ e.preventDefault(); document.getElementById('hide-tray-btn')?.click(); }}
  }});
}}
"""

HELP_TEXT = """
| Shortcut | Action |
|----------|--------|
| `Ctrl+K` | Focus input (closes the menu) |
| `Ctrl+M` | Open / close the menu drawer |
| `Ctrl+L` | Clear chat |
| `Ctrl+S` | Suggest mode (queue review-tier commands) |
| `Ctrl+E` | Execute mode (auto-run safe + review-tier) |
| `Ctrl+H` | Hide to tray |
| `Esc` | Close the menu — or hide to tray when it's already closed |
| `↑` | Previous command |

**Command safety:** `ls`, `pwd`, `cat`, `tree`, `git status`, etc. auto-run. `rm`, `sudo`, `mv` queue for approval.

**Queue buttons:** **Execute** (highlighted risky cmd once) · **Approve All** (safe queue) · **Modify** (edit in input) · **Dismiss**

**Slash:** `/forge task` · `/bg task` (background agent) · `/research <topic>` (multi-engine research + summarize + offer to save) · `/tasks` · `/task <id>` · `/continue [id]` (resume a stalled task) · `/cancel task <id>` · `/create project <name> using <stack>` · `/build app <spec>` · `/download pdf|web <url>` · `/ingest web <url>` · `/queue` · `/approve run` · `/approve all` · `/reject` · `/kb status` · `/memory` · `/recall` · `/learn` · `/learn now` · `/reflect` · `/self` · `/self plan` · `/self improve` · `/self patterns` · `/self lessons` · `/self log` · `/ask code <q>` · `/approve edit` · `/metrics` · `/good` · `/bad`

**Speed:** `/model fast` · `/model large` · `/model auto` (dynamic switching) · `/gpu status` (AMD GPU acceleration)

**Models in RAM:** `/model loaded` (resident now) · `/model sequential on|off` (one heavy model at a time — lower CPU/heat)

**Dual models:** Primus (fast) · Forge (coder) — router delegates on intent score; only one stays loaded at a time
"""


def _missing_deps_script() -> str:
    """Install commands for only missing required/optional components."""
    deps = _host.check_dependencies(refresh=True)
    lines = ["# Fix missing Primus components — paste into terminal", ""]
    for d in deps:
        if not d["ok"]:
            req = "required" if d.get("required") else "optional"
            lines.append(f"# {req}: {d['name']}")
            lines.append(d["install"])
            lines.append("")
    if len(lines) <= 2:
        lines.append("# All components OK — nothing to install.")
    return "\n".join(lines)


def _quick_install_script() -> str:
    """One-block install script for the Menu → Status copy box."""
    ollama = _host.get_ollama_details()
    lines = [
        "# Primus quick install — paste into terminal",
        "sudo apt update && sudo apt install -y " + " ".join(_host.SYSTEM_APT_PACKAGES[:8]),
        "curl -fsSL https://ollama.com/install.sh | sh",
        "ollama serve &",
        f"ollama pull {_host.CFG.get('model', _host.DEFAULT_MODEL)}",
        f"ollama pull {_host.CFG.get('forge_model', _host.FORGE_MODEL)}",
        "ollama pull nomic-embed-text",
        f"cd {_host.SCRIPT_PATH.parent}",
        "uv pip install " + " ".join(_host.PYTHON_PACKAGES + _host.OPTIONAL_PYTHON_PACKAGES),
        "uv run python admin_assistant.py --install-desktop",
        "uv run python admin_assistant.py --browser --tray",
    ]
    if not ollama["model_ready"] and ollama["reachable"]:
        lines.insert(6, f"ollama pull {ollama['configured']}")
    for d in _host.PrimusSession_deps_cache:
        if d["required"] and not d["ok"]:
            lines.append(f"# fix: {d['name']}")
            lines.append(d["install"])
    return "\n".join(lines)


def _allow_export_downloads() -> None:
    """Let the Export button actually hand the file over.

    Gradio 5 refuses to serve a returned file from outside the working dir / system temp dir,
    so Export raised InvalidPathError on ~/.primus/exports and the download never appeared.
    launch() isn't ours to change, but it falls back to GRADIO_ALLOWED_PATHS when no
    allowed_paths kwarg is given — so whitelist exactly that one directory here.

    Scope on purpose: EXPORT_DIR and nothing else. Whatever sits in it becomes fetchable by
    anyone who can reach the server, which is fine while Primus is bound to localhost — but
    `--share` or a 0.0.0.0 bind puts those exports on the network. If some other file ever
    fails to download, move the file into EXPORT_DIR (or hand back a temp copy); do not widen
    this list to ~/.primus, knowledge, config, or $HOME to compensate.
    """
    target = str(_host.EXPORT_DIR)
    current = [p.strip() for p in os.environ.get("GRADIO_ALLOWED_PATHS", "").split(",") if p.strip()]
    if target not in current:
        os.environ["GRADIO_ALLOWED_PATHS"] = ",".join([*current, target])


def export_chat(history: list) -> Optional[str]:
    if not history:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _host.EXPORT_DIR / f"primus_{stamp}.md"
    lines = [f"# Primus — {stamp}\n"]
    for item in history:
        if isinstance(item, dict) and not item.get("metadata"):
            lines.append(f"## {item.get('role','?').upper()}\n{item.get('content','')}\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def render_queue_md() -> str:
    q = _host.PrimusSession.pending_queue
    cmd = _host.PrimusSession.pending_shell_command or _host.PrimusSession.pending_suggestion
    if not q and not cmd:
        return ""
    lines = ["**Command queue**"]
    safe_n = sum(1 for i in q if not i.get("dangerous") and _host.assess_command_risk(i.get("command", ""))[0] == "safe")
    for item in q[-6:]:
        risk = item.get("risk") or ("dangerous" if item.get("dangerous") else "review")
        flag = "⚠" if risk == "dangerous" else ("○" if risk == "safe" else "◐")
        lines.append(f"- {flag} `{item.get('command', '')[:80]}` _({risk})_")
    if cmd and not any(x.get("command") == cmd for x in q):
        lines.append(f"- ○ `{cmd[:80]}`")
    lines.append("")
    if safe_n:
        lines.append(f"**Approve All** runs {safe_n} safe command(s). **Execute** runs the highlighted risky command.")
    else:
        lines.append("**Execute** runs the highlighted command (may require approval for risky ops).")
    return "\n".join(lines)


def render_queue_bar_md() -> str:
    """One-line summary for the slim approval bar above the composer.

    Deliberately terser than ``render_queue_md`` (which still backs ``/queue`` in chat): the
    bar has to stay a single quiet line, so it names the command that Execute would run plus
    how much else is waiting.
    """
    q = _host.PrimusSession.pending_queue
    cmd = _host.PrimusSession.pending_shell_command or _host.PrimusSession.pending_suggestion
    if not q and not cmd:
        return ""
    target = cmd or (q[0].get("command", "") if q else "")
    safe_n = sum(
        1 for i in q
        if not i.get("dangerous") and _host.assess_command_risk(i.get("command", ""))[0] == "safe"
    )
    risky = [i for i in q if i.get("dangerous") or i.get("risk") == "dangerous"]
    label = "Destructive command waiting" if (risky or _host.PrimusSession.pending_shell_command) else "Command waiting"
    bits = [f"**{label}** — `{target[:88]}`"]
    extra = max(0, len(q) - 1)
    if extra:
        bits.append(f"{extra} more queued")
    if safe_n:
        bits.append(f"Approve all runs {safe_n} safe")
    return "  ·  ".join(bits)


def _model_dropdown_choices() -> list[str]:
    ollama = _host.get_ollama_details()
    installed = list(ollama.get("models") or [])
    recommended = [m for m, _ in _host.RECOMMENDED_MODELS]
    seen: set[str] = set()
    out: list[str] = []
    for m in recommended + installed + [_host.CFG.get("model", _host.DEFAULT_MODEL), _host.CFG.get("forge_model", _host.FORGE_MODEL)]:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


# ===========================================================================
# Speech-to-Text (voice input) — local, offline-first, via faster-whisper
#
# Pipeline: mic (Gradio Audio → temp .wav filepath) → faster-whisper transcribe
# (with VAD to trim silence/noise) → text dropped into the chat input, then either
# auto-submitted or left for editing. The model is lazy-loaded and cached; device
# auto-detects CUDA/ROCm and falls back to CPU/int8 so it always works.
# ===========================================================================

_whisper_model: Any = None
_whisper_model_key: str = ""
_whisper_lock = threading.Lock()
WHISPER_MODEL_SIZES = ["tiny", "base", "small", "medium"]


def _detect_whisper_device() -> tuple[str, str]:
    """Resolve (device, compute_type) honoring config, auto-detecting a GPU when asked."""
    dev = str(_host.CFG.get("stt_device", "auto")).lower()
    comp = str(_host.CFG.get("stt_compute_type", "auto")).lower()
    if dev == "auto":
        dev = "cpu"
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():  # CUDA or ROCm-built torch both report here
                dev = "cuda"
        except Exception:  # noqa: BLE001
            dev = "cpu"
    if comp == "auto":
        comp = "float16" if dev == "cuda" else "int8"
    return dev, comp


def get_whisper_model() -> Any:
    """Lazy-load + cache the faster-whisper model for the configured size/device.

    Raises ImportError if faster-whisper isn't installed; otherwise always returns a model
    (falling back to CPU/int8 if GPU init fails). Re-creates the model only when size/device change.
    """
    global _whisper_model, _whisper_model_key
    size = str(_host.CFG.get("stt_model_size", "base"))
    dev, comp = _detect_whisper_device()
    key = f"{size}:{dev}:{comp}"
    with _whisper_lock:
        if _whisper_model is not None and _whisper_model_key == key:
            return _whisper_model
        from faster_whisper import WhisperModel  # type: ignore

        try:
            model = WhisperModel(size, device=dev, compute_type=comp)
            _whisper_model_key = key
        except Exception as exc:  # noqa: BLE001 — GPU/init failure → safe CPU fallback
            _host.log.warning("Whisper init on %s/%s failed (%s) — falling back to cpu/int8", dev, comp, exc)
            model = WhisperModel(size, device="cpu", compute_type="int8")
            _whisper_model_key = f"{size}:cpu:int8"
        _whisper_model = model
        _host.log.info("Loaded faster-whisper '%s' (%s)", size, _whisper_model_key)
        return model


def transcribe_audio(filepath: Optional[str]) -> tuple[str, str]:
    """Transcribe a recorded audio file → (text, status). Never raises.

    Returns ("", reason) when nothing usable was produced so the UI can show clean feedback.
    """
    if not _host.CFG.get("stt_enabled", True):
        return "", "Voice input is turned off in settings."
    if not filepath:
        return "", "No audio captured — check your microphone."
    try:
        model = get_whisper_model()
    except ImportError:
        return "", "Voice needs faster-whisper — run `uv pip install faster-whisper`."
    except Exception as exc:  # noqa: BLE001
        _host.log.warning("Whisper load failed: %s", exc)
        return "", f"Couldn't load the voice model ({str(exc)[:80]})."

    _host.PrimusSession.emit_think("Voice", "Transcribing…", "running")
    try:
        lang = str(_host.CFG.get("stt_language", "")).strip() or None
        segments, _info = model.transcribe(
            filepath,
            language=lang,
            vad_filter=bool(_host.CFG.get("stt_vad", True)),
            beam_size=int(_host.CFG.get("stt_beam_size", 5)),
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as exc:  # noqa: BLE001
        _host.log.warning("Transcription failed: %s", exc)
        _host.PrimusSession.emit_think("Voice", "Failed", "error")
        return "", f"Transcription failed ({str(exc)[:80]})."

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        _host.PrimusSession.emit_think("Voice", "No speech detected", "done")
        return "", "I didn't catch any speech — try again, a bit closer to the mic."
    _host.PrimusSession.emit_think("Voice", f"Heard: {text[:60]}", "done")
    return text, f"Heard: {text[:60]}{'…' if len(text) > 60 else ''}"


def prewarm_whisper() -> None:
    """Load the STT model in the background so the first voice input is snappy. Never raises."""
    if not _host.CFG.get("stt_enabled", True):
        return

    def _worker() -> None:
        try:
            get_whisper_model()
        except Exception as exc:  # noqa: BLE001
            _host.log.debug("Whisper prewarm skipped: %s", exc)

    threading.Thread(target=_worker, daemon=True, name="primus-whisper-prewarm").start()


# --- Project/chat UI helpers (presentation only) ------------------------------------
def _typing_slices(text: str, *, max_steps: int = 130) -> list[int]:
    """Cut points for a fast typewriter reveal of `text`.

    Short replies reveal ~1 char/tick (smooth, fast); long replies reveal in bigger chunks
    so the whole animation stays well under a second regardless of length. UI-only helper.
    """
    n = len(text)
    if n <= 1:
        return [n]
    chunk = max(1, -(-n // max_steps))  # ceil(n / max_steps)
    cuts = list(range(chunk, n, chunk))
    cuts.append(n)
    return cuts


def _chat_choices() -> list[tuple[str, str]]:
    """Radio choices for the chat list: (label, id). General Chat first, then projects."""
    out: list[tuple[str, str]] = []
    for p in _host.ChatProjects.list():
        out.append((p.get("name", "?"), p.get("id")))
    return out


def _active_project_desc() -> str:
    return _host.ChatProjects.get(_host.ChatProjects.active_id()).get("description", "")


def _sched_choices() -> list[tuple[str, str]]:
    """Dropdown choices for existing scheduled tasks: (label, id)."""
    return [
        (f"{t.get('name', '?')}  ·  {_host._schedule_label(t)}", t.get("id"))
        for t in _host.ScheduledTaskManager.list()
    ]


def _output_choices() -> list[str]:
    """Recent output document filenames for the viewer dropdown."""
    return [p.name for p in _host.list_agent_outputs()]


def _right_status_md() -> str:
    """Compact right-panel status: active chat scope, models, and acceleration."""
    active = _host.ChatProjects.get(_host.ChatProjects.active_id())
    lines = [f"**Chat:** {active.get('name', 'General Chat')}"]
    if active.get("id") != _host.GENERAL_CHAT_ID:
        lines.append("_Scope: project-focused recall · full memory still available._")
    else:
        lines.append("_Scope: full memory + knowledge._")
    lines += [
        "",
        f"**Primus:** `{_host.CFG.get('model', _host.DEFAULT_MODEL)}`",
        f"**Forge:** `{_host.CFG.get('forge_model', _host.FORGE_MODEL)}`",
    ]
    fast = _host.CFG.get("fast_fallback_model") or _host.CFG.get("primus_fast_model")
    if fast:
        lines.append(f"**Fast:** `{fast}`")
    try:
        mode, detail = _host.gpu_effective_mode()
        lines += ["", f"**Acceleration:** {mode}"]
        if detail:
            lines.append(f"_{detail}_")
    except Exception:  # noqa: BLE001
        pass
    try:
        accounts = _host.connections_health_line()
        if accounts:
            lines += ["", accounts]
    except Exception:  # noqa: BLE001
        pass
    # Background / scheduled task health at a glance.
    try:
        bg = _host.BackgroundTaskManager.list()
        running = sum(1 for t in bg if t.get("status") in ("queued", "running"))
        failed = sum(1 for t in bg if t.get("status") == "failed")
        sched = [t for t in _host.ScheduledTaskManager.list() if t.get("enabled", True)]
        bits = []
        if running:
            bits.append(f"▶ {running} running")
        if failed:
            bits.append(f"✗ {failed} failed")
        if sched:
            bits.append(f"🗓 {len(sched)} scheduled")
        if bits:
            lines += ["", "**Tasks:** " + " · ".join(bits)]
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


# Quiet empty state for the conversation — one line, no marketing hero.
EMPTY_STATE = "Ask anything. `/help` for commands, the menu for everything else."

# How long the top bar may reuse the last Ollama probe when it is only ticking, not reacting.
_TOPBAR_TTL = 20.0
_probe_cache: tuple[float, dict, bool] = (0.0, {}, False)


def _status_probe(max_age: float = 0.0) -> tuple[dict, bool]:
    """(ollama details, runtime_ready). Re-probes at most once per ``max_age`` seconds.

    get_ollama_details() is an HTTP call, so the top-bar timer reads through this instead of
    polling the daemon every tick; a stale answer just means the dot shows the last known state.
    """
    global _probe_cache
    now = time.monotonic()
    ts, details, ok = _probe_cache
    if max_age and details and (now - ts) < max_age:
        return details, ok
    try:
        details = _host.get_ollama_details()
    except Exception:  # noqa: BLE001
        details = {}
    try:
        ok = bool(_host.runtime_ready())
    except Exception:  # noqa: BLE001
        ok = False
    _probe_cache = (now, details, ok)
    return details, ok


def _topbar_html(max_age: float = 0.0) -> str:
    """Thin top bar: wordmark, one live status dot (online / model), active chat name."""
    ollama, ok = _status_probe(max_age)
    dot = "ok" if ok else ("warn" if ollama.get("reachable") else "bad")
    model = str(_host.CFG.get("model", _host.DEFAULT_MODEL)).split(":")[0]
    try:
        chat = _host.ChatProjects.get(_host.ChatProjects.active_id()).get("name", "General Chat")
    except Exception:  # noqa: BLE001
        chat = "General Chat"
    title = str(ollama.get("message") or ("online" if ok else "setup incomplete"))
    return (
        '<div class="pr-brand">'
        '<span class="pr-mark">PRIMUS</span>'
        f'<span class="pr-status" title="{_esc(title)}">'
        f'<span class="pr-dot {dot}"></span>{_esc(model)}</span>'
        f'<span class="pr-chat">{_esc(chat)}</span>'
        "</div>"
    )


def _topbar_tick() -> Any:
    """Slow timer target: keeps the status dot honest between explicit refreshes.

    Only the probe is cached — the chat name is always rendered live, so a tick can never
    revert the top bar to a stale project after a switch/create.
    """
    return gr.update(value=_topbar_html(max_age=_TOPBAR_TTL))


# ---- Chat attachments (📎 in the composer) ------------------------------------------------
# Caps for the text carried inline with the next message. Anything past the cap is still
# indexed into the knowledge base by the background ingest, so retrieval can reach it.
ATTACH_MAX_FILES = 3
ATTACH_PER_FILE_CAP = 8_000    # chars inline per file (small documents arrive whole)
ATTACH_TOTAL_CAP = 12_000      # chars inline across all files in one turn


def _attachment_enrichment(processed: str, attachments: list) -> tuple[str, str]:
    """Fold pending attachments into the agent's input.

    Returns (agent_message, display_suffix): the agent gets the inline text in a clearly
    delimited block after the operator's words; the chat bubble only gets a small attachment line,
    so transcripts stay clean. Module-level (not a build_ui closure) so it stays testable.
    """
    if not attachments:
        return processed, ""
    blocks: list[str] = []
    names: list[str] = []
    budget = ATTACH_TOTAL_CAP
    for att in attachments:
        name = att.get("name", "document")
        text = (att.get("text") or "")[: max(0, budget)]
        budget -= len(text)
        names.append(name)
        tail = ""
        if att.get("truncated") or len(text) < int(att.get("chars", 0)):
            tail = (" (excerpt — the full file was indexed into the knowledge base; "
                    "use search_knowledge for the rest)")
        blocks.append(
            f"[Attached document: {name} — {int(att.get('chars', 0)):,} chars{tail}]\n"
            f"{text}\n[End of {name}]"
        )
        if budget <= 0:
            break
    agent_message = (
        processed
        + _render_op(
            "\n\n{op_cap} attached document(s) to this message. Their text follows — read it "
            "and answer {op_pos} question about it directly; do not ask {op} to paste anything.\n\n"
        )
        + "\n\n".join(blocks)
    )
    return agent_message, "📎 *Attached: " + ", ".join(names) + "*"


# ---- Brain budget (drawer accordion; writes the same JSON the governor reads) --------------
_BRAIN_BLOCK_MARKERS = (
    "Brain weekly budget reached",
    "Brain per-turn call limit reached",
    "I kept circling on this",
)
_WEEKDAY_NAMES = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
)


def _brain_budget_data() -> dict[str, Any]:
    """Governor snapshot — source of truth at call time. Never invent a second file."""
    from primus.core.brain_governor import load_budget  # noqa: PLC0415

    return load_budget()


def _brain_loop_break_label(raw: Optional[dict[str, Any]] = None) -> str:
    """Read-only: show last loop-break only if the JSON already stored one."""
    blob = raw
    if blob is None:
        from primus.core.brain_governor import budget_path  # noqa: PLC0415

        try:
            blob = json.loads(budget_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            blob = {}
    if not isinstance(blob, dict):
        return "—"
    for key in ("last_loop_break", "last_loop_break_at", "last_loop"):
        val = blob.get(key)
        if val not in (None, ""):
            return str(val)
    return "—"


def _brain_pending(
    weekly_usd: Any,
    per_turn_calls: Any,
    override_grant_max: Any,
    usd_per_call_estimate: Any,
    base: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    data = dict(base or _brain_budget_data())
    if weekly_usd is not None:
        data["weekly_usd"] = float(weekly_usd)
    if per_turn_calls is not None:
        data["per_turn_calls"] = int(per_turn_calls)
    if override_grant_max is not None:
        data["override_grant_max"] = int(override_grant_max)
    if usd_per_call_estimate is not None:
        data["usd_per_call_estimate"] = float(usd_per_call_estimate)
    return data


def _brain_runout_weekday(
    weekly: float, spent: float, estimate: float, per_turn: int, *, now: Optional[datetime] = None
) -> str:
    """Projected weekday the cap is hit, from the live dials (before Save)."""
    now = now or datetime.now()
    remaining = max(0.0, weekly - spent)
    if remaining <= 1e-9:
        return _WEEKDAY_NAMES[now.weekday()]
    est = float(estimate)
    if est <= 1e-12:
        return "—"
    remain_calls = remaining / est
    pace = max(1, int(per_turn) if per_turn else 1)
    days = remain_calls / pace
    runout = now.date() + timedelta(days=days)
    name = _WEEKDAY_NAMES[runout.weekday()]
    if runout > now.date() + timedelta(days=6):
        return f"next {name}"
    return name


def _brain_calc_md(
    weekly_usd: Any,
    per_turn_calls: Any,
    override_grant_max: Any,  # noqa: ARG001 — kept so .change can pass all dials
    usd_per_call_estimate: Any,
    *,
    now: Optional[datetime] = None,
) -> str:
    """Live calculator: updates as the operator types/drags, before Save."""
    data = _brain_budget_data()
    try:
        weekly = float(weekly_usd if weekly_usd is not None else data.get("weekly_usd", 10.0))
    except (TypeError, ValueError):
        weekly = float(data.get("weekly_usd", 10.0))
    try:
        per_turn = int(per_turn_calls if per_turn_calls is not None else data.get("per_turn_calls", 3))
    except (TypeError, ValueError):
        per_turn = int(data.get("per_turn_calls", 3))
    try:
        estimate = float(
            usd_per_call_estimate
            if usd_per_call_estimate is not None
            else data.get("usd_per_call_estimate", 0.02)
        )
    except (TypeError, ValueError):
        estimate = float(data.get("usd_per_call_estimate", 0.02))
    spent = float(data.get("spent_usd", 0.0) or 0.0)
    remaining = max(0.0, weekly - spent)
    if estimate > 1e-12:
        max_calls = int(weekly / estimate)
        remain_calls = int(remaining / estimate)
    else:
        max_calls = 0
        remain_calls = 0
    runout = _brain_runout_weekday(weekly, spent, estimate, per_turn, now=now)
    return (
        f"**Max calls this week:** {max_calls}\n\n"
        f"**Estimated run-out:** {runout}\n\n"
        f"**Spent vs cap:** ${spent:.2f} / ${weekly:.2f}\n\n"
        f"At current rates, that's roughly {remain_calls} calls before you hit the cap."
    )


def _brain_activity_md(data: Optional[dict[str, Any]] = None) -> str:
    snap = data or _brain_budget_data()
    calls = int(snap.get("calls_this_week", 0) or 0)
    spent = float(snap.get("spent_usd", 0.0) or 0.0)
    return (
        f"**Calls this week:** {calls}\n\n"
        f"**Spent:** ${spent:.2f}\n\n"
        f"**Last loop-break:** {_brain_loop_break_label()}"
    )


def _brain_json_preview(
    weekly_usd: Any,
    per_turn_calls: Any,
    override_grant_max: Any,
    usd_per_call_estimate: Any,
) -> str:
    try:
        pending = _brain_pending(
            weekly_usd, per_turn_calls, override_grant_max, usd_per_call_estimate
        )
    except (TypeError, ValueError):
        pending = _brain_budget_data()
    return "```json\n" + json.dumps(pending, indent=2) + "\n```"


def _brain_live(
    weekly_usd: Any,
    per_turn_calls: Any,
    override_grant_max: Any,
    usd_per_call_estimate: Any,
) -> tuple[str, str, str]:
    return (
        _brain_calc_md(weekly_usd, per_turn_calls, override_grant_max, usd_per_call_estimate),
        _brain_activity_md(),
        _brain_json_preview(weekly_usd, per_turn_calls, override_grant_max, usd_per_call_estimate),
    )


def _brain_budget_save(
    weekly_usd: Any,
    per_turn_calls: Any,
    override_grant_max: Any,
    usd_per_call_estimate: Any,
) -> tuple[Any, ...]:
    """Validate dials and write the JSON the governor already reads."""
    from primus.core.brain_governor import load_budget, save_budget  # noqa: PLC0415

    errors: list[str] = []
    try:
        weekly = float(weekly_usd)
        if weekly < 0:
            errors.append("weekly_usd must be ≥ 0")
    except (TypeError, ValueError):
        errors.append("weekly_usd must be a number ≥ 0")
        weekly = None
    try:
        per_turn = int(per_turn_calls)
        if per_turn < 1:
            errors.append("per_turn_calls must be ≥ 1")
    except (TypeError, ValueError):
        errors.append("per_turn_calls must be an integer ≥ 1")
        per_turn = None
    try:
        grant = int(override_grant_max)
        if grant < 0:
            errors.append("override_grant_max must be ≥ 0")
    except (TypeError, ValueError):
        errors.append("override_grant_max must be an integer ≥ 0")
        grant = None
    try:
        estimate = float(usd_per_call_estimate) if usd_per_call_estimate is not None else None
        if estimate is not None and estimate < 0:
            errors.append("usd_per_call_estimate must be ≥ 0")
    except (TypeError, ValueError):
        errors.append("usd_per_call_estimate must be a number ≥ 0")
        estimate = None

    data = load_budget()
    if errors:
        calc, activity, preview = _brain_live(
            weekly_usd, per_turn_calls, override_grant_max, usd_per_call_estimate
        )
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            calc,
            activity,
            preview,
            "Not saved — " + "; ".join(errors),
        )

    data["weekly_usd"] = weekly
    data["per_turn_calls"] = per_turn
    data["override_grant_max"] = grant
    if estimate is not None:
        data["usd_per_call_estimate"] = estimate
    save_budget(data)
    saved = load_budget()
    calc, activity, preview = _brain_live(
        saved["weekly_usd"],
        saved["per_turn_calls"],
        saved["override_grant_max"],
        saved["usd_per_call_estimate"],
    )
    return (
        saved["weekly_usd"],
        saved["per_turn_calls"],
        saved["override_grant_max"],
        saved["usd_per_call_estimate"],
        calc,
        activity,
        preview,
        "Saved — governor will use these caps on the next brain call.",
    )


def _brain_budget_reload() -> tuple[Any, ...]:
    """Reload dials from disk (restart / CFG reload)."""
    data = _brain_budget_data()
    calc, activity, preview = _brain_live(
        data.get("weekly_usd"),
        data.get("per_turn_calls"),
        data.get("override_grant_max"),
        data.get("usd_per_call_estimate"),
    )
    return (
        data.get("weekly_usd", 10.0),
        data.get("per_turn_calls", 3),
        data.get("override_grant_max", 3),
        data.get("usd_per_call_estimate", 0.02),
        calc,
        activity,
        preview,
        "",
    )


def _append_brain_cap_hint(answer: str) -> str:
    """One line, only when the governor actually blocked this turn."""
    text = (answer or "").rstrip()
    if not text or not any(m in text for m in _BRAIN_BLOCK_MARKERS):
        return answer
    if "brain calls and is stuck" in text:
        return answer
    from primus.core.brain_governor import current_turn, load_budget  # noqa: PLC0415

    turn = current_turn()
    budget = load_budget()
    used = int(turn.calls) if turn.calls > 0 else int(turn.limit or budget.get("per_turn_calls") or 3)
    grant = max(0, int(budget.get("override_grant_max", 3) or 0))
    if grant < 1:
        return answer
    more = min(2, grant)
    hint = f"Primus used {used} brain calls and is stuck. Try {more} more?"
    return f"{text}\n\n{hint}"


def build_ui(graphs: dict[str, Any], model_name: str) -> tuple[Any, Any, str]:
    """Build Gradio UI; returns (demo, theme, css) for Gradio 6 launch()."""
    _allow_export_downloads()
    settings = _host.load_settings()
    default_mode = settings.get("execution_mode", "execute")
    compact = bool(_host.CFG.get("compact_mode", False))
    # The conversation fills the viewport between the top bar and the composer dock.
    chat_h = "calc(100vh - 168px)" if compact else "calc(100vh - 184px)"
    _host.PrimusSession.mode = _host.ExecutionMode(default_mode)

    _host.PrimusSession.last_health = _host.get_ollama_details().get("message", "")

    # Restrained metal theme: near-black surfaces, hairline borders, one platinum accent.
    # gr.themes.Base exists in every Gradio version; each factory is tried in turn so the
    # UI never fails to build (fonts/vars degrade to the stylesheet below).
    theme = None
    for theme_factory in (
        lambda: gr.themes.Base(
            primary_hue="slate",
            secondary_hue="slate",
            neutral_hue="slate",
            # Google fonts only: Gradio emits an @font-face pointing at static/fonts/<name>
            # for every *plain string* in these lists, and it ships no such file — a stock
            # Gradio app 404s on system-ui/ui-sans-serif for exactly this reason. The
            # fallback chain lives in --p-font/--p-mono in the stylesheet instead.
            font=[gr.themes.GoogleFont("Inter")],
            font_mono=[gr.themes.GoogleFont("IBM Plex Mono")],
        ).set(
            body_background_fill="#07080B",
            body_background_fill_dark="#07080B",
            background_fill_primary="#0C0E13",
            background_fill_primary_dark="#0C0E13",
            background_fill_secondary="#0C0E13",
            background_fill_secondary_dark="#0C0E13",
            block_background_fill="transparent",
            block_background_fill_dark="transparent",
            block_border_width="0px",
            block_shadow="none",
            body_text_color="#E8EAED",
            body_text_color_dark="#E8EAED",
            body_text_color_subdued="#8B909A",
            body_text_color_subdued_dark="#8B909A",
            border_color_primary="rgba(255,255,255,0.06)",
            border_color_primary_dark="rgba(255,255,255,0.06)",
            input_background_fill="#0C0E13",
            input_background_fill_dark="#0C0E13",
            button_primary_background_fill="#1C222C",
            button_primary_background_fill_dark="#1C222C",
            button_primary_text_color="#E8EAED",
            button_primary_text_color_dark="#E8EAED",
            button_secondary_background_fill="#0F1218",
            button_secondary_background_fill_dark="#0F1218",
            button_secondary_text_color="#E8EAED",
            button_secondary_text_color_dark="#E8EAED",
        ),
        lambda: gr.themes.Base(primary_hue="slate", secondary_hue="slate", neutral_hue="slate"),
        lambda: gr.themes.Default(),
    ):
        try:
            theme = theme_factory()
            break
        except Exception as exc:  # noqa: BLE001
            _host.log.warning("Theme init failed (%s) — trying next", exc)
            theme = None

    ready = _host.runtime_ready()
    banner_html = ""
    if not ready:
        banner_html = (
            '<div class="readiness-banner">'
            "Setup incomplete — open <b>Menu → Status</b> for install commands. "
            "Chat is limited until Ollama + models are ready."
            "</div>"
        )

    # theme and css belong on Blocks (NOT launch()); applying them here ensures the
    # dark/compact styling is always present even when launch kwargs are minimized.
    blocks_kwargs: dict[str, Any] = {
        "title": _host.APP_NAME,
        "elem_classes": ["compact-mode"] if compact else [],
        "css": PRIMUS_CSS,
    }
    if theme is not None:
        blocks_kwargs["theme"] = theme
    with gr.Blocks(**blocks_kwargs) as demo:
        # ================= top bar: wordmark · status dot · one Menu button =================
        with gr.Row(elem_id="primus-topbar"):
            # The Menu button and the drawer's close button are driven entirely client-side
            # (see _keyboard_js) so opening the drawer never round-trips or re-renders.
            gr.Button("≡", elem_id="menu-btn", variant="secondary", size="sm")
            topbar = gr.HTML(_topbar_html(), elem_id="primus-brand")

        # ================= drawer: everything that isn't the conversation =================
        # A CSS slide-over. Every component below stays mounted (and wired) whether the
        # drawer is open or closed — the JS in _keyboard_js only toggles a body class.
        with gr.Column(elem_id="primus-drawer"):
            with gr.Row(elem_id="drawer-head"):
                gr.HTML('<div class="drawer-title">Menu</div>')
                gr.Button("✕", elem_id="drawer-close", variant="secondary", size="sm")

            # ---- 1 · Chats ----
            with gr.Accordion("Chats", open=True, elem_classes=["drawer-sec"]):
                with gr.Column(elem_classes=["drawer-body"]):
                    chat_selector = gr.Radio(
                        choices=_chat_choices(),
                        value=_host.ChatProjects.active_id(),
                        label="",
                        elem_id="chat-list",
                    )
                    proj_status = gr.Markdown("", elem_id="proj-status")
                    with gr.Accordion("New project", open=False, elem_classes=["drawer-sub"]):
                        with gr.Column(elem_classes=["drawer-body"]):
                            new_proj_name = gr.Textbox(
                                label="Name", placeholder="e.g. Website rebuild", lines=1
                            )
                            new_proj_desc = gr.Textbox(
                                label="Description / scope (optional)", lines=2,
                                placeholder="What this chat focuses on…",
                            )
                            create_proj_btn = gr.Button("Create", variant="primary", size="sm")
                    with gr.Accordion("Edit project", open=False, elem_classes=["drawer-sub"]):
                        with gr.Column(elem_classes=["drawer-body"]):
                            proj_desc_box = gr.Textbox(
                                label="Description / scope", lines=3, value=_active_project_desc()
                            )
                            rename_box = gr.Textbox(label="Rename to", lines=1)
                            with gr.Row():
                                save_desc_btn = gr.Button("Save", size="sm")
                                rename_btn = gr.Button("Rename", size="sm")
                            del_proj_btn = gr.Button("Delete project", variant="stop", size="sm")

            # ---- 2 · Mode (+ session actions and workflow shortcuts) ----
            # Chats and Mode are the two sections worth scanning on open; the rest stay
            # collapsed so the 352px column never turns into a wall.
            with gr.Accordion("Mode", open=True, elem_classes=["drawer-sec"]):
                with gr.Column(elem_classes=["drawer-body"]):
                    mode_radio = gr.Radio(
                        [("Suggest", "suggest"), ("Execute", "execute")],
                        value=default_mode,
                        label="",
                        elem_id="mode-radio",
                    )
                    gr.Markdown(
                        "_Suggest queues review-tier commands. Execute runs safe + review-tier "
                        "and queues anything dangerous._",
                        elem_classes=["drawer-note"],
                    )
                    wf_dd = gr.Dropdown(
                        _host.WORKFLOW_SUGGESTIONS, label="Workflows", allow_custom_value=True
                    )
                    with gr.Row():
                        hist_btn = gr.Button("Last command", elem_id="hist-prev", size="sm")
                        clear_btn = gr.Button("Clear chat", elem_id="clear-btn", size="sm")
                        export_btn = gr.Button("Export", size="sm")

            # ---- Brain budget (drawer accordion, not a page) ----
            # Writes {APP_DIR}/brain_budget.json — the same file the governor reads on
            # the next ask_grok. Collapsed like Inbox; Mode stays the scan-on-open pair.
            try:
                _bb0 = _brain_budget_reload()
            except Exception:  # noqa: BLE001 — never let the budget accordion break page build
                _bb0 = (
                    10.0, 3, 3, 0.02,
                    "At current rates, that's roughly 500 calls before you hit the cap.",
                    "**Calls this week:** 0\n\n**Spent:** $0.00\n\n**Last loop-break:** —",
                    "```json\n{}\n```",
                    "",
                )
            with gr.Accordion("Brain budget", open=False, elem_classes=["drawer-sec"]):
                with gr.Column(elem_classes=["drawer-body"]):
                    bb_weekly = gr.Slider(
                        minimum=0,
                        maximum=250,
                        step=0.25,
                        value=float(_bb0[0]),
                        label="Weekly budget (USD)",
                    )
                    bb_per_turn = gr.Slider(
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=int(_bb0[1]),
                        label="Per-turn calls",
                    )
                    bb_override = gr.Slider(
                        minimum=0,
                        maximum=20,
                        step=1,
                        value=int(_bb0[2]),
                        label="Override grant max",
                    )
                    bb_calc_md = gr.Markdown(_bb0[4], elem_id="brain-budget-calc")
                    bb_activity_md = gr.Markdown(_bb0[5], elem_id="brain-budget-activity")
                    with gr.Accordion("Advanced", open=False, elem_classes=["drawer-sub"]):
                        with gr.Column(elem_classes=["drawer-body"]):
                            try:
                                bb_estimate = gr.Number(
                                    value=float(_bb0[3]),
                                    label="USD per call (estimate)",
                                    minimum=0,
                                    precision=4,
                                )
                            except TypeError:
                                bb_estimate = gr.Number(
                                    value=float(_bb0[3]),
                                    label="USD per call (estimate)",
                                )
                            bb_json_md = gr.Markdown(_bb0[6], elem_classes=["drawer-note"])
                    bb_save_btn = gr.Button("Save", variant="primary", size="sm")
                    bb_status_md = gr.Markdown(_bb0[7] or "", elem_classes=["drawer-note"])
                    gr.Markdown(
                        "_Save writes the JSON the governor already reads. Caps apply on the next "
                        "brain call._",
                        elem_classes=["drawer-note"],
                    )

            # ---- 3 · Status ----
            with gr.Accordion("Status", open=False, elem_classes=["drawer-sec"]):
                with gr.Column(elem_classes=["drawer-body"]):
                    status_bar = gr.HTML(f'<div class="status-bar">{_host.build_status_bar()}</div>')
                    right_status_md = gr.Markdown(_right_status_md())
                    with gr.Row():
                        right_refresh_btn = gr.Button("Refresh status", size="sm")
                        refresh_btn = gr.Button("Refresh ticker", size="sm")
                    setup_dashboard = gr.HTML(_host.render_setup_dashboard_html())
                    model_choices = _model_dropdown_choices()
                    primus_model_dd = gr.Dropdown(
                        model_choices,
                        value=_host.CFG.get("model", _host.DEFAULT_MODEL),
                        label="Primus model",
                        allow_custom_value=True,
                    )
                    forge_model_dd = gr.Dropdown(
                        model_choices,
                        value=_host.CFG.get("forge_model", _host.FORGE_MODEL),
                        label="Forge model",
                        allow_custom_value=True,
                    )
                    fast_model_dd = gr.Dropdown(
                        model_choices,
                        value=_host.CFG.get("fast_fallback_model", "llama3.2:3b"),
                        label="Fast model (small/quantized)",
                        allow_custom_value=True,
                    )
                    apply_models_btn = gr.Button("Apply models", variant="primary", size="sm")
                    with gr.Accordion("GPU / acceleration", open=False, elem_classes=["drawer-sub"]):
                        with gr.Column(elem_classes=["drawer-body"]):
                            gpu_md = gr.Markdown(_host.gpu_status_text())
                            prefer_gpu_cb = gr.Checkbox(
                                label="Prefer GPU (auto CPU fallback)",
                                value=bool(_host.CFG.get("prefer_gpu", True)),
                            )
                            gpu_backend_dd = gr.Dropdown(
                                ["auto", "rocm", "vulkan", "cpu"],
                                value=_host.gpu_backend_value(),
                                label="GPU backend",
                            )
                            gpu_layers_tb = gr.Textbox(
                                value=str(_host.CFG.get("gpu_layers", -1)),
                                label="GPU layers (-1 all · 0 CPU · N)",
                            )
                            hsa_tb = gr.Textbox(
                                value=str(_host.CFG.get("hsa_override_gfx_version", "auto")),
                                label="HSA gfx override (auto / blank / 11.0.2)",
                            )
                            with gr.Row():
                                apply_gpu_btn = gr.Button("Apply GPU", variant="primary", size="sm")
                                refresh_gpu_btn = gr.Button("Refresh", size="sm")
                            rocm_help_btn = gr.Button("Install ROCm for AMD GPU", size="sm")
                            recover_gpu_btn = gr.Button(
                                "Recover Ollama GPU (auto-retry)",
                                variant="primary", size="sm", elem_id="recover-gpu-btn",
                            )
                            recover_gpu_log = gr.Textbox(
                                label="GPU recovery log (live)",
                                value="",
                                lines=10,
                                max_lines=18,
                                interactive=False,
                                visible=False,
                                elem_id="recover-gpu-log",
                            )
                            gr.Markdown(
                                "_GPU is used by default when available; Primus falls back to CPU "
                                "automatically. **Recover Ollama GPU** force-restarts `ollama serve` "
                                "with the reliable Vulkan settings and retries until `ollama ps` "
                                "confirms GPU — an in-flight reply may be interrupted._",
                                elem_classes=["drawer-note"],
                            )
                    with gr.Accordion("Install & system", open=False, elem_classes=["drawer-sub"]):
                        with gr.Column(elem_classes=["drawer-body"]):
                            setup_md = gr.Markdown(_host.render_setup_status_markdown())
                            setup_cmds = gr.Textbox(
                                label="Copy-paste install commands",
                                value=_quick_install_script(),
                                lines=6,
                                max_lines=12,
                                interactive=True,
                            )
                            fix_cmds = gr.Textbox(
                                label="Missing components only",
                                value=_missing_deps_script(),
                                lines=5,
                                interactive=True,
                            )
                            refresh_setup_btn = gr.Button("Refresh status", variant="primary", size="sm")
                            install_desktop_btn = gr.Button("Install menu launcher", size="sm")
                            install_autostart_btn = gr.Button("Enable login autostart", size="sm")
                            remove_autostart_btn = gr.Button("Disable autostart", size="sm")

            # ---- 4 · Knowledge ----
            with gr.Accordion("Knowledge", open=False, elem_classes=["drawer-sec"]):
                with gr.Column(elem_classes=["drawer-body"]):
                    # No file_types filter: the browser picker can silently reject valid files
                    # (and chokes on multi-dot names). We accept everything and validate in code.
                    try:
                        kb_drop = gr.File(
                            label="Drop files to teach Primus · PDF, TXT, MD, DOCX, ODT, PY, JSON, CSV…",
                            file_count="multiple",
                            height=140,
                            elem_id="kb-dropzone",
                        )
                    except TypeError:
                        kb_drop = gr.File(
                            label="Drop files to teach Primus · PDF, TXT, MD, DOCX, ODT, PY, JSON, CSV…",
                            file_count="multiple",
                            elem_id="kb-dropzone",
                        )
                    kb_category = gr.Radio(
                        choices=list(_host.KB_CATEGORY_LABELS),
                        value="Learned / New Information",
                        label="Category / collection",
                    )
                    kb_project = gr.Dropdown(
                        choices=list(_host.KB_PROJECT_CHOICES),
                        value=list(_host.KB_PROJECT_CHOICES)[0],
                        label="Project",
                        visible=False,
                    )
                    # Tag uploaded docs to a project chat (metadata-level → biases that chat's recall).
                    kb_chat_dd = gr.Dropdown(
                        choices=_chat_choices(),
                        value=_host.GENERAL_CHAT_ID,
                        label="Assign to chat (tags docs for that project's recall)",
                    )
                    with gr.Row():
                        process_kb_btn = gr.Button("Process all", variant="primary", size="sm")
                        clear_drop_btn = gr.Button("Clear", size="sm")
                    kb_status_md = gr.Markdown("", elem_id="kb-status")
                    with gr.Accordion("Fetch from the web", open=False, elem_classes=["drawer-sub"]):
                        with gr.Column(elem_classes=["drawer-body"]):
                            web_url_tb = gr.Textbox(
                                label="URL", placeholder="https://… (webpage or hosted PDF)",
                            )
                            web_project_dd = gr.Dropdown(
                                choices=_chat_choices(), value=_host.GENERAL_CHAT_ID, label="Tag to chat",
                            )
                            web_ingest_btn = gr.Button("Download & ingest", variant="primary", size="sm")
                            with gr.Row():
                                web_pdf_btn = gr.Button("Save PDF only", size="sm")
                                web_page_btn = gr.Button("Page → PDF", size="sm")
                            web_status_md = gr.Markdown("", elem_id="web-status")
                    with gr.Accordion("Knowledge base", open=False, elem_classes=["drawer-sub"]):
                        with gr.Column(elem_classes=["drawer-body"]):
                            kb_md = gr.Markdown(_host.render_kb_dashboard())
                            with gr.Row():
                                refresh_kb_btn = gr.Button("Refresh", size="sm")
                                index_projects_btn = gr.Button("Index projects", size="sm")
                            approve_learn_btn = gr.Button("Save queued web → KB", size="sm")
                            kb_log = gr.Textbox(label="Knowledge log", interactive=False, max_lines=2)

            # ---- 5 · Scheduled ----
            with gr.Accordion("Scheduled", open=False, elem_classes=["drawer-sec"]):
                with gr.Column(elem_classes=["drawer-body"]):
                    gr.Markdown(
                        "_Autonomous agents run on the heavy model with all tools; each run saves "
                        "a report to `~/.primus/agent_outputs/`._",
                        elem_classes=["drawer-note"],
                    )
                    sched_existing_dd = gr.Dropdown(
                        choices=_sched_choices(), label="Edit existing (blank = create new)",
                        value=None, allow_custom_value=False,
                    )
                    sched_name = gr.Textbox(label="Task name", placeholder="e.g. Daily AI news brief")
                    sched_instructions = gr.Textbox(
                        label="Instructions / prompt", lines=4,
                        placeholder="Exactly what the agent should do…",
                    )
                    sched_project = gr.Dropdown(
                        choices=_chat_choices(), value=_host.GENERAL_CHAT_ID,
                        label="Project scope (memory focus)",
                    )
                    sched_type = gr.Radio(
                        [("Once", "once"), ("Daily", "daily"), ("Weekly", "weekly"), ("Interval", "interval")],
                        value="once", label="Schedule",
                    )
                    sched_runat = gr.Textbox(
                        label="Once — date & time", placeholder="YYYY-MM-DD HH:MM",
                    )
                    sched_time = gr.Textbox(
                        label="Daily/Weekly — time", value="09:00", placeholder="HH:MM",
                    )
                    sched_weekday = gr.Dropdown(
                        choices=[(d, i) for i, d in enumerate(_host._WEEKDAYS)],
                        value=0, label="Weekly — day",
                    )
                    sched_interval = gr.Number(
                        value=60, label="Interval — minutes", precision=0,
                    )
                    sched_enabled = gr.Checkbox(label="Enabled", value=True)
                    with gr.Row():
                        sched_save_btn = gr.Button("Create / update", variant="primary", size="sm")
                        sched_runnow_btn = gr.Button("Run now", size="sm")
                    with gr.Row():
                        sched_toggle_btn = gr.Button("Enable/disable", size="sm")
                        sched_new_btn = gr.Button("New / clear", size="sm")
                        sched_delete_btn = gr.Button("Delete", variant="stop", size="sm")
                    sched_status = gr.Markdown("")
                    with gr.Accordion("Schedule & outputs", open=False, elem_classes=["drawer-sub"]):
                        with gr.Column(elem_classes=["drawer-body"]):
                            sched_list_md = gr.Markdown(_host.render_scheduled_tasks_md())
                            sched_refresh_btn = gr.Button("Refresh schedule", size="sm")
                            sched_outputs_md = gr.Markdown(_host.render_recent_outputs_md())
                            with gr.Row():
                                sched_output_pick = gr.Dropdown(
                                    choices=_output_choices(), label="View an output document",
                                )
                                sched_output_refresh = gr.Button("↻", size="sm", scale=0)
                            sched_output_view = gr.Markdown("")

            # ---- 6 · Voice & window ----
            with gr.Accordion("Voice & window", open=False, elem_classes=["drawer-sec"]):
                with gr.Column(elem_classes=["drawer-body"]):
                    # Built defensively so an unusual Gradio Audio API can never break UI
                    # construction/launch. mic stays None if unavailable.
                    mic = None
                    voice_cb = None
                    voice_status = None
                    if bool(_host.CFG.get("stt_enabled", True)):
                        try:
                            with gr.Column(elem_id="voice-row"):
                                try:
                                    mic = gr.Audio(
                                        sources=["microphone"], type="filepath",
                                        label="Voice — tap to record, tap again to send",
                                        show_label=True, elem_id="primus-mic",
                                    )
                                except TypeError:
                                    # Older Gradio used singular `source=`.
                                    mic = gr.Audio(
                                        source="microphone", type="filepath",
                                        label="Voice — tap to record, tap again to send",
                                        elem_id="primus-mic",
                                    )
                                voice_cb = gr.Checkbox(
                                    label="Auto-send", value=bool(_host.CFG.get("stt_autosubmit", True)),
                                    elem_id="voice-autosend",
                                )
                                voice_status = gr.Markdown("", elem_id="voice-status")
                        except Exception as exc:  # noqa: BLE001
                            _host.log.warning("Voice input UI unavailable (%s) — continuing without mic", exc)
                            mic = voice_cb = voice_status = None
                    stt_enable_cb = gr.Checkbox(
                        label="Voice input (Speech-to-Text)",
                        value=bool(_host.CFG.get("stt_enabled", True)),
                    )
                    stt_model_dd = gr.Dropdown(
                        WHISPER_MODEL_SIZES,
                        value=_host.CFG.get("stt_model_size", "base"),
                        label="Voice model (accuracy ↔ speed)",
                    )
                    stt_lang_tb = gr.Textbox(
                        value=_host.CFG.get("stt_language", ""),
                        label="Voice language (blank = auto)",
                        placeholder="en",
                    )
                    apply_stt_btn = gr.Button("Apply voice", size="sm")
                    stt_status = gr.Markdown("")
                    with gr.Accordion("Window", open=False, elem_classes=["drawer-sub"]):
                        with gr.Column(elem_classes=["drawer-body"]):
                            # elem_id lets the load-JS apply the density change immediately
                            # (the handler below only persists it).
                            compact_cb = gr.Checkbox(
                                label="Compact", value=compact, elem_id="compact-cb"
                            )
                            pin_cb = gr.Checkbox(
                                label="Pin on top", value=bool(settings.get("always_on_top"))
                            )
                            with gr.Row():
                                hide_btn = gr.Button("Tray", size="sm", elem_id="hide-tray-btn")
                                win_btn = gr.Button("Win", size="sm")
                    with gr.Accordion("Response display", open=False, elem_classes=["drawer-sub"]):
                        with gr.Column(elem_classes=["drawer-body"]):
                            typing_effect_cb = gr.Checkbox(
                                label="Typewriter effect (animate responses)",
                                value=bool(_host.CFG.get("typing_effect", True)),
                            )
                            typing_speed_slider = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=int(_host.CFG.get("typing_speed", 18)),
                                label="Typing speed (1 fast · 100 slow)",
                            )
                            apply_display_btn = gr.Button("Apply display", size="sm")
                            typing_status = gr.Markdown("", elem_id="typing-status")

            # ---- 7 · Connections ----
            with gr.Accordion("Connections", open=False, elem_classes=["drawer-sec"]):
                with gr.Column(elem_classes=["drawer-body"]):
                    connections_md = gr.Markdown(_host.connections_status_markdown())
                    conn_refresh_btn = gr.Button("Refresh accounts", variant="primary", size="sm")
                    conn_add_name = gr.Textbox(label="Name", placeholder="e.g. work-gmail")
                    conn_add_type = gr.Dropdown(
                        ["gmail", "calendar", "slack", "discord"], value="gmail", label="Type",
                    )
                    conn_add_secret = gr.Textbox(
                        label="Token / creds path",
                        placeholder="xoxb-… (Slack) or /path/to/client_secret.json (Google)",
                        type="password",
                    )
                    conn_add_btn = gr.Button("Add / update", size="sm")
                    conn_remove_name = gr.Textbox(
                        label="Remove account (name)", placeholder="name from the table above",
                    )
                    conn_remove_btn = gr.Button("Remove", size="sm", variant="stop")
                    conn_action_md = gr.Markdown("")
                    gr.Markdown(
                        "_Secrets are stored owner-only (chmod 600) under `~/.primus/`. For "
                        "Gmail/Calendar, give the OAuth client-secret JSON path, then run "
                        "`gmail_auth` / `calendar_auth` in chat. For Slack, paste the bot token._",
                        elem_classes=["drawer-note"],
                    )

            # ---- 8 · Inbox (Gmail) ----
            # First-class inbox in the drawer. Works in two modes: live (OAuth token present)
            # and demo fixture (no token → examples/gmail/fixture_inbox.json). All actions go
            # through the existing gmail_* tools via thin _host.inbox_* bridges — no duplicated
            # API logic. Drafts respect Suggest/Execute; fixture drafts are preview-only.
            with gr.Accordion("Inbox", open=False, elem_classes=["drawer-sec"]):
                with gr.Column(elem_classes=["drawer-body"]):
                    inbox_status_md = gr.Markdown(_host.inbox_status_markdown())
                    with gr.Accordion("Setup help", open=False, elem_classes=["drawer-sub"]):
                        gr.Markdown(_host.inbox_setup_markdown(), elem_classes=["drawer-note"])
                    with gr.Row():
                        inbox_connect_btn = gr.Button("Connect", variant="primary", size="sm")
                        inbox_refresh_btn = gr.Button("Refresh", size="sm")
                    inbox_query = gr.Textbox(
                        label="Query (Gmail syntax)",
                        value="is:unread newer_than:7d",
                        placeholder="is:unread newer_than:7d",
                    )
                    # Pre-populate only in fixture mode (local JSON, instant). Live mode does
                    # a network round-trip, so it waits for an explicit Refresh click instead
                    # of slowing down page build.
                    _inbox_init_table, _inbox_init_rows, _inbox_init_note = ([], [], "")
                    try:
                        if "fixture" in _host.inbox_status_markdown().lower():
                            _inbox_init_table, _inbox_init_rows, _inbox_init_note = (
                                _host.inbox_list_ui("is:unread newer_than:7d")
                            )
                    except Exception:  # noqa: BLE001 — never let the inbox break page build
                        pass
                    inbox_rows_state = gr.State(_inbox_init_rows)
                    inbox_msg_id = gr.State("")
                    inbox_list = gr.Dataframe(
                        headers=["", "Date", "From", "Subject"],
                        value=_inbox_init_table,
                        col_count=(4, "fixed"),
                        interactive=False,
                        wrap=True,
                        elem_id="inbox-list",
                    )
                    inbox_read = gr.Markdown(
                        "_Select a message above._", elem_id="inbox-read"
                    )
                    inbox_reply = gr.Textbox(
                        label="Draft reply", lines=3,
                        placeholder="Write a reply — saved as a Gmail draft, never auto-sent.",
                    )
                    with gr.Row():
                        inbox_prefill_btn = gr.Button("Suggest reply", size="sm")
                        inbox_draft_btn = gr.Button("Create draft", size="sm")
                        inbox_chat_btn = gr.Button("Open in chat", size="sm")
                    inbox_action_md = gr.Markdown("", elem_classes=["drawer-note"])

            # ---- 9 · Activity / agents ----
            with gr.Accordion("Activity", open=False, elem_classes=["drawer-sec"]):
                with gr.Column(elem_classes=["drawer-body"]):
                    think_panel = gr.HTML(_host.render_thinking_html())
                    with gr.Accordion("Background agents", open=False, elem_classes=["drawer-sub"]):
                        with gr.Column(elem_classes=["drawer-body"]):
                            bg_panel = gr.HTML(
                                _host.render_background_agents_html(), elem_id="bg-agents-panel"
                            )
                            bg_task_tb = gr.Textbox(
                                label="", placeholder="Spin off a long task… (runs on Forge)",
                                lines=2, elem_id="bg-task-input",
                            )
                            with gr.Row():
                                bg_run_btn = gr.Button("Run in background", variant="primary", size="sm")
                                bg_refresh_btn = gr.Button("↻", size="sm", scale=0)
                                bg_clear_btn = gr.Button("Clear done", size="sm", scale=0)
                            bg_status = gr.Markdown("", elem_id="bg-submit-status")
                    with gr.Accordion("Background tasks", open=False, elem_classes=["drawer-sub"]):
                        with gr.Column(elem_classes=["drawer-body"]):
                            gr.Markdown(
                                "_Heavy/multi-step requests auto-run here so the chat never blocks. "
                                "Results post back into the conversation when ready._",
                                elem_classes=["drawer-note"],
                            )
                            tasks_panel = gr.Markdown(_host.render_background_tasks_md())
                            tasks_refresh_btn = gr.Button("Refresh", size="sm")
                            with gr.Row():
                                tasks_cancel_tb = gr.Textbox(label="", placeholder="task id to cancel")
                                tasks_cancel_btn = gr.Button("Cancel", size="sm", scale=0)
                            tasks_status = gr.Markdown("", elem_id="tasks-status")
                    with gr.Accordion("Coding", open=False, elem_classes=["drawer-sub"]):
                        with gr.Column(elem_classes=["drawer-body"]):
                            gr.Markdown(
                                "_Builds run on Forge in the background → `~/Projects/`._",
                                elem_classes=["drawer-note"],
                            )
                            code_name_tb = gr.Textbox(
                                label="Project name", placeholder="e.g. invoice-api",
                            )
                            code_stack_tb = gr.Textbox(
                                label="Tech stack", value="Python + FastAPI",
                                placeholder="Python + FastAPI / React + TypeScript / Python CLI…",
                            )
                            code_desc_tb = gr.Textbox(
                                label="Description / features", lines=2,
                                placeholder="What it does + key features…",
                            )
                            code_create_btn = gr.Button("Create project", variant="primary", size="sm")
                            code_spec_tb = gr.Textbox(
                                label="Build app from spec", lines=3,
                                placeholder="Describe the full app to build…",
                            )
                            code_build_btn = gr.Button("Build app", size="sm")
                            code_status = gr.Markdown("", elem_id="code-submit-status")

            # ---- reference ----
            with gr.Accordion("Shortcuts & commands", open=False, elem_classes=["drawer-sec"]):
                with gr.Column(elem_classes=["drawer-body"]):
                    gr.Markdown(HELP_TEXT, elem_classes=["drawer-note"])

        # ================= centre: the conversation, nothing else =================
        with gr.Column(elem_id="primus-main"):
            with gr.Column(elem_id="primus-stage"):
                if banner_html:
                    gr.HTML(banner_html)
                try:
                    initial_history = _host.initial_chat_history()
                except Exception as exc:  # noqa: BLE001
                    _host.log.warning("Chat history load failed (%s) — starting empty", exc)
                    initial_history = []
                chat_kwargs: dict[str, Any] = dict(
                    value=initial_history,
                    type="messages",
                    height=chat_h,
                    min_height=220,
                    show_label=False,
                    elem_id="primus-chat",
                )
                try:
                    chatbot = gr.Chatbot(**chat_kwargs, placeholder=EMPTY_STATE)
                except TypeError:
                    # Older Gradio: no placeholder / min_height — height stays a plain CSS string.
                    chat_kwargs.pop("min_height", None)
                    try:
                        chatbot = gr.Chatbot(**chat_kwargs)
                    except TypeError:
                        chat_kwargs["height"] = 580
                        chatbot = gr.Chatbot(**chat_kwargs)

        # ================= dock: approval bar (when pending) + composer =================
        with gr.Column(elem_id="primus-dock"):
            # Never bury a pending destructive action: this bar shows above the composer
            # even with the menu closed.
            with gr.Row(visible=False, elem_id="approval-bar") as cmd_row:
                cmd_preview = gr.Markdown("", elem_id="cmd-preview")
                run_btn = gr.Button("Execute", variant="primary", size="sm")
                approve_all_btn = gr.Button("Approve all", size="sm")
                modify_btn = gr.Button("Modify", size="sm")
                dismiss_btn = gr.Button("Dismiss", size="sm")
            with gr.Row(elem_id="primus-composer"):
                # Attach documents straight into the chat: the extracted text rides along with
                # the next message (and a copy is indexed into the knowledge base).
                attach_btn = gr.UploadButton(
                    "📎",
                    file_count="multiple",
                    file_types=sorted(_host.TEXT_INGEST_EXTENSIONS),
                    elem_id="attach-btn",
                    scale=0,
                    variant="secondary",
                    size="sm",
                )
                msg = gr.Textbox(
                    placeholder="Ask Primus…  (/help)",
                    show_label=False,
                    elem_id="primus-input",
                    container=False,
                    lines=1,
                    max_lines=4,
                )
                send_btn = gr.Button("Send", variant="primary", elem_id="send-btn", scale=0)
                halt_btn = gr.Button("Halt", variant="stop", elem_id="halt-btn", scale=0)
            # Quiet activity line under the composer — the target of every handler's note
            # ("Working…", "Typing…", "Ready", errors) that used to be a labelled Log box.
            log_tb = gr.Textbox(
                value="", show_label=False, interactive=False, max_lines=1,
                container=False, elem_id="log-line",
            )

        export_file = gr.File(visible=False)
        # Documents attached via the composer's 📎 button, waiting for the next send. Each entry:
        # {name, text, chars, truncated, path}. Consumed (cleared) by the next submitted turn.
        pending_attachments = gr.State([])
        # Id of the chat whose transcript the chatbot is currently holding. It rides in the
        # same payload as the messages, so every save can be addressed to the chat the
        # messages actually came from instead of to whatever chat is active when the save
        # runs — that mismatch is what wrote one chat's transcript into another's file.
        chat_owner = gr.State(_host.ChatProjects.active_id())
        mode_suggest_btn = gr.Button("Suggest", elem_id="mode-suggest", visible=False)
        mode_exec_btn = gr.Button("Execute", elem_id="mode-exec", visible=False)

        def _status():
            return gr.update(value=f'<div class="status-bar">{_host.build_status_bar()}</div>')

        def _think():
            return gr.update(value=_host.render_thinking_html())

        def _cmd_ui():
            md = render_queue_bar_md()
            if md:
                return gr.update(visible=True), gr.update(value=md)
            return gr.update(visible=False), gr.update(value="")

        def set_mode(mode):
            _host.PrimusSession.mode = _host.ExecutionMode(mode)
            s = _host.load_settings()
            s["execution_mode"] = mode
            _host.save_settings(s)
            if mode == "suggest":
                label = "Suggest — safe auto-runs; review-tier queued"
            else:
                label = "Execute — safe auto-runs; review-tier runs; dangerous queued"
            return label

        def handle_slash(message, history, chat_id=None):
            raw = message.strip()
            low = raw.lower()

            def reply(text, note=""):
                history.extend([{"role": "user", "content": raw}, {"role": "assistant", "content": text}])
                return None, history, note

            if low == "/help":
                return reply(HELP_TEXT, "Help")
            kb_out = _host.handle_kb_command(raw)
            if kb_out is not None:
                return reply(kb_out, "KB")
            if low.startswith("/forge"):
                task = raw.split(None, 1)[1].strip() if len(raw.split(None, 1)) > 1 else ""
                if not task:
                    forge = _host.CFG.get("forge_model", _host.FORGE_MODEL)
                    primus = _host.CFG.get("model", _host.DEFAULT_MODEL)
                    return reply(
                        f"Usage: `/forge your coding task`\n\n"
                        f"**Primus** `{primus}` — fast chat/admin\n"
                        f"**Forge** `{forge}` — coding/debug (auto-delegated when beneficial)",
                        "Forge",
                    )
                _host.PrimusSession.force_forge_next = True
                return task, history, "Forge → forced"
            if low.startswith("/bg") or low.startswith("/background"):
                task = raw.split(None, 1)[1].strip() if len(raw.split(None, 1)) > 1 else ""
                if not task:
                    return reply(
                        "Usage: `/bg <task>` — spin off a long-running background agent "
                        f"(runs on Forge `{_host.CFG.get('forge_model', _host.FORGE_MODEL)}` with all tools, "
                        "independent of chat timeouts). Watch progress in **Menu → Activity → "
                        "Background agents**.",
                        "Background",
                    )
                entry = _host.BackgroundAgentManager.submit(task)
                return reply(
                    f"🛰 Background agent **{entry['id']}** started:\n\n> {task}\n\n"
                    "It runs independently — keep chatting. Track it in **Menu → Activity**.",
                    "Background",
                )
            if low == "/tasks" or low.startswith("/tasks "):
                return reply(_host.render_background_tasks_md(), "Tasks")
            if low == "/continue" or low.startswith("/continue "):
                # Resume a stalled/interrupted background task from its last step (reloads its goal
                # + task memory so it doesn't restart from scratch). `/continue` alone picks the most
                # recent resumable task; `/continue <id>` targets a specific one.
                tid = raw.split(None, 1)[1].strip() if len(raw.split(None, 1)) > 1 else ""
                entry = _host.BackgroundTaskManager.resume(tid)
                if not entry:
                    return reply(
                        "Nothing to continue — no matching background task found. Use `/tasks` to list them.",
                        "Tasks",
                    )
                return reply(
                    f"↻ Resuming as **{entry['id']}** from the last completed step — it won't restart "
                    "the analysis. Track it in the **Background agents** panel.",
                    "Tasks",
                )
            if low.startswith("/cancel task"):
                tid = raw.split("task", 1)[1].strip()
                if not tid:
                    return reply("Usage: `/cancel task <id>` (e.g. `/cancel task task-ab12cd`).", "Tasks")
                ok = _host.BackgroundTaskManager.cancel(tid)
                return reply(
                    f"Cancelled background task **{tid}**." if ok
                    else f"Couldn't cancel **{tid}** (unknown or already finished).",
                    "Tasks",
                )
            if low.startswith("/task"):
                tid = raw.split(None, 1)[1].strip() if len(raw.split(None, 1)) > 1 else ""
                if not tid:
                    return reply("Usage: `/task <id>` — details for a background task. See `/tasks`.", "Tasks")
                t = _host.BackgroundTaskManager.get(tid)
                if not t:
                    return reply(f"No background task with id **{tid}**. Use `/tasks` to list them.", "Tasks")
                icon = _host._BG_TASK_ICONS.get(t.get("status", ""), "•")
                details = [
                    f"### Task `{t.get('id')}` — {icon} {t.get('status', '')}",
                    f"- **Type:** {t.get('task_type', 'general')}",
                    f"- **Request:** {t.get('user_request', '')}",
                    f"- **Created:** {t.get('created_at', '')}",
                ]
                if t.get("started_at"):
                    details.append(f"- **Started:** {t.get('started_at')}")
                if t.get("completed_at"):
                    details.append(f"- **Completed:** {t.get('completed_at')}")
                if t.get("last_update"):
                    details.append(f"- **Last update:** {t.get('last_update')}")
                if t.get("output_path"):
                    details.append(f"- **Full output:** `{t.get('output_path')}`")
                if t.get("error"):
                    details.append(f"- **Error:** {t.get('error')}")
                if t.get("result_summary"):
                    details.append(f"\n**Result:**\n\n{t.get('result_summary')}")
                return reply("\n".join(details), "Tasks")
            if low.startswith("/create project") or low.startswith("/create-project"):
                rest = raw.split("project", 1)[1].strip() if "project" in raw else ""
                stack = "Python"
                name = rest
                m = re.split(r"\s+using\s+", rest, maxsplit=1, flags=re.IGNORECASE)
                if len(m) == 2:
                    name, stack = m[0].strip(), m[1].strip()
                name = name.strip().strip('"').strip("'")
                if not name:
                    return reply(
                        "Usage: `/create project <name> using <tech stack>` — e.g. "
                        "`/create project invoice-api using Python + FastAPI`.",
                        "Coding",
                    )
                instr = (
                    f"Use the create_full_project tool to scaffold a project. "
                    f"project_name='{name}', tech_stack='{stack}', "
                    f"description='A {stack} project named {name}.', features=''. "
                    "Then briefly report the file tree and how to run it."
                )
                entry = _host.BackgroundAgentManager.submit(instr, scope=_host.current_scope_hint())
                return reply(
                    f"🛠 Building project **{name}** ({stack}) in the background "
                    f"(**{entry['id']}**). It saves to `~/Projects/`. Track it in the "
                    "**Background agents** panel.",
                    "Coding",
                )
            if low.startswith("/build app"):
                spec = raw.split("app", 1)[1].strip() if "app" in raw else ""
                if not spec:
                    return reply(
                        "Usage: `/build app <description of what the app should do>`.",
                        "Coding",
                    )
                app_name = "_".join(spec.split()[:4]).lower() or "app"
                instr = (
                    f"Use the build_application_from_spec tool. app_name='{app_name}', "
                    f"tech_stack='Python', detailed_spec='''{spec}'''. "
                    "After building, compile-check and report the result."
                )
                entry = _host.BackgroundAgentManager.submit(instr, scope=_host.current_scope_hint())
                return reply(
                    f"🛠 Building app from your spec in the background (**{entry['id']}**). "
                    "It saves to `~/Projects/`. Track it in the **Background agents** panel.",
                    "Coding",
                )
            if low.startswith("/download"):
                parts = raw.split()
                args = parts[1:]
                kind = args[0].lower() if args else ""
                url = next((a for a in args if a.lower().startswith(("http://", "https://"))), "")
                if not url:
                    return reply(
                        "Usage: `/download pdf <url>` (hosted PDF) or `/download web <url>` "
                        "(render a webpage to PDF). Files save to `~/.primus/knowledge/downloads/`.",
                        "Download",
                    )
                if kind == "web":
                    out = _host.download_webpage_as_pdf.func(url) if hasattr(_host.download_webpage_as_pdf, "func") else _host.download_webpage_as_pdf(url)
                else:
                    out = _host.download_pdf_from_url.func(url) if hasattr(_host.download_pdf_from_url, "func") else _host.download_pdf_from_url(url)
                return reply(out, "Download")
            if low.startswith("/ingest"):
                url = next(
                    (a for a in raw.split() if a.lower().startswith(("http://", "https://"))), ""
                )
                if not url:
                    return reply(
                        "Usage: `/ingest web <url>` — download a webpage/PDF and add it to the "
                        "knowledge base. Optionally tag it by appending a project name.",
                        "Ingest",
                    )
                out = _host.ingest_web_document.func(url) if hasattr(_host.ingest_web_document, "func") else _host.ingest_web_document(url)
                return reply(out, "Ingest")
            if low.startswith("/research"):
                topic = raw.split(None, 1)[1].strip() if len(raw.split(None, 1)) > 1 else ""
                if not topic:
                    return reply(
                        "Usage: `/research <topic>` — multi-engine search, reads the top sources, "
                        "and summarizes with citations. Offers to save the best sources to your "
                        "knowledge base (`/approve learn`). Runs in the background so you can keep "
                        "chatting.",
                        "Research",
                    )
                instr = (
                    f"Use the research_topic tool with topic='{topic}', max_sources=5, "
                    "ingest_offer=True. Present its full markdown briefing to the operator verbatim, "
                    "including the Sources section and the save-to-KB offer."
                )
                entry = _host.BackgroundAgentManager.submit(instr, scope=_host.current_scope_hint())
                return reply(
                    f"🔬 Researching **{topic}** in the background (**{entry['id']}**) — multi-engine "
                    "search + reading the top sources. I'll post the cited briefing here when it's "
                    "ready; then say `/approve learn` to save the best sources. Track it in the "
                    "**Background agents** panel.",
                    "Research",
                )
            if low == "/primus":
                return reply(
                    f"Primus (`{_host.CFG.get('model', _host.DEFAULT_MODEL)}`) handles general tasks; "
                    f"Forge auto-activates for coding. Use `/forge` to force coder model.",
                    "Primus",
                )
            if low == "/index_projects":
                try:
                    return reply(_host.index_projects(background=True), "Index")
                except Exception as exc:
                    return reply(f"Index failed: {exc}", "Error")
            if low == "/clear":
                _save_chat_for(chat_id or _host.ChatProjects.active_id(), [])
                try:
                    _host.get_memory_system().clear_session()
                except Exception:
                    pass
                return None, [], "Cleared"
            if low == "/export":
                p = export_chat(history)
                return reply(f"Exported: {p}" if p else "Empty", "Export")
            if low == "/memory":
                try:
                    return reply(_host.get_memory_system().display_recent_memories(), "Memory")
                except Exception as exc:
                    return reply(_host.memory_context_block() + f"\n\n*(memory system: {exc})*", "Memory")
            if low == "/tasks":
                tasks = _host.load_task_history()
                body = "\n".join(
                    f"- [{t.get('date','')[:10]}] {t.get('task','')[:70]}"
                    for t in tasks[-15:]
                ) or "(no tasks logged)"
                return reply(f"**Past tasks**\n{body}", "Tasks")
            if low == "/status":
                try:
                    model_line = f"\n\n**Model policy:** {_host.DynamicModelManager.status()}"
                except Exception:
                    model_line = ""
                body = f"{_host.build_status_bar()}{model_line}\n\n{_host.render_setup_status_markdown()}"
                return reply(body, "Status")
            if low.startswith("/gpu"):
                try:
                    return reply(_host.gpu_status_text(), "GPU")
                except Exception as exc:
                    return reply(f"GPU status unavailable: {exc}", "Error")
            if low.startswith("/model"):
                return reply(_host.handle_model_slash(raw), "Models")
            if low == "/reflect":
                try:
                    return reply(_host.get_memory_system().reflection_report(), "Reflect")
                except Exception as exc:
                    return reply(f"Reflection report failed: {exc}", "Error")
            if low in ("/metrics", "/stats", "/performance"):
                try:
                    return reply(_host.get_metrics().report(), "Metrics")
                except Exception as exc:
                    return reply(f"Metrics unavailable: {exc}", "Error")
            if low in ("/good", "/👍") or low.startswith(("/feedback good", "/good ")):
                note = raw.split(None, 2)[-1] if low.startswith("/feedback") and len(raw.split()) > 2 else (
                    raw.split(None, 1)[1] if len(raw.split()) > 1 and not low.startswith("/feedback") else "")
                try:
                    msg_out = _host.get_metrics().record_feedback(True, note)
                except Exception as exc:
                    msg_out = f"(feedback noted; metrics error: {exc})"
                return reply(msg_out, "Feedback 👍")
            if low in ("/bad", "/👎") or low.startswith(("/feedback bad", "/bad ")):
                note = raw.split(None, 2)[-1] if low.startswith("/feedback") and len(raw.split()) > 2 else (
                    raw.split(None, 1)[1] if len(raw.split()) > 1 and not low.startswith("/feedback") else "")
                try:
                    msg_out = _host.get_metrics().record_feedback(False, note)
                    # Negative feedback is a strong learning signal — capture it for reflection too.
                    if note:
                        _host.get_memory_system().add_session_fact(
                            f"Feedback to improve on: {note[:200]}", tags=["preferences", "feedback"]
                        )
                except Exception as exc:
                    msg_out = f"(feedback noted; metrics error: {exc})"
                return reply(msg_out, "Feedback 👎")
            if low in ("/summarize", "/summarize session", "/digest", "/learn now"):
                try:
                    ms = _host.get_memory_system()
                    digest = ms.learning_digest(on_demand=True)
                    cons = ms.consolidate()
                    return reply(f"{digest}\n\n_{cons}_", "Learned session")
                except Exception as exc:
                    return reply(f"Session learning failed: {exc}", "Error")
            if low.startswith("/learn"):
                payload = raw.split(None, 1)[1].strip() if len(raw.split(None, 1)) > 1 else ""
                if not payload:
                    return reply(
                        "Usage: `/learn your fact` · `/learn important key fact` · `/learn ~/file.md` · "
                        "`/learn now` (distil this session)",
                        "Learn",
                    )
                try:
                    force_important = payload.lower().startswith("important")
                    if force_important:
                        payload = payload[len("important"):].strip()
                        if not payload:
                            return reply("Usage: `/learn important your key fact here`", "Learn")
                        result = _host.get_memory_system().add_memory(
                            payload,
                            tags=_host.detect_memory_tags(payload),
                            importance=1.0,
                            source="user:important",
                        )
                        _host.get_kb().learn_text(
                            payload,
                            kind="manual",
                            source="user:important",
                            importance=1.0,
                            tags=_host.detect_memory_tags(payload),
                        )
                        return reply(result, "Learned (important)")
                    if payload.startswith("~") or payload.startswith("/") or payload.startswith("."):
                        path = _host._resolve_path(payload.split()[0])
                        result = _host.get_kb().ingest_file(path)
                    else:
                        result = _host.get_memory_system().add_memory(
                            payload,
                            tags=_host.detect_memory_tags(payload),
                            importance=0.85,
                            source="user:/learn",
                        )
                        _host.get_kb().learn_text(
                            payload,
                            kind="manual",
                            source="user:/learn",
                            importance=0.85,
                            tags=_host.detect_memory_tags(payload),
                        )
                    return reply(result, "Learned")
                except Exception as exc:
                    return reply(f"Learn failed: {exc}", "Error")
            if low.startswith("/recall") or low.startswith("/search"):
                query = raw.split(None, 1)[1].strip() if len(raw.split(None, 1)) > 1 else ""
                if not query:
                    return reply("Usage: `/recall your topic` — searches long-term memory + knowledge base", "Recall")
                try:
                    ms = _host.get_memory_system()
                    ltm = ms.recall_ltm(query)
                    kb_docs = _host.get_kb().search(query, exclude_kinds={"ltm"})
                    return reply(ms.format_recall_results(ltm, kb_docs), "Recall")
                except Exception as exc:
                    return reply(f"Recall error: {exc}", "Error")
            if low.startswith("/forget"):
                query = raw.split(None, 1)[1].strip() if len(raw.split(None, 1)) > 1 else ""
                if not query:
                    return reply("Usage: `/forget mem-xxxxxxxx` or `/forget topic phrase`", "Forget")
                try:
                    return reply(_host.get_memory_system().forget(query), "Forgot")
                except Exception as exc:
                    return reply(f"Forget failed: {exc}", "Error")
            if low.startswith("/index"):
                payload = raw.split(None, 1)[1].strip() if len(raw.split(None, 1)) > 1 else ""
                try:
                    if not payload:
                        _host.start_background_index()
                        paths = _host.CFG.get("ingest_paths") or []
                        return reply(
                            f"Background indexing started for:\n"
                            + "\n".join(f"- `{p}`" for p in paths)
                            or "*(no paths configured)*",
                            "Index",
                        )
                    path, err = _host._guard_path(payload.split()[0], must_exist=True)
                    if err:
                        return reply(err, "Error")
                    if path.is_dir():
                        _host.start_background_index([str(path)])
                        return reply(f"Background indexing started for `{path}`", "Index")
                    result = _host.get_kb().ingest_file(path)
                    return reply(result, "Index")
                except Exception as exc:
                    return reply(f"Index failed: {exc}", "Error")
            if low == "/setup":
                return reply(_host.render_setup_status_markdown(), "Setup")
            if low == "/queue":
                return reply(render_queue_md() or "Queue empty.", "Queue")
            if low == "/approve learn":
                results = _host.PrimusSession.flush_kb_learn_queue()
                body = "\n".join(results) if results else "No web snippets queued."
                return reply(body, "KB saved")
            if low == "/reject learn":
                n = len(_host.PrimusSession.pending_kb_learn)
                _host.PrimusSession.pending_kb_learn = []
                return reply(f"Discarded {n} queued KB snippet(s).", "Rejected")
            if low == "/approve edit":
                try:
                    return reply(_host.apply_pending_self_edit(), "Edit applied")
                except Exception as exc:
                    return reply(f"Self-edit apply failed: {exc}", "Error")
            if low == "/reject edit":
                had = _host.PrimusSession.pending_edit is not None
                if had:
                    pe = _host.PrimusSession.pending_edit
                    _host.SelfImprovementLog.mark_outcome(pe.get("log_id", ""), "rejected")
                    desc = pe.get("description", "self-edit")
                    try:
                        _host.SelfImprovementMemory.add(
                            lesson=f"Proposal rejected: {desc}. Reconsider the approach next time.",
                            category=_host._infer_lesson_category(desc),
                            effectiveness="rejected",
                            attempt=desc,
                            outcome="The operator rejected the proposal; not applied.",
                        )
                    except Exception:  # noqa: BLE001
                        pass
                _host.PrimusSession.pending_edit = None
                return reply(
                    "Discarded the staged self-edit." if had else "No self-edit was staged.",
                    "Rejected",
                )
            if low in ("/self improve", "/self improvements", "/self suggest"):
                try:
                    body = _host.analyze_self.invoke({"aspect": "suggestions"})
                except Exception as exc:
                    body = f"Couldn't generate suggestions: {exc}"
                return reply(body, "Self-improve")
            if low in ("/self log", "/self history"):
                return reply(_host.SelfImprovementLog.summary(), "Self-history")
            if low in ("/self plan", "/self roadmap"):
                try:
                    return reply(_host.analyze_self.invoke({"aspect": "plan"}), "Self-plan")
                except Exception as exc:
                    return reply(f"Plan unavailable: {exc}", "Self")
            if low in ("/self lessons", "/self memory"):
                return reply(_host.SelfImprovementMemory.summary(), "Self-lessons")
            if low.startswith("/self learn "):
                lesson = raw.split(" ", 2)[2].strip() if len(raw.split(" ", 2)) > 2 else ""
                if not lesson:
                    return reply("Usage: `/self learn <a lesson about improving yourself>`.", "Self")
                entry = _host.SelfImprovementMemory.add(lesson)
                return reply(
                    f"Logged lesson: \"{entry.get('lesson', lesson)[:160]}\"." if entry
                    else "Couldn't log an empty lesson.",
                    "Self-learn",
                )
            if low.startswith("/ask code"):
                q = raw[len("/ask code"):].strip()
                if not q:
                    return reply("Usage: `/ask code <what to find in my source>`.", "Code")
                try:
                    return reply(_host.search_own_codebase.invoke({"query": q}), "Code")
                except Exception as exc:
                    return reply(f"Code search failed: {exc}", "Code")
            if low in ("/self patterns", "/self usage"):
                try:
                    return reply(_host.analyze_self.invoke({"aspect": "patterns"}), "Self-patterns")
                except Exception as exc:
                    return reply(f"Patterns unavailable: {exc}", "Self")
            if low == "/self":
                try:
                    body = _host.analyze_self.invoke({"aspect": "overview"})
                except Exception:
                    body = f"Source: `{_host.SCRIPT_PATH}`"
                hist = _host.SelfImprovementLog.recent(3)
                if hist:
                    icon = {"applied": "✅", "rejected": "❌", "proposed": "⏳"}
                    body += "\n\n**Recent self-improvements:** " + " · ".join(
                        f"{icon.get(h.get('status'), '•')} {h.get('description', '')[:40]}" for h in hist
                    )
                body += (
                    "\n\n_Try `/self plan` (prioritized roadmap), `/self improve` (suggestions), "
                    "`/self patterns` (usage by area), `/self lessons` (what I've learned), "
                    "`/self log` (proposal history), or `/ask code <query>` (search my source)._"
                )
                if _host.PrimusSession.pending_edit:
                    body += (
                        f"\n\n**Staged edit:** {_host.PrimusSession.pending_edit.get('description', '')} "
                        "— type `/approve edit` to apply or `/reject edit` to discard."
                    )
                return reply(body, "Self")
            if low == "/approve":
                _host.PrimusSession.allow_dangerous_once = True
                return raw, history, "Approved once"
            if low == "/approve all":
                out = _host.run_queued_safe_commands()
                return reply(out, "Approved safe")
            if low == "/approve run":
                cmd = _host.PrimusSession.pending_shell_command or _host.PrimusSession.pending_suggestion
                if not cmd:
                    return reply("Nothing queued.", "Empty")
                old = _host.PrimusSession.mode
                _host.PrimusSession.mode = _host.ExecutionMode.EXECUTE
                _host.PrimusSession.allow_dangerous_once = True
                out = _host.run_shell(cmd, force=True)
                _host.PrimusSession.mode = old
                return reply(f"```bash\n{cmd}\n```\n\n{out}", "Ran")
            if low == "/reject":
                _host.PrimusSession.pending_shell_command = None
                _host.PrimusSession.pending_suggestion = None
                _host.PrimusSession.pending_queue = []
                return reply("Queue cleared.", "Rejected")
            return raw, history, ""

        # ---- Chat-switch safety ----------------------------------------------------------
        # A transcript is always written to the chat it came from, never to "whatever chat is
        # active by the time the write runs". Two mechanisms, because Gradio gives no ordering
        # guarantee between a switch and work already in flight:
        #
        #   1. `chat_owner` (a gr.State) rides along in the same payload as the messages, so a
        #      turn submitted before a switch still carries the id of the chat it was typed in.
        #      Every save goes through _save_chat_for(<that id>, …).
        #   2. `_switch_epoch` is bumped by every switch/create/delete. A stream that finishes
        #      after a switch still persists to its own chat, but stops painting the chatbot so
        #      it cannot stamp the outgoing transcript over the chat now on screen.
        #
        # Without these, a save that resolved the path from the live active id wrote the
        # outgoing transcript into the incoming project file and destroyed it.
        _switch_state = {"epoch": 0}

        def _chat_epoch() -> int:
            return _switch_state["epoch"]

        def _bump_chat_epoch() -> None:
            _switch_state["epoch"] += 1

        def _chat_exists(pid: str) -> bool:
            """True for General or a still-registered project (ChatProjects.get falls back to
            General for unknown ids, so it can't answer this)."""
            if not pid or pid == _host.GENERAL_CHAT_ID:
                return True
            return any(p.get("id") == pid for p in _host.ChatProjects.list())

        def _save_chat_for(pid: str, history: list) -> None:
            """Persist a transcript to the chat that produced it."""
            pid = pid or _host.GENERAL_CHAT_ID
            if not _chat_exists(pid):
                # Late save from a chat that has since been deleted — dropping it keeps us
                # from resurrecting an orphan chat_<id>.json.
                _host.log.debug("Dropped chat save for unknown/deleted chat %s", pid)
                return
            _host.save_chat_history(history, _host.chat_history_path(pid))

        def _load_chat_for(pid: str) -> list:
            return _host.sanitize_chat_history(
                _host.load_chat_history(_host.chat_history_path(pid or _host.GENERAL_CHAT_ID))
            )

        def _paint_for(pid: str, history: list):
            """Return `history` only if that chat is still on screen, else leave the chatbot be.

            Stops a handler that was queued before a switch from repainting the outgoing
            transcript over the chat the user moved to (which the next save would then
            persist under the wrong id).
            """
            if (pid or _host.GENERAL_CHAT_ID) == _host.ChatProjects.active_id():
                return history
            return gr.update()

        def _flush_outgoing(pid: str, history: list) -> None:
            """Flush the chat being left to its own file, before the active id moves.

            Writes only when the live transcript is a strict extension of the stored one.
            The chatbot can hold a capped, content-truncated slice of a long history (see
            initial_chat_history) and every turn is already persisted as it happens, so this
            is a safety net that must never be able to shrink the chat it leaves behind.
            """
            try:
                hist = _host.sanitize_chat_history(history or [])
                stored = _load_chat_for(pid)
                if len(hist) <= len(stored):
                    return

                def _key(m):
                    return (m.get("role"), m.get("content"))

                if [_key(m) for m in stored] != [_key(m) for m in hist[: len(stored)]]:
                    _host.log.debug(
                        "Outgoing transcript for %s doesn't extend its stored history — "
                        "not flushing", pid,
                    )
                    return
                _save_chat_for(pid, hist)
            except Exception as exc:  # noqa: BLE001 — a flush must never block a switch
                _host.log.warning("Could not flush outgoing chat %s: %s", pid, exc)

        # ---- Chat attachments (📎 in the composer) -------------------------------------
        # The operator drops a PDF/docx/txt/… on the composer; we extract the text immediately and
        # hold it in `pending_attachments`. The next submitted turn carries the text inline so
        # Primus actually reads it, and a background copy is indexed into the knowledge base so
        # anything past the inline cap stays reachable through retrieval later.

        def _ingest_attachments_bg(paths: list[str], chat_id: str) -> None:
            """Best-effort KB copy of chat attachments — never blocks or breaks the chat turn."""
            try:
                extra: list[str] = []
                if chat_id and chat_id != _host.GENERAL_CHAT_ID:
                    proj = _host.ChatProjects.get(chat_id)
                    extra = [f"project:{chat_id}", *(proj.get("tags") or [])]
                summary = _host.ingest_uploaded_files(
                    paths, "Learned / New Information", "", extra_tags=[*extra, "chat-attachment"]
                )
                _host.log.info("Chat attachment indexed: %s", (summary or "").splitlines()[0][:120])
            except Exception as exc:  # noqa: BLE001
                _host.log.warning("Chat attachment KB index failed (turn unaffected): %s", exc)

        def on_attach(files, existing, chat_id=None):
            """Extract text from picked documents and queue them for the next message."""
            existing = list(existing or [])
            # Gradio hands back different shapes across versions: str path, NamedString, dict.
            if files is None:
                files = []
            elif not isinstance(files, (list, tuple)):
                files = [files]
            paths: list[str] = []
            for f in files:
                if isinstance(f, str):
                    p = f
                elif isinstance(f, dict):
                    p = f.get("path") or f.get("name")
                else:
                    p = getattr(f, "name", None) or getattr(f, "path", None) or str(f)
                if p:
                    paths.append(p)
            if not paths:
                return gr.update(), gr.update()

            notes: list[str] = []
            good_paths: list[str] = []
            for p in paths:
                src = Path(str(p))
                name = src.name or "document"
                if src.suffix.lower() not in _host.TEXT_INGEST_EXTENSIONS:
                    notes.append(f"{name}: unsupported type")
                    continue
                text, err = _host.load_document_text(src)
                if err or not (text or "").strip():
                    notes.append(f"{name}: {err or 'no extractable text'}")
                    continue
                text = text.strip()
                truncated = len(text) > ATTACH_PER_FILE_CAP
                existing.append({
                    "name": name,
                    "text": text[:ATTACH_PER_FILE_CAP],
                    "chars": len(text),
                    "truncated": truncated,
                    "path": str(src),
                })
                good_paths.append(str(src))
                notes.append(f"{name} ({len(text):,} chars{' → first 8k inline' if truncated else ''})")
            existing = existing[-ATTACH_MAX_FILES:]

            if good_paths:
                threading.Thread(
                    target=_ingest_attachments_bg, args=(good_paths, chat_id or ""),
                    name="primus-attach-ingest", daemon=True,
                ).start()

            held = [a["name"] for a in existing]
            note = "📎 " + ", ".join(notes)
            if held:
                note += "  ·  attached: " + ", ".join(held) + " — included in your next message"
            return existing, note

        def respond(message, history, mode, chat_id=None, attachments=None):
            """Streaming responder: instant acknowledgment, then the real answer when ready."""
            # Owner of `history`, from the same payload — not the live active id, which may
            # already point at a chat the user switched to while this turn was in flight.
            chat_id = chat_id or _host.ChatProjects.active_id()
            epoch = _chat_epoch()

            def _paint(hist):
                """Only paint the chatbot while this turn's chat is still the one on screen."""
                return hist if _chat_epoch() == epoch else gr.update()

            if not str(message).strip():
                # Drop-a-file-and-hit-Send: with attachments pending, an empty box means
                # "tell me about this document".
                if attachments:
                    message = "I attached a document — summarize it and tell me what it is."
                else:
                    cr, cm = _cmd_ui()
                    yield (_paint(_host.sanitize_chat_history(history)), "", cr, cm, "",
                           _status(), _think(), gr.update())
                    return

            history = _host.sanitize_chat_history(history or [])
            # Surface any background-task results that finished since the last turn so they
            # appear in-line before we process the new message (file already has them too).
            try:
                _bgp = _host.BackgroundTaskManager.drain_pending(
                    str(_host.chat_history_path(chat_id))
                )
                if _bgp:
                    history = _host.sanitize_chat_history(list(history) + _bgp)
            except Exception:
                pass
            processed, history, note = handle_slash(message, history, chat_id)
            if processed is None:
                history = _host.sanitize_chat_history(history)
                _save_chat_for(chat_id, history)
                cr, cm = _cmd_ui()
                # Slash commands don't consume attachments — they stay queued for the next
                # real message.
                yield _paint(history), "", cr, cm, note, _status(), _think(), gr.update()
                return

            # Fold any pending attachments in here: the agent reads the full inline text,
            # the chat bubble only shows a small attachment line, and the state is cleared
            # on the next yield so a queued follow-up can't re-send the same file.
            agent_message, attach_suffix = _attachment_enrichment(processed, attachments)
            display_message = message + ("\n\n" + attach_suffix if attach_suffix else "")

            _host.PrimusSession.push_history(message)
            _host.PrimusSession.reset_history_nav()
            _host.PrimusSession.mode = _host.ExecutionMode(mode)

            # --- Auto-background: detect heavy / multi-step work and run it async so the chat
            #     turn never blocks or times out. Primus acks immediately; the result is injected
            #     back into this chat when ready. Skips meta-instructions and quick asks. ---
            try:
                _route_agent = _host.ModelRouter.analyze(processed).agent_id
            except Exception:
                _route_agent = ""
            _is_long, _ttype = _host.classify_long_task(processed, route_agent=_route_agent)
            if _is_long and not _host.is_meta_instruction(processed) and graphs and _host.runtime_ready():
                chat_path = str(_host.chat_history_path(chat_id))
                entry = _host.BackgroundTaskManager.submit(
                    agent_message, task_type=_ttype, scope=_host.current_scope_hint(),
                    chat_path=chat_path, mode=mode,
                )
                if entry:
                    ack_msg = (
                        f"Got it — spinning up background task **{entry['id']}** for this "
                        f"(*{_ttype}*). I'll post the results right here when it's done, so keep "
                        "chatting in the meantime.\n\n"
                        "_Track or cancel it with `/tasks`, `/task " f"{entry['id']}" "`, or "
                        "`/cancel task " f"{entry['id']}" "` — also in the Background Tasks panel._"
                    )
                    final = _host.sanitize_chat_history(
                        list(history)
                        + [
                            {"role": "user", "content": display_message},
                            {"role": "assistant", "content": ack_msg},
                        ]
                    )
                    _save_chat_for(chat_id, final)
                    cr, cm = _cmd_ui()
                    yield (_paint(final), "", cr, cm, f"Queued {entry['id']}",
                           _status(), _think(), [])
                    return

            # --- Immediate acknowledgment: clears input + shows a 'working' bubble right away ---
            ack = _host.acknowledgment_text(processed)
            ack_history = _host.sanitize_chat_history(
                list(history)
                + [
                    {"role": "user", "content": display_message},
                    {"role": "assistant", "content": ack},
                ]
            )
            cr, cm = _cmd_ui()
            yield _paint(ack_history), "", cr, cm, "Working…", _status(), _think(), []

            # --- Do the actual work (the heavy/background part) ---
            t_start = time.time()
            is_meta = False
            # Stamp the turn start so the background watchdog can recover it if it hangs.
            # invoke_primus() also calls begin_turn() (idempotent); the finally guarantees the
            # active-turn flag is cleared so the watchdog never misfires on an idle session.
            _host.PrimusSession.begin_turn()
            try:
                if not graphs or not _host.runtime_ready():
                    answer = _host._agent_unavailable_message()
                    think_steps = []
                # --- Fast meta-instruction path (highest priority): preference/style directives
                #     are saved + confirmed instantly, skipping routing and the full agent loop. ---
                elif _host.is_meta_instruction(processed):
                    is_meta = True
                    _host.PrimusSession.clear_thinking()
                    _host.PrimusSession.emit_think("Preference", "Saved directly — no agent loop", "done")
                    answer = _host.handle_meta_instruction(processed)
                    think_steps = list(_host.PrimusSession.thinking_steps)
                else:
                    try:
                        answer, think_steps = _host.invoke_primus(
                            graphs,
                            agent_message,
                            history,
                            mode,
                            force_forge=_host.PrimusSession.force_forge_next,
                            scope=_host.current_scope_hint(),  # project-chat recall bias ("" = General)
                        )
                    except Exception as exc:
                        _host.log.exception("invoke_primus failed")
                        answer = (
                            f"**Primus error** — `{exc}`\n\n"
                            "The UI stays online. Check logs or Menu → Status."
                        )
                        think_steps = list(_host.PrimusSession.thinking_steps)
                _host.PrimusSession.force_forge_next = False
            finally:
                _host.PrimusSession.end_turn()

            # --- Meta-learning: record which path answered and how fast (background-safe) ---
            elapsed = time.time() - t_start
            try:
                _host.get_metrics().record_response(_host._response_path_label(), elapsed * 1000.0)
            except Exception:
                pass
            # --- Dynamic model switching: react to slow simple turns ---
            try:
                _host.DynamicModelManager.note_result(
                    agent_id=_host.PrimusSession.active_agent,
                    intent=_host.PrimusSession.last_intent,
                    path=_host.PrimusSession.last_path,
                    elapsed_sec=elapsed,
                )
            except Exception:
                pass

            # --- Final polish: guarantee clean, human-readable text reaches the chat ---
            answer = _host.polish_response(answer) or "That's sorted. What's next?"

            # --- Proactive anticipation: optionally offer one smart next step ---
            # (Skip for meta/preference confirmations — they're complete on their own.)
            if not is_meta:
                try:
                    answer = _host.append_followup(processed, answer)
                except Exception:
                    pass
            # One-line cap hint — only when the governor actually blocked this turn.
            try:
                answer = _append_brain_cap_hint(answer)
            except Exception:  # noqa: BLE001
                pass

            # --- Replace the ack bubble with the final answer (fast typewriter reveal) ---
            base = list(history)
            base.append({"role": "user", "content": display_message})
            if bool(_host.CFG.get("show_thoughts_in_chat", False)):
                base.extend(_host.build_thought_messages(think_steps))
            base = _host.sanitize_chat_history(base)
            cr, cm = _cmd_ui()

            # Stream the assistant reply character-by-character so it 'types' into the chat.
            # Purely presentational: the full answer is persisted once, after the animation.
            if bool(_host.CFG.get("typing_effect", True)) and answer:
                # Slider value (1 fast … 100 slow) maps directly to ms-per-character.
                delay = max(0.0, float(_host.CFG.get("typing_speed", 18)) / 1000.0)
                for cut in _typing_slices(answer):
                    partial = base + [{"role": "assistant", "content": answer[:cut]}]
                    # Don't reset the input while streaming — send gr.update() (no value) so any
                    # text the user pre-types survives, and force interactive=True so the box
                    # stays editable. Only the initial ack yield clears the submitted message.
                    yield (_paint(partial), gr.update(interactive=True), cr, cm, "Typing…",
                           _status(), _think(), [])
                    if delay and cut < len(answer) and not _host.PrimusSession.is_cancelled():
                        time.sleep(delay)

            final = _host.sanitize_chat_history(base + [{"role": "assistant", "content": answer}])
            _save_chat_for(chat_id, final)
            try:
                _host.maybe_store_interaction(message, answer)
            except Exception:
                pass

            # Preserve any message the user pre-typed while the answer was streaming (don't wipe
            # msg back to ""); the next submit's acknowledgment yield clears the box.
            yield (_paint(final), gr.update(interactive=True), cr, cm, "Ready",
                   _status(), _think(), [])

        def run_pending(history, mode, chat_id=None):
            chat_id = chat_id or _host.ChatProjects.active_id()
            cmd = _host.PrimusSession.pending_shell_command or _host.PrimusSession.pending_suggestion
            if not cmd:
                cr, cm = _cmd_ui()
                return _paint_for(chat_id, _host.sanitize_chat_history(history)), cr, cm, "Empty"
            old = _host.PrimusSession.mode
            _host.PrimusSession.mode = _host.ExecutionMode.EXECUTE
            _host.PrimusSession.allow_dangerous_once = True
            out = _host.run_shell(cmd, force=True)
            _host.PrimusSession.mode = _host.ExecutionMode(mode)
            history = _host.sanitize_chat_history(history or [])
            history.append({"role": "assistant", "content": f"```bash\n{cmd}\n```\n\n{out}"})
            history = _host.sanitize_chat_history(history)
            _save_chat_for(chat_id, history)
            cr, cm = _cmd_ui()
            return _paint_for(chat_id, history), cr, cm, "Executed"

        def approve_all_pending(history, mode, chat_id=None):
            chat_id = chat_id or _host.ChatProjects.active_id()
            out = _host.run_queued_safe_commands()
            history = _host.sanitize_chat_history(history or [])
            if out and out not in ("No commands queued.",):
                history.append({"role": "assistant", "content": out})
                history = _host.sanitize_chat_history(history)
                _save_chat_for(chat_id, history)
            cr, cm = _cmd_ui()
            note = "Approved safe queue" if "Approved safe" in out else out[:60]
            return _paint_for(chat_id, history), cr, cm, note

        def modify_cmd():
            cmd = _host.PrimusSession.pending_shell_command or _host.PrimusSession.pending_suggestion or ""
            if not cmd and _host.PrimusSession.pending_queue:
                cmd = _host.PrimusSession.pending_queue[0].get("command", "")
            return gr.update(value=cmd), "Edit command in input, then Send"

        def dismiss():
            _host.PrimusSession.pending_shell_command = None
            _host.PrimusSession.pending_suggestion = None
            _host.PrimusSession.pending_queue = []
            return gr.update(visible=False), gr.update(value=""), "Dismissed"

        def clear_all(chat_id=None):
            # Clears the chat the button was pressed in (from the payload), so a clear that
            # was queued before a switch can't wipe the chat the user just opened.
            chat_id = chat_id or _host.ChatProjects.active_id()
            _save_chat_for(chat_id, [])
            try:
                _host.get_memory_system().clear_session()
            except Exception:
                pass
            _host.PrimusSession.pending_shell_command = None
            _host.PrimusSession.pending_suggestion = None
            _host.PrimusSession.pending_queue = []
            _host.PrimusSession.clear_thinking()
            return (_paint_for(chat_id, []), gr.update(visible=False), gr.update(value=""),
                    "Cleared", _status(), _think())

        def do_export(history):
            p = export_chat(history or [])
            if p:
                return p, gr.update(value=p, visible=True), f"Saved {p}"
            return None, gr.update(visible=False), "Empty"

        def refresh_setup_tab():
            return (
                _host.render_setup_dashboard_html(),
                _host.render_setup_status_markdown(),
                _quick_install_script(),
                _missing_deps_script(),
                gr.update(choices=_model_dropdown_choices()),
                gr.update(choices=_model_dropdown_choices()),
                gr.update(choices=_model_dropdown_choices()),
                _host.gpu_status_text(),
                _topbar_html(),
            )

        def ui_apply_models(primus_m, forge_m, fast_m):
            msg = _host.set_config_models(
                primus=str(primus_m) if primus_m else None,
                forge=str(forge_m) if forge_m else None,
            )
            fast_note = ""
            fast_m = str(fast_m).strip() if fast_m else ""
            if fast_m and fast_m != _host.CFG.get("fast_fallback_model"):
                _host.CFG["fast_fallback_model"] = fast_m
                _host.CFG["primus_fast_model"] = fast_m
                _host.save_config_file()
                try:
                    _host.init_agent_graphs(force=True)  # rebuild incl. fast-model graph
                except Exception as exc:  # noqa: BLE001
                    _host.log.warning("Graph rebuild after fast-model change failed: %s", exc)
                fast_note = f"  ·  Fast model → `{fast_m}`"
            return (
                msg + fast_note,
                _host.render_setup_dashboard_html(),
                _host.render_setup_status_markdown(),
                f'<div class="status-bar">{_host.build_status_bar()}</div>',
            )

        def ui_apply_gpu(prefer, backend, layers, hsa):
            _host.CFG["prefer_gpu"] = bool(prefer)
            _host.set_gpu_backend(str(backend or "auto"))  # keeps gpu_backend + legacy key in sync
            try:
                _host.CFG["gpu_layers"] = int(str(layers).strip() or -1)
            except (TypeError, ValueError):
                _host.CFG["gpu_layers"] = -1
            _host.CFG["hsa_override_gfx_version"] = str(hsa or "").strip()
            _host.save_config_file()
            _host._gpu.bust_status_cache()  # bust cache
            try:
                _host.setup_gpu_acceleration()
            except Exception as exc:  # noqa: BLE001
                _host.log.warning("GPU apply failed: %s", exc)
            # Rebuild graphs so the new num_gpu / CPU-only policy takes effect immediately.
            try:
                _host.init_agent_graphs(force=True)
            except Exception as exc:  # noqa: BLE001
                _host.log.warning("Graph rebuild after GPU change failed: %s", exc)
            return _host.gpu_status_text()

        def ui_refresh_gpu():
            _host._gpu.bust_status_cache()
            return _host.gpu_status_text()

        def ui_recover_gpu():
            """Stream GPU recovery to the log panel WITHOUT blocking Gradio.

            The recovery loop (process control, sleeps, model load) runs in a background daemon
            thread; this handler only drains its queue and yields, so it can never freeze or crash
            the Primus/Gradio process — and recovery keeps running even if this request disconnects.
            """
            import queue as _queue

            final_log = "Starting GPU recovery…"
            yield gr.update(value=final_log, visible=True), gr.update()
            try:
                q = _host._gpu.recover_ollama_gpu_threaded(max_attempts=8)
            except Exception as exc:  # noqa: BLE001
                _host.log.warning("GPU recovery failed to start: %s", exc)
                yield gr.update(value=f"✗ Could not start recovery: {exc}", visible=True), gr.update()
                return

            while True:
                try:
                    item = q.get(timeout=240)
                except _queue.Empty:
                    # Heartbeat so the panel shows we're still alive on a very slow step.
                    yield gr.update(value=final_log + "\n   ⏳ still working…", visible=True), gr.update()
                    continue
                if item is None:  # sentinel → finished
                    break
                log_text, done, _success = item
                final_log = log_text
                if done:
                    # Refresh the (slower) status markdown only on the terminal update.
                    _host._gpu.bust_status_cache()
                    try:
                        status = _host.gpu_status_text()
                    except Exception:  # noqa: BLE001
                        status = gr.update()
                    yield gr.update(value=log_text, visible=True), status
                else:
                    yield gr.update(value=log_text, visible=True), gr.update()

        def ui_rocm_help():
            """One-click guidance for enabling ROCm on AMD (esp. Ryzen AI iGPUs)."""
            gpu = _host._detect_amd_gpu_name() or "(no AMD GPU detected)"
            ryzen = _host._is_ryzen_ai_igpu(gpu)
            hsa_line = (
                f"export HSA_OVERRIDE_GFX_VERSION={_host._RYZEN_AI_HSA_DEFAULT}   # Ryzen AI iGPU override\n"
                if ryzen else ""
            )
            return (
                f"## Install ROCm for AMD GPU acceleration\n"
                f"- **Detected GPU:** {gpu}\n\n"
                "**1) Monitoring tools (quick win):**\n"
                "```bash\nsudo apt install -y rocm-smi radeontop vulkan-tools\n```\n"
                "**2) Full ROCm runtime (official AMD installer):**\n"
                "```bash\n"
                "# See https://rocm.docs.amd.com for your distro, then:\n"
                "sudo apt update && sudo apt install -y rocm-hip-libraries rocminfo\n"
                "sudo usermod -aG render,video $USER   # log out/in afterwards\n"
                "```\n"
                "**3) Point Ollama at the GPU and restart the server:**\n"
                "```bash\n"
                f"{hsa_line}"
                "OLLAMA_FLASH_ATTENTION=1 ollama serve\n"
                "```\n"
                "**4) Verify offload:** send a message, then run `ollama ps` — the PROCESSOR "
                "column should show `GPU` (or a CPU/GPU split). Re-open this panel to see the live mode."
            )

        def ui_install_desktop():
            try:
                p = _host.install_desktop_entry()
                return f"Installed: {p}", _host.render_setup_status_markdown()
            except OSError as exc:
                return f"Failed: {exc}", _host.render_setup_status_markdown()

        def ui_install_autostart():
            try:
                p = _host.install_autostart(True)
                return f"Autostart enabled: {p}", _host.render_setup_status_markdown()
            except OSError as exc:
                return f"Failed: {exc}", _host.render_setup_status_markdown()

        def ui_remove_autostart():
            _host.install_autostart(False)
            return "Autostart disabled.", _host.render_setup_status_markdown()

        def refresh_kb_tab():
            return _host.render_kb_dashboard(), "Refreshed."

        def toggle_project(category):
            return gr.update(visible=(category == "Projects"))

        def ui_process_uploads(files, category, project, chat_id="", progress=gr.Progress()):
            # Gradio hands back different shapes across versions: a single item or a list of
            # NamedString/objects with .name, plain path strings, or dicts {"path"/"name": …}.
            if files is None:
                files = []
            elif not isinstance(files, (list, tuple)):
                files = [files]
            file_paths: list[str] = []
            for f in files:
                p = None
                if isinstance(f, str):
                    p = f
                elif isinstance(f, dict):
                    p = f.get("path") or f.get("name")
                else:
                    p = getattr(f, "name", None) or getattr(f, "path", None) or str(f)
                if p:
                    file_paths.append(p)
            if not file_paths:
                return (
                    gr.update(value="⚠ Drop one or more files first, then pick a category."),
                    _host.render_kb_dashboard(),
                    gr.update(),
                )
            progress(0.02, desc="Starting…")

            def _cb(frac: float, desc: str) -> None:
                progress(min(0.99, frac), desc=desc)

            # Project-chat tagging: when a project chat is selected, tag docs with its id +
            # name so that chat's scoped recall preferentially surfaces them later.
            extra_tags: list[str] = []
            if chat_id and chat_id != _host.GENERAL_CHAT_ID:
                proj = _host.ChatProjects.get(chat_id)
                extra_tags = [f"project:{chat_id}", *(proj.get("tags") or [])]
            try:
                summary = _host.ingest_uploaded_files(
                    file_paths, category, project or "", progress_cb=_cb, extra_tags=extra_tags
                )
            except Exception as exc:  # noqa: BLE001
                _host.log.exception("Upload ingestion failed")
                summary = f"⚠ Ingestion failed: {exc}"
            progress(1.0, desc="Done")
            return gr.update(value=summary), _host.render_kb_dashboard(), gr.update(value=None)

        def ui_clear_drop():
            return gr.update(value=None), gr.update(value="")

        def _tool_call(fn, *args):
            return (fn.func if hasattr(fn, "func") else fn)(*args)

        def ui_web_ingest(url, chat_id):
            if not (url or "").strip():
                return "Enter a URL first.", _host.render_kb_dashboard()
            out = _tool_call(_host.ingest_web_document, url, "", chat_id or "")
            return out, _host.render_kb_dashboard()

        def ui_web_pdf(url):
            if not (url or "").strip():
                return "Enter a URL first.", _host.render_kb_dashboard()
            return _tool_call(_host.download_pdf_from_url, url), _host.render_kb_dashboard()

        def ui_web_page_pdf(url):
            if not (url or "").strip():
                return "Enter a URL first.", _host.render_kb_dashboard()
            return _tool_call(_host.download_webpage_as_pdf, url), _host.render_kb_dashboard()

        def ui_index_projects():
            msg = _host.index_projects(background=True)
            return _host.render_kb_dashboard(), msg

        def ui_approve_learn():
            results = _host.PrimusSession.flush_kb_learn_queue()
            msg = "\n".join(results) if results else "No queued snippets."
            return _host.render_kb_dashboard(), msg[:500]

        def do_halt():
            _host.PrimusSession.request_halt()
            return "■ Halt requested — stopping current task as soon as possible."

        def voice_input(audio_path, autosend, history, mode, chat_id=None, attachments=None):
            """Mic → transcribe → drop into the input, then auto-send (or leave for editing).

            Outputs match `outs` + [mic, voice_status]; mic is reset each time so the next
            recording starts clean. When autosend is on, it streams straight through `respond`.
            """
            history = _host.sanitize_chat_history(history or [])
            text, status = transcribe_audio(audio_path)
            if not text:
                cr, cm = _cmd_ui()
                # Keep the typed box untouched; just report what happened and reset the mic.
                yield (history, gr.update(), cr, cm, status, _status(), _think(),
                       gr.update(), gr.update(value=None), status)
                return
            if autosend:
                # Stream through the normal responder exactly like a typed message (same
                # chat id, so the transcript lands in the chat the mic was used in).
                for upd in respond(text, history, mode, chat_id, attachments):
                    yield (*upd, gr.update(value=None), f"🗣 {text[:50]}")
            else:
                cr, cm = _cmd_ui()
                yield (history, gr.update(value=text), cr, cm, status, _status(), _think(),
                       gr.update(), gr.update(value=None),
                       f"Heard — review & press Send: _{text[:60]}_")

        # ---- Project / chat organization handlers (UI/session layer only) ----
        def _switch_chat(pid, history, owner_id):
            """Activate a chat and load its own transcript; refresh scope panels.

            Strict order, because getting it wrong destroys a transcript:
              1. flush the transcript we are holding to the chat that owns it (`owner_id`,
                 which travelled in this same payload — NOT the live active id),
              2. invalidate in-flight streams,
              3. move the active id,
              4. load the incoming chat from its own file.
            """
            new_id = pid or _host.GENERAL_CHAT_ID
            owner_id = owner_id or _host.ChatProjects.active_id()
            # 1 — outgoing transcript, addressed to its own chat only. Skipped when the payload
            # already belongs to the incoming chat, which is what a programmatic selector
            # update (create/rename/delete) looks like.
            if owner_id != new_id:
                _flush_outgoing(owner_id, history)
            _bump_chat_epoch()                                          # 2
            proj = _host.ChatProjects.set_active(new_id)                # 3
            active_id = proj.get("id", _host.GENERAL_CHAT_ID)
            _host.PrimusSession.active_project_id = active_id
            hist = _load_chat_for(active_id)                            # 4
            return (
                hist,
                active_id,                                              # chat_owner state
                gr.update(value=proj.get("description", "")),
                _right_status_md(),
                f"Active chat: **{proj.get('name', 'General Chat')}**",
                _topbar_html(max_age=_TOPBAR_TTL),
            )

        def _create_project(name, desc, history, owner_id):
            entry = _host.ChatProjects.create(name, desc)
            if not entry:
                return (
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                    "Enter a project name first.", gr.update(),
                )
            # Same order as _switch_chat: the chat we are leaving is flushed to its own file
            # before the active id moves, and the new project is loaded from its own (empty)
            # file — never seeded with the transcript still sitting in the chatbot.
            owner_id = owner_id or _host.GENERAL_CHAT_ID
            if owner_id != entry["id"]:
                _flush_outgoing(owner_id, history)
            _bump_chat_epoch()
            _host.PrimusSession.active_project_id = entry["id"]
            hist = _load_chat_for(entry["id"])
            return (
                gr.update(choices=_chat_choices(), value=entry["id"]),  # chat_selector
                hist,                                                   # chatbot
                entry["id"],                                            # chat_owner state
                gr.update(value=entry.get("description", "")),          # proj_desc_box
                gr.update(value=""),                                    # new_proj_name
                gr.update(value=""),                                    # new_proj_desc
                _right_status_md(),                                     # right_status_md
                f"Created **{entry['name']}** — now active.",           # proj_status
                _topbar_html(max_age=_TOPBAR_TTL),                      # topbar
            )

        def _rename_project(new_name):
            pid = _host.ChatProjects.active_id()
            if pid == _host.GENERAL_CHAT_ID:
                return (gr.update(), gr.update(), _right_status_md(),
                        "General Chat can't be renamed.", gr.update())
            ok = _host.ChatProjects.rename(pid, new_name)
            return (
                gr.update(choices=_chat_choices(), value=pid),
                gr.update(value=""),
                _right_status_md(),
                "Renamed." if ok else "Enter a new name first.",
                _topbar_html(max_age=_TOPBAR_TTL),
            )

        def _save_desc(desc):
            pid = _host.ChatProjects.active_id()
            if pid == _host.GENERAL_CHAT_ID:
                return _right_status_md(), "General Chat scope is fixed (full access)."
            _host.ChatProjects.set_description(pid, desc)
            return _right_status_md(), "Scope description saved."

        def _delete_project():
            pid = _host.ChatProjects.active_id()
            if pid == _host.GENERAL_CHAT_ID:
                return (gr.update(), gr.update(), gr.update(), gr.update(), _right_status_md(),
                        "General Chat can't be deleted.", gr.update())
            name = _host.ChatProjects.get(pid).get("name", "project")
            _host.ChatProjects.delete(pid)  # switches active → General; removes its transcript
            # Nothing to flush (the chat is gone); the epoch bump stops any stream still
            # running for it from painting General, and _save_chat_for drops its late save.
            _bump_chat_epoch()
            _host.PrimusSession.active_project_id = _host.GENERAL_CHAT_ID
            hist = _load_chat_for(_host.GENERAL_CHAT_ID)
            return (
                gr.update(choices=_chat_choices(), value=_host.GENERAL_CHAT_ID),
                hist,
                _host.GENERAL_CHAT_ID,                                  # chat_owner state
                gr.update(value=_active_project_desc()),
                _right_status_md(),
                f"Deleted **{name}** — back to General Chat.",
                _topbar_html(max_age=_TOPBAR_TTL),
            )

        def _refresh_right():
            return _right_status_md()

        # ---- Background agent handlers (UI/session layer only) ----
        def _bg_submit(task):
            task = (task or "").strip()
            if not task:
                return _host.render_background_agents_html(), gr.update(), "Enter a task to run in the background."
            entry = _host.BackgroundAgentManager.submit(task)
            note = f"Started **{entry['id']}** — tracking below. Keep chatting."
            return _host.render_background_agents_html(), gr.update(value=""), note

        def _bg_refresh():
            n = _host.BackgroundAgentManager.active_count()
            note = f"{n} agent(s) active." if n else "No active agents."
            return _host.render_background_agents_html(), note

        def _bg_clear():
            removed = _host.BackgroundAgentManager.clear_finished()
            return _host.render_background_agents_html(), f"Cleared {removed} finished task(s)."

        def _tasks_refresh():
            n = _host.BackgroundTaskManager.active_count()
            note = f"{n} background task(s) active." if n else "No active background tasks."
            return _host.render_background_tasks_md(), note

        def _tasks_cancel(tid):
            tid = (tid or "").strip()
            if not tid:
                return _host.render_background_tasks_md(), gr.update(), "Enter a task id to cancel."
            ok = _host.BackgroundTaskManager.cancel(tid)
            note = f"Cancelled **{tid}**." if ok else f"Couldn't cancel **{tid}**."
            return _host.render_background_tasks_md(), gr.update(value=""), note

        def _tasks_tick():
            """Timer tick: refresh the tasks panel and surface finished results into the chat."""
            panel = _host.render_background_tasks_md()
            # Resolve the chat once so the path we check and the file we load can't straddle a
            # switch (this only ever reads, and only for the chat that is on screen).
            pid = _host.ChatProjects.active_id()
            chat_path = str(_host.chat_history_path(pid))
            if _host.BackgroundTaskManager.has_updates_for(chat_path):
                # The result was already appended to the chat file by the worker; reload it
                # into the live chatbot, then mark the update consumed so we don't loop.
                history = _load_chat_for(pid)
                _host.BackgroundTaskManager.consume_updates(chat_path)
                return panel, history
            return panel, gr.update()

        def _code_create(name, stack, desc):
            name = (name or "").strip()
            if not name:
                return _host.render_background_agents_html(), "Enter a project name."
            stack = (stack or "Python").strip()
            instr = (
                f"Use the create_full_project tool. project_name='{name}', "
                f"tech_stack='{stack}', description='{(desc or '').strip()}', features=''. "
                "Then report the file tree and how to run it."
            )
            entry = _host.BackgroundAgentManager.submit(instr, scope=_host.current_scope_hint())
            return (
                _host.render_background_agents_html(),
                f"🛠 Building **{name}** ({stack}) → `~/Projects/` (**{entry['id']}**).",
            )

        def _code_build(spec):
            spec = (spec or "").strip()
            if not spec:
                return _host.render_background_agents_html(), "Describe the app to build."
            app_name = "_".join(spec.split()[:4]).lower() or "app"
            instr = (
                f"Use the build_application_from_spec tool. app_name='{app_name}', "
                f"tech_stack='Python', detailed_spec='''{spec}'''. "
                "After building, compile-check and report the result."
            )
            entry = _host.BackgroundAgentManager.submit(instr, scope=_host.current_scope_hint())
            return (
                _host.render_background_agents_html(),
                f"🛠 Building app from spec → `~/Projects/` (**{entry['id']}**).",
            )

        # ---- Scheduled task handlers (scheduling layer only) ----
        def _sched_fields(name, instr, proj, stype, runat, tod, wd, interval, enabled):
            return dict(
                name=name, instructions=instr, project_id=proj or _host.GENERAL_CHAT_ID,
                schedule_type=stype or "once", run_at=runat, time_of_day=tod or "09:00",
                weekday=int(wd or 0), interval_minutes=int(interval or 60),
                enabled=bool(enabled),
            )

        def _sched_save(selected, name, instr, proj, stype, runat, tod, wd, interval, enabled):
            fields = _sched_fields(name, instr, proj, stype, runat, tod, wd, interval, enabled)
            if not (fields["name"].strip() and fields["instructions"].strip()):
                return (gr.update(), _host.render_scheduled_tasks_md(), "Enter a name and instructions.")
            if selected:
                ok = _host.ScheduledTaskManager.update(selected, **fields)
                note = "Updated." if ok else "Could not update (not found)."
            else:
                t = _host.ScheduledTaskManager.create(**fields)
                note = f"Created **{t.get('name','')}** (`{t.get('id','')}`)." if t else "Create failed."
            return (gr.update(choices=_sched_choices()), _host.render_scheduled_tasks_md(), note)

        def _sched_load(selected):
            t = _host.ScheduledTaskManager.get(selected) if selected else {}
            if not t:
                # One no-op per field in _sched_form — the last one is the Enabled checkbox,
                # so it must be gr.update() and not a value, or clearing the picker unticks it.
                return tuple(gr.update() for _ in range(9))
            return (
                gr.update(value=t.get("name", "")),
                gr.update(value=t.get("instructions", "")),
                gr.update(value=t.get("project_id", _host.GENERAL_CHAT_ID)),
                gr.update(value=t.get("schedule_type", "once")),
                gr.update(value=t.get("run_at", "")),
                gr.update(value=t.get("time_of_day", "09:00")),
                gr.update(value=int(t.get("weekday", 0))),
                gr.update(value=int(t.get("interval_minutes", 60))),
                gr.update(value=bool(t.get("enabled", True))),
            )

        def _sched_delete(selected):
            if not selected:
                return gr.update(), _host.render_scheduled_tasks_md(), "Select a task to delete."
            _host.ScheduledTaskManager.delete(selected)
            return gr.update(choices=_sched_choices(), value=None), _host.render_scheduled_tasks_md(), "Deleted."

        def _sched_toggle(selected):
            if not selected:
                return _host.render_scheduled_tasks_md(), "Select a task first."
            state = _host.ScheduledTaskManager.toggle(selected)
            return _host.render_scheduled_tasks_md(), ("Enabled." if state else "Disabled.")

        def _sched_run_now(selected):
            if not selected:
                return _host.render_scheduled_tasks_md(), _host.render_background_agents_html(), "Select a task to run."
            _host.ScheduledTaskManager.run_now(selected)
            return (_host.render_scheduled_tasks_md(), _host.render_background_agents_html(),
                    "Running now in the background — check Menu → Activity.")

        def _sched_new():
            return (
                gr.update(value=None), gr.update(value=""), gr.update(value=""),
                gr.update(value=_host.GENERAL_CHAT_ID), gr.update(value="once"),
                gr.update(value=""), gr.update(value="09:00"), gr.update(value=0),
                gr.update(value=60), gr.update(value=True), "Cleared — ready for a new task.",
            )

        def _sched_refresh():
            return _host.render_scheduled_tasks_md(), _host.render_recent_outputs_md(), gr.update(choices=_output_choices())

        def _sched_view_output(name):
            if not name:
                return ""
            p = _host.AGENT_OUTPUTS_DIR / name
            try:
                if p.exists() and p.parent == _host.AGENT_OUTPUTS_DIR:
                    return p.read_text(encoding="utf-8")[:20000]
            except OSError as exc:
                return f"_Could not read output: {exc}_"
            return "_Output not found._"

        outs = [chatbot, msg, cmd_row, cmd_preview, log_tb, status_bar, think_panel,
                pending_attachments]
        voice_outs = outs + ([mic, voice_status] if mic is not None else [])
        mode_radio.change(set_mode, [mode_radio], [log_tb])

        def _mode_hotkey(mode):
            """Ctrl+S / Ctrl+E: the hidden buttons the load-JS clicks now actually set the mode."""
            return gr.update(value=mode), set_mode(mode)

        mode_suggest_btn.click(partial(_mode_hotkey, "suggest"), None, [mode_radio, log_tb])
        mode_exec_btn.click(partial(_mode_hotkey, "execute"), None, [mode_radio, log_tb])
        # chat_owner is an input everywhere a transcript is read or written: it names the chat
        # the messages in the same payload belong to, which survives a switch landing mid-turn.
        msg.submit(respond, [msg, chatbot, mode_radio, chat_owner, pending_attachments], outs)
        send_btn.click(respond, [msg, chatbot, mode_radio, chat_owner, pending_attachments], outs)
        attach_btn.upload(
            on_attach, [attach_btn, pending_attachments, chat_owner],
            [pending_attachments, log_tb],
        )
        # Voice input: fires when the user stops the mic recording (push-to-talk style).
        # Wired defensively — only if the Audio component built and exposes the event.
        if mic is not None and hasattr(mic, "stop_recording"):
            try:
                mic.stop_recording(
                    voice_input, [mic, voice_cb, chatbot, mode_radio, chat_owner,
                                  pending_attachments], voice_outs
                )
            except Exception as exc:  # noqa: BLE001
                _host.log.warning("Could not wire mic.stop_recording (%s)", exc)

            def _set_autosend(v):
                _host.CFG["stt_autosubmit"] = bool(v)
                _host.save_config_file()
                return "Voice auto-send on." if v else "Voice fills the box — press Send to confirm."

            voice_cb.change(_set_autosend, [voice_cb], [voice_status])
        # queue=False → the halt fires immediately even while respond is streaming in the queue.
        halt_btn.click(do_halt, None, [log_tb], queue=False)
        wf_dd.change(lambda t: t or "", [wf_dd], [msg])
        run_btn.click(
            run_pending, [chatbot, mode_radio, chat_owner],
            [chatbot, cmd_row, cmd_preview, log_tb],
        )
        approve_all_btn.click(
            approve_all_pending, [chatbot, mode_radio, chat_owner],
            [chatbot, cmd_row, cmd_preview, log_tb],
        )
        modify_btn.click(modify_cmd, None, [msg, log_tb])
        dismiss_btn.click(dismiss, None, [cmd_row, cmd_preview, log_tb])
        clear_btn.click(
            clear_all, [chat_owner],
            [chatbot, cmd_row, cmd_preview, log_tb, status_bar, think_panel],
        )
        # --- Project / chat organization wiring (topbar shows the active chat name) ---
        # The switch/create handlers take the outgoing transcript plus its owner id so they can
        # flush it to its own file before moving the active id, and they hand back the new owner
        # id together with the incoming transcript.
        chat_selector.change(
            _switch_chat, [chat_selector, chatbot, chat_owner],
            [chatbot, chat_owner, proj_desc_box, right_status_md, proj_status, topbar],
        )
        create_proj_btn.click(
            _create_project,
            [new_proj_name, new_proj_desc, chatbot, chat_owner],
            [chat_selector, chatbot, chat_owner, proj_desc_box, new_proj_name, new_proj_desc,
             right_status_md, proj_status, topbar],
        )
        rename_btn.click(
            _rename_project, [rename_box],
            [chat_selector, rename_box, right_status_md, proj_status, topbar],
        )
        save_desc_btn.click(_save_desc, [proj_desc_box], [right_status_md, proj_status])
        del_proj_btn.click(
            _delete_project, None,
            [chat_selector, chatbot, chat_owner, proj_desc_box, right_status_md, proj_status,
             topbar],
        )
        right_refresh_btn.click(_refresh_right, None, [right_status_md])
        # --- Background agents wiring ---
        bg_run_btn.click(_bg_submit, [bg_task_tb], [bg_panel, bg_task_tb, bg_status])
        bg_task_tb.submit(_bg_submit, [bg_task_tb], [bg_panel, bg_task_tb, bg_status])
        bg_refresh_btn.click(_bg_refresh, None, [bg_panel, bg_status])
        bg_clear_btn.click(_bg_clear, None, [bg_panel, bg_status])
        tasks_refresh_btn.click(_tasks_refresh, None, [tasks_panel, tasks_status])
        tasks_cancel_btn.click(
            _tasks_cancel, [tasks_cancel_tb], [tasks_panel, tasks_cancel_tb, tasks_status]
        )
        code_create_btn.click(
            _code_create, [code_name_tb, code_stack_tb, code_desc_tb], [bg_panel, code_status]
        )
        code_build_btn.click(_code_build, [code_spec_tb], [bg_panel, code_status])
        # Live refresh of the background-agents panel (every few seconds) when supported by
        # the installed Gradio; degrades gracefully to the manual ↻ button on older versions.
        try:
            bg_timer = gr.Timer(3.0)
            bg_timer.tick(_host.render_background_agents_html, None, [bg_panel])
            # Separate, slightly slower timer: refresh the auto-tasks panel and push any
            # completed results into the live chat (the file already has them persisted).
            tasks_timer = gr.Timer(5.0)
            tasks_timer.tick(_tasks_tick, None, [tasks_panel, chatbot])
            # Slow tick so the top-bar status dot tracks Ollama going up/down (TTL-cached).
            topbar_timer = gr.Timer(20.0)
            topbar_timer.tick(_topbar_tick, None, [topbar])
        except Exception as exc:  # noqa: BLE001
            _host.log.debug("gr.Timer unavailable (%s) — background panels use manual refresh", exc)
        # --- Scheduled task wiring ---
        _sched_form = [
            sched_name, sched_instructions, sched_project, sched_type,
            sched_runat, sched_time, sched_weekday, sched_interval, sched_enabled,
        ]
        sched_save_btn.click(
            _sched_save, [sched_existing_dd, *_sched_form],
            [sched_existing_dd, sched_list_md, sched_status],
        )
        sched_existing_dd.change(_sched_load, [sched_existing_dd], _sched_form)
        sched_delete_btn.click(
            _sched_delete, [sched_existing_dd], [sched_existing_dd, sched_list_md, sched_status]
        )
        sched_toggle_btn.click(_sched_toggle, [sched_existing_dd], [sched_list_md, sched_status])
        sched_runnow_btn.click(
            _sched_run_now, [sched_existing_dd], [sched_list_md, bg_panel, sched_status]
        )
        sched_new_btn.click(_sched_new, None, [sched_existing_dd, *_sched_form, sched_status])
        sched_refresh_btn.click(
            _sched_refresh, None, [sched_list_md, sched_outputs_md, sched_output_pick]
        )
        sched_output_refresh.click(
            _sched_refresh, None, [sched_list_md, sched_outputs_md, sched_output_pick]
        )
        sched_output_pick.change(_sched_view_output, [sched_output_pick], [sched_output_view])
        # Start the scheduler loop (idempotent) so due tasks dispatch even while chatting.
        _host.ScheduledTaskManager.ensure_scheduler_started()
        # Mark any tasks left mid-flight by a previous shutdown as 'interrupted'.
        _host.BackgroundTaskManager.reconcile_on_start()
        export_btn.click(do_export, [chatbot], [export_file, export_file, log_tb])
        hist_btn.click(lambda _: _host.PrimusSession.prev_command(), [msg], [msg])
        refresh_btn.click(_status, None, [status_bar])
        pin_cb.change(_host.set_always_on_top, [pin_cb], [log_tb])
        compact_cb.change(_host.set_compact_mode, [compact_cb], [log_tb])
        hide_btn.click(lambda: _host.hide_app_window(), None, [log_tb])
        win_btn.click(lambda: _host.show_app_window(), None, [log_tb])
        _setup_outs = [
            setup_dashboard, setup_md, setup_cmds, fix_cmds,
            primus_model_dd, forge_model_dd, fast_model_dd, gpu_md, topbar,
        ]
        refresh_setup_btn.click(refresh_setup_tab, None, _setup_outs)

        # ---- Connected Accounts handlers ----
        def ui_conn_refresh():
            return _host.connections_refresh_markdown(), _right_status_md()

        def ui_conn_add(name, ctype, secret):
            msg = _host.connections_add_ui(name, ctype, secret)
            return (
                _host.connections_status_markdown(), msg,
                gr.update(value=""), _right_status_md(),
            )

        def ui_conn_remove(name):
            msg = _host.connections_remove_ui(name)
            return (
                _host.connections_status_markdown(), msg,
                gr.update(value=""), _right_status_md(),
            )

        conn_refresh_btn.click(ui_conn_refresh, None, [connections_md, right_status_md])
        conn_add_btn.click(
            ui_conn_add,
            [conn_add_name, conn_add_type, conn_add_secret],
            [connections_md, conn_action_md, conn_add_secret, right_status_md],
        )
        conn_remove_btn.click(
            ui_conn_remove,
            [conn_remove_name],
            [connections_md, conn_action_md, conn_remove_name, right_status_md],
        )

        # ---- Inbox handlers (thin bridges over the gmail_* tools; see registry.inbox_*_ui) ----
        def ui_inbox_refresh(query):
            table, rows, note = _host.inbox_list_ui(query)
            status = _host.inbox_status_markdown()
            action = f"_{note}_" if note else ""
            return table, rows, status, action, "_Select a message above._", ""

        def ui_inbox_connect():
            msg = _host.inbox_connect_ui()
            return msg, _host.inbox_status_markdown()

        def ui_inbox_select(rows, evt: gr.SelectData):
            md, mid = _host.inbox_read_ui(rows, evt.index if evt else None)
            return md, mid

        def ui_inbox_prefill(mid):
            if not mid:
                return ""
            return _host.inbox_reply_prefill_ui(mid)

        def ui_inbox_draft(mid, body):
            return _host.inbox_draft_reply_ui(mid, body)

        def ui_inbox_open_chat(rows, mid):
            if not mid:
                return gr.update()
            subj = ""
            for r in rows or []:
                if r.get("id") == mid:
                    subj = r.get("subject", "")
                    break
            return gr.update(value=f"Summarize this email [{mid}]: {subj}")

        inbox_refresh_btn.click(
            ui_inbox_refresh,
            [inbox_query],
            [inbox_list, inbox_rows_state, inbox_status_md, inbox_action_md,
             inbox_read, inbox_msg_id],
        )
        inbox_connect_btn.click(
            ui_inbox_connect, None, [inbox_action_md, inbox_status_md]
        )
        inbox_list.select(
            ui_inbox_select, [inbox_rows_state], [inbox_read, inbox_msg_id]
        )
        inbox_prefill_btn.click(ui_inbox_prefill, [inbox_msg_id], [inbox_reply])
        inbox_draft_btn.click(
            ui_inbox_draft, [inbox_msg_id, inbox_reply], [inbox_action_md]
        )
        inbox_chat_btn.click(
            ui_inbox_open_chat, [inbox_rows_state, inbox_msg_id], [msg]
        )

        # ---- Brain budget (same JSON the governor reads; live calc before Save) ----
        _bb_dials = [bb_weekly, bb_per_turn, bb_override, bb_estimate]
        _bb_live_out = [bb_calc_md, bb_activity_md, bb_json_md]
        for _bb_comp in _bb_dials:
            _bb_comp.change(_brain_live, _bb_dials, _bb_live_out)
            if hasattr(_bb_comp, "input"):
                _bb_comp.input(_brain_live, _bb_dials, _bb_live_out)
        bb_save_btn.click(
            _brain_budget_save,
            _bb_dials,
            [*_bb_dials, bb_calc_md, bb_activity_md, bb_json_md, bb_status_md],
        )

        apply_models_btn.click(
            ui_apply_models,
            [primus_model_dd, forge_model_dd, fast_model_dd],
            [log_tb, setup_dashboard, setup_md, status_bar],
        )
        apply_gpu_btn.click(
            ui_apply_gpu, [prefer_gpu_cb, gpu_backend_dd, gpu_layers_tb, hsa_tb], [gpu_md]
        )
        refresh_gpu_btn.click(ui_refresh_gpu, None, [gpu_md])
        rocm_help_btn.click(ui_rocm_help, None, [gpu_md])
        recover_gpu_btn.click(
            ui_recover_gpu,
            None,
            [recover_gpu_log, gpu_md],
            js=(
                "() => { if (!confirm("
                "'Recover Ollama GPU?\\n\\n"
                "This force-restarts the local Ollama server (stopping any running instance and the "
                "systemd ollama.service) with Vulkan GPU settings, then retries up to 8 times until "
                "ollama ps confirms GPU acceleration.\\n\\n"
                "Any in-progress reply will be interrupted. Continue?'"
                ")) { throw new Error('cancelled'); } return []; }"
            ),
        )

        def ui_apply_stt(enabled, size, lang):
            global _whisper_model, _whisper_model_key
            _host.CFG["stt_enabled"] = bool(enabled)
            _host.CFG["stt_model_size"] = size or "base"
            _host.CFG["stt_language"] = (lang or "").strip()
            _host.save_config_file()
            # Drop any cached model so the new size/lang takes effect, then prewarm.
            with _whisper_lock:
                _whisper_model = None
                _whisper_model_key = ""
            if _host.CFG["stt_enabled"]:
                prewarm_whisper()
                return f"Voice updated → model `{_host.CFG['stt_model_size']}`, lang `{_host.CFG['stt_language'] or 'auto'}`. Loading…"
            return "Voice input disabled."

        apply_stt_btn.click(ui_apply_stt, [stt_enable_cb, stt_model_dd, stt_lang_tb], [stt_status])

        def ui_apply_display(typing_on, speed):
            _host.CFG["typing_effect"] = bool(typing_on)
            _host.CFG["typing_speed"] = int(max(1, min(int(speed or 18), 100)))
            _host.save_config_file()
            if not _host.CFG["typing_effect"]:
                return "Typewriter effect off — responses appear all at once."
            return f"Typewriter effect on — speed {_host.CFG['typing_speed']} (~{_host.CFG['typing_speed']} ms/char)."

        apply_display_btn.click(
            ui_apply_display, [typing_effect_cb, typing_speed_slider], [typing_status]
        )
        install_desktop_btn.click(ui_install_desktop, None, [log_tb, setup_md])
        install_autostart_btn.click(ui_install_autostart, None, [log_tb, setup_md])
        remove_autostart_btn.click(ui_remove_autostart, None, [log_tb, setup_md])
        refresh_kb_btn.click(refresh_kb_tab, None, [kb_md, kb_log])
        index_projects_btn.click(ui_index_projects, None, [kb_md, kb_log])
        approve_learn_btn.click(ui_approve_learn, None, [kb_md, kb_log])
        kb_category.change(toggle_project, [kb_category], [kb_project])
        process_kb_btn.click(
            ui_process_uploads,
            [kb_drop, kb_category, kb_project, kb_chat_dd],
            [kb_status_md, kb_md, kb_drop],
        )
        clear_drop_btn.click(ui_clear_drop, None, [kb_drop, kb_status_md])
        web_ingest_btn.click(ui_web_ingest, [web_url_tb, web_project_dd], [web_status_md, kb_md])
        web_pdf_btn.click(ui_web_pdf, [web_url_tb], [web_status_md, kb_md])
        web_page_btn.click(ui_web_page_pdf, [web_url_tb], [web_status_md, kb_md])
        demo.load(refresh_setup_tab, None, _setup_outs)
        demo.load(_host.render_background_agents_html, None, [bg_panel])
        demo.load(
            _brain_budget_reload,
            None,
            [*_bb_dials, bb_calc_md, bb_activity_md, bb_json_md, bb_status_md],
        )
        demo.load(None, None, None, js=_keyboard_js())

    return demo, theme, CYBER_CSS


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _fallback_setup_html() -> str:
    """Minimal HTML page when Gradio is not installed."""
    deps = _host.check_dependencies(refresh=True)
    rows = []
    for d in deps:
        icon = "✅" if d["ok"] else "❌"
        fix = "" if d["ok"] else f'<pre>{d["install"]}</pre>'
        rows.append(f"<tr><td>{icon}</td><td>{d['name']}</td><td>{fix}</td></tr>")
    script = _quick_install_script()
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Primus Setup</title>
<style>
body{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#07080B;color:#E8EAED;
  padding:24px;max-width:760px;margin:auto;font-size:13px;line-height:1.55}}
h1{{font-size:12px;font-weight:500;letter-spacing:.14em;color:#8B909A}}
h3{{font-size:11px;font-weight:500;letter-spacing:.10em;text-transform:uppercase;color:#8B909A;margin:0 0 8px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid rgba(255,255,255,.08);padding:7px;vertical-align:top;text-align:left}}
th{{color:#8B909A;font-weight:500}}
pre{{margin:4px 0;color:#B08A4A;white-space:pre-wrap}}
code{{color:#C6CDD4}}
.box{{background:#0C0E13;border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:14px;margin-top:20px}}
</style></head><body>
<h1>PRIMUS — SETUP</h1>
<p>Gradio is not installed — showing minimal setup page. Install Gradio for the full UI.</p>
<table><tr><th></th><th>Component</th><th>Fix</th></tr>{"".join(rows)}</table>
<div class="box"><h3>Quick install</h3><pre>{script}</pre></div>
<p>Then run: <code>uv run python admin_assistant.py --browser --tray</code></p>
</body></html>"""


def run_fallback_setup_server(host: str, port: int) -> None:
    """Serve setup page on stdlib HTTP when Gradio unavailable."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    html = _fallback_setup_html().encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    url = f"http://{host}:{port}"
    print(f"\n◈ Primus setup page (no Gradio): {url}\n")
    print("Install: uv pip install gradio")
    print("Then restart for the full UI.\n")
    HTTPServer((host, port), Handler).serve_forever()
