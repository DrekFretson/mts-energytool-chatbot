/* Виджет чата МТС EnergyTool для сайта: кнопка в углу и панель с чатом во фрейме.
   Подключение: <script src="https://АДРЕС_СЕРВИСА/widget.js" defer></script>
   Атрибуты: data-open="true" - открыть панель при первом заходе; data-debug="true" - показать переключатель отладки. */
(function () {
  "use strict";
  var script = document.currentScript;
  var origin = new URL(script.src, location.href).origin;
  var STATE = "energytool-chat-open";
  var frame = null;

  var style = document.createElement("style");
  style.textContent =
    ".et-chat-button{position:fixed;right:20px;bottom:20px;z-index:2147483000;width:60px;height:60px;border:0;border-radius:50%;" +
    "background:#e30611;color:#fff;cursor:pointer;box-shadow:0 6px 20px rgba(0,0,0,.25);display:flex;align-items:center;justify-content:center}" +
    ".et-chat-button:hover{background:#c8050f}.et-chat-button:focus-visible{outline:3px solid #1d2023;outline-offset:3px}" +
    ".et-chat-button svg{width:28px;height:28px;fill:currentColor}" +
    ".et-chat-panel{position:fixed;right:20px;bottom:92px;z-index:2147483000;width:400px;max-width:calc(100vw - 40px);" +
    "height:min(640px,calc(100vh - 112px));border-radius:16px;overflow:hidden;background:#fff;box-shadow:0 12px 40px rgba(0,0,0,.28)}" +
    ".et-chat-panel[hidden]{display:none}.et-chat-panel iframe{width:100%;height:100%;border:0;display:block}" +
    "@media (max-width:520px){.et-chat-panel{inset:0;width:100%;max-width:none;height:100%;border-radius:0}" +
    ".et-chat-button.et-open{display:none}}";
  document.head.appendChild(style);

  var button = document.createElement("button");
  button.type = "button";
  button.className = "et-chat-button";
  button.setAttribute("aria-label", "Открыть чат с консультантом EnergyTool");
  button.setAttribute("aria-expanded", "false");
  button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 4h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H9l-5 4v-4H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"/></svg>';

  var panel = document.createElement("div");
  panel.className = "et-chat-panel";
  panel.id = "et-chat-panel";
  panel.setAttribute("role", "region");
  panel.setAttribute("aria-label", "Чат с консультантом МТС EnergyTool");
  panel.hidden = true;
  button.setAttribute("aria-controls", "et-chat-panel");

  function remember(value) {
    try { sessionStorage.setItem(STATE, value); } catch (error) { /* хранилище может быть отключено */ }
  }

  function remembered() {
    try { return sessionStorage.getItem(STATE); } catch (error) { return null; }
  }

  function focusChat() {
    if (!frame || panel.hidden) return;
    frame.focus();
    frame.contentWindow.postMessage({ type: "energytool-chat-open" }, origin);
  }

  function open() {
    if (!frame) {
      // Фрейм создаётся при первом открытии: страница сайта не ждёт загрузки чата.
      frame = document.createElement("iframe");
      frame.title = "Чат с консультантом МТС EnergyTool";
      frame.src = origin + "/chat?embed=1" + (script.dataset.debug === "true" ? "&debug=1" : "");
      frame.addEventListener("load", focusChat);
      panel.appendChild(frame);
    }
    panel.hidden = false;
    button.classList.add("et-open");
    button.setAttribute("aria-expanded", "true");
    button.setAttribute("aria-label", "Свернуть чат с консультантом EnergyTool");
    remember("1");
    focusChat();
  }

  function close() {
    panel.hidden = true;
    button.classList.remove("et-open");
    button.setAttribute("aria-expanded", "false");
    button.setAttribute("aria-label", "Открыть чат с консультантом EnergyTool");
    remember("0");
    button.focus();
  }

  button.addEventListener("click", function () { if (panel.hidden) open(); else close(); });
  document.addEventListener("keydown", function (event) { if (event.key === "Escape" && !panel.hidden) close(); });
  window.addEventListener("message", function (event) {
    if (frame && event.source === frame.contentWindow && event.origin === origin && event.data && event.data.type === "energytool-chat-close") close();
  });

  function mount() {
    document.body.appendChild(panel);
    document.body.appendChild(button);
    var state = remembered();
    if (state === "1" || (state === null && script.dataset.open === "true" && window.innerWidth > 520)) open();
  }

  if (document.body) mount(); else document.addEventListener("DOMContentLoaded", mount);
})();
