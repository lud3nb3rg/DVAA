/* Minimal progressive-enhancement helper.
 *
 * Deliberately vendored rather than pulled from a CDN: a lab about LLM supply
 * chain risk should not import a script it cannot vouch for.
 *
 * Usage:
 *   <form data-swap="#target">  submit via fetch, replace #target's HTML
 *   <form data-swap="#target" data-append>  append instead of replace
 *   <... data-busy>             gets .is-busy while a request is in flight
 *
 * With JavaScript disabled the forms still submit and the action still happens,
 * but the response is the bare fragment rather than a whole page - challenge
 * routes render partials unconditionally. Enhancement, not a fallback.
 */
(function () {
  "use strict";

  function scrollLog(el) {
    const log = el.querySelector(".chat-log") || el.closest(".chat-log");
    if (log) log.scrollTop = log.scrollHeight;
  }

  async function submit(form) {
    const targetSel = form.getAttribute("data-swap");
    const target = document.querySelector(targetSel);
    if (!target) return form.submit();

    const busy = form.querySelectorAll("[data-busy], button[type=submit]");
    busy.forEach((b) => {
      b.classList.add("is-busy");
      b.disabled = true;
    });

    const indicator = form.querySelector(".htmx-indicator");
    if (indicator) form.classList.add("htmx-request");

    try {
      const resp = await fetch(form.action, {
        method: (form.method || "POST").toUpperCase(),
        body: new FormData(form),
        headers: { "X-DVAA-Fragment": "1" },
      });
      const html = await resp.text();

      if (form.hasAttribute("data-append")) {
        target.insertAdjacentHTML("beforeend", html);
      } else {
        target.innerHTML = html;
      }

      if (!form.hasAttribute("data-keep")) {
        form.querySelectorAll("textarea, input[type=text]").forEach((i) => {
          if (!i.hasAttribute("data-keep")) i.value = "";
        });
      }
      scrollLog(target);
      target.dispatchEvent(new CustomEvent("dvaa:swapped", { bubbles: true }));
    } catch (err) {
      target.insertAdjacentHTML(
        "beforeend",
        '<div class="notice notice-bad">Request failed: ' +
          String(err).replace(/[<>&]/g, "") +
          "</div>"
      );
    } finally {
      busy.forEach((b) => {
        b.classList.remove("is-busy");
        b.disabled = false;
      });
      form.classList.remove("htmx-request");
    }
  }

  document.addEventListener("submit", function (ev) {
    const form = ev.target;
    if (!(form instanceof HTMLFormElement)) return;
    if (!form.hasAttribute("data-swap")) return;
    // An inline onsubmit="return confirm(...)" cancels the event but does not
    // stop it bubbling here. Without this check the fetch fires anyway and the
    // confirmation dialog is decoration.
    if (ev.defaultPrevented) return;
    ev.preventDefault();
    submit(form);
  });

  /* <select data-autosubmit> submits its form when the value changes.
   *
   * Here rather than as an inline onchange= because two challenge pages are
   * served under a Content-Security-Policy with `script-src 'self'`, which
   * blocks inline handlers. A handler that only works on the pages without a
   * policy is worse than no handler: it fails silently, on the page whose whole
   * subject is what a CSP does and does not stop.
   */
  document.addEventListener("change", function (ev) {
    const el = ev.target;
    if (!(el instanceof HTMLSelectElement)) return;
    if (!el.hasAttribute("data-autosubmit")) return;
    const form = el.closest("form");
    if (form) form.submit();
  });

  /* Ctrl/Cmd+Enter sends the composer. */
  document.addEventListener("keydown", function (ev) {
    if (ev.key !== "Enter" || !(ev.ctrlKey || ev.metaKey)) return;
    const ta = ev.target;
    if (!(ta instanceof HTMLTextAreaElement)) return;
    const form = ta.closest("form");
    if (form) form.requestSubmit();
  });
})();
