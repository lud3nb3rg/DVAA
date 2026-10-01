/* LLM10 instrumentation: poll the collector so exfiltrated hits appear without
 * a manual refresh. This is the challenge's own plumbing, not part of the
 * vulnerability being taught.
 *
 * It lives in a file rather than in a <script> block on the page because the
 * High and Impossible levels serve that page under a Content-Security-Policy
 * with `script-src 'self'`. An inline block would be the first thing that
 * policy killed - and a challenge whose own instrumentation has to be exempted
 * from its defence is a challenge that is not really running the defence.
 */
(function () {
  "use strict";

  const box = document.getElementById("llm10-collector");
  if (!box) return;

  const url = box.getAttribute("data-collector-url");
  if (!url) return;

  setInterval(async function () {
    try {
      const resp = await fetch(url);
      if (resp.ok) box.innerHTML = await resp.text();
    } catch (err) {
      /* the lab is not running; nothing useful to say here */
    }
  }, 2500);
})();
