// Behaviour tests for the web page, run in jsdom: `cd tests/ui && npm install && npm test`.
// (Checks structure and interaction, not how it looks.)
const { JSDOM } = require("jsdom");
const fs = require("fs");
const html = fs.readFileSync(require("path").join(__dirname, "..", "..", "lexeuropa", "web_static", "index.html"), "utf8");
const INFO = {chunks: 6872, instruments: [
  {celex:"32016R0679",title:"Regulation (EU) 2016/679 (General Data Protection Regulation)",doc_type:"regulation",case_number:null,chunks:389,alias:"gdpr"},
  {celex:"12016P/TXT",title:"Charter of Fundamental Rights of the European Union",doc_type:"treaty",case_number:null,chunks:91,alias:"eu charter"},
  {celex:"62012CJ0131",title:"Google Spain SL and Google Inc. v AEPD",doc_type:"judgment",case_number:"C-131/12",chunks:100,alias:null}]};
const RESULT = {question:"Q?", language:"en", generator:"fallback", generator_model:"Qwen/Qwen2.5-72B-Instruct", answers_question:false, relevance_reason:"It covers a different point.", elapsed_s:41.2,
  disclaimer:"Not legal advice.",
  retrieved:[{id:"A:1",celex:"32002L0058",citation:"Article 13(1) ePrivacy Directive",instrument_title:"Directive 2002/58/EC",text:"x".repeat(900)},
             {id:"A:2",celex:"32002L0058",citation:"Article 13(2) ePrivacy Directive",instrument_title:"Directive 2002/58/EC",text:"short text"},
             {id:"A:3",celex:"12016P/TXT",citation:"Article 8 Charter",instrument_title:"Charter",text:"unused"}],
  claims:[{text:"First statement.",citations:["A:1"],label:"supported",reason:"ok",citation_labels:{}},
          {text:"Second statement.",citations:["A:2","A:1"],label:"unsupported",reason:"Does not match the source.",citation_labels:{}},
          {text:"No source claim.",citations:[],label:"uncertain",reason:"no citation",citation_labels:{}}]};

function stream(lines) {
  const enc = new TextEncoder(); const chunks = lines.map(l => enc.encode(l + "\n")); let i = 0;
  return { ok: true, status: 200, body: { getReader: () => ({ read: () => Promise.resolve(i < chunks.length ? {done:false, value: chunks[i++]} : {done:true}) }) } };
}
async function boot(hash, askLines, fail) {
  const dom = new JSDOM(html, { runScripts: "dangerously", url: "http://127.0.0.1:8765/" + (hash||""), pretendToBeVisual: true,
    beforeParse(w) {
      w.TextEncoder = TextEncoder; w.TextDecoder = TextDecoder; w.scrollTo = () => {};
      w.HTMLElement.prototype.scrollIntoView = function(){ this.dataset.scrolled = "1"; };
      w.fetch = (url, opts) => {
        if (url === "/api/info") return Promise.resolve({ ok: true, json: () => Promise.resolve(INFO) });
        if (fail) return Promise.reject(new Error("net"));
        return Promise.resolve(stream(askLines));
      };
    }});
  await new Promise(r => setTimeout(r, 50));
  return dom;
}
function assert(c, m) { if (!c) { console.error("FAIL:", m); process.exitCode = 1; } else console.log("ok  -", m); }
const tick = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  // 1. empty state + info
  let d = await boot("");
  let doc = d.window.document;
  assert(doc.body.getAttribute("data-view") === "empty", "starts in the empty view");
  assert(doc.querySelectorAll("#try-list button").length === 4, "four example questions");
  assert(doc.querySelectorAll("#law option").length === 3, "law selector: All + GDPR + Charter (judgments excluded)");
  assert(doc.getElementById("cov-body").textContent.includes("3 documents"), "coverage dialog filled from /api/info");
  assert(doc.querySelector(".brand svg circle") !== null, "logo rendered");

  // 2. built-in example (#example)
  d = await boot("#example"); doc = d.window.document;
  assert(doc.body.getAttribute("data-view") === "result", "#example shows a result");
  assert(doc.querySelectorAll("li.stmt").length === 2, "example has 2 statements");
  assert(doc.querySelectorAll("button.cite").length === 2, "one citation marker per statement");
  assert(doc.querySelector(".example-tag") !== null, "example is clearly labelled");
  assert(doc.querySelector(".verdict").textContent.includes("Both statements are supported"), "verdict sentence");
  assert(doc.querySelectorAll("#sources .src").length === 3, "3 sources in the pane (2 cited + 1 other)");
  assert(doc.querySelectorAll("#sources .src .num").length === 2, "only cited sources are numbered");
  doc.querySelectorAll("button.cite")[1].click();
  assert(doc.body.classList.contains("sheet-open"), "clicking a marker opens the sources sheet");
  assert(doc.querySelectorAll("#sources .src.flash").length === 1, "clicking a marker highlights its source");
  assert(!doc.body.innerHTML.includes("<script>alert"), "no raw HTML injection");

  // 3. streamed ask with stages + fallback + off-topic warning + unsupported statement
  const lines = ['{"stage":"retrieving"}','{"stage":"generating"}','{"stage":"reviewing"}','{"stage":"second_opinion"}','{"stage":"checking"}', JSON.stringify({result: RESULT})];
  d = await boot("", lines); doc = d.window.document;
  doc.getElementById("question").value = "What is Article 17?";
  doc.getElementById("form").dispatchEvent(new d.window.Event("submit", {cancelable:true}));
  await tick(120);
  assert(doc.body.getAttribute("data-view") === "result", "streamed ask ends in the result view");
  assert(doc.querySelectorAll("li.stmt").length === 3, "3 statements rendered");
  assert(doc.querySelectorAll(".note").length === 2, "fallback note + off-topic warning shown");
  assert(doc.querySelector(".note.bad").textContent.includes("different point"), "relevance reason shown");
  assert(doc.querySelector(".verdict").textContent.includes("1 of 3 statements supported"), "mixed verdict");
  assert(doc.querySelector(".stmt .why") !== null, "reason shown for non-supported statement");
  assert(doc.querySelectorAll("#sources .src .num").length === 2, "two cited sources numbered");
  assert(doc.querySelector("#sources .src .row").textContent.includes("Cited in statement 1, 2"), "cited-by list is right");
  assert(doc.querySelectorAll("#sources .others .src").length === 1, "uncited source tucked under 'also retrieved'");
  assert(JSON.parse(d.window.localStorage.getItem("lexeuropa.history.v1")).length === 1, "answer saved to history");
  assert(doc.querySelectorAll("#hist li").length === 1, "history list rendered");
  doc.querySelector(".hist-del").click();
  assert(doc.querySelectorAll("#hist li").length === 0, "history item can be removed");

  // 4. error paths
  d = await boot("", ['{"stage":"retrieving"}','{"error":"generation request failed: HTTP 503"}']); doc = d.window.document;
  doc.getElementById("question").value = "What is Article 17?"; doc.getElementById("form").dispatchEvent(new d.window.Event("submit", {cancelable:true}));
  await tick(120);
  assert(doc.querySelector(".note.bad") && doc.querySelector(".note.bad").textContent.includes("503"), "server error shown");
  assert(doc.body.textContent.includes("starting up"), "503 gets the cold-start hint");
  assert(doc.getElementById("ask-btn").disabled === false, "Ask button re-enabled after an error");
  d = await boot("", [], true); doc = d.window.document;
  doc.getElementById("question").value = "What is Article 17?"; doc.getElementById("form").dispatchEvent(new d.window.Event("submit", {cancelable:true}));
  await tick(120);
  assert(doc.body.textContent.includes("Could not reach the local LexEuropa server"), "network failure message");
  console.log(process.exitCode ? "\nSOME CHECKS FAILED" : "\nall UI checks passed");
})();
