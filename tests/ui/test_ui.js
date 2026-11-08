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
      w.__asks = [];
      w.fetch = (url, opts) => {
        if (url === "/api/ask") w.__asks.push(JSON.parse(opts.body));
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
  assert(doc.querySelector("select") === null, "no native <select> left (everything is styled consistently)");
  assert(doc.getElementById("picker-pop").hidden === true, "law picker starts closed");
  doc.getElementById("picker-btn").click();
  assert(doc.getElementById("picker-pop").hidden === false && doc.getElementById("picker-btn").getAttribute("aria-expanded") === "true", "clicking the picker opens it");
  assert(doc.querySelectorAll(".picker-opt").length === 3, "picker lists All + GDPR + Charter (judgments excluded)");
  assert(doc.querySelectorAll(".picker-group").length === 2, "picker groups: Legislation, Treaties and the Charter");
  const pq = doc.getElementById("picker-q");
  pq.value = "char"; pq.dispatchEvent(new d.window.Event("input"));
  assert(doc.querySelectorAll(".picker-opt").length === 1, "typing filters the picker");
  pq.dispatchEvent(new d.window.KeyboardEvent("keydown", {key: "Enter", bubbles: true}));
  assert(doc.getElementById("picker-val").textContent.includes("Charter"), "Enter selects the highlighted law");
  assert(doc.getElementById("picker-pop").hidden === true, "picker closes after choosing");
  doc.getElementById("picker-btn").click();
  doc.querySelectorAll(".picker-opt")[0].dispatchEvent(new d.window.MouseEvent("mousedown", {bubbles: true, cancelable: true}));
  assert(doc.getElementById("picker-val").textContent === "All sources", "choosing 'All sources' resets it");
  doc.getElementById("picker-btn").click();
  pq.dispatchEvent(new d.window.KeyboardEvent("keydown", {key: "Escape", bubbles: true}));
  assert(doc.getElementById("picker-pop").hidden === true, "Escape closes the picker");
  const la = doc.getElementById("marg-l-la").textContent, ra = doc.getElementById("marg-r-la").textContent;
  assert(la.length > 5 && ra.length > 5 && la !== ra, "two different Latin maxims fill the margins");
  assert(doc.getElementById("marg-l-en").textContent.length > 5, "each maxim has an English translation");
  assert(doc.querySelector(".hero .ring polygon") !== null && doc.querySelectorAll(".hero .ring polygon").length === 12, "star-ring watermark has 12 stars");
  assert(doc.querySelector("link[rel=icon]").getAttribute("href").startsWith("data:image/svg+xml"), "favicon is inline");
  assert(doc.getElementById("show-example").classList.contains("btn"), "example button uses the shared button style");
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
  await tick(30);
  doc.getElementById("picker-btn").click();
  doc.querySelectorAll(".picker-opt")[1].dispatchEvent(new d.window.MouseEvent("mousedown", {bubbles: true, cancelable: true}));
  doc.getElementById("question").value = "What is Article 17?";
  doc.getElementById("form").dispatchEvent(new d.window.Event("submit", {cancelable:true}));
  await tick(120);
  assert(doc.body.getAttribute("data-view") === "result", "streamed ask ends in the result view");
  assert(d.window.__asks[0].instrument === "gdpr" && d.window.__asks[0].stream === true, "the chosen law and stream flag are sent with the question");
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
