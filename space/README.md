---
title: LexEuropa Live Demo
emoji: ⚖️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: EU legal research with checked citations. Sign in with your free Hugging Face account.
hf_oauth: true
hf_oauth_scopes:
  - inference-api
hf_oauth_expiration_minutes: 480
---

# LexEuropa live demo

The real [LexEuropa](https://github.com/lumenworksco/-LexEuropa) pipeline (hybrid retrieval over 81 EU instruments,
a generator, per-claim citation verification), running live. Project page: <https://lex.braunf.com>.

**How it is paid for.** This Space holds no Hugging Face credentials of its own. You sign in with Hugging Face
(scope `inference-api`) and every model call for *your* questions is made with *your* token, so your own free monthly
inference credit pays. The token stays on the server for your session only (never in the page's JavaScript), is never
logged, and is scrubbed from every reply. Free accounts get a small allowance, so expect a handful of questions a month.

**What answers.** A stock Qwen model (`Qwen3-235B-A22B-Instruct-2507`), not the fine-tuned model (which only its owner can reach), and the
answer says so. Questions are sent to Hugging Face-hosted models; do not enter confidential or personal information.
LexEuropa is a research prototype and gives no legal advice.
