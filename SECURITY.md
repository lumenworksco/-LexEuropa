# Security policy

## Supported versions

Only the latest release and the `main` branch receive fixes.

## Reporting a vulnerability

Please do **not** open a public issue. Use GitHub's private reporting:
<https://github.com/lumenworksco/-LexEuropa/security/advisories/new>

You can expect an acknowledgement within a few days. This is a small research project maintained by one person, so there is no bug bounty, but valid reports are credited in the fix.

## What to know about the design

- The web interface (`python3 -m lexeuropa.cli serve`) has **no login** and therefore binds to `127.0.0.1` only. It rejects requests whose `Host` header is not a loopback name (DNS rebinding), accepts only `application/json` POSTs, limits request and question size, sends a strict Content-Security-Policy and never logs question text. Do not expose it to a network.
- Questions are sent to Hugging Face-hosted models (embeddings, generation, the judge and the fallback) unless you point the configuration at your own servers. Do not enter confidential or personal information.
- Secrets live in `.env` (git-ignored). Use a fine-grained, inference-only Hugging Face token, not a write-scoped one.
- The GitHub Pages demo is a static page with a built-in example answer; it contains no backend and sends nothing anywhere.

## Out of scope

The accuracy of generated legal text is a quality issue, not a security issue: please use the "Wrong or misleading answer" issue form for that.
