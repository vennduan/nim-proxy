---
type: Decision
title: In-gateway web_search execution for the Messages bridge
description: The web_search_20250305 server tool is executed by the proxy itself — a bounded execute-then-resend loop against an operator-configured provider — instead of echoing tool_use to the client or rejecting the offer.
tags: [anthropic-bridge, web-search, configuration, server-tools]
timestamp: 2026-09-19T00:00:00Z
---

# In-gateway `web_search` execution for the Messages bridge

## Context

The Anthropic Messages bridge (T1–T9) rejected every server-tool offer with a
typed 400 (`server_tools_unsupported`), because a server tool has no OpenAI
chat analogue and no one in the pipeline could run it. Claude Code and similar
agent harnesses offer `web_search_20250305` by default; rejecting the offer
means the harness falls back to slower, key-carrying client-side search — or
just cannot search at all through this gateway.

nim4cc solved the same problem with an execute-then-resend loop
(`create_anthropic_message_with_server_tools`): the model asks for a search,
the gateway runs it, the result is folded back into the conversation as a
tool message, and the round is re-sent — bounded, with usage accumulated
across rounds. We follow that pattern because it keeps the loop stateless
(every state lives in the request's message list) and never surfaces the raw
upstream to the client.

## Options

1. **Keep rejecting `web_search` with the typed 400** (status quo). Clean,
   zero surface area, but the gateway stays search-blind for every agent
   harness that wants it.
2. **Echo `tool_use` to the client and let the client run the search** (like
   client tools). Contradicts the tool's `web_search_` family semantics —
   Anthropic clients treat the `web_search` family as gateway-executed and
   will not run it themselves.
3. **In-gateway execution against an operator-configured provider** (chosen).
   The provider endpoint is new config in the UI-managed store, so the
   gateway is keyless-by-default (Bing RSS, the nim4cc pattern) and the
   operator can point it at any keyless or keyed RSS/JSON endpoint.

## Decision

- `web_search_20250305` (any `web_search_*` dated shape) is now accepted by
  the bridge and executed in-gateway on the **buffered** path. A streaming
  request carrying the offer streams through the translator; its
  `server_tool_use` blocks carry no `caller` echo, documenting that the
  gateway runs the search on the follow-up non-streaming call.
- The loop is bounded: at most 4 upstream rounds per request, a per-call
  `max_uses` cap checked before any I/O, `user_location` folded into the
  query string (plain provider, no location parameter), result count capped
  by `max_results`.
- **No provider failure is silent.** A fetch timeout, non-2xx, or empty
  body becomes a typed `web_search_tool_result` error block
  (`search_unavailable`); a missing/oversized query is `missing_query` /
  `query_too_long`; the iteration cap is a typed `max_iterations_exceeded`
  `tool_result` error block with `stop_reason` staying `tool_use` so the
  client knows more calls were wanted.
- `encrypted_content` and `retrieved_at` are emitted to match the official
  wire shape. The blob is **synthetic** (`nimsearch_` + URL-safe base64 of
  the plaintext payload the gateway built, the same approach as nim4cc):
  the model parses the plaintext `results` array, never the blob, so we
  make no claim of cryptographic integrity.
- New config keys `web_search.{base_url, format, max_results,
  timeout_secs}` in the UI-managed store, managed through
  `POST /api/settings/web-search` (operator/admin). Defaults: keyless
  Bing RSS endpoint, `rss` format, 5 results, 30 s timeout. This is a wire
  contract — covered by unit tests (`check_web_search_validates_endpoint_and_bounds`
  and bounds table), the two-mock e2e suite, and a CHANGELOG entry.

## Consequences

- The gateway now makes outbound requests to an operator-chosen third-party
  endpoint: that endpoint is a new trust-boundary hop. The store validation
  applies the same guard as the upstream base URL (http(s) only, no
  link-local hosts), but the operator picks the destination.
- Usage accounting: `usage.server_tool_use.web_search_requests` is emitted
  on any request that executed at least one provider call; token usage is
  the sum across all rounds of the loop.
- The 4-round bound and the `max_uses` cap mean a looping model cannot
  exhaust the provider's quota from one request; the typed cap block keeps
  the client informed instead of silently stopping.
- Stability gate per the plan: repeated mock-harness runs with no
  unhandled panics, provider timeout → clean `search_unavailable`, result
  count bounded by config — all covered by
  `messages_web_search_*` e2e tests.
