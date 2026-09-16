---
type: Decision
title: Capture request-shape & response-quality metrics (counts, never content)
description: The proxy already deserializes every request and streams every response, so agent-behavior and model-quality signal is in hand but unread. Record it as bounded-cardinality metrics to turn the proxy into a benchmarking / agent-observability tool.
tags: [metrics, benchmarking, observability, cardinality, privacy]
timestamp: 2026-07-02T00:00:00Z
---

# Capture request-shape & response-quality metrics (counts, never content)

## Context

nim-proxy sits in the request path for every harness (OpenCode, Codex, n8n)
and every model, so it can *see* what agents actually do: how many tools they
offer, how deep their conversations run, how they set sampling params, where
models truncate, how much a reasoning model "thinks". The
[streaming pipeline](../architecture/streaming-pipeline.md) already
deserializes the request body once into a `serde_json::Value` (`proxy.rs`,
`handle`) and scans every SSE event / buffered `usage` object. That signal was
being discarded — the dashboard only ever showed tokens and latency.

The tension: capturing more is nearly free, but the proxy is security-hardened
(see [input-sanitizing-and-xss](input-sanitizing-and-xss.md)) and metric-label
cardinality is a real exposure — an attacker who controls a label value can
explode the registry.

## Options

1. **Do nothing extra** — keep tokens/latency only. Leaves the most interesting
   data (agent behavior, truncation, reasoning cost) invisible.
2. **Capture everything as labels** — e.g. `temperature="0.7"`, `messages="12"`.
   Trivial to read, but every distinct value is a new time series: unbounded
   cardinality, a registry-explosion vector.
3. **Capture as bounded metrics** — counts/sizes/params go to **histograms**
   (fixed buckets, no per-value series); categorical signal goes to **labels
   only when the value set is a fixed enum** (`finish_reason`, `tool_choice`
   mode, `stream` bool), clamped server-side so an odd upstream value collapses
   to `other`. Record **counts and sizes only — never message content.**

## Choice

Option 3. New metrics, split by the label that makes them useful:

- **Per client (harness behavior)** — `nimproxy_request_messages`,
  `nimproxy_request_tools`, `nimproxy_request_max_tokens`,
  `nimproxy_request_temperature` (histograms); `nimproxy_stream_requests_total`
  `{stream}` and `nimproxy_json_mode_total`. Powers the Clients view.
- **Per model (quality)** — `nimproxy_finish_reason_total` `{reason}` (→
  truncation rate), `nimproxy_tool_calls_total`, `nimproxy_reasoning_tokens_total`,
  `nimproxy_tpot_seconds` (mean inter-token latency). Powers the Models view,
  including its head-to-head scorecard section.
- **Global enum** — `nimproxy_tool_choice_total` `{mode}`.

## Addendum: endpoint label and tool-type counter (Messages bridge)

When the Anthropic Messages bridge (`POST /v1/messages`, see
[trust-boundary map](../architecture/http-trust-boundary-map.md)) landed,
the shape family had to be attributable to the surface that produced it, or
the dashboard could not distinguish an OpenCode OpenAI session from an
Anthropic-native harness.

- The whole request-shape family gains a frozen `endpoint` label: `chat`
  (every pre-existing series now carries it) and `messages` (the bridge).
  `nimproxy_stream_requests_total` becomes
  `{client, endpoint, stream}`; `nimproxy_tool_choice_total` becomes
  `{endpoint, mode}`; `nimproxy_json_mode_total` becomes `{client, endpoint}`;
  the request histogram family gains `{client, endpoint}`. Existing series
  change shape (a new label dimension) — the rename note for
  `nimproxy_lane_benched_total` is the precedent.
- A new counter `nimproxy_tool_type_total` counts offered tool families over
  a frozen vocabulary — `bash`, `text_editor`, `memory`, `web_search`,
  `computer`, `custom` — plus `server_rejected` for a bridge request that
  asked for a server tool the pipeline will never see. Offered tool families
  are read off the converted chat payload, so the same count lands on both
  endpoints; `server_rejected` fires exactly once, on the rejected bridge
  request.
- The bridge's proxy-own `gateway_timeout` re-maps the upstream 504 into a
  504 with proxy code `rate_limited`: a deadline cut by the proxy is a
  pacing-class failure for the client, and the client's retry semantics key
  off that code. Upstream re-shape-mapping keeps the status but never the
  proxy code.

Request shape is read from the already-parsed body at `Ctx` construction, so no
second deserialize. Response quality uses the private typed
[NIM observations](../architecture/nim-observations.md) component for both
buffered JSON and bounded SSE frames, then records only validated final
observations. Finish labels remain bounded to known values plus `other`;
invalid or unavailable numeric observations are omitted from existing metrics.

## Consequences

- The dashboard becomes a benchmarking / agent-observability tool: five
  persona-aligned tabs (see [dashboard](../architecture/dashboard.md)),
  including a head-to-head model scorecard (a section of Models) and a
  Clients view that fingerprints each agent's tool intensity, conversation
  depth, and sampling.
- Cardinality stays bounded by construction: no client-controlled free-text
  reaches a label; params live in histogram buckets; enums are clamped. The
  typed observer controls and exact-byte proxy E2E cover bounded response
  classification; request-shape tests continue to assert that `stream` is only
  ever `true`/`false`.
- Privacy posture is explicit and documented: **counts and sizes, never
  message content.** Nothing that could carry prompt text is recorded.
- Shape is labeled by *client*, not *model* — the harness determines tool use
  and sampling, so per-harness is the meaningful cut and avoids a
  client×model×buckets cardinality blow-up.
