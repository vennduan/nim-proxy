# Plan — Anthropic Messages bridge (`POST /v1/messages`)

**Status:** Proposed. Scope agreed with the owner on 2026-09-16: implement
protocol bridging in Rust, borrowing the approach of
[nim4cc](https://github.com/vennduan/nim4cc) but keeping nim-proxy's own
key-pool, pacing, and governance model. Tasks are for stepwise execution;
each task is one worktree + task branch + PR.

**Base branch:** `main`. Verified 2026-09-16: the stabilization
integration PR (#72, `release/v0.6.6` → `main`) is merged; `main` at
330efc4 is up to date.

**Integration branch:** `main` — no release line is open, so this
feature's task branches PR directly against `main`.

## Purpose

Claude Code and other Anthropic-native harnesses speak the Messages API
(`POST /v1/messages`, Anthropic SSE event stream). Public NIM has no such
endpoint — verified by probing `integrate.api.nvidia.com` directly:
`/v1/messages`, `/anthropic/v1/messages`, `/api/v1/messages` all 404. The
NeMo Gym docs describe a `/v1/messages` handler, but that is Gym's own
model-server layer, not the public API.

This work adds a bridge so any Messages client can run against NIM through
nim-proxy: request/response/SSE translation, Claude Code client-tool
schemas, and thinking-block synthesis. Everything the client sees on
`/v1/messages` is a new, deliberately added contract; the existing
`/v1/chat/completions` pass-through is untouched.

## Invariant routing (wire format does not move)

- New routes and response types are **additions**. Existing OpenAI-style
  behavior on every other path is frozen; no existing `nimproxy_*` series,
  status code, or label value moves.
- `POST /v1/messages` stops being a transparent pass-through (upstream has
  no such route; today it relays a 404). That is an intentional,
  documented contract change: OpenAPI gains two operations, and the
  [trust-boundary map](../../knowledge/architecture/http-trust-boundary-map.md)
  gains the row.
- The bridge does **not** accept a user-supplied NIM key per request
  (nim4cc's model). Keys come from the pool; pacing, retry, heartbeats,
  and governor behavior are inherited. This is the one structural
  difference from the reference implementation, on purpose.
- Data is never localized: model ids, tool names, and metric label values
  pass through untouched.
- Unsupported Anthropic capabilities are rejected with a typed 400, never
  silently swallowed (nim4cc's posture): `mcp_toolset`, server-side tools
  other than a later `web_search` milestone, `allowed_callers` without
  `direct`.

## Architecture

New module `src/bridge.rs` (request + response conversion, tool schemas,
thinking config); a small handler in `src/proxy.rs`; a route in
`src/routes.rs`. Pipeline reuse:

```
client ── POST /v1/messages ──> bridge handler
                                  │ auth (shared /v1 client-key logic)
                                  │ request transform:  body → chat payload
                                  │   + tool metadata + thinking config
                                  │ path rewritten to /v1/chat/completions
                                  ▼
        existing pipeline: model permit → dispatcher slot → heartbeat
        wait loop → retry/failover/cooldown → governor (all inherited)
                                  │
                  ┌───────────────┴────────────────┐
                  ▼ non-stream                      ▼ stream
           buffer + JSON transform         chunk-level SSE translator
           chat completion → Anthropic    (message_start → blocks →
           message + error shape          message_delta → message_stop)
```

Design notes (Ponytail-checked):

- **The bridge inherits the entire pool strategy, not just the key.** The
  bridge handler funnels into the *existing* pipeline — model-pressure
  permits, the global FIFO dispatcher, sticky-lane affinity, lane cooldowns
  and retry/failover, the governor, heartbeats, deadlines, shedding — so a
  Messages client gets identical pacing guarantees to a chat client. No
  bridge-specific queue, key selection, or backoff code.
- **No new state machine for affinity.** The existing conversation-affinity
  hash consumes `messages[0..2]`; the bridge computes it on the *converted*
  chat payload, so sticky-lane and prefix-cache behavior is identical to a
  chat client.
- **Usage injection already fits.** The bridge controls the upstream body,
  so `stream_options.include_usage` injection and the 400-fallback retry
  apply unchanged; no new exemption list.
- **SSE translator is a stateful chunk→event transducer**, in the spirit
  of the existing bounded `sse_scan` observer: it buffers partial SSE lines
  and emits Anthropic events incrementally. Real streaming, not nim4cc's
  buffer-then-replay: first-token latency stays at upstream TTFT, and
  heartbeats keep flowing while slots are pending (comment lines are legal
  in Anthropic SSE and are ignored by clients).
- **Non-streaming 4xx/5xx from upstream** are re-shape-mapped into the
  Anthropic error envelope (`{"type":"error","error":{"type","message"}}`)
  with the same status code, so clients get a parseable error. Auth and
  overload failures from the proxy itself keep their existing OpenAI-style
  envelopes (documented exception, no new machinery).
- **Metric additions are frozen identifiers.** New request-shape series for
  the bridge follow the existing
  [request-shape decision](../../knowledge/decisions/request-shape-metrics.md):
  counts, bounded labels, no content. Because existing series' label values
  are wire contract, the bridge adds one new bounded label
  `endpoint={"chat","messages"}` to the shape family plus a new tool-type
  label drawn from the frozen vocabulary below. This is a deliberate
  pre-1.0 contract change: it gets a decision note, a test, and the release
  note.
- **No history/SQLite.** `previous_response_id` and dashboard work is a
  separate, cuttable milestone; the core bridge is stateless per request
  (Claude Code sends the full history every turn).

## Label vocabulary (frozen after M1)

Sourced from the Anthropic Messages API contract as shipped by the local
Claude Code CLI 2.1.143 (`/c/Users/Venn/.local/bin/claude.exe`, 2026-05-18
build) — its embedded docs are a one-hand source for what real harnesses
send:

- **Tool types** (the `type` string of a `tools[]` entry):
  `bash_20250124`, `text_editor_20250124` / `text_editor_20250429` /
  `text_editor_20250728`, `memory_20250818`, `web_search_20250305` /
  `web_search_20260209`, `computer_*` (the CLI ships `computer_batch`
  variants; accept the `computer_` prefix family), `mcp_toolset`, and
  custom tools with no `type`. The bridge matches on prefix, not exact
  version.
- **Metric label values** (bounded): `nimproxy_tool_type_total`
  `type={"bash","text_editor","memory","web_search","computer","custom","server_rejected"}` —
  derived from the prefix, unknown/`mcp_toolset`-style rejected server
  tools count as `server_rejected` (one bucket, not one per tool name:
  cardinality must stay bounded); `nimproxy_stream_requests_total` and
  friends gain `endpoint={"chat","messages"}`.
- **Beta headers**: accept and ignore `interleaved-thinking-2025-05-14`,
  `fine-grained-tool-streaming-2025-05-14`, `long-context-…` (no new label
  dimensions for betas — a single bounded `beta` counter is enough, or
  nothing: default is *nothing*, decide in T8).
- **Content block types**: `text`, `thinking`, `redacted_thinking`,
  `tool_use`, `server_tool_use`, `tool_result` (+ suffixed variants
  `web_search_tool_result`, `text_editor_code_execution_tool_result` —
  matched by suffix, not enumerated).

## Task list

Each task: isolated worktree, task branch off the confirmed base branch,
PR against the integration branch, red→green commit for behavioral work,
`cargo fmt --check` before push.

### M0 — Contracts and scope

- **T0 Decision doc + plan record.**
  - Outcome: this file reviewed and committed; `knowledge/` entry for the
    decision lands with M1 (not before proof).
  - Proof: plan file in `docs/plans/`; owner sign-off recorded here.
  - Constraint: no code.
  - Ponytail: rung 1 — one Markdown file.
- **T1 Trust-boundary map row + OpenAPI shape contract.**
  - Outcome: `POST /v1/messages` row drafted in the
    [trust-boundary map](../../knowledge/architecture/http-trust-boundary-map.md)
    (phase, auth = /v1 client-key auth, wire types, side effects = full
    pipeline, OpenAPI operations `createAnthropicMessage` +
    `anthropicSSEStream`).
  - Proof: link check; the map table renders; no row for any existing route
    changed.
  - Constraint: existing rows byte-identical.
  - Ponytail: rung 1 — table rows + two named types.

### M1 — Core bridge, non-streaming (MVP: text + custom tools)

- **T2 Request transform `bridge::to_chat_payload()`.**
  - Outcome: Anthropic body → chat payload: `system` (string or blocks)
    prepended; `messages` with `text` / `image` / `document` blocks →
    OpenAI content parts; custom `tools` (`input_schema`) → functions;
    `tool_choice` (`auto`/`any`/`none`/`{"type":"tool","name"}`) mapped;
    `stop_sequences` → `stop`; `temperature`/`top_p`/`max_tokens`
    pass-through; `seed` mapped when present; unsupported server tools
    rejected 400 with the typed message.
  - Proof: unit tests, one fixture pair per mapping, **red first** (fixture
    written, function stubbed), including a negative fixture for
    `mcp_toolset` and `allowed_callers` without `direct`.
  - Constraint: no changes outside `src/bridge.rs` + tests.
  - Ponytail: rung 3 — serde_json only; no new deps.
- **T3 Response transform `bridge::chat_to_message()`.**
  - Outcome: chat completion → Anthropic message: text blocks, `tool_use`
    blocks (id normalized, input parsed), `stop_reason` mapping
    (`stop`→`end_turn`, `length`→`max_tokens`, `tool_calls`→`tool_use`),
    `usage` mapping (`prompt_tokens`→`input_tokens`,
    `completion_tokens`→`output_tokens`), `model` echoed from request,
    message id derived from upstream id.
  - Proof: unit tests with fixtures including a tool-call response and a
    truncated (`length`) response; red first.
  - Constraint: pure function, no I/O.
  - Ponytail: rung 3.
- **T4 Route wiring + e2e.**
  - Outcome: `POST /v1/messages` route ahead of the `/v1/{*path}` wildcard;
    handler shares the existing setup-required check, in-flight guard,
    client-key auth, and deadline parsing (extracted, not copied); non-2xx
    upstream re-shape-mapped into the Anthropic error envelope; OpenAPI
    regenerated; trust-boundary map row lands for real.
  - Proof: e2e against `scripts/mock_nim.py` — a Messages request arrives
    at the mock as `/v1/chat/completions`, returns a chat body, and the
    client sees a valid Anthropic message; 401/429/400 paths asserted;
    `openapi.json` diff shows additions only.
  - Constraint: chat path bytes unchanged (existing e2e suite green
    unmodified).
  - Also lands the request-shape emission for the bridge (existing
    `record_shape` family + `endpoint` label + `nimproxy_tool_type_total`
    with the frozen vocabulary above), per the decision-note requirement
    in the invariants section.
  - Ponytail: rung 5 — axum route + extracted helpers.

### M2 — Streaming bridge

- **T5 SSE translator `bridge::StreamTranslator`.**
  - Outcome: chunk-level transducer that turns upstream chat SSE chunks
    into the Anthropic event sequence (`message_start`, per-block
    `content_block_start`/`delta`/`stop`, `message_delta` carrying
    `stop_reason` + usage, `message_stop`), tolerating arbitrary chunk
    boundaries (mid-line, mid-JSON) and forwarding usage when injected.
  - Proof: unit tests feeding byte-exact fixture chunk streams (clean
    splits + hostile splits); a fixture asserting heartbeat comment lines
    pass through; red first.
  - Constraint: translator has bounded state (no unbounded accumulation —
    the block/usage limits of the existing observer apply).
  - Ponytail: rung 3 — one stateful struct over `Bytes`.
- **T6 Stream wiring + tool-call streaming.**
  - Outcome: `stream: true` requests run through the existing wait/heartbeat
    loop, then the translator; `tool_calls` deltas render as `tool_use`
    blocks with `input_json_delta`; a mid-stream upstream error after the
    `message_start` is emitted as an in-stream Anthropic `error` event,
    matching the existing in-stream error posture.
  - Proof: e2e streaming test with the mock NIM emitting chunked tool-call
    SSE; assert the full Anthropic event order and that heartbeats appear
    only before commitment; Claude Code smoke test against the live proxy
    (recorded request/response, not a live NIM key in fixtures).
  - Constraint: no change to the chat stream path; disconnect semantics
    (slot release on client hang-up) preserved.
  - Ponytail: rung 5 — reuses the streaming task's mpsc channel.

### M3 — Claude Code completeness

- **T7 Client tool schemas.**
  - Outcome: `bash_*`, `text_editor_*`, `computer_*`, `memory_*` tool
    types carry generated function schemas (ported, trimmed, from nim4cc's
    `build_*_tool_schema`; descriptions in the catalog style used by the
    proxy, not free Chinese text), so a Claude Code harness gets usable
    tool definitions; `caller: {type: "direct"}` echoed on `tool_use`.
  - Proof: unit tests asserting each schema is valid against its OpenAI
    function schema shape and that round-trip `tool_use` → `tool_calls` →
    `tool_use` is identity; negative fixture: a `computer_` tool with
    `allowed_callers: ["programmatic"]` is rejected 400.
  - Constraint: schema strings are frozen identifiers — no prose edits
    after this task lands.
  - Ponytail: rung 7 (last) — ~300 lines of static schema data.
- **T8 Thinking blocks.**
  - Outcome: `thinking` request param validated (budget vs `max_tokens`,
  temperature/top_p constraints — nim4cc's rules); when upstream returns
  `reasoning_content`, it becomes a leading `thinking` block with a
  synthetic `signature`; interleaved-thinking beta accepted but mapped to
  the same single-block shape.
  - Proof: unit tests on a fixture with `reasoning_content`; e2e with a
  reasoning model against the mock; 400 path for `budget_tokens >=
  max_tokens` without the interleaved beta.
  - Constraint: signature is synthetic and documented as such; no claim
  of cryptographic thinking integrity.
  - Ponytail: rung 3.
- **T9 Multi-turn tool round-trip.**
  - Outcome: assistant `tool_use` blocks + user `tool_result` blocks
  (including `*_tool_result` suffixed variants and `is_error`) convert to
  OpenAI `assistant`/`tool` messages and back, preserving ordering
  (multiple parallel tool calls, results out of order).
  - Proof: fixture round-trip test `anthropic → chat → anthropic` with a
    3-tool conversation; red first; also a `tool_result` with block-array
    content.
  - Constraint: stateless; no cross-request storage.
  - Ponytail: rung 3.

### M4 — Cuttable follow-ups (each its own plan section when started)

- **T10 `web_search` server tool execution — owner: implement, but drop if
  unstable.** In-gateway execute-then-resend loop (nim4cc's
  `create_anthropic_message_with_server_tools` pattern: bounded iteration
  count, accumulated usage, per-call `max_uses` respected, `user_location`
  folded into the query; all upstream errors surface as
  `web_search_tool_result` error blocks, never silent). Design notes:
  - Search provider is injected config in the UI-managed store — default
    keyless RSS (nim4cc's `bing.com/search?format=rss` approach,
    `WEB_SEARCH_RSS_URL` equivalent; bing RSS is unstable, so the config
    accepts any RSS/JSON endpoint the owner can key). New config keys =
    wire contract: decision + tests + release note required.
  - Stability gate (commit before merging T10): repeated runs against a
    real provider through the mock harness — zero unhandled panics,
    provider timeout → clean `search_unavailable` block, result count
    bounded by config. If any gate fails twice, park T10 (one-line note in
    the plan) and ship the rest: dev/R&D scenarios rarely need in-loop
    web search, and a half-reliable web_search is worse than a clean 400.
  - `encrypted_content` / `retrieved_at` are emitted to match the official
    wire shape (the model parses the plaintext `results` payload the
    gateway builds, same as nim4cc).
- **T11 OpenAI Responses bridge** (`POST /v1/responses`,
  `previous_response_id` backed by the existing history store) — only if a
  harness needs it; the Messages bridge covers Claude Code, which is the
  stated driver.

### Cross-cutting proofs and gates

- Every M1–M3 behavioral task: committed red→green regression, exact
  proof named before the edit.
- `cargo fmt --check` + the test-strategy-routed suite (unit, e2e) before
  each push; load test (`scripts/loadtest.py`) re-run against a 40 RPM
  lane after M2 to prove zero upstream rate violations under a
  Messages-streaming workload (invariant 3).
- Release note: "New `POST /v1/messages` Anthropic-compatible bridge;
  existing `/v1/*` behavior unchanged."
- Knowledge ingest (after proof): decision `anthropic-bridge`, component
  `messages-bridge`, trust-boundary map row, index updates; log entries
  added newest-first per M-task.

## Open questions (owner)

1. ~~Base/integration branch~~ — resolved: `main` (verified up to date,
   PR #72 merged).
2. ~~M4.T10 web_search scope~~ — resolved 2026-09-16: implement with a
   hard stability gate; park if it fails twice (dev scenarios rarely need
   in-loop search). See T10.
3. ~~Metric label vocabulary~~ — resolved from the local Claude Code CLI
   2.1.143 (one-hand source, see "Label vocabulary" section); the
   `endpoint` label change lands in T4 with its decision note. Beta
   headers get no new label dimensions by default (T8 decides between
   "nothing" and one bounded `beta` counter — lean nothing).
