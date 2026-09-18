//! Request handling: strict pass-through to NIM with three additions the
//! upstream doesn't give us — per-key rate-limit pacing, retry on 429/5xx,
//! and SSE comment heartbeats so agent harnesses (OpenCode etc.) keep the
//! connection open instead of aborting while we wait for a slot. Every
//! request is measured on the way through (see README for the metric list).

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures_util::StreamExt;
use metrics::{counter, gauge, histogram};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::dispatch::Slot;
use crate::governor::{self, ModelPermit};
use crate::observation::{
    observe_buffered, usage_observation_metrics, FinishResult, Observation, ResponseObservations,
    SseObserver, StreamOutcome,
};
use crate::{AppState, Config};
use serde_json::Value;

/// Per-request metric labels, resolved once up front.
#[derive(Clone)]
struct Ctx {
    client: String,
    model: String,
    path: String,
    /// Generation endpoint the request was served on: the OpenAI-wire chat
    /// surface or the Anthropic Messages bridge that funnels into it. Bounded
    /// to the two values of the `endpoint` label vocabulary.
    endpoint: &'static str,
    started: Instant,
}

/// Cap on distinct `model` label values tracked, past which new models are
/// bucketed to "other" so an attacker can't explode metric cardinality.
const MODEL_LABEL_CAP: usize = 256;
const DEADLINE_HEADER: &str = "x-nim-proxy-deadline-ms";

#[derive(Clone, Copy)]
struct RequestDeadline(Instant);

fn parse_request_deadline(
    headers: &HeaderMap,
    accepted: Instant,
) -> Result<Option<RequestDeadline>, ()> {
    let mut values = headers.get_all(DEADLINE_HEADER).iter();
    let Some(value) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() {
        return Err(());
    }
    let raw = value.to_str().map_err(|_| ())?;
    if raw.is_empty() || !raw.bytes().all(|b| b.is_ascii_digit()) {
        return Err(());
    }
    let millis = raw.parse::<u64>().map_err(|_| ())?;
    accepted
        .checked_add(Duration::from_millis(millis))
        .map(RequestDeadline)
        .map(Some)
        .ok_or(())
}

fn wait_deadline(cfg: &Config) -> Instant {
    Instant::now() + cfg.max_wait
}

/// Reduce an arbitrary client-supplied string to a safe metric-label / log
/// value: keep a conservative charset (which model ids use), drop everything
/// else (quotes, braces, newlines, control/ANSI — the injection vectors for
/// Prometheus exposition, structured logs, and terminals), and cap length.
fn sanitize_label(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/' | ':'))
        .take(64)
        .collect();
    if cleaned.is_empty() {
        "none".to_owned()
    } else {
        cleaned
    }
}

/// Sanitize a model id and bound its cardinality: known models pass through,
/// but once `MODEL_LABEL_CAP` distinct values have been seen, further new
/// ones collapse to "other".
fn label_model(state: &AppState, raw: &str) -> String {
    let s = sanitize_label(raw);
    let mut seen = state.model_labels.lock().unwrap();
    bounded_label(&mut seen, s, MODEL_LABEL_CAP)
}

/// Cardinality guard: return `s` if already seen or under the cap (recording
/// it), else "other". Pure so it can be tested without an AppState.
fn bounded_label(seen: &mut std::collections::HashSet<String>, s: String, cap: usize) -> String {
    if seen.contains(&s) {
        s
    } else if seen.len() < cap {
        seen.insert(s.clone());
        s
    } else {
        "other".to_owned()
    }
}

/// Bound the `path` label to the known OpenAI endpoints; anything else
/// (arbitrary sub-paths a client can hit under /v1/) becomes "other".
fn label_path(path: &str) -> String {
    match path {
        "/v1/chat/completions"
        | "/v1/completions"
        | "/v1/embeddings"
        | "/v1/models"
        | "/v1/rankings" => path.to_owned(),
        _ => "other".to_owned(),
    }
}

/// Statuses worth waiting out: rate limit and transient server-side trouble.
fn retryable(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504)
}

/// Backoff for a lane in cooldown: honor Retry-After when present.
fn backoff_for(resp: &reqwest::Response) -> Duration {
    resp.headers()
        .get(header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(10))
}

/// Join the global FIFO queue for a rate-limit slot, invoking `on_wait` every
/// heartbeat interval so streaming callers can keep their client alive.
/// Returns None if the queue rejects us (no slot before the deadline) or
/// `on_wait` reports the client is gone.
async fn reserve_slot(
    state: &AppState,
    heartbeat: Duration,
    deadline: Instant,
    prefer: Option<usize>,
    mut on_wait: impl FnMut() -> bool,
) -> Option<Slot> {
    let queued = Instant::now();
    let mut rx = state.dispatch.acquire(deadline, prefer);
    loop {
        tokio::select! {
            slot = &mut rx => {
                histogram!("nimproxy_queue_wait_seconds").record(queued.elapsed().as_secs_f64());
                if let Ok(slot) = &slot {
                    counter!("nimproxy_lane_requests_total", "lane" => slot.lane.to_string())
                        .increment(1);
                }
                return slot.ok();
            }
            _ = tokio::time::sleep(heartbeat) => {
                if !on_wait() {
                    return None;
                }
            }
        }
    }
}

/// Wait for a model-pressure permit (the governor's worker-concurrency gate),
/// heartbeating so streaming callers keep their client alive. `Ok(None)`
/// means the request isn't gated (governor off, non-generation path, or no
/// model to scope by); `Err(())` means the deadline passed or the client left.
async fn acquire_model_permit(
    state: &AppState,
    cfg: &Config,
    ctx: &Ctx,
    deadline: Instant,
    mut on_wait: impl FnMut() -> bool,
) -> Result<Option<ModelPermit>, ()> {
    let gated = cfg.governor.enabled
        && ctx.model != "none"
        && matches!(
            ctx.path.as_str(),
            "/v1/chat/completions" | "/v1/completions"
        );
    if !gated {
        return Ok(None);
    }
    let pinned = cfg.governor.overrides.get(&ctx.model).copied();
    let mut next_heartbeat = Instant::now() + cfg.heartbeat;
    loop {
        if let Some(p) = state.governor.admit(&ctx.model, pinned) {
            return Ok(Some(p));
        }
        if Instant::now() + governor::POLL > deadline {
            return Err(());
        }
        tokio::time::sleep(governor::POLL).await;
        if Instant::now() >= next_heartbeat {
            if !on_wait() {
                return Err(());
            }
            next_heartbeat = Instant::now() + cfg.heartbeat;
        }
    }
}

/// Put the granting lane in cooldown after the upstream told us to back
/// off. Routes through the slot's own pool, so a cooldown that races a
/// settings-driven pool swap lands on the (possibly retired) generation that
/// made the grant.
fn enter_cooldown(slot: &Slot, status: &str, backoff: Duration) {
    counter!("nimproxy_lane_cooldown_total", "lane" => slot.lane.to_string(), "status" => status.to_owned())
        .increment(1);
    slot.pool.penalize(slot.lane, backoff);
}

fn record_request(ctx: &Ctx, status: &str) {
    counter!(
        "nimproxy_requests_total",
        "client" => ctx.client.clone(),
        "model" => ctx.model.clone(),
        "path" => ctx.path.clone(),
        "status" => status.to_owned(),
    )
    .increment(1);
    tracing::info!(
        "{:<6} {} {} {} ({} ms)",
        status,
        ctx.client,
        ctx.model,
        ctx.path,
        ctx.started.elapsed().as_millis()
    );
}

fn record_deadline(ctx: &Ctx) {
    counter!(
        "nimproxy_deadline_exceeded_total",
        "client" => ctx.client.clone(),
        "model" => ctx.model.clone(),
        "path" => ctx.path.clone(),
    )
    .increment(1);
    record_request(ctx, "deadline");
}

fn record_tokens(ctx: &Ctx, prompt: Option<u64>, completion: Option<u64>, source: &str) {
    if let Some(p) = prompt {
        counter!("nimproxy_prompt_tokens_total", "client" => ctx.client.clone(), "model" => ctx.model.clone())
            .increment(p);
    }
    if let Some(c) = completion {
        counter!(
            "nimproxy_completion_tokens_total",
            "client" => ctx.client.clone(),
            "model" => ctx.model.clone(),
            "source" => source.to_owned(),
        )
        .increment(c);
    }
}

/// The request's tool-selection mode, bounded to a small enum. Called only
/// when the request offers tools, so a missing `tool_choice` means the
/// provider default (auto).
fn tool_choice_mode(v: &serde_json::Value) -> &'static str {
    match v.get("tool_choice") {
        Some(serde_json::Value::String(s)) => match s.as_str() {
            "auto" => "auto",
            "none" => "none",
            "required" => "required",
            _ => "other",
        },
        Some(serde_json::Value::Object(_)) => "named",
        _ => "auto",
    }
}

/// Count tools offered in a request body (`tools`, or legacy `functions`).
fn count_tools(v: &serde_json::Value) -> Option<usize> {
    v.get("tools")
        .and_then(|t| t.as_array())
        .map(|a| a.len())
        .or_else(|| {
            v.get("functions")
                .and_then(|t| t.as_array())
                .map(|a| a.len())
        })
}

/// Whether the request asks for structured (JSON) output.
fn is_json_mode(v: &serde_json::Value) -> bool {
    v.get("response_format")
        .and_then(|rf| rf.get("type"))
        .and_then(|t| t.as_str())
        .is_some_and(|t| t == "json_object" || t == "json_schema")
}

/// Record request-shape metrics: what the harness asked for (stream flag,
/// conversation depth, tools offered, sampling params, output cap, JSON mode).
/// Counts and sizes only — never message content. All heavy values go to
/// histograms, never labels, so cardinality stays bounded. The `endpoint`
/// label cuts the same shape between the OpenAI-wire chat surface ("chat")
/// and the Anthropic Messages bridge ("messages"); it is an addition to an
/// existing series (pre-1.0 contract change, see
/// knowledge/decisions/request-shape-metrics.md).
fn record_shape(ctx: &Ctx, parsed: Option<&serde_json::Value>, wants_stream: bool) {
    // Labeled by client: request shape reflects the calling client, not the
    // model — this is what powers the Clients view ("what is each agent
    // doing"). "Harness" is retired vocabulary; see
    // knowledge/decisions/standard-vocabulary.md.
    counter!(
        "nimproxy_stream_requests_total",
        "client" => ctx.client.clone(),
        "endpoint" => ctx.endpoint,
        "stream" => if wants_stream { "true" } else { "false" }.to_owned(),
    )
    .increment(1);
    let Some(v) = parsed else { return };
    if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
        histogram!(
            "nimproxy_request_messages",
            "client" => ctx.client.clone(),
            "endpoint" => ctx.endpoint,
        )
        .record(msgs.len() as f64);
    }
    if let Some(n) = count_tools(v) {
        histogram!(
            "nimproxy_request_tools",
            "client" => ctx.client.clone(),
            "endpoint" => ctx.endpoint,
        )
        .record(n as f64);
        counter!(
            "nimproxy_tool_choice_total",
            "endpoint" => ctx.endpoint,
            "mode" => tool_choice_mode(v).to_owned(),
        )
        .increment(1);
    }
    if let Some(mt) = v
        .get("max_tokens")
        .and_then(|x| x.as_u64())
        .or_else(|| v.get("max_completion_tokens").and_then(|x| x.as_u64()))
    {
        histogram!(
            "nimproxy_request_max_tokens",
            "client" => ctx.client.clone(),
            "endpoint" => ctx.endpoint,
        )
        .record(mt as f64);
    }
    if let Some(t) = v.get("temperature").and_then(|x| x.as_f64()) {
        histogram!(
            "nimproxy_request_temperature",
            "client" => ctx.client.clone(),
            "endpoint" => ctx.endpoint,
        )
        .record(t);
    }
    if is_json_mode(v) {
        counter!(
            "nimproxy_json_mode_total",
            "client" => ctx.client.clone(),
            "endpoint" => ctx.endpoint,
        )
        .increment(1);
    }
}

/// The Anthropic server-tool family label, drawn from the frozen bridge
/// vocabulary: tool types derive to a bounded family name, and rejected
/// server tools (unknown types, `mcp_toolset`, programmatic-only callers)
/// count as `server_rejected` — one bucket, never one per tool name.
fn tool_type_label(family: crate::bridge::ToolFamily) -> &'static str {
    match family {
        crate::bridge::ToolFamily::Bash => "bash",
        crate::bridge::ToolFamily::TextEditor => "text_editor",
        crate::bridge::ToolFamily::Memory => "memory",
        crate::bridge::ToolFamily::WebSearch => "web_search",
        crate::bridge::ToolFamily::WebFetch => "web_search",
        crate::bridge::ToolFamily::Computer => "computer",
        crate::bridge::ToolFamily::Custom => "custom",
    }
}

/// The streaming translator's request metadata: the client's model for the
/// `message_start` echo, plus the client-offered tool names (non-Web
/// families — the same set the buffered response transform echoes
/// `caller: {"type": "direct"}` on).
fn messages_stream_meta(
    payload: &crate::bridge::ChatPayload,
    request_model: &str,
) -> crate::bridge::StreamMeta {
    let use_family = |f: crate::bridge::ToolFamily| {
        !matches!(
            f,
            crate::bridge::ToolFamily::WebSearch | crate::bridge::ToolFamily::WebFetch
        )
    };
    crate::bridge::StreamMeta {
        client_tool_names: payload
            .tool_meta
            .iter()
            .filter(|m| use_family(m.family))
            .map(|m| m.name.clone())
            .collect(),
        model: request_model.to_owned(),
    }
}

/// Record only finalized typed observations. Invalid and unavailable upstream
/// values are deliberately absent from the existing metrics.
fn record_observations(
    ctx: &Ctx,
    observations: &ResponseObservations,
) -> Option<(u64, &'static str)> {
    for metric in usage_observation_metrics(&observations.usage) {
        counter!(
            "nimproxy_usage_observations_total",
            "field" => metric.field,
            "result" => metric.result,
        )
        .increment(1);
    }
    let prompt = match observations.usage.prompt_tokens {
        Observation::Measured(value) => Some(value),
        _ => None,
    };
    let (completion, source) = match observations.usage.completion_tokens {
        Observation::Measured(value) => (Some(value), "usage"),
        Observation::Estimated(value) => (Some(value), "estimate"),
        Observation::Unavailable | Observation::Invalid => (None, "usage"),
    };
    record_tokens(ctx, prompt, completion, source);
    for finish in &observations.finish_reasons {
        let FinishResult::Measured(reason) = &finish.result else {
            continue;
        };
        counter!(
            "nimproxy_finish_reason_total",
            "model" => ctx.model.clone(),
            "reason" => reason.metric_label(),
        )
        .increment(1);
    }
    if let Observation::Measured(reasoning) = observations.usage.reasoning_tokens {
        if reasoning > 0 {
            counter!("nimproxy_reasoning_tokens_total", "model" => ctx.model.clone())
                .increment(reasoning);
        }
    }
    if let Observation::Measured(tool_calls) = observations.tool_calls {
        if tool_calls > 0 {
            counter!("nimproxy_tool_calls_total", "model" => ctx.model.clone())
                .increment(tool_calls);
        }
    }
    completion.map(|value| (value, source))
}

/// Take the one stream-owned observer, if upstream streaming began, and
/// account it exactly once. The deadline arm and normal relay exits race for
/// this ownership rather than duplicating observer state or metrics.
fn finalize_sse_observer(
    ctx: &Ctx,
    observer: &Arc<Mutex<Option<SseObserver>>>,
    outcome: StreamOutcome,
) -> Option<(u64, &'static str)> {
    let observations = observer.lock().unwrap().take()?.finish(outcome);
    record_observations(ctx, &observations)
}

fn upstream_request(
    http: &reqwest::Client,
    base_url: &str,
    method: &Method,
    path_query: &str,
    headers: &HeaderMap,
    key: &str,
    body: &Bytes,
) -> reqwest::RequestBuilder {
    let url = format!("{base_url}{path_query}");
    let mut req = http
        .request(method.clone(), url)
        .header(header::AUTHORIZATION, format!("Bearer {key}"));
    for name in [header::CONTENT_TYPE, header::ACCEPT] {
        if let Some(v) = headers.get(&name) {
            req = req.header(name, v);
        }
    }
    if !body.is_empty() {
        req = req.body(body.clone());
    }
    req
}

/// Everything both /v1 entry points share before the pipeline: the
/// setup-required gate, the in-flight cap, client-key auth, and deadline
/// parsing. The in-flight guard is returned with the success value so the
/// streaming path can move it into its spawned task.
async fn shared_guard(
    state: &Arc<AppState>,
    cfg: &Config,
    headers: &HeaderMap,
    accepted: Instant,
) -> Result<
    (
        String,
        crate::dispatch::InflightGuard,
        Option<RequestDeadline>,
    ),
    Response,
> {
    // Shed load past the in-flight cap so a connection flood can't grow the
    // queue unbounded. The guard decrements on every exit path; the
    // streaming path moves it into its spawned task so a live stream keeps
    // occupying its slot until the stream actually ends.
    let inflight = state
        .inflight
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        + 1;
    let inflight_guard = crate::dispatch::InflightGuard::new(state.clone());
    if inflight > cfg.max_inflight {
        counter!("nimproxy_shed_total").increment(1);
        return Err(overloaded(cfg.max_inflight));
    }

    // Client auth: open mode admits everyone as "local"; keyed mode hashes
    // the presented bearer and compares against the stored SHA-256 digests
    // (the store never holds a usable token). Comparisons are constant-time.
    let client = match &cfg.clients {
        None => "local".to_owned(),
        Some(clients) => {
            let token = headers
                .get(header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer "))
                .unwrap_or("");
            let digest = crate::auth::sha256_hex(token);
            let mut matched = None;
            for (stored_digest, name) in clients {
                if crate::auth::ct_eq(&digest, stored_digest) {
                    matched = Some(name.clone());
                }
            }
            match matched {
                Some(name) => name,
                None => {
                    counter!("nimproxy_unauthorized_total").increment(1);
                    tokio::time::sleep(Duration::from_millis(250)).await;
                    return Err(unauthorized());
                }
            }
        }
    };

    match parse_request_deadline(headers, accepted) {
        Ok(request_deadline) => Ok((client, inflight_guard, request_deadline)),
        Err(()) => Err(invalid_deadline()),
    }
}

/// Single entry point for every /v1/* call.
pub async fn handle(
    State(state): State<Arc<AppState>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let accepted = Instant::now();
    // Fail closed until first-time setup completes: nothing proxies, and the
    // error tells the operator exactly why.
    if state
        .setup_required
        .load(std::sync::atomic::Ordering::SeqCst)
    {
        return crate::auth::setup_required_json();
    }

    // One consistent config view for this request's whole lifetime; a
    // concurrent settings save affects only requests that arrive after it.
    let cfg = state.cfg();

    let (client, inflight_guard, request_deadline) =
        match shared_guard(&state, &cfg, &headers, accepted).await {
            Ok(guard) => guard,
            Err(response) => return response,
        };

    let path_query = uri
        .path_and_query()
        .map(|pq| pq.as_str().to_owned())
        .unwrap_or_else(|| uri.path().to_owned());

    let mut parsed = serde_json::from_slice::<serde_json::Value>(&body).ok();
    let raw_model = parsed
        .as_ref()
        .and_then(|v| v.get("model").and_then(|m| m.as_str()))
        .unwrap_or("none");
    let ctx = Ctx {
        client,
        model: label_model(&state, raw_model),
        path: label_path(uri.path()),
        endpoint: "chat",
        started: Instant::now(),
    };

    // Answer the model-catalog probe from cache: harnesses poll it and it
    // shouldn't burn rate-limit budget on every poll.
    if method == Method::GET && uri.path() == "/v1/models" {
        if let Some(deadline) = request_deadline {
            return match tokio::time::timeout_at(deadline.0.into(), models(state, cfg)).await {
                Ok(resp) => {
                    record_request(&ctx, resp.status().as_str());
                    resp
                }
                Err(_) => {
                    record_deadline(&ctx);
                    deadline_exceeded()
                }
            };
        }
        let resp = models(state, cfg).await;
        record_request(&ctx, resp.status().as_str());
        return resp;
    }

    let wants_stream = parsed
        .as_ref()
        .and_then(|v| v.get("stream").and_then(|s| s.as_bool()))
        .unwrap_or(false);
    let prefer = parsed
        .as_ref()
        .and_then(|v| affinity(v, state.pool().len()));

    // Fingerprint what the harness asked for (generation endpoints only).
    if ctx.path == "/v1/chat/completions" || ctx.path == "/v1/completions" {
        record_shape(&ctx, parsed.as_ref(), wants_stream);
    }

    // Usage injection: streamed responses only report exact token usage when
    // asked via stream_options, so ask on the client's behalf. `fallback`
    // keeps the untouched body for a one-shot retry if the model rejects it.
    let mut body = body;
    let mut fallback = None;
    if wants_stream && !cfg.strict_passthrough && uri.path() == "/v1/chat/completions" {
        let injectable = parsed
            .as_ref()
            .is_some_and(|v| v.is_object() && v.get("stream_options").is_none())
            && !state.no_inject.lock().unwrap().contains(&ctx.model);
        if injectable {
            // `parsed` is unused after this point, so move it rather than deep-
            // cloning the whole request body (the full conversation) to inject
            // one field.
            let mut v = parsed.take().unwrap();
            v["stream_options"] = serde_json::json!({ "include_usage": true });
            fallback = Some(std::mem::replace(
                &mut body,
                Bytes::from(serde_json::to_vec(&v).expect("serialize injected body")),
            ));
        }
    }

    if wants_stream {
        let wait_deadline = wait_deadline(&cfg);
        streaming(
            state,
            cfg,
            ctx,
            method,
            path_query,
            headers,
            body,
            prefer,
            fallback,
            inflight_guard,
            request_deadline,
            wait_deadline,
            None,
        )
    } else {
        let wait_deadline = wait_deadline(&cfg);
        let deadline_ctx = ctx.clone();
        let work = buffered(
            state,
            cfg,
            ctx,
            method,
            path_query,
            headers,
            body,
            prefer,
            wait_deadline,
        );
        if let Some(deadline) = request_deadline {
            match tokio::time::timeout_at(deadline.0.into(), work).await {
                Ok(response) => response,
                Err(_) => {
                    record_deadline(&deadline_ctx);
                    deadline_exceeded()
                }
            }
        } else {
            work.await
        }
    }
}

/// The Anthropic Messages bridge: `POST /v1/messages` converts to the chat
/// payload, funnels through the exact same pipeline as the OpenAI-wire
/// surface, and converts the response back — or re-shape-maps a non-2xx
/// upstream into the Anthropic error envelope. Proxy-own failures
/// (setup-required, 401, 429-shed, 400 deadline, BridgeError) keep the
/// OpenAI-style envelope: documented exception, no new machinery.
pub async fn handle_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let accepted = Instant::now();
    if state
        .setup_required
        .load(std::sync::atomic::Ordering::SeqCst)
    {
        return crate::auth::setup_required_json();
    }
    let cfg = state.cfg();
    let (client, inflight_guard, request_deadline) =
        match shared_guard(&state, &cfg, &headers, accepted).await {
            Ok(guard) => guard,
            Err(response) => return response,
        };

    let anthropic = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .filter(|v| v.is_object());
    let anthropic = match anthropic {
        Some(anthropic) => anthropic,
        None => {
            return json_response(
                StatusCode::BAD_REQUEST,
                Bytes::from(
                    serde_json::to_vec(&proxy_error_json(
                        "invalid_json",
                        "request body must be a JSON object",
                    ))
                    .expect("static body serializes"),
                ),
            );
        }
    };

    let request_model = anthropic
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("none")
        .to_owned();

    let anthropic_beta = headers
        .get("anthropic-beta")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let payload = match crate::bridge::to_chat_payload(&anthropic, anthropic_beta) {
        Ok(payload) => payload,
        Err(
            bridge_error @ crate::bridge::BridgeError {
                code: "server_tools_unsupported",
                ..
            },
        ) => {
            // A rejected server tool is the one tool offer the client asked
            // for that the pipeline will never see: it counts as
            // `server_rejected` so the frozen vocabulary stays complete.
            counter!(
                "nimproxy_tool_type_total",
                "type" => "server_rejected",
            )
            .increment(1);
            return json_response(
                StatusCode::BAD_REQUEST,
                Bytes::from(
                    serde_json::to_vec(&serde_json::json!({
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "code": bridge_error.code,
                            "message": bridge_error.message
                        }
                    }))
                    .expect("static shape serializes"),
                ),
            );
        }
        Err(crate::bridge::BridgeError { code, message }) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                Bytes::from(
                    serde_json::to_vec(&serde_json::json!({
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "code": code,
                            "message": message
                        }
                    }))
                    .expect("static shape serializes"),
                ),
            );
        }
    };

    let chat = payload.json.clone();
    let wants_stream = chat
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);
    let prefer = affinity(&chat, state.pool().len());

    let ctx = Ctx {
        client,
        model: label_model(&state, request_model.as_str()),
        path: label_path("/v1/chat/completions"),
        endpoint: "messages",
        started: Instant::now(),
    };

    // Request-shape emission for the bridge endpoint: the shape family reads
    // the converted chat body. Tool offers are counted per entry over the
    // frozen tool-type vocabulary; a request rejected by
    // `to_chat_payload` above for a server tool counts as `server_rejected`
    // exactly once.
    record_shape(&ctx, Some(&chat), wants_stream);
    for meta in &payload.tool_meta {
        counter!(
            "nimproxy_tool_type_total",
            "type" => tool_type_label(meta.family),
        )
        .increment(1);
    }

    // Usage injection: streamed responses only report exact token usage when
    // asked via stream_options, so ask on the client's behalf. `fallback`
    // keeps the untouched body for a one-shot retry if the model rejects it.
    let mut body = Bytes::from(serde_json::to_vec(&chat).expect("chat payload serializes"));
    let mut fallback: Option<Bytes> = None;
    if wants_stream && !cfg.strict_passthrough {
        let injectable = !state.no_inject.lock().unwrap().contains(&ctx.model);
        if injectable {
            let mut v = chat;
            v["stream_options"] = serde_json::json!({ "include_usage": true });
            fallback = Some(std::mem::replace(
                &mut body,
                Bytes::from(serde_json::to_vec(&v).expect("injected payload serializes")),
            ));
        }
    }
    let method = Method::POST;
    let path_query = "/v1/chat/completions".to_owned();

    if wants_stream {
        // The bridge streams through the same wait/heartbeat loop; upstream
        // chunks are translated into the Anthropic event sequence (T5/T6),
        // and a stream that never commits to an event replays the upstream
        // bytes untouched (the uncommitted-passthrough contract).
        return streaming(
            state,
            cfg.clone(),
            ctx,
            method,
            path_query,
            headers,
            body,
            prefer,
            fallback,
            inflight_guard,
            request_deadline,
            wait_deadline(&cfg),
            Some(messages_stream_meta(&payload, request_model.as_str())),
        );
    }

    if let Some(meta) = &payload.web_search {
        // In-gateway `web_search` executor (T10): the bounded
        // execute-then-resend loop. A streaming request with a web_search
        // offer never reaches this branch — it streams through the
        // translator above, whose `tool_use` blocks carry no
        // `caller: "direct"` echo for the server family, documenting that
        // the gateway runs the search on a follow-up non-streaming call.
        return web_search_executor(
            &state,
            &cfg,
            &ctx,
            method,
            &path_query,
            &headers,
            &request_model,
            &payload,
            meta,
            request_deadline,
        )
        .await;
    }

    let work = buffered(
        state,
        cfg.clone(),
        ctx.clone(),
        method,
        path_query,
        headers,
        body,
        prefer,
        wait_deadline(&cfg),
    );
    let response = match request_deadline {
        Some(deadline) => match tokio::time::timeout_at(deadline.0.into(), work).await {
            Ok(response) => response,
            Err(_) => {
                record_deadline(&ctx);
                return deadline_exceeded();
            }
        },
        None => work.await,
    };
    anthropic_finish(response, &ctx, &request_model, &payload.tool_meta).await
}

/// Finish a bridge response: a success is harvested for observations and
/// converted to the Anthropic message; a non-2xx is re-shape-mapped into
/// the Anthropic error envelope with the same status code, so the client
/// gets a parseable error. Unparseable success bodies pass through
/// untouched — the pipeline already settled the request, so a 2xx is never
/// dropped.
async fn anthropic_finish(
    resp: Response,
    ctx: &Ctx,
    request_model: &str,
    tool_meta: &[crate::bridge::ToolMeta],
) -> Response {
    let status = resp.status();
    if status.is_success() {
        let bytes = match axum::body::to_bytes(resp.into_body(), usize::MAX).await {
            Ok(bytes) => bytes,
            Err(_) => {
                // Body stalled: surface the gateway error rather than a
                // truncated success.
                return bad_gateway();
            }
        };
        match serde_json::from_slice::<serde_json::Value>(&bytes) {
            Ok(completion) => {
                record_observations(ctx, &observe_buffered(&bytes));
                let message = crate::bridge::chat_to_message(&completion, request_model, tool_meta);
                return json_response(
                    status,
                    Bytes::from(
                        serde_json::to_vec(&message).expect("Anthropic message serializes"),
                    ),
                );
            }
            Err(_) => return json_response(status, bytes),
        }
    }
    // Non-2xx upstream into the Anthropic error envelope: same status code,
    // parseable `{"type":"error","error":{"type","message"}}`. The
    // upstream's JSON body supplies the message; the status class becomes
    // the error type.
    let raw = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .ok();
    let detail = raw
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message"))
                .or_else(|| v.get("message"))
                .or_else(|| v.get("detail"))
                .and_then(|m| m.as_str().map(str::to_owned))
        })
        .unwrap_or_else(|| "upstream request failed".to_owned());
    let kind = match status.as_u16() {
        400 => "invalid_request_error",
        401 | 403 => "authentication_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        500 => "api_error",
        503 => "overloaded_error",
        504 => "timeout_error",
        _ => "api_error",
    };
    record_request(ctx, status.as_str());
    json_response(
        status,
        Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "type": "error",
                "error": { "type": kind, "message": detail }
            }))
            .expect("static shape serializes"),
        ),
    )
}

/// In-gateway `web_search` executor (T10): the request's `web_search`
/// server tool is run here — the model's `tool_use` comes back, the
/// configured provider is called, and the results are folded back into
/// the conversation as a tool message before the round is re-sent. Bounded
/// by `WEB_SEARCH_MAX_ITERATIONS`; per-call `max_uses` and the
/// request's `user_location` ride on the metadata. Every provider failure
/// surfaces as a `web_search_tool_result` error block, never silent.
///
/// Each upstream round is a full pacing-loop exchange (the executor runs
/// in the request's shadow: the same pool, the same retry, the same
/// deadline), so the request stays under the governance model end to end.
/// Only the buffered path executes: a streaming request with a `web_search`
/// offer streams through the translator, whose `tool_use` blocks carry no
/// `caller: "direct"` echo for the server family — the gateway documents
/// that it runs the search, on a follow-up non-streaming call.
#[allow(clippy::too_many_arguments)]
async fn web_search_executor(
    state: &Arc<AppState>,
    cfg: &Arc<Config>,
    ctx: &Ctx,
    method: Method,
    path_query: &str,
    headers: &HeaderMap,
    request_model: &str,
    payload: &crate::bridge::ChatPayload,
    meta: &crate::bridge::search::WebSearchMeta,
    request_deadline: Option<RequestDeadline>,
) -> Response {
    const WEB_SEARCH_MAX_ITERATIONS: u32 = 4;

    let mut messages = payload
        .json
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let search_cfg = cfg.web_search.clone();
    let mut rounds: Vec<Value> = vec![];
    let mut outward_blocks: Vec<Value> = vec![];
    let mut last_completion: Option<Value> = None;
    let mut last_bytes: Option<Bytes> = None;
    let mut search_requests: u64 = 0;
    let mut uses: u32 = 0;

    for _round in 0..WEB_SEARCH_MAX_ITERATIONS {
        let mut chat_json = payload.json.clone();
        chat_json["messages"] = Value::Array(messages.clone());
        let body = Bytes::from(serde_json::to_vec(&chat_json).expect("chat payload serializes"));
        let work = buffered(
            state.clone(),
            cfg.clone(),
            ctx.clone(),
            method.clone(),
            path_query.to_owned(),
            headers.clone(),
            body,
            None,
            wait_deadline(cfg),
        );
        let response = match request_deadline {
            Some(deadline) => match tokio::time::timeout_at(deadline.0.into(), work).await {
                Ok(response) => response,
                Err(_) => {
                    record_deadline(ctx);
                    return deadline_exceeded();
                }
            },
            None => work.await,
        };

        let status = response.status();
        if !status.is_success() {
            return anthropic_finish(response, ctx, request_model, &payload.tool_meta).await;
        }
        let Ok(bytes) = axum::body::to_bytes(response.into_body(), usize::MAX).await else {
            return bad_gateway();
        };
        let Ok(completion) = serde_json::from_slice::<Value>(&bytes) else {
            // A 2xx body that is not a chat completion: the pipeline settled
            // the request, so the passthrough contract applies.
            return json_response(status, bytes);
        };
        let message = crate::bridge::chat_to_message_with_web_search(
            &completion,
            request_model,
            &payload.tool_meta,
            Some(meta),
        );
        if crate::bridge::web_search_pending(&completion, meta) {
            // Execute every web_search call this round named, then fold
            // the results back in for the next round.
            let calls = completion
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|c| c.first())
                .and_then(|c| c.get("message"))
                .and_then(|m| m.get("tool_calls"))
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let mut results: Vec<(Value, Value)> = vec![];
            for call in &calls {
                let name = call
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if name != meta.name {
                    continue;
                }
                let id = call
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|i| !i.is_empty())
                    .map(str::to_owned)
                    .unwrap_or_else(|| "srvtoolu_bridge_auto".to_owned());
                let input = call
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(Value::as_str)
                    .and_then(|raw| serde_json::from_str(raw).ok())
                    .unwrap_or(Value::Null);
                let (outward, model_payload, made) = crate::bridge::search::execute(
                    &state.http,
                    &search_cfg,
                    &id,
                    &input,
                    meta,
                    &mut uses,
                )
                .await;
                search_requests += u64::from(made);
                results.push((outward, model_payload));
            }
            let assistant_content = message
                .get("content")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let round = crate::bridge::web_search_round_messages(&assistant_content, &results);
            let usage =
                crate::bridge::web_search_usage(std::slice::from_ref(&message), search_requests);
            let mut folded = Value::Object(serde_json::Map::new());
            folded["usage"] = usage;
            rounds.push(folded);
            messages.extend(round);
            outward_blocks.extend(results.iter().map(|(outward, _)| outward.clone()));
            last_completion = Some(completion);
            last_bytes = Some(bytes);
            continue;
        }

        // Terminal round: no further web_search calls. The `message` built
        // above is the final one; its usage already covers this round. The
        // executor's outward result blocks from every executed search ride
        // the content array, so the client sees what ran.
        let mut final_message = message;
        let mut all_rounds = rounds.clone();
        all_rounds.push(final_message.clone());
        let usage = crate::bridge::web_search_usage(&all_rounds, search_requests);
        final_message["usage"] = usage;
        let mut content = final_message
            .get("content")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        content.extend(outward_blocks);
        final_message["content"] = Value::Array(content);
        record_observations(ctx, &observe_buffered(&bytes));
        return json_response(
            status,
            Bytes::from(serde_json::to_vec(&final_message).expect("Anthropic message serializes")),
        );
    }

    // Iteration cap reached while the model still wanted to search: stop,
    // and surface the typed marker block instead of a silent cut-off.
    let base = last_completion
        .as_ref()
        .map(|c| {
            crate::bridge::chat_to_message_with_web_search(
                c,
                request_model,
                &payload.tool_meta,
                Some(meta),
            )
        })
        .unwrap_or_else(|| {
            let mut m = serde_json::json!({
                "id": "msg_bridge_cap",
                "type": "message",
                "role": "assistant",
                "model": request_model,
                "stop_reason": "tool_use",
                "stop_sequence": Value::Null,
            });
            m["content"] = Value::Array(Vec::new());
            m
        });
    let mut final_message = base;
    let cap_block = crate::bridge::web_search_cap_block();
    let mut content = final_message
        .get("content")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    content.push(cap_block);
    content.extend(outward_blocks);
    final_message["content"] = Value::Array(content);
    final_message["stop_reason"] = Value::String("tool_use".to_owned());
    // Every executed call already folded its usage into `rounds` (the cap
    // path only exists when all four calls wanted to search again), so no
    // extra round is added here — the last call must not count twice.
    let usage = crate::bridge::web_search_usage(&rounds, search_requests);
    final_message["usage"] = usage;
    if let Some(bytes) = &last_bytes {
        record_observations(ctx, &observe_buffered(bytes));
    }
    json_response(
        StatusCode::OK,
        Bytes::from(serde_json::to_vec(&final_message).expect("Anthropic message serializes")),
    )
}

/// Sticky-lane hint: hash the conversation's identity (model + the first two
/// messages — typically the system prompt and first user turn, stable across
/// every turn of an agent session) so a conversation keeps hitting the same key
/// while it has capacity, keeping any upstream prefix cache warm. Purely an
/// optimization; correctness never depends on which key serves a request.
fn affinity(body: &serde_json::Value, lanes: usize) -> Option<usize> {
    let messages = body.get("messages")?.as_array()?;
    let mut h = DefaultHasher::new();
    body.get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .hash(&mut h);
    for msg in messages.iter().take(2) {
        msg.to_string().hash(&mut h);
    }
    Some((h.finish() % lanes as u64) as usize)
}

/// Non-streaming: pace, retry, and return the upstream response verbatim.
#[allow(clippy::too_many_arguments)]
async fn buffered(
    state: Arc<AppState>,
    cfg: Arc<Config>,
    ctx: Ctx,
    method: Method,
    path_query: String,
    headers: HeaderMap,
    body: Bytes,
    prefer: Option<usize>,
    deadline: Instant,
) -> Response {
    let _active = crate::dispatch::scopeguard(|| gauge!("nimproxy_active_requests").decrement(1.0));
    gauge!("nimproxy_active_requests").increment(1.0);
    loop {
        // Two admission gates: a model-pressure permit (worker concurrency,
        // held through the whole upstream exchange — dropped on every exit
        // from this iteration), then an RPM slot.
        let Ok(_permit) = acquire_model_permit(&state, &cfg, &ctx, deadline, || true).await else {
            record_request(&ctx, "504");
            return gateway_timeout(&cfg, state.pool().len());
        };
        let Some(slot) = reserve_slot(&state, cfg.heartbeat, deadline, prefer, || true).await
        else {
            record_request(&ctx, "504");
            return gateway_timeout(&cfg, state.pool().len());
        };
        let sent_at = Instant::now();
        // A non-streaming request gets an overall timeout so a stalled body read
        // can't pin an in-flight slot forever (streaming has no such cap).
        let resp = match upstream_request(
            &state.http,
            &cfg.base_url,
            &method,
            &path_query,
            &headers,
            &slot.key,
            &body,
        )
        .timeout(cfg.request_timeout)
        .send()
        .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(lane = slot.lane, error = %e, "upstream connection error, retrying");
                enter_cooldown(&slot, "connect", Duration::from_secs(5));
                continue;
            }
        };
        if retryable(resp.status()) && Instant::now() < deadline {
            let status = resp.status();
            let backoff = backoff_for(&resp);
            // Sniff the error body: worker exhaustion is model-scoped (shared
            // across every key), so cooling down the lane would just burn healthy
            // key capacity on a failover that cannot help.
            let detail = resp.text().await.unwrap_or_default();
            if governor::is_worker_exhausted(&detail) {
                state
                    .governor
                    .note_exhausted(&ctx.model, cfg.governor.overrides.get(&ctx.model).copied());
                continue; // permit drops here; re-admission waits out the drain
            }
            tracing::info!(lane = slot.lane, %status, ?backoff, "lane in cooldown, retrying");
            enter_cooldown(&slot, status.as_str(), backoff);
            continue;
        }
        histogram!("nimproxy_upstream_seconds", "model" => ctx.model.clone())
            .record(sent_at.elapsed().as_secs_f64());
        record_request(&ctx, resp.status().as_str());
        return relay(resp, &ctx).await;
    }
}

/// Streaming: commit to a 200 SSE response immediately and emit `: heartbeat`
/// comment lines (ignored by every OpenAI SSE client) while we wait for a
/// slot or ride out 429/5xx, then pipe the upstream stream through. On the
/// Messages bridge the translator turns upstream chunks into Anthropic
/// events and drops the comment frames once the stream commits (T6).
#[allow(clippy::too_many_arguments)]
fn streaming(
    state: Arc<AppState>,
    cfg: Arc<Config>,
    ctx: Ctx,
    method: Method,
    path_query: String,
    headers: HeaderMap,
    mut body: Bytes,
    prefer: Option<usize>,
    mut fallback: Option<Bytes>,
    inflight_guard: impl Send + 'static,
    request_deadline: Option<RequestDeadline>,
    deadline: Instant,
    anthropic_sse: Option<crate::bridge::StreamMeta>,
) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(16);
    // The bridge endpoint (T6) translates upstream chunks into Anthropic
    // events; the OpenAI-wire chat path forwards bytes untouched (None).
    let translator: std::sync::Arc<std::sync::Mutex<Option<crate::bridge::StreamTranslator>>> =
        std::sync::Arc::new(std::sync::Mutex::new(
            anthropic_sse.map(crate::bridge::StreamTranslator::with_meta),
        ));

    tokio::spawn(async move {
        // Holds the handler's in-flight slot until this task — the request's
        // real lifetime — exits, so max_inflight bounds live streams too.
        let _inflight = inflight_guard;
        let _active =
            crate::dispatch::scopeguard(|| gauge!("nimproxy_active_requests").decrement(1.0));
        gauge!("nimproxy_active_requests").increment(1.0);
        let deadline_tx = tx.clone();
        let deadline_ctx = ctx.clone();
        let tr = translator.clone();
        let observer = Arc::new(Mutex::new(None));
        let deadline_observer = observer.clone();
        let work = async move {
            let send = |b: &'static str| {
                let tx = tx.clone();
                let tr = tr.clone();
                // Static control frames — no per-send alloc/copy. Bridge
                // streams (T6) stay silent once an Anthropic event
                // committed: heartbeats appear only before commitment.
                async move {
                    if let Some(t) = tr.lock().unwrap().as_mut() {
                        if !t.committed() {
                            // Uncommitted: control frames stay byte-exact
                            // for the replay-on-finish passthrough
                            // contract.
                            t.push(&Bytes::from_static(b.as_bytes()));
                        }
                        return true;
                    }
                    tx.send(Ok(Bytes::from_static(b.as_bytes()))).await.is_ok()
                }
            };
            if !send(": connected\n\n").await {
                record_request(&ctx, "disconnect");
                return;
            }
            loop {
                // Model-pressure permit first (worker concurrency), then an RPM
                // slot — both heartbeating so the harness doesn't hang up. The
                // permit spans the whole upstream exchange and drops on every
                // exit from this iteration.
                let Ok(_permit) = acquire_model_permit(&state, &cfg, &ctx, deadline, || {
                    send_control_frame(&tx, &tr)
                })
                .await
                else {
                    record_request(&ctx, "504");
                    let _ = tx
                        .send(Ok(sse_error(
                            "proxy timed out waiting for an upstream slot",
                        )))
                        .await;
                    return;
                };
                let slot = reserve_slot(&state, cfg.heartbeat, deadline, prefer, || {
                    send_control_frame(&tx, &tr)
                })
                .await;
                let Some(slot) = slot else {
                    record_request(&ctx, "504");
                    let _ = tx
                        .send(Ok(sse_error(
                            "proxy timed out waiting for an upstream slot",
                        )))
                        .await;
                    return;
                };

                let sent_at = Instant::now();
                let resp = match upstream_request(
                    &state.http,
                    &cfg.base_url,
                    &method,
                    &path_query,
                    &headers,
                    &slot.key,
                    &body,
                )
                .send()
                .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!(lane = slot.lane, error = %e, "upstream connection error, retrying");
                        enter_cooldown(&slot, "connect", Duration::from_secs(5));
                        continue;
                    }
                };

                // A 400 right after we injected stream_options usually means this
                // model rejects the field: remember that and retry untouched.
                if resp.status() == reqwest::StatusCode::BAD_REQUEST && fallback.is_some() {
                    tracing::info!(model = %ctx.model, "model rejected stream_options; retrying without injection");
                    state.no_inject.lock().unwrap().insert(ctx.model.clone());
                    body = fallback.take().unwrap();
                    continue;
                }

                if retryable(resp.status()) {
                    if Instant::now() >= deadline {
                        record_request(&ctx, "504");
                        let _ = tx
                            .send(Ok(sse_error("upstream unavailable, retries exhausted")))
                            .await;
                        return;
                    }
                    let status = resp.status();
                    let backoff = backoff_for(&resp);
                    // Worker exhaustion is model-scoped: back off the model via
                    // the governor, never the lane (see `buffered`).
                    let detail = resp.text().await.unwrap_or_default();
                    if governor::is_worker_exhausted(&detail) {
                        state.governor.note_exhausted(
                            &ctx.model,
                            cfg.governor.overrides.get(&ctx.model).copied(),
                        );
                    } else {
                        tracing::info!(lane = slot.lane, %status, ?backoff, "lane in cooldown, retrying");
                        enter_cooldown(&slot, status.as_str(), backoff);
                    }
                    if !send(": retrying\n\n").await {
                        record_request(&ctx, "disconnect");
                        return;
                    }
                    continue;
                }

                if !resp.status().is_success() {
                    // Non-retryable upstream error after we already committed to
                    // SSE: surface it as an in-stream error event.
                    let status = resp.status();
                    let detail = resp.text().await.unwrap_or_default();
                    tracing::warn!(%status, "upstream rejected request");
                    record_request(&ctx, status.as_str());
                    let _ = tx
                        .send(Ok(stream_error_frame(
                            &tr,
                            "upstream_unavailable",
                            &format!("upstream error {status}: {detail}"),
                        )))
                        .await;
                    return;
                }

                *observer.lock().unwrap() = Some(SseObserver::default());
                let mut first_chunk: Option<Instant> = None;
                let mut chunks = resp.bytes_stream();
                loop {
                    // Two ways out of a blocked upstream read: the stall cutoff
                    // (a stalled upstream would otherwise hold the client
                    // forever), and the client hanging up (`tx.closed()`) — which
                    // must free the in-flight slot promptly, not at the cutoff
                    // (and with stream_idle 0 there is no cutoff: a hung upstream
                    // would pin the slot until restart).
                    let upstream_read = async {
                        if cfg.stream_idle.is_zero() {
                            Ok(chunks.next().await)
                        } else {
                            tokio::time::timeout(cfg.stream_idle, chunks.next()).await
                        }
                    };
                    let next = tokio::select! {
                        _ = tx.closed() => {
                            finalize_sse_observer(&ctx, &observer, StreamOutcome::Disconnected);
                            record_request(&ctx, "disconnect");
                            return;
                        }
                        read = upstream_read => match read {
                            Ok(n) => n,
                            Err(_) => {
                                finalize_sse_observer(&ctx, &observer, StreamOutcome::Truncated);
                                tracing::warn!(model = %ctx.model, idle = ?cfg.stream_idle, "upstream stream stalled");
                                record_request(&ctx, "stall");
                                let _ = tx
                                    .send(Ok(stream_error_frame(
                                        &tr,
                                        "upstream_unavailable",
                                        "upstream stream stalled",
                                    )))
                                    .await;
                                return;
                            }
                        }
                    };
                    let Some(chunk) = next else { break };
                    match chunk {
                        Ok(b) => {
                            if first_chunk.is_none() {
                                first_chunk = Some(Instant::now());
                                histogram!("nimproxy_ttft_seconds", "model" => ctx.model.clone())
                                    .record(sent_at.elapsed().as_secs_f64());
                            }
                            observer
                                .lock()
                                .unwrap()
                                .as_mut()
                                .expect("stream observer initialized")
                                .push(&b);
                            // T6: bridge streams hand the client Anthropic
                            // events, never raw upstream chunks; the chat
                            // path forwards the chunk untouched.
                            let frames = {
                                let mut guard = tr.lock().unwrap();
                                match guard.as_mut() {
                                    Some(t) => t.push(&b),
                                    None => vec![b],
                                }
                            };
                            for frame in frames {
                                if tx.send(Ok(frame)).await.is_err() {
                                    finalize_sse_observer(
                                        &ctx,
                                        &observer,
                                        StreamOutcome::Disconnected,
                                    );
                                    record_request(&ctx, "disconnect");
                                    return; // client hung up
                                }
                            }
                        }
                        Err(e) => {
                            finalize_sse_observer(&ctx, &observer, StreamOutcome::Truncated);
                            tracing::warn!(error = %e, "upstream stream broke mid-response");
                            record_request(&ctx, "stream_error");
                            let _ = tx
                                .send(Ok(stream_error_frame(
                                    &tr,
                                    "upstream_unavailable",
                                    "upstream stream interrupted",
                                )))
                                .await;
                            return;
                        }
                    }
                }

                // T6: close the Anthropic stream (message_delta +
                // message_stop); an uncommitted stream replays its retained
                // upstream bytes instead — the passthrough contract.
                let tail = tr
                    .lock()
                    .unwrap()
                    .as_mut()
                    .map_or(Vec::new(), |t| t.finish());
                for frame in tail {
                    if tx.send(Ok(frame)).await.is_err() {
                        finalize_sse_observer(&ctx, &observer, StreamOutcome::Disconnected);
                        record_request(&ctx, "disconnect");
                        return;
                    }
                }
                let completion = finalize_sse_observer(&ctx, &observer, StreamOutcome::Completed);
                if let (Some(first), Some((c, source))) = (first_chunk, completion) {
                    let gen_secs = first.elapsed().as_secs_f64();
                    if gen_secs > 0.1 && c > 0 {
                        histogram!("nimproxy_tokens_per_second", "model" => ctx.model.clone(), "source" => source)
                        .record(c as f64 / gen_secs);
                        // Mean inter-token latency (time-per-output-token).
                        histogram!("nimproxy_tpot_seconds", "model" => ctx.model.clone())
                            .record(gen_secs / c as f64);
                    }
                }
                // Total upstream time for streaming, for parity with the buffered
                // path (which records upstream_seconds directly).
                histogram!("nimproxy_upstream_seconds", "model" => ctx.model.clone())
                    .record(sent_at.elapsed().as_secs_f64());
                record_request(&ctx, "200");
                return;
            }
        };
        if let Some(request_deadline) = request_deadline {
            tokio::select! {
                _ = tokio::time::sleep_until(request_deadline.0.into()) => {
                    finalize_sse_observer(&deadline_ctx, &deadline_observer, StreamOutcome::Deadline);
                    record_deadline(&deadline_ctx);
                    let _ = deadline_tx.try_send(Ok(stream_error_frame(
                        &translator,
                        "deadline_exceeded",
                        "proxy request deadline exceeded",
                    )));
                }
                _ = work => {}
            }
        } else {
            work.await;
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap()
}

/// /v1/models, cached so harness catalog polls cost zero rate budget. The
/// lock is held across the refresh so concurrent misses make one upstream
/// call (followers see the fresh cache when they get the lock).
async fn models(state: Arc<AppState>, cfg: Arc<Config>) -> Response {
    let mut cache = state.models_cache.lock().await;
    if let Some((at, body)) = cache.as_ref() {
        if at.elapsed() < cfg.models_ttl {
            return json_response(StatusCode::OK, body.clone());
        }
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    let Some(slot) = reserve_slot(&state, cfg.heartbeat, deadline, None, || true).await else {
        return gateway_timeout(&cfg, state.pool().len());
    };
    match fetch_models(&state.http, &cfg.base_url, &slot.key).await {
        Ok(resp) if resp.status().is_success() => {
            let body = resp.bytes().await.unwrap_or_default();
            *cache = Some((Instant::now(), body.clone()));
            json_response(StatusCode::OK, body)
        }
        Ok(resp) => {
            if retryable(resp.status()) {
                enter_cooldown(&slot, resp.status().as_str(), backoff_for(&resp));
            }
            let status =
                StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let body = resp.bytes().await.unwrap_or_default();
            json_response(status, body)
        }
        Err(e) => {
            tracing::warn!(error = %e, "models fetch failed");
            gateway_timeout(&cfg, state.pool().len())
        }
    }
}

/// The raw model-catalog fetch with an explicit key — shared by the cached
/// `/v1/models` path above and the setup wizard's key-validation probe
/// (which must bypass both the pool and the cache).
pub async fn fetch_models(
    http: &reqwest::Client,
    base_url: &str,
    key: &str,
) -> reqwest::Result<reqwest::Response> {
    http.get(format!("{base_url}/v1/models"))
        .bearer_auth(key)
        .send()
        .await
}

/// Return an upstream response to the client as-is, harvesting the `usage`
/// object for token accounting on the way past.
async fn relay(resp: reqwest::Response, ctx: &Ctx) -> Response {
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_owned();
    let body = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            // Body stalled past the request timeout, or the connection dropped
            // mid-body. Surface a clear gateway error rather than a truncated
            // "success" with an empty body.
            tracing::warn!(error = %e, "upstream body read failed");
            return bad_gateway();
        }
    };
    if status.is_success() {
        record_observations(ctx, &observe_buffered(&body));
    }
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, content_type)
        .body(Body::from(body))
        .unwrap()
}

/// The proxy's standard error envelope: `{"error":{message,type,code}}`.
fn proxy_error_json(code: &str, message: impl Into<String>) -> serde_json::Value {
    serde_json::json!({
        "error": { "message": message.into(), "type": "proxy_error", "code": code }
    })
}

fn sse_error_with_code(code: &str, message: &str) -> Bytes {
    let event = proxy_error_json(code, message);
    Bytes::from(format!("data: {event}\n\ndata: [DONE]\n\n"))
}

fn sse_error(message: &str) -> Bytes {
    sse_error_with_code("upstream_unavailable", message)
}

/// Terminal in-stream error frame: the OpenAI-wire proxy envelope on the
/// chat path; once the bridge translator committed to an Anthropic event,
/// an Anthropic `error` event instead (T6: after `message_start` the
/// client no longer parses OpenAI SSE).
fn stream_error_frame(
    translator: &std::sync::Arc<std::sync::Mutex<Option<crate::bridge::StreamTranslator>>>,
    code: &str,
    message: &str,
) -> Bytes {
    if let Some(t) = translator.lock().unwrap().as_mut() {
        if t.committed() {
            return t.error_event(code, message);
        }
    }
    sse_error_with_code(code, message)
}

/// Wait-loop heartbeat (`: heartbeat` comment frame) for the wait/permit
/// loops. Bridge streams translate through the translator, so a heartbeat
/// appears only before the stream commits: retained byte-exact while
/// uncommitted, dropped structurally afterwards (T6). Chat streams forward
/// the frame to the client directly.
fn send_control_frame(
    tx: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    translator: &std::sync::Arc<std::sync::Mutex<Option<crate::bridge::StreamTranslator>>>,
) -> bool {
    let (committed, is_bridge) = {
        let mut guard = translator.lock().unwrap();
        match guard.as_mut() {
            Some(t) => (t.committed(), true),
            None => (false, false),
        }
    };
    if !is_bridge {
        return tx
            .try_send(Ok(Bytes::from_static(b": heartbeat\n\n")))
            .is_ok();
    }
    if !committed {
        let mut guard = translator.lock().unwrap();
        if let Some(t) = guard.as_mut() {
            t.push(&Bytes::from_static(b": heartbeat\n\n"));
        }
    }
    true
}

fn json_response(status: StatusCode, body: Bytes) -> Response {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap()
}

fn unauthorized() -> Response {
    let body = proxy_error_json(
        "unauthorized",
        "missing or invalid proxy API key (Authorization: Bearer ...)",
    );
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer")],
        axum::Json(body),
    )
        .into_response()
}

fn invalid_deadline() -> Response {
    let body = proxy_error_json(
        "invalid_deadline",
        "X-Nim-Proxy-Deadline-Ms must be one unsigned decimal millisecond value",
    );
    (StatusCode::BAD_REQUEST, axum::Json(body)).into_response()
}

fn deadline_exceeded() -> Response {
    let body = proxy_error_json("deadline_exceeded", "proxy request deadline exceeded");
    (StatusCode::GATEWAY_TIMEOUT, axum::Json(body)).into_response()
}

fn overloaded(max_inflight: usize) -> Response {
    let body = proxy_error_json(
        "overloaded",
        format!("proxy at capacity ({max_inflight} concurrent requests); retry shortly"),
    );
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::RETRY_AFTER, "5")],
        axum::Json(body),
    )
        .into_response()
}

fn bad_gateway() -> Response {
    let body = proxy_error_json("bad_gateway", "upstream response failed or timed out");
    (StatusCode::BAD_GATEWAY, axum::Json(body)).into_response()
}

fn gateway_timeout(cfg: &Config, pool_len: usize) -> Response {
    let body = proxy_error_json(
        "rate_limited",
        format!(
            "no upstream slot became available within {}s (all {} keys saturated)",
            cfg.max_wait.as_secs(),
            pool_len
        ),
    );
    (StatusCode::GATEWAY_TIMEOUT, axum::Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::{
        bounded_label, count_tools, is_json_mode, label_path, sanitize_label, tool_choice_mode,
    };
    use std::collections::HashSet;

    #[test]
    fn sanitize_strips_injection_chars() {
        // Quotes, braces, angle brackets, newlines, ANSI escapes all removed.
        assert_eq!(sanitize_label("meta/llama-3.3-70b"), "meta/llama-3.3-70b");
        assert_eq!(sanitize_label("a\"} fake_metric 1"), "afake_metric1");
        assert_eq!(
            sanitize_label("<img src=x onerror=alert(1)>"),
            "imgsrcxonerroralert1"
        );
        assert_eq!(sanitize_label("line1\nline2"), "line1line2");
        assert_eq!(sanitize_label("\x1b[31mred"), "31mred");
        assert_eq!(sanitize_label(""), "none");
        assert_eq!(sanitize_label("!!!"), "none");
    }

    #[test]
    fn sanitize_caps_length() {
        let long = "a".repeat(200);
        assert_eq!(sanitize_label(&long).len(), 64);
    }

    #[test]
    fn bounded_label_caps_cardinality() {
        let mut seen = HashSet::new();
        assert_eq!(bounded_label(&mut seen, "m1".into(), 2), "m1");
        assert_eq!(bounded_label(&mut seen, "m2".into(), 2), "m2");
        // Third distinct value exceeds the cap -> "other".
        assert_eq!(bounded_label(&mut seen, "m3".into(), 2), "other");
        // Already-seen values still pass through after the cap.
        assert_eq!(bounded_label(&mut seen, "m1".into(), 2), "m1");
    }

    #[test]
    fn path_label_is_allowlisted() {
        assert_eq!(label_path("/v1/chat/completions"), "/v1/chat/completions");
        assert_eq!(label_path("/v1/embeddings"), "/v1/embeddings");
        assert_eq!(label_path("/v1/anything-else"), "other");
        assert_eq!(label_path("/v1/../etc"), "other");
    }

    #[test]
    fn tool_choice_mode_maps_unknown_strings_to_other() {
        // auto / none / required / named are covered elsewhere; an unrecognized
        // string must collapse to the bounded "other" label, not pass through.
        assert_eq!(
            tool_choice_mode(&serde_json::json!({"tool_choice": "banana"})),
            "other"
        );
    }

    #[test]
    fn tool_choice_and_shape_readers() {
        let auto = serde_json::json!({"tools": [{}], "tool_choice": "auto"});
        assert_eq!(tool_choice_mode(&auto), "auto");
        let named = serde_json::json!({"tool_choice": {"type": "function"}});
        assert_eq!(tool_choice_mode(&named), "named");
        // tools present, no explicit choice -> provider default (auto)
        assert_eq!(
            tool_choice_mode(&serde_json::json!({"tools": [{}]})),
            "auto"
        );

        assert_eq!(
            count_tools(&serde_json::json!({"tools": [{}, {}, {}]})),
            Some(3)
        );
        assert_eq!(
            count_tools(&serde_json::json!({"functions": [{}]})),
            Some(1)
        );
        assert_eq!(count_tools(&serde_json::json!({"model": "x"})), None);

        assert!(is_json_mode(
            &serde_json::json!({"response_format": {"type": "json_object"}})
        ));
        assert!(!is_json_mode(
            &serde_json::json!({"response_format": {"type": "text"}})
        ));
        assert!(!is_json_mode(&serde_json::json!({"model": "x"})));
    }
}

/// Fuzzing-only surface (see fuzz/). Thin wrappers so the fuzz targets can
/// exercise the untrusted-byte parsers without widening the visibility of
/// the real items.
#[cfg(fuzzing)]
#[doc(hidden)]
pub mod fuzz {
    /// Drive the private SSE observer with arbitrary bytes, twice: once as a
    /// single chunk and once re-fragmented at an input-derived boundary. Must
    /// never panic; the observer bounds its one unfinished event internally.
    pub fn sse_scan(data: &[u8]) {
        let mut whole = crate::observation::SseObserver::default();
        whole.push(data);
        let _ = whole.finish(crate::observation::StreamOutcome::Completed);

        let step = data.first().map_or(3, |b| (*b as usize % 17) + 1);
        let mut frag = crate::observation::SseObserver::default();
        for chunk in data.chunks(step) {
            frag.push(chunk);
        }
        let _ = frag.finish(crate::observation::StreamOutcome::Completed);
    }

    /// The sanitizer's output invariants ARE the security property: bounded
    /// length, never empty, and only chars that are inert in the Prometheus
    /// exposition format, access logs, and persisted history.
    pub fn sanitize_label(data: &[u8]) {
        let raw = String::from_utf8_lossy(data);
        let out = super::sanitize_label(&raw);
        assert!(!out.is_empty(), "sanitized label must never be empty");
        assert!(out.len() <= 64, "sanitized label must be length-capped");
        assert!(
            out.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/' | ':')),
            "sanitized label must stay in the safe charset"
        );
    }
}
