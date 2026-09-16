//! Anthropic Messages bridge: request-side protocol conversion.
//!
//! Converts an Anthropic `POST /v1/messages` body into the OpenAI chat
//! payload the pipeline sends upstream, plus the request-side metadata the
//! response transform and metrics need (tool metadata, thinking mode).
//! The conversion is stateless; nothing here sees a pool key.
//! Unsupported Anthropic server-side capabilities are rejected with a
//! typed `BridgeError` (400), never silently dropped.

use serde_json::{json, Value};

/// Request-side rejection / validation error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BridgeError {
    pub code: &'static str,
    pub message: String,
}

impl BridgeError {
    fn new(code: &'static str, message: &'static str) -> Self {
        Self {
            code,
            message: message.to_string(),
        }
    }
}

/// Tool family a `tools[]` entry resolved into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolFamily {
    Bash,
    TextEditor,
    Memory,
    Computer,
    WebSearch,
    WebFetch,
    Custom,
}

/// Tool metadata carried from the request transform into the response
/// transform and the tool-type metric.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolMeta {
    pub family: ToolFamily,
}

/// Thinking mode resolved from the request. Full parameter validation
/// (budget vs max_tokens, temperature/top_p constraints) is the thinking
/// task; this carries the resolved mode.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ThinkingMeta {
    pub enabled: bool,
    pub budget_tokens: Option<u64>,
}

/// Result of the request transform.
#[derive(Clone, Debug)]
pub struct ChatPayload {
    /// The OpenAI chat-completions request body to send upstream.
    pub json: Value,
    // Forward-looking fields: the streaming translator (T6), the client tool
    // schemas (T7), and thinking validation (T8) consume these. T4 routes on
    // `json`/`tool_meta` only, so they stay unread in the lib build for now.
    #[allow(dead_code)]
    /// Normalized OpenAI function tool definitions.
    pub tools: Vec<Value>,
    /// Family per entry of `tools`, in order.
    pub tool_meta: Vec<ToolMeta>,
    #[allow(dead_code)]
    /// Mapped `tool_choice`, if present.
    pub tool_choice: Option<Value>,
    #[allow(dead_code)]
    /// `true` when `tool_choice` forces a specific tool.
    pub forced_tool: bool,
    #[allow(dead_code)]
    pub thinking: ThinkingMeta,
}

/// Id state for one request: synthesized tool-use ids and the lockstep
/// pairing of id-less `tool_result` blocks to them.
///
/// A `tool_use` without an `id` gets `toolu_bridge_N` from a per-request
/// counter (deterministic, never random). A `tool_result` without a
/// `tool_use_id`/`id` reuses the oldest not-yet-paired synthesized id,
/// because a valid conversation lists the `tool_use` before its
/// `tool_result`; only a result with no preceding call in the request
/// gets its own synthesized id.
struct IdState {
    next: u64,
    /// Synthesized ids not yet claimed by a `tool_result`.
    pending: Vec<String>,
}

impl IdState {
    fn new() -> Self {
        Self {
            next: 0,
            pending: vec![],
        }
    }

    fn synth(&mut self) -> String {
        let id = format!("toolu_bridge_{}", self.next);
        self.next += 1;
        id
    }

    /// Id for the `tool_use` side: an explicit id passes through
    /// untouched (invariant: identifiers stay frozen); an absent one is
    /// synthesized and queued for lockstep pairing.
    fn use_id(&mut self, block: &Value) -> String {
        let explicit = block
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_string);
        match explicit {
            Some(id) => id,
            None => {
                let id = self.synth();
                self.pending.push(id.clone());
                id
            }
        }
    }

    /// Id a `tool_result` pairs back to: `tool_use_id` first, then the
    /// block's own `id`; otherwise the oldest pending synthesized id;
    /// otherwise a fresh one (an id-less result with no id-less call
    /// ahead of it in the request).
    fn result_id(&mut self, block: &Value) -> String {
        let explicit = block
            .get("tool_use_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .or_else(|| {
                block
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .map(str::to_string)
            });
        if let Some(id) = explicit {
            return id;
        }
        if self.pending.is_empty() {
            self.synth()
        } else {
            self.pending.remove(0)
        }
    }
}

fn object(v: Option<&Value>) -> Option<&serde_json::Map<String, Value>> {
    v.and_then(Value::as_object)
}

/// Anthropic content, in all its wire shapes, as a block list: a string
/// becomes one text block, `null`/absent becomes none, and a block list is
/// normalized (string entries become text blocks).
fn content_blocks(content: Option<&Value>) -> Vec<Value> {
    match content {
        None | Some(Value::Null) => vec![],
        Some(Value::String(s)) => vec![json!({ "type": "text", "text": s })],
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                if let Some(s) = item.as_str() {
                    json!({ "type": "text", "text": s })
                } else {
                    item.clone()
                }
            })
            .collect(),
        Some(other) => vec![other.clone()],
    }
}

/// The text an Anthropic content block contributes when folded into a
/// chat message: text-family blocks yield their text, images/documents
/// yield a placeholder (the upstream model cannot see them through this
/// bridge), and redacted thinking contributes nothing.
fn block_text(block: &Value) -> Option<String> {
    match block.get("type").and_then(Value::as_str) {
        Some("text") | Some("input_text") | Some("output_text") => block
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_string),
        Some("image") => Some("[image content omitted]".into()),
        Some("document") => Some("[document content omitted]".into()),
        Some("redacted_thinking") => None,
        Some("thinking") => block
            .get("thinking")
            .and_then(Value::as_str)
            .map(str::to_string),
        _ => block
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_string),
    }
}

/// `system` accepts a string or a content-block array.
fn system_text(v: Option<&Value>) -> Option<String> {
    match v {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(items)) => {
            let joined: Vec<String> = items
                .iter()
                .filter_map(block_text)
                .filter(|t| !t.is_empty())
                .collect();
            let joined = joined.join("\n");
            (!joined.is_empty()).then_some(joined)
        }
        Some(other) => Some(other.to_string()),
    }
}

fn is_tool_result(block: &Value) -> bool {
    block
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|t| t == "tool_result" || t.ends_with("_tool_result"))
}

/// The result text a `tool_result` block maps to; a string result is
/// passed through, a block array is folded, and `is_error` results are
/// flagged so the model can see the failure.
fn tool_result_text(block: &Value) -> String {
    let raw = match block.get("content") {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(block_text)
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        Some(other) => other.to_string(),
    };
    if block
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        format!("tool returned an error: {raw}")
    } else {
        raw
    }
}

fn tool_use_to_call(block: &Value, ids: &mut IdState) -> Value {
    let id = ids.use_id(block);
    let name = block
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    // OpenAI `arguments` is a JSON string; the block's input is an object.
    let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
    let arguments = match input {
        Value::String(s) => s,
        other => other.to_string(),
    };
    json!({
        "id": id,
        "type": "function",
        "function": { "name": name, "arguments": arguments }
    })
}

fn tool_family(tool_type: Option<&str>) -> ToolFamily {
    match tool_type {
        Some(t) if t.starts_with("bash_") => ToolFamily::Bash,
        Some(t) if t.starts_with("text_editor_") => ToolFamily::TextEditor,
        Some(t) if t.starts_with("memory_") => ToolFamily::Memory,
        Some(t) if t.starts_with("computer_") => ToolFamily::Computer,
        Some(t) if t.starts_with("web_search_") => ToolFamily::WebSearch,
        Some(t) if t.starts_with("web_fetch_") => ToolFamily::WebFetch,
        _ => ToolFamily::Custom,
    }
}

/// Fail closed on a tool `type` the gateway cannot classify: unknown
/// server-side families (advisor_, code_execution_, ...) must be rejected,
/// not dropped or mislabeled `custom`. A missing/`custom` type is fine —
/// that is an ordinary client tool.
fn check_unknown_tool_type(tool_type: Option<&str>) -> Result<(), BridgeError> {
    if let Some(t) = tool_type {
        let known = t.starts_with("bash_")
            || t.starts_with("text_editor_")
            || t.starts_with("memory_")
            || t.starts_with("computer_")
            || t.starts_with("web_search_")
            || t.starts_with("web_fetch_")
            || t == "custom";
        if !known {
            return Err(reject_server_tool(t));
        }
    }
    Ok(())
}

fn family_default_name(family: ToolFamily) -> String {
    match family {
        ToolFamily::Bash => "bash".into(),
        ToolFamily::TextEditor => "text_editor".into(),
        ToolFamily::Memory => "memory".into(),
        ToolFamily::Computer => "computer".into(),
        ToolFamily::WebSearch => "web_search".into(),
        ToolFamily::WebFetch => "web_fetch".into(),
        ToolFamily::Custom => "tool".into(),
    }
}

fn reject_server_tool(type_or_name: &str) -> BridgeError {
    BridgeError {
        code: "server_tools_unsupported",
        message: format!(
            "server-side tool {type_or_name} is not supported by this gateway; use client tools"
        ),
    }
}

/// The caller gate: a tool restricted to `programmatic` callers needs a
/// server runtime this gateway does not have, so it is rejected, not
/// silently served.
fn check_allowed_callers(v: Option<&Value>) -> Result<(), BridgeError> {
    let Some(callers) = v else { return Ok(()) };
    let direct = matches!(
        callers,
        Value::String(s) if s == "direct"
    ) || callers
        .as_array()
        .is_some_and(|items| items.iter().any(|i| i.as_str() == Some("direct")));
    if direct {
        Ok(())
    } else {
        Err(BridgeError {
            code: "server_tools_unsupported",
            message: "tool allows only programmatic callers, which this gateway cannot execute"
                .into(),
        })
    }
}

fn resolve_thinking(v: Option<&Value>) -> ThinkingMeta {
    let Some(v) = v else {
        return ThinkingMeta::default();
    };
    let (kind, budget) = match v {
        Value::Bool(b) => (*b, None),
        Value::Object(o) => {
            let kind = match o.get("type").and_then(Value::as_str) {
                Some("enabled") => true,
                Some("disabled") => false,
                _ => o.get("enabled").and_then(Value::as_bool).unwrap_or(false),
            };
            let budget = o
                .get("budget_tokens")
                .or_else(|| o.get("budgetTokens"))
                .and_then(Value::as_u64);
            (kind, budget)
        }
        _ => (false, None),
    };
    ThinkingMeta {
        enabled: kind,
        budget_tokens: budget,
    }
}

/// `tool_choice` in the Anthropic dialect, in both the string and the
/// object forms, mapped to OpenAI. `any` and a named tool force a tool
/// call; `none` means no tools at all.
fn map_tool_choice(v: Option<&Value>) -> Result<(Option<Value>, bool), BridgeError> {
    let Some(v) = v else {
        return Ok((None, false));
    };
    Ok(match v {
        Value::String(s) => match s.as_str() {
            "auto" => (Some(json!("auto")), false),
            "any" | "required" => (Some(json!("required")), true),
            "none" => (Some(json!("none")), false),
            other => (Some(Value::String(other.to_string())), false),
        },
        Value::Object(o) => match o.get("type").and_then(Value::as_str) {
            Some("auto") => (Some(json!("auto")), false),
            Some("any") => (Some(json!("required")), true),
            Some("none") => (Some(json!("none")), false),
            Some("tool") => {
                let Some(name) = o.get("name").and_then(Value::as_str) else {
                    return Err(BridgeError::new(
                        "invalid_tool_choice",
                        "tool_choice {\"type\":\"tool\"} requires a name",
                    ));
                };
                (
                    Some(json!({
                        "type": "function",
                        "function": { "name": name }
                    })),
                    true,
                )
            }
            _ => (None, false),
        },
        _ => (None, false),
    })
}

/// Convert an Anthropic Messages request body to an OpenAI chat payload.
///
/// `system` (string or blocks) becomes a leading system message; assistant
/// `tool_use` blocks become `tool_calls`; user `tool_result` blocks become
/// `role: "tool"` messages flushed after any pending text; client tool
/// prefixes map onto OpenAI functions (the generated schemas arrive in the
/// tool-schema task — the family name is carried in `tool_meta`);
/// `stop_sequences` becomes `stop`.
///
/// Rejected with a typed error: `mcp_servers`, `mcp_toolset` tools, and
/// `allowed_callers` lacking `direct`.
pub fn to_chat_payload(body: &Value) -> Result<ChatPayload, BridgeError> {
    let mut ids = IdState::new();
    let Some(model) = body.get("model").and_then(Value::as_str) else {
        return Err(BridgeError::new("missing_model", "model is required"));
    };
    if body.get("mcp_servers").is_some_and(|v| !v.is_null()) {
        return Err(reject_server_tool("mcp_servers"));
    }

    let mut messages: Vec<Value> = vec![];
    if let Some(system) = system_text(body.get("system")) {
        messages.push(json!({ "role": "system", "content": system }));
    }

    for raw in body
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match raw {
            Value::String(text) => messages.push(json!({ "role": "user", "content": text })),
            Value::Object(msg) => {
                let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
                let blocks = content_blocks(msg.get("content"));
                if role == "assistant" {
                    let mut text: Vec<String> = vec![];
                    let mut thinking_text: Vec<String> = vec![];
                    let mut tool_calls: Vec<Value> = vec![];
                    for block in &blocks {
                        match block.get("type").and_then(Value::as_str) {
                            Some("tool_use") | Some("server_tool_use") => {
                                tool_calls.push(tool_use_to_call(block, &mut ids));
                            }
                            Some("thinking") => {
                                let t = block
                                    .get("thinking")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .trim()
                                    .to_string();
                                if !t.is_empty() {
                                    thinking_text.push(t);
                                }
                            }
                            Some("redacted_thinking") => {}
                            _ => {
                                if let Some(t) = block_text(block) {
                                    text.push(t);
                                }
                            }
                        }
                    }
                    let mut merged = String::new();
                    for t in thinking_text {
                        // tag preserved thinking so the upstream model can
                        // distinguish it from its own prior output
                        let chunk = format!("<thinking>{t}</thinking>");
                        if !merged.is_empty() {
                            merged.push('\n');
                        }
                        merged.push_str(&chunk);
                    }
                    for chunk in &text {
                        if !merged.is_empty() {
                            merged.push('\n');
                        }
                        merged.push_str(chunk);
                    }
                    let text = merged;
                    if text.is_empty() && tool_calls.is_empty() {
                        continue;
                    }
                    let content: Value = if tool_calls.is_empty() {
                        Value::String(text)
                    } else if text.is_empty() {
                        Value::Null
                    } else {
                        Value::String(text)
                    };
                    let mut m = json!({ "role": "assistant", "content": content });
                    if !tool_calls.is_empty() {
                        m["tool_calls"] = json!(tool_calls);
                    }
                    messages.push(m);
                } else {
                    let target_role = if role == "developer" || role == "system" {
                        "system"
                    } else {
                        "user"
                    };
                    let mut pending_text: Vec<String> = vec![];
                    let flush =
                        |messages: &mut Vec<Value>, pending: &mut Vec<String>, target: &str| {
                            let joined = pending.join("\n");
                            if !joined.is_empty() {
                                messages.push(json!({ "role": target, "content": joined }));
                            }
                            pending.clear();
                        };
                    for block in &blocks {
                        if is_tool_result(block) {
                            flush(&mut messages, &mut pending_text, target_role);
                            let id = ids.result_id(block);
                            let result = tool_result_text(block);
                            if result.is_empty() {
                                messages.push(json!({
                                    "role": "tool",
                                    "tool_call_id": id,
                                    "content": "(no output)"
                                }));
                            } else {
                                messages.push(json!({
                                    "role": "tool",
                                    "tool_call_id": id,
                                    "content": result,
                                }));
                            }
                        } else if let Some(t) = block_text(block) {
                            pending_text.push(t);
                        }
                    }
                    flush(&mut messages, &mut pending_text, target_role);
                }
            }
            _ => {}
        }
    }

    let mut tools: Vec<Value> = vec![];
    let mut tool_meta: Vec<ToolMeta> = vec![];
    for raw in body
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(tool) = object(Some(raw)) else {
            continue;
        };
        let tool_type = tool.get("type").and_then(Value::as_str);
        if matches!(tool_type, Some("mcp_toolset") | Some("mcp")) {
            return Err(reject_server_tool(tool_type.unwrap_or("mcp_toolset")));
        }
        check_unknown_tool_type(tool_type)?;
        check_allowed_callers(tool.get("allowed_callers"))?;
        let family = tool_family(tool_type);
        let name = tool
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| (family != ToolFamily::Custom).then(|| family_default_name(family)));
        let Some(name) = name else {
            // A custom tool with no name carries no callable schema.
            continue;
        };
        let schema = tool
            .get("input_schema")
            .cloned()
            .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
        tools.push(json!({
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description").cloned().unwrap_or(Value::Null),
                "parameters": schema
            }
        }));
        tool_meta.push(ToolMeta { family });
    }

    let (tool_choice, forced_tool) = map_tool_choice(body.get("tool_choice"))?;
    let thinking = resolve_thinking(body.get("thinking"));

    let mut payload = json!({ "model": model, "messages": messages });
    if let Some(v) = body.get("max_tokens").filter(|v| !v.is_null()) {
        payload["max_tokens"] = v.clone();
    }
    if let Some(v) = body.get("temperature").filter(|v| !v.is_null()) {
        payload["temperature"] = v.clone();
    }
    if let Some(v) = body.get("top_p").filter(|v| !v.is_null()) {
        payload["top_p"] = v.clone();
    }
    if let Some(v) = body.get("stop_sequences").filter(|v| !v.is_null()) {
        payload["stop"] = v.clone();
    }
    if let Some(v) = body.get("stream").filter(|v| !v.is_null()) {
        payload["stream"] = v.clone();
    }
    if let Some(v) = body.get("seed").filter(|v| !v.is_null()) {
        payload["seed"] = v.clone();
    }
    if !tools.is_empty() {
        payload["tools"] = json!(tools);
    }
    if let Some(tc) = &tool_choice {
        payload["tool_choice"] = tc.clone();
    }
    // `parallel_tool_use` is a top-level request field;
    // `tool_choice.disable_parallel_tool_use` inverts onto OpenAI's
    // `parallel_tool_calls`. Emitted only when tools are offered.
    if !tools.is_empty() {
        let parallel = body
            .get("parallel_tool_use")
            .and_then(Value::as_bool)
            .or_else(|| {
                body.get("tool_choice")
                    .and_then(|tc| tc.get("disable_parallel_tool_use"))
                    .and_then(Value::as_bool)
                    .map(|disabled| !disabled)
            });
        if let Some(p) = parallel {
            payload["parallel_tool_calls"] = json!(p);
        }
    }

    Ok(ChatPayload {
        json: payload,
        tools,
        tool_meta,
        tool_choice,
        forced_tool,
        thinking,
    })
}

/// Response-side transform (T3): upstream chat completion → Anthropic
/// message JSON returned to the Messages client. Pure — no I/O, no pool
/// keys. The request model name is echoed back, not the upstream one.
///
/// `stop_reason` mapping: `length` → `max_tokens`; `tool_calls` or any
/// finish reason with emitted tool_use blocks → `tool_use`; otherwise
/// `end_turn` (fail-closed: an unknown reason with no tool blocks still
/// reports a terminal stop).
pub fn chat_to_message(completion: &Value, req_model: &str) -> Value {
    let choices = completion
        .get("choices")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let choice = choices.first();

    let mut content: Vec<Value> = vec![];
    let mut tool_call_ids: Vec<String> = vec![];

    if let Some(choice) = choice {
        let message = choice.get("message").cloned().unwrap_or(Value::Null);
        if let Some(text) = message.get("content").and_then(Value::as_str) {
            if !text.is_empty() {
                content.push(json!({ "type": "text", "text": text }));
            }
        }
        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            for (i, call) in calls.iter().enumerate() {
                let func = call.get("function").cloned().unwrap_or(Value::Null);
                let name = func
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let arguments = func.get("arguments").cloned().unwrap_or(Value::Null);
                let input = match &arguments {
                    Value::String(raw) => {
                        serde_json::from_str(raw).unwrap_or_else(|_| arguments.clone())
                    }
                    Value::Null => json!({}),
                    other => other.clone(),
                };
                let explicit: Option<String> = call
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .map(str::to_string);
                let id = match explicit {
                    Some(id) => id,
                    None => format!("toolu_bridge_call_{i}"),
                };
                tool_call_ids.push(id.clone());
                content.push(json!({
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": input
                }));
            }
        }
    }

    let finish = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(Value::as_str);
    let stop_reason = if finish == Some("length") {
        "max_tokens"
    } else if finish == Some("tool_calls") || (!tool_call_ids.is_empty() && finish != Some("stop"))
    {
        "tool_use"
    } else {
        "end_turn"
    };

    let usage = completion.get("usage").cloned().unwrap_or(Value::Null);
    let in_tok = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let out_tok = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let upstream_id = completion
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .unwrap_or("");
    let msg_id = format!("msg_{upstream_id}");

    json!({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": req_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": Value::Null,
        "usage": { "input_tokens": in_tok, "output_tokens": out_tok }
    })
}

#[cfg(test)]
mod tests {
    fn cm(completion: &Value) -> Value {
        chat_to_message(completion, "client-model")
    }

    #[test]
    fn green_chat_to_message_minimal_text() {
        let m = cm(&json!({
            "id": "chatcmpl-1",
            "model": "nim-m",
            "choices": [{ "message": { "content": "hi" }, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 5, "completion_tokens": 7 }
        }));
        assert_eq!(m["id"], "msg_chatcmpl-1");
        assert_eq!(m["type"], "message");
        assert_eq!(m["role"], "assistant");
        assert_eq!(
            m["model"], "client-model",
            "request model echoed, not upstream"
        );
        assert_eq!(m["content"], json!([{ "type": "text", "text": "hi" }]));
        assert_eq!(m["stop_reason"], "end_turn");
        assert_eq!(m["stop_sequence"], Value::Null);
        assert_eq!(m["usage"], json!({ "input_tokens": 5, "output_tokens": 7 }));
    }

    #[test]
    fn green_chat_to_message_tool_calls() {
        let m = cm(&json!({
            "id": "chatcmpl-2",
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "bash", "arguments": "{\"cmd\":\"ls\"}" }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }));
        assert_eq!(
            m["content"],
            json!([
                { "type": "tool_use", "id": "call_1", "name": "bash", "input": { "cmd": "ls" } }
            ])
        );
        assert_eq!(m["stop_reason"], "tool_use");
    }

    #[test]
    fn green_chat_to_message_length_maps_to_max_tokens() {
        let m = cm(&json!({
            "choices": [{ "message": { "content": "trunc" }, "finish_reason": "length" }]
        }));
        assert_eq!(m["stop_reason"], "max_tokens");
    }

    #[test]
    fn green_chat_to_message_bad_arguments_preserved_as_string() {
        let m = cm(&json!({
            "choices": [{
                "message": {
                    "tool_calls": [{ "id": "c", "function": { "name": "f", "arguments": "oops" } }]
                },
                "finish_reason": "tool_calls"
            }]
        }));
        assert_eq!(m["content"][0]["type"], "tool_use");
        assert_eq!(
            m["content"][0]["input"], "oops",
            "unparseable arguments kept as the raw string"
        );
    }

    #[test]
    fn green_chat_to_message_missing_usage_is_zero() {
        let m =
            cm(&json!({ "choices": [{ "message": { "content": "x" }, "finish_reason": "stop" }] }));
        assert_eq!(m["usage"], json!({ "input_tokens": 0, "output_tokens": 0 }));
    }

    #[test]
    fn green_chat_to_message_missing_upstream_id_fallback() {
        let m =
            cm(&json!({ "choices": [{ "message": { "content": "x" }, "finish_reason": "stop" }] }));
        assert_eq!(
            m["id"], "msg_",
            "deterministic fallback when upstream omits the id"
        );
    }

    #[test]
    fn green_chat_to_message_text_and_tool_calls_both() {
        let m = cm(&json!({
            "choices": [{
                "message": {
                    "content": "running now",
                    "tool_calls": [{ "id": "c2", "function": { "name": "f", "arguments": "1" } }]
                },
                "finish_reason": "content_filter"
            }]
        }));
        assert_eq!(
            m["content"][0],
            json!({ "type": "text", "text": "running now" })
        );
        assert_eq!(m["content"][1]["type"], "tool_use");
        assert_eq!(m["content"][1]["input"], 1);
        // unknown finish_reason with tool_use blocks present resolves to tool_use
        assert_eq!(m["stop_reason"], "tool_use");
    }

    #[test]
    fn green_chat_to_message_no_choices_is_empty_message() {
        let m = cm(&json!({ "id": "u", "choices": [] }));
        assert_eq!(m["content"], json!([]));
        assert_eq!(m["stop_reason"], "end_turn");
    }

    #[test]
    fn green_chat_to_message_call_id_normalized() {
        let m = cm(&json!({
            "choices": [{
                "message": {
                    "tool_calls": [
                        { "function": { "name": "a", "arguments": "{}" } },
                        { "id": "", "function": { "name": "b", "arguments": "{}" } }
                    ]
                },
                "finish_reason": "tool_calls"
            }]
        }));
        assert_eq!(m["content"][0]["id"], "toolu_bridge_call_0");
        assert_eq!(m["content"][1]["id"], "toolu_bridge_call_1");
    }

    use super::*;

    fn conv(body: &Value) -> ChatPayload {
        to_chat_payload(body).expect("transform succeeds")
    }

    #[test]
    fn red_model_is_required() {
        let r = to_chat_payload(&json!({ "messages": [] }));
        assert!(
            matches!(
                r,
                Err(BridgeError {
                    code: "missing_model",
                    ..
                })
            ),
            "{r:?}"
        );
    }

    #[test]
    fn green_minimal_chat_payload() {
        let p = conv(&json!({
            "model": "deepseek-ai/deepseek-r1",
            "max_tokens": 1024,
            "messages": [{ "role": "user", "content": "hello" }]
        }));
        assert_eq!(p.json["model"], "deepseek-ai/deepseek-r1");
        assert_eq!(p.json["max_tokens"], 1024);
        let msgs = p.json["messages"].as_array().expect("messages");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "hello");
        assert!(p.tools.is_empty());
        assert!(p.json.get("tools").is_none());
        assert!(!p.forced_tool);
    }

    #[test]
    fn green_system_string_and_blocks() {
        let p = conv(&json!({
            "model": "m",
            "system": "be brief",
            "messages": [{ "role": "user", "content": [{ "type": "text", "text": "hi" }] }]
        }));
        let msgs = p.json["messages"].as_array().expect("messages");
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "be brief");
        assert_eq!(msgs[1]["content"], "hi");

        let p = conv(&json!({
            "model": "m",
            "system": [{ "type": "text", "text": "a" }, { "type": "text", "text": "b" }],
            "messages": []
        }));
        let msgs = p.json["messages"].as_array().expect("messages");
        assert_eq!(msgs[0]["content"], "a\nb");
    }

    #[test]
    fn green_custom_tool_shapes() {
        let p = conv(&json!({
            "model": "m",
            "tools": [{
                "name": "get_weather",
                "description": "weather",
                "input_schema": { "type": "object", "properties": { "city": { "type": "string" } } }
            }],
            "messages": []
        }));
        assert_eq!(p.tools.len(), 1);
        assert_eq!(p.tools[0]["type"], "function");
        assert_eq!(p.tools[0]["function"]["name"], "get_weather");
        assert_eq!(p.tools[0]["function"]["parameters"]["type"], "object");
        assert_eq!(
            p.json["tools"].as_array().expect("tools on payload").len(),
            1
        );
        assert!(p.tool_meta.iter().all(|m| m.family == ToolFamily::Custom));
    }

    #[test]
    fn green_client_tool_prefix_families() {
        for (type_str, family) in [
            ("bash_20250124", ToolFamily::Bash),
            ("text_editor_20250728", ToolFamily::TextEditor),
            ("memory_20250818", ToolFamily::Memory),
            ("computer_use_20250124", ToolFamily::Computer),
            ("web_search_20250305", ToolFamily::WebSearch),
        ] {
            let p = conv(&json!({
                "model": "m",
                "tools": [{ "type": type_str, "name": "tool1" }],
                "messages": []
            }));
            assert_eq!(
                p.tool_meta.first().expect("meta").family,
                family,
                "{type_str}"
            );
        }
    }

    #[test]
    fn red_mcp_servers_rejected() {
        let r = to_chat_payload(&json!({
            "model": "m",
            "mcp_servers": { "github": {} },
            "messages": []
        }));
        assert!(
            matches!(
                r,
                Err(BridgeError {
                    code: "server_tools_unsupported",
                    ..
                })
            ),
            "{r:?}"
        );
    }

    #[test]
    fn red_mcp_toolset_rejected() {
        let r = to_chat_payload(&json!({
            "model": "m",
            "tools": [{ "type": "mcp_toolset", "mcp_server_name": "github" }],
            "messages": []
        }));
        assert!(
            matches!(
                r,
                Err(BridgeError {
                    code: "server_tools_unsupported",
                    ..
                })
            ),
            "{r:?}"
        );
    }

    #[test]
    fn red_programmatic_callers_rejected() {
        let r = to_chat_payload(&json!({
            "model": "m",
            "tools": [{
                "type": "bash_20250124",
                "name": "bash",
                "allowed_callers": ["programmatic"]
            }],
            "messages": []
        }));
        assert!(
            matches!(
                r,
                Err(BridgeError {
                    code: "server_tools_unsupported",
                    ..
                })
            ),
            "{r:?}"
        );
    }

    #[test]
    fn green_callers_including_direct_accepted() {
        let p = conv(&json!({
            "model": "m",
            "tools": [{
                "type": "bash_20250124",
                "name": "bash",
                "allowed_callers": ["direct", "programmatic"]
            }],
            "messages": []
        }));
        assert_eq!(p.tool_meta[0].family, ToolFamily::Bash);
    }

    #[test]
    fn green_assistant_tool_use_becomes_tool_calls() {
        let p = conv(&json!({
            "model": "m",
            "messages": [
              { "role": "user", "content": "list files" },
              { "role": "assistant", "content": [
                  { "type": "text", "text": "checking" },
                  { "type": "tool_use", "id": "toolu_1", "name": "bash", "input": { "command": "ls" } }
              ]},
              { "role": "user", "content": [
                  { "type": "tool_result", "tool_use_id": "toolu_1", "content": "a\nb" }
              ]}
            ]
        }));
        let msgs = p.json["messages"].as_array().expect("messages");
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "checking");
        assert_eq!(msgs[1]["tool_calls"].as_array().expect("calls").len(), 1);
        let call = &msgs[1]["tool_calls"][0];
        assert_eq!(call["id"], "toolu_1");
        assert_eq!(call["function"]["name"], "bash");
        assert_eq!(call["function"]["arguments"], "{\"command\":\"ls\"}");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "toolu_1");
        assert_eq!(msgs[2]["content"], "a\nb");
    }

    #[test]
    fn green_tool_use_without_id_gets_bridge_id() {
        let p = conv(&json!({
            "model": "m",
            "messages": [{
                "role": "assistant",
                "content": [{ "type": "tool_use", "name": "f", "input": {} }]
            }]
        }));
        let msgs = p.json["messages"].as_array().expect("messages");
        let id = msgs[0]["tool_calls"][0]["id"].as_str().expect("id");
        assert_eq!(id, "toolu_bridge_0");
    }

    #[test]
    fn green_tool_result_error_flag_prefixes() {
        let p = conv(&json!({
            "model": "m",
            "messages": [
              { "role": "assistant", "content": [
                  { "type": "tool_use", "id": "toolu_9", "name": "f", "input": {} }
              ]},
              { "role": "user", "content": [
                  { "type": "tool_result", "tool_use_id": "toolu_9", "content": "boom", "is_error": true }
              ]}
            ]
        }));
        let msgs = p.json["messages"].as_array().expect("messages");
        assert!(
            msgs[1]["content"]
                .as_str()
                .expect("tool content")
                .starts_with("tool returned an error"),
            "{:?}",
            msgs[1]["content"]
        );
    }

    #[test]
    fn green_tool_result_without_id_pairs_synthesized_id() {
        let p = conv(&json!({
            "model": "m",
            "messages": [
              {
                "role": "assistant",
                "content": [
                    { "type": "tool_use", "name": "f", "input": {} },
                    { "type": "text", "text": "calling f" }
                ]
              },
              {
                "role": "user",
                "content": [
                    { "type": "tool_result", "content": "note" },
                    { "type": "text", "text": "now what?" }
                ]
              }
            ]
        }));
        let msgs = p.json["messages"].as_array().expect("messages");
        // assistant message first (tool_use with no id -> synthesized),
        // then the tool message must pair to the same id, then the text.
        let call_id = msgs[0]["tool_calls"][0]["id"].as_str().expect("call id");
        assert_eq!(
            call_id, "toolu_bridge_0",
            "synthesized pair is deterministic per request"
        );
        assert!(call_id.starts_with("toolu_bridge_"), "{call_id}");
        let tool_msg = msgs.iter().find(|m| m["role"] == "tool").expect("tool msg");
        assert_eq!(tool_msg["tool_call_id"], call_id, "ids must pair");
        assert_eq!(tool_msg["content"], "note");
        let user_msg = msgs.iter().find(|m| m["role"] == "user").expect("user msg");
        assert_eq!(user_msg["content"], "now what?");
    }

    #[test]
    fn green_tool_result_without_id_standalone_gets_placeholder() {
        let p = conv(&json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [{ "type": "tool_result" }]
            }]
        }));
        let msgs = p.json["messages"].as_array().expect("messages");
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["content"], "(no output)");
    }

    #[test]
    fn red_unknown_server_tool_type_rejected() {
        for type_str in [
            "advisor_20250729",
            "code_execution_20250825",
            "python_20250414",
        ] {
            let r = to_chat_payload(&json!({
                "model": "m",
                "tools": [{ "type": type_str, "name": "x" }],
                "messages": []
            }));
            assert!(
                matches!(
                    r,
                    Err(BridgeError {
                        code: "server_tools_unsupported",
                        ..
                    })
                ),
                "{type_str}: {r:?}"
            );
        }
    }

    #[test]
    fn red_tool_choice_named_without_name_rejected() {
        let r = to_chat_payload(&json!({
            "model": "m",
            "tools": [{ "name": "f" }],
            "tool_choice": { "type": "tool" },
            "messages": []
        }));
        assert!(
            matches!(
                r,
                Err(BridgeError {
                    code: "invalid_tool_choice",
                    ..
                })
            ),
            "{r:?}"
        );
    }

    #[test]
    fn green_thinking_blocks_tagged_in_assistant_history() {
        let p = conv(&json!({
            "model": "m",
            "messages": [{
                "role": "assistant",
                "content": [
                    { "type": "thinking", "thinking": "hmm let me see" },
                    { "type": "text", "text": "answer" }
                ]
            }]
        }));
        let msgs = p.json["messages"].as_array().expect("messages");
        assert_eq!(
            msgs[0]["content"],
            "<thinking>hmm let me see</thinking>\nanswer"
        );
    }

    #[test]
    fn green_tool_choice_mapping() {
        let p = conv(
            &json!({ "model": "m", "tools": [{ "name": "f" }], "tool_choice": "any", "messages": [] }),
        );
        assert_eq!(p.tool_choice.as_ref().expect("choice"), &json!("required"));
        assert!(p.forced_tool);
        assert_eq!(p.json["tool_choice"], json!("required"));

        let p = conv(&json!({
            "model": "m", "tools": [{ "name": "f" }],
            "tool_choice": { "type": "tool", "name": "f" },
            "messages": []
        }));
        assert_eq!(p.json["tool_choice"]["function"]["name"], "f");
        assert!(p.forced_tool);

        let p = conv(
            &json!({ "model": "m", "tools": [{ "name": "f" }], "tool_choice": "none", "messages": [] }),
        );
        assert_eq!(
            p.json["tool_choice"],
            json!("none"),
            "none must ban tool calls, not vanish"
        );
        assert!(!p.forced_tool);
    }

    #[test]
    fn green_param_passthrough_and_thinking_presence() {
        let p = conv(&json!({
            "model": "m",
            "max_tokens": 32,
            "temperature": 0.3,
            "top_p": 0.9,
            "seed": 7,
            "stop_sequences": ["\n\n"],
            "thinking": { "type": "enabled", "budget_tokens": 256 },
            "messages": []
        }));
        assert_eq!(p.json["max_tokens"], 32);
        assert_eq!(p.json["temperature"], 0.3);
        assert_eq!(p.json["top_p"], 0.9);
        assert_eq!(p.json["seed"], 7);
        assert_eq!(p.json["stop"], json!(["\n\n"]));
        assert!(p.thinking.enabled);
        assert_eq!(p.thinking.budget_tokens, Some(256));
        assert!(p.json.get("thinking").is_none());
    }

    #[test]
    fn green_stream_flag_passes_through() {
        let p = conv(&json!({ "model": "m", "stream": true, "messages": [] }));
        assert_eq!(p.json["stream"], true);

        let p = conv(&json!({ "model": "m", "stream": false, "messages": [] }));
        assert_eq!(p.json["stream"], false);

        let p = conv(&json!({ "model": "m", "messages": [] }));
        assert!(p.json.get("stream").is_none());
    }

    #[test]
    fn green_parallel_tool_use_maps_to_openai() {
        let p = conv(&json!({
            "model": "m",
            "tools": [{ "name": "f" }],
            "parallel_tool_use": true,
            "messages": []
        }));
        assert_eq!(p.json["parallel_tool_calls"], true);

        let p = conv(&json!({
            "model": "m",
            "tools": [{ "name": "f" }],
            "tool_choice": { "type": "auto", "disable_parallel_tool_use": true },
            "messages": []
        }));
        assert_eq!(p.json["parallel_tool_calls"], false);
    }

    #[test]
    fn green_parallel_tool_use_ignored_without_tools() {
        let p = conv(&json!({ "model": "m", "parallel_tool_use": true, "messages": [] }));
        assert!(p.json.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn green_image_document_omitted() {
        let p = conv(&json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "image", "source": {} },
                    { "type": "text", "text": "what is this?" }
                ]
            }]
        }));
        let msgs = p.json["messages"].as_array().expect("messages");
        // user-side multi-part content folds into one joined string, the
        // OpenAI-compatible shape (chat upstreams don't take part arrays
        // for user messages through NIM).
        assert_eq!(msgs[0]["content"], "[image content omitted]\nwhat is this?");
    }

    #[test]
    fn green_developer_role_maps_to_system() {
        let p = conv(&json!({
            "model": "m",
            "messages": [{ "role": "developer", "content": "house rules" }]
        }));
        let msgs = p.json["messages"].as_array().expect("messages");
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "house rules");
    }
}
