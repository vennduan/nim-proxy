//! Anthropic Messages bridge: request-side protocol conversion.
//!
//! Converts an Anthropic `POST /v1/messages` body into the OpenAI chat
//! payload the pipeline sends upstream, plus the request-side metadata the
//! response transform and metrics need (tool metadata, thinking mode).
//! The conversion is stateless; nothing here sees a pool key.
//! Unsupported Anthropic server-side capabilities are rejected with a
//! typed `BridgeError` (400), never silently dropped.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub mod search;
mod sse;
pub use sse::{StreamMeta, StreamTranslator};

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
    /// Normalized name the tool offer reached the upstream under.
    pub name: String,
    pub family: ToolFamily,
}

/// Thinking mode resolved and validated from the request.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ThinkingMeta {
    pub enabled: bool,
    pub budget_tokens: Option<u64>,
    /// True when the request carried the interleaved-thinking beta with
    /// tools offered — the one case `budget_tokens >= max_tokens` is legal.
    pub interleaved: bool,
}

/// Floor on an enabled thinking budget.
pub const MIN_THINKING_BUDGET: u64 = 1024;

/// The beta flag that admits interleaved thinking with tools.
pub const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";

/// Result of the request transform.
#[derive(Clone, Debug)]
pub struct ChatPayload {
    /// The OpenAI chat-completions request body to send upstream.
    pub json: Value,
    // Forward-looking fields: the streaming translator (T6) consumes these;
    // the bridge route reads `json` and `tool_meta` only, so the rest stay
    // unread in the lib build.
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
    /// The validated thinking mode for the response transform.
    pub thinking: ThinkingMeta,
    /// Execution metadata for the offered `web_search` server tool (T10):
    /// `None` when the request offers no web_search. Drives the
    /// in-gateway execute-then-resend loop and the `server_tool_use`
    /// response transform.
    pub web_search: Option<search::WebSearchMeta>,
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
        // Anthropic's canonical client-tool name, so echoed `tool_use`
        // blocks match what the Anthropic API itself would have returned.
        ToolFamily::TextEditor => "str_replace_based_edit_tool".into(),
        ToolFamily::Memory => "memory".into(),
        ToolFamily::Computer => "computer".into(),
        ToolFamily::WebSearch => "web_search".into(),
        ToolFamily::WebFetch => "web_fetch".into(),
        ToolFamily::Custom => "tool".into(),
    }
}

/// Catalog-style description fallbacks for the client tool families: a tool
/// offer that carries no description still reaches the upstream with usable
/// copy, instead of a null.
fn client_tool_description(family: ToolFamily) -> Option<&'static str> {
    match family {
        ToolFamily::Bash => Some("Execute shell commands in a persistent bash session."),
        ToolFamily::TextEditor => Some("View and edit text files with command-based operations."),
        ToolFamily::Computer => Some(
            "Interact with a computer UI using screenshots, clicks, typing, keys, \
             scrolling, and drag actions.",
        ),
        ToolFamily::Memory => {
            Some("Read and edit persistent memory files with command-based operations.")
        }
        _ => None,
    }
}

/// An integer pair property: `view_range`, `coordinate`, and friends.
fn int_pair(description: &str) -> Value {
    json!({
        "type": "array",
        "items": { "type": "integer" },
        "minItems": 2,
        "maxItems": 2,
        "description": description
    })
}

fn bash_tool_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute in the persistent bash session."
            },
            "restart": {
                "type": "boolean",
                "description": "Restart the persistent bash session before running the next command."
            }
        }
    })
}

fn text_editor_tool_schema(tool_type: Option<&str>) -> Value {
    let legacy = matches!(
        tool_type,
        Some(t) if t.ends_with("20241022") || t.ends_with("20250124")
    );
    let mut commands = vec!["view", "create", "str_replace", "insert"];
    if legacy {
        commands.push("undo_edit");
    }
    json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": commands,
                "description": "The editor operation to perform."
            },
            "path": { "type": "string", "description": "Absolute or relative path to the target file." },
            "view_range": int_pair("Inclusive start/end line numbers for view operations."),
            "file_text": { "type": "string", "description": "Full file contents when creating a file." },
            "old_str": { "type": "string", "description": "Existing text to replace." },
            "new_str": { "type": "string", "description": "Replacement text for str_replace." },
            "insert_line": { "type": "integer", "description": "Line number to insert text before." },
            "insert_text": { "type": "string", "description": "Text to insert." }
        }
    })
}

fn memory_tool_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["view", "create", "str_replace", "insert", "delete", "rename"],
                "description": "The memory operation to perform under the memory directory."
            },
            "path": { "type": "string", "description": "Path to the memory file." },
            "new_path": { "type": "string", "description": "New path when renaming a memory file." },
            "view_range": int_pair("Inclusive start/end line numbers for view operations."),
            "file_text": { "type": "string", "description": "Full file contents when creating a memory file." },
            "old_str": { "type": "string", "description": "Existing text to replace." },
            "new_str": { "type": "string", "description": "Replacement text for str_replace." },
            "insert_line": { "type": "integer", "description": "Line number to insert text before." },
            "insert_text": { "type": "string", "description": "Text to insert." }
        }
    })
}

fn computer_tool_schema(tool_type: Option<&str>) -> Value {
    let mut actions = vec![
        "screenshot",
        "left_click",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "mouse_move",
        "left_click_drag",
        "left_mouse_down",
        "left_mouse_up",
        "scroll",
        "type",
        "key",
        "hold_key",
        "wait",
    ];
    if tool_type.is_some_and(|t| t.ends_with("20251124")) {
        actions.push("zoom");
    }
    json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": actions,
                "description": "The computer action to perform."
            },
            "coordinate": int_pair("X/Y coordinate for click and move actions."),
            "start_coordinate": int_pair("Start coordinate for drag actions."),
            "end_coordinate": int_pair("End coordinate for drag actions."),
            "text": { "type": "string", "description": "Text to type or zoom target text." },
            "key": { "type": "string", "description": "Keyboard key or key chord to press." },
            "duration": { "type": "number", "description": "Optional wait duration in seconds." },
            "scroll_direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
                "description": "Scroll direction."
            },
            "scroll_amount": { "type": "integer", "description": "Scroll distance in pixels or wheel units." },
            "region": {
                "type": "array",
                "items": { "type": "integer" },
                "minItems": 4,
                "maxItems": 4,
                "description": "Optional region [left, top, width, height] for screenshots."
            },
            "modifiers": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Modifier keys to hold during the action."
            }
        }
    })
}

/// The generated `parameters` for a client tool family, when the request
/// offers no explicit `input_schema`. Server-side and custom families have
/// no generated schema — their offers fall back to the empty object.
fn client_tool_parameters(family: ToolFamily, tool_type: Option<&str>) -> Option<Value> {
    match family {
        ToolFamily::Bash => Some(bash_tool_schema()),
        ToolFamily::TextEditor => Some(text_editor_tool_schema(tool_type)),
        ToolFamily::Memory => Some(memory_tool_schema()),
        ToolFamily::Computer => Some(computer_tool_schema(tool_type)),
        // The server-executed family: the model is told it runs in-gateway,
        // with the one documented input the executor reads (T10).
        ToolFamily::WebSearch => Some(web_search_tool_schema()),
        _ => None,
    }
}

fn web_search_tool_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The web search query to execute."
            }
        },
        "required": ["query"]
    })
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

/// Resolve and validate the `thinking` param. The rules ported from
/// nim4cc: enabled requires an integer `budget_tokens` at or above
/// `MIN_THINKING_BUDGET`; a custom `temperature` is rejected; `top_p` is
/// restricted to [0.95, 1.0]; forced `tool_choice` and an assistant
/// prefill (last message is assistant with any content besides
/// redacted_thinking) are rejected; `budget_tokens >= max_tokens` is
/// rejected unless the interleaved-thinking beta is present with tools.
/// Single call site passing the request's own fields; not a struct.
#[allow(clippy::too_many_arguments)]
pub fn resolve_thinking(
    v: Option<&Value>,
    max_tokens: Option<&Value>,
    temperature: Option<&Value>,
    top_p: Option<&Value>,
    forced_tool: bool,
    tools: Option<&Value>,
    messages: Option<&Value>,
    beta: &str,
) -> Result<ThinkingMeta, BridgeError> {
    fn invalid(code: &'static str, message: &str) -> BridgeError {
        BridgeError {
            code,
            message: message.to_string(),
        }
    }
    let interleaved = beta
        .split(',')
        .map(|f| f.trim())
        .any(|f| f == INTERLEAVED_THINKING_BETA);
    let offers_tools = tools
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty());
    let interleaved = interleaved && offers_tools;
    let Some(v) = v else {
        return Ok(ThinkingMeta {
            interleaved,
            ..Default::default()
        });
    };
    let (kind, budget) = match v {
        Value::Bool(b) => (*b, None),
        Value::Object(o) => {
            let kind = match o.get("type").and_then(Value::as_str) {
                Some("enabled" | "enable" | "on" | "true") => true,
                Some("disabled" | "disable" | "off" | "false") => false,
                Some(t) => {
                    return Err(invalid(
                        "invalid_thinking",
                        &format!("thinking.type only supports enabled or disabled, got {t:?}"),
                    ))
                }
                None => o.get("enabled").and_then(Value::as_bool).unwrap_or(false),
            };
            let budget = o
                .get("budget_tokens")
                .or_else(|| o.get("budgetTokens"))
                .and_then(Value::as_u64);
            (kind, budget)
        }
        _ => {
            return Err(invalid(
                "invalid_thinking",
                "thinking must be a boolean or an object",
            ))
        }
    };
    if !kind {
        return Ok(ThinkingMeta::default());
    }
    let Some(budget) = budget else {
        return Err(invalid(
            "invalid_thinking",
            "thinking.enabled requires an integer budget_tokens",
        ));
    };
    if budget < MIN_THINKING_BUDGET {
        return Err(invalid(
            "invalid_thinking",
            &format!("thinking.budget_tokens must be at least {MIN_THINKING_BUDGET}"),
        ));
    }
    if let Some(max) = max_tokens.and_then(Value::as_u64) {
        if budget >= max && !interleaved {
            return Err(invalid(
                "invalid_thinking",
                "thinking.budget_tokens must be below max_tokens unless the \
                 interleaved-thinking beta is enabled with tools",
            ));
        }
    }
    if temperature.is_some() {
        return Err(invalid(
            "invalid_thinking",
            "thinking mode does not support a custom temperature",
        ));
    }
    if let Some(p) = top_p.and_then(Value::as_f64) {
        if !(0.95..=1.0).contains(&p) {
            return Err(invalid(
                "invalid_thinking",
                "thinking mode restricts top_p to 0.95..=1.0",
            ));
        }
    }
    if forced_tool {
        return Err(invalid(
            "invalid_thinking",
            "thinking mode does not support a forced tool choice",
        ));
    }
    if let Some(msgs) = messages.and_then(Value::as_array) {
        if let Some(last) = msgs.last() {
            let prefill = last
                .get("role")
                .and_then(Value::as_str)
                .is_some_and(|r| r == "assistant")
                && last
                    .get("content")
                    .and_then(Value::as_array)
                    .is_some_and(|blocks| {
                        blocks.iter().any(|b| {
                            b.get("type").and_then(Value::as_str) != Some("redacted_thinking")
                        })
                    });
            if prefill {
                return Err(invalid(
                    "invalid_thinking",
                    "thinking mode does not support an assistant prefill as the last message",
                ));
            }
        }
    }
    Ok(ThinkingMeta {
        enabled: true,
        budget_tokens: Some(budget),
        interleaved,
    })
}

/// The synthetic signature on an emitted `thinking` block: a tagged
/// SHA-256 digest of the request model and the thinking text. It is
/// documented as synthetic — this is an identity marker, not a
/// cryptographic attestation of the thinking.
fn synthetic_thinking_signature(model: &str, text: &str) -> String {
    use base64::{engine::general_purpose::URL_SAFE, Engine};
    let digest = Sha256::digest(format!("{model}\n{text}").as_bytes());
    let encoded = URL_SAFE.encode(digest).trim_end_matches('=').to_string();
    format!("nimthinking_{encoded}")
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
/// Rejected with a typed error: `mcp_servers`, `mcp_toolset` tools,
/// `allowed_callers` lacking `direct`, and invalid `thinking`
/// combinations. `beta` is the raw `anthropic-beta` header value.
pub fn to_chat_payload(body: &Value, beta: &str) -> Result<ChatPayload, BridgeError> {
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
    let mut web_search: Option<search::WebSearchMeta> = None;
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
            .or_else(|| client_tool_parameters(family, tool_type))
            .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
        let description = tool
            .get("description")
            .cloned()
            .or_else(|| client_tool_description(family).map(|d| Value::String(d.to_string())))
            .unwrap_or(Value::Null);
        tools.push(json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": schema
            }
        }));
        tool_meta.push(ToolMeta {
            name: name.clone(),
            family,
        });
        if family == ToolFamily::WebSearch {
            // The executor's per-request caps: `max_uses` bounds how many
            // searches this request may run; `user_location` is folded
            // into the query. Non-conforming values are dropped, never
            // guessed — the executor then fails the over-cap call with a
            // typed result block instead of searching.
            let max_uses = tool
                .get("max_uses")
                .and_then(Value::as_u64)
                .filter(|v| *v > 0 && *v <= u64::from(u32::MAX))
                .map(|v| v as u32);
            let user_location = tool.get("user_location").cloned().filter(|v| v.is_object());
            web_search = Some(search::WebSearchMeta {
                name: name.clone(),
                max_uses,
                user_location,
            });
        }
    }

    let (tool_choice, forced_tool) = map_tool_choice(body.get("tool_choice"))?;
    let thinking = resolve_thinking(
        body.get("thinking"),
        body.get("max_tokens"),
        body.get("temperature"),
        body.get("top_p"),
        forced_tool,
        body.get("tools"),
        body.get("messages"),
        beta,
    )?;

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
        web_search,
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
pub fn chat_to_message(completion: &Value, req_model: &str, tool_meta: &[ToolMeta]) -> Value {
    chat_to_message_with_web_search(completion, req_model, tool_meta, None)
}

/// The response transform with the in-gateway `web_search` executor
/// enabled (T10): an upstream tool call that names the offered web_search
/// tool comes back as a `server_tool_use` block (no `caller` echo — the
/// gateway runs it), not as a `tool_use`.
pub fn chat_to_message_with_web_search(
    completion: &Value,
    req_model: &str,
    tool_meta: &[ToolMeta],
    web_search: Option<&search::WebSearchMeta>,
) -> Value {
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
        // Reasoning models surface their chain of thought in
        // `reasoning_content`; that becomes a leading `thinking` block
        // carrying a synthetic signature (documented as such — an
        // identity marker, not cryptographic thinking integrity).
        if let Some(reasoning) = message.get("reasoning_content").and_then(Value::as_str) {
            let trimmed = reasoning.trim();
            if !trimmed.is_empty() {
                content.push(json!({
                    "type": "thinking",
                    "thinking": trimmed,
                    "signature": synthetic_thinking_signature(req_model, trimmed)
                }));
            }
        }
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
                let is_server_call = matches!(
                    web_search,
                    Some(meta) if meta.name == name
                );
                let mut block = json!({
                    "type": if is_server_call { "server_tool_use" } else { "tool_use" },
                    "id": id,
                    "name": name,
                    "input": input
                });
                // This gateway's tool_use blocks are all run by the caller,
                // so the direct caller is echoed (nim4cc's `caller:
                // {type: "direct"}`). The one exception is an in-gateway
                // `web_search` (T10): the call comes back as
                // `server_tool_use` and the gateway's executor — not the
                // client — runs it, so it carries no caller. Web-family
                // offers that the request did not pair with an executor
                // (no `web_search` metadata) keep the legacy tool_use.
                let client_offered = !is_server_call
                    && tool_meta.iter().any(|m| {
                        m.name == name
                            && !matches!(m.family, ToolFamily::WebSearch | ToolFamily::WebFetch)
                    });
                if client_offered {
                    block["caller"] = json!({ "type": "direct" });
                }
                content.push(block);
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

/// True while the executor's loop must continue: the upstream response's
/// first-choice `tool_calls` still names the offered web_search tool.
pub fn web_search_pending(completion: &Value, meta: &search::WebSearchMeta) -> bool {
    completion
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("tool_calls"))
        .and_then(Value::as_array)
        .is_some_and(|calls| {
            calls.iter().any(|call| {
                call.get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                    == Some(meta.name.as_str())
            })
        })
}

/// Fold one executor round back into the chat dialect: the assistant turn
/// (its text + every tool call, client or server) plus a `role: "tool"`
/// message per executed web_search result. The result's `tool_call_id` is
/// the `tool_use_id` the executor carried, and its content is the plaintext
/// payload the model parses (`{"query", "results"}` — the `results` array
/// with snippets, never the opaque outward block).
pub fn web_search_round_messages(
    assistant_content: &[Value],
    results: &[(Value, Value)],
) -> Vec<Value> {
    let mut ids = IdState::new();
    let mut text: Vec<String> = vec![];
    let mut thinking: Vec<String> = vec![];
    let mut calls: Vec<Value> = vec![];
    for block in assistant_content {
        match block.get("type").and_then(Value::as_str) {
            Some("thinking") => {
                let t = block
                    .get("thinking")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !t.is_empty() {
                    thinking.push(t);
                }
            }
            Some("text") => {
                let t = block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !t.is_empty() {
                    text.push(t);
                }
            }
            Some("tool_use") | Some("server_tool_use") => {
                calls.push(tool_use_to_call(block, &mut ids));
            }
            _ => {}
        }
    }
    let mut merged = String::new();
    for t in &thinking {
        // Tagged so the upstream model can tell preserved thinking apart
        // from its own — same fold as the request transform.
        let chunk = format!("<thinking>{t}</thinking>");
        if !merged.is_empty() {
            merged.push('\n');
        }
        merged.push_str(&chunk);
    }
    for t in &text {
        if !merged.is_empty() {
            merged.push('\n');
        }
        merged.push_str(t);
    }
    let mut assistant = json!({ "role": "assistant" });
    if !merged.is_empty() {
        assistant["content"] = Value::String(merged);
    }
    if !calls.is_empty() {
        assistant["tool_calls"] = json!(calls);
    }
    let mut out = vec![assistant];
    for (outward, model_payload) in results {
        let id = outward
            .get("tool_use_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        out.push(json!({
            "role": "tool",
            "tool_call_id": id,
            "content": model_payload.to_string(),
        }));
    }
    out
}

/// The loop-exhaustion marker: the iteration cap was hit while the model
/// still wanted web_search, so execution stops and the client sees this
/// typed error block instead of a silent stop.
pub fn web_search_cap_block() -> Value {
    json!({
        "type": "tool_result",
        "tool_use_id": "srvtoolu_bridge_cap",
        "content": [{
            "type": "tool_result_error",
            "error_code": "max_iterations_exceeded",
            "message": "the web_search iteration limit was reached; no further searches were executed"
        }],
        "is_error": true,
    })
}

/// One round's accumulated usage: the round messages' `input_tokens` and
/// `output_tokens` summed, plus the executor's provider requests under
/// `server_tool_use.web_search_requests` (emitted only when > 0 — the
/// official shape keeps the object absent for search-free rounds).
pub fn web_search_usage(rounds: &[Value], web_search_requests: u64) -> Value {
    let mut input = 0u64;
    let mut output = 0u64;
    for r in rounds {
        let usage = r.get("usage").cloned().unwrap_or(Value::Null);
        input += usage
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        output += usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
    }
    let mut usage = json!({ "input_tokens": input, "output_tokens": output });
    if web_search_requests > 0 {
        usage["server_tool_use"] = json!({ "web_search_requests": web_search_requests });
    }
    usage
}

#[cfg(test)]
mod tests {
    fn cm(completion: &Value) -> Value {
        chat_to_message(completion, "client-model", &[])
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
        to_chat_payload(body, "").expect("transform succeeds")
    }

    #[test]
    fn red_model_is_required() {
        let r = to_chat_payload(&json!({ "messages": [] }), "");
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
            let meta = p.tool_meta.first().expect("meta");
            assert_eq!(meta.family, family, "{type_str}");
            assert_eq!(meta.name, "tool1", "{type_str}: explicit name stored");
            // The explicit name rides the wire; the default name is only for
            // offers that name nothing.
            assert_eq!(p.tools[0]["function"]["name"], "tool1");
        }
    }

    #[test]
    fn green_family_default_names_without_offered_names() {
        for (type_str, default_name) in [
            ("bash_20250124", "bash"),
            ("text_editor_20250728", "str_replace_based_edit_tool"),
            ("memory_20250818", "memory"),
            ("computer_use_20250124", "computer"),
        ] {
            let p = conv(&json!({
                "model": "m",
                "tools": [{ "type": type_str }],
                "messages": []
            }));
            assert_eq!(p.tools[0]["function"]["name"], default_name, "{type_str}");
            assert_eq!(p.tool_meta[0].name, default_name, "{type_str}");
        }
    }

    #[test]
    fn red_mcp_servers_rejected() {
        let r = to_chat_payload(
            &json!({
                "model": "m",
                "mcp_servers": { "github": {} },
                "messages": []
            }),
            "",
        );
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
        let r = to_chat_payload(
            &json!({
                "model": "m",
                "tools": [{ "type": "mcp_toolset", "mcp_server_name": "github" }],
                "messages": []
            }),
            "",
        );
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
        let r = to_chat_payload(
            &json!({
                "model": "m",
                "tools": [{
                    "type": "bash_20250124",
                    "name": "bash",
                    "allowed_callers": ["programmatic"]
                }],
                "messages": []
            }),
            "",
        );
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
            let r = to_chat_payload(
                &json!({
                    "model": "m",
                    "tools": [{ "type": type_str, "name": "x" }],
                    "messages": []
                }),
                "",
            );
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
        let r = to_chat_payload(
            &json!({
                "model": "m",
                "tools": [{ "name": "f" }],
                "tool_choice": { "type": "tool" },
                "messages": []
            }),
            "",
        );
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
            "max_tokens": 4096,
            "top_p": 0.95,
            "seed": 7,
            "stop_sequences": ["\n\n"],
            "thinking": { "type": "enabled", "budget_tokens": 2048 },
            "messages": []
        }));
        assert_eq!(p.json["max_tokens"], 4096);
        assert_eq!(p.json["top_p"], 0.95);
        assert_eq!(p.json["seed"], 7);
        assert_eq!(p.json["stop"], json!(["\n\n"]));
        assert!(p.thinking.enabled);
        assert_eq!(p.thinking.budget_tokens, Some(2048));
        assert!(!p.thinking.interleaved);
        assert!(p.json.get("thinking").is_none());
    }

    #[test]
    fn green_thinking_reasoning_content_emits_leading_block() {
        let m = cm(&json!({
            "id": "chatcmpl-r1",
            "model": "nim-m",
            "choices": [{
                "message": { "reasoning_content": "  let me think  ", "content": "hi" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2 }
        }));
        let blocks = m["content"].as_array().expect("content");
        assert_eq!(blocks.len(), 2, "thinking leads, then text: {blocks:?}");
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["thinking"], "let me think", "trimmed, not raw");
        let sig = blocks[0]["signature"].as_str().expect("signature");
        assert!(sig.starts_with("nimthinking_"), "{sig:?}");
        assert!(!sig.contains('='), "urlsafe base64, padding stripped");
        assert_eq!(blocks[1], json!({ "type": "text", "text": "hi" }));
    }

    #[test]
    fn green_thinking_signature_is_deterministic_and_model_bound() {
        let sig = synthetic_thinking_signature("model-a", "text");
        assert_eq!(sig, synthetic_thinking_signature("model-a", "text"));
        assert_ne!(sig, synthetic_thinking_signature("model-b", "text"));
        assert_ne!(sig, synthetic_thinking_signature("model-a", "other"));
    }

    #[test]
    fn green_thinking_blank_reasoning_is_not_emitted() {
        let m = cm(&json!({
            "id": "chatcmpl-r2",
            "model": "nim-m",
            "choices": [{
                "message": { "reasoning_content": "   ", "content": "hi" },
                "finish_reason": "stop"
            }],
            "usage": {}
        }));
        let blocks = m["content"].as_array().expect("content");
        assert_eq!(blocks.len(), 1, "blank reasoning must not create a block");
        assert_eq!(blocks[0]["type"], "text");
    }

    #[test]
    fn green_interleaved_beta_licenses_budget_at_max_tokens() {
        let body = json!({
            "model": "m",
            "max_tokens": 2048,
            "thinking": { "type": "enabled", "budget_tokens": 2048 },
            "tools": [{ "name": "f" }],
            "messages": []
        });
        let p = to_chat_payload(&body, INTERLEAVED_THINKING_BETA)
            .expect("budget == max is legal under the interleaved beta");
        assert!(p.thinking.enabled);
        assert!(p.thinking.interleaved, "beta + tools offered");

        // Same request, no beta: the `>=` boundary must fall back to 400.
        let r = to_chat_payload(&body, "");
        assert!(
            matches!(
                r,
                Err(BridgeError {
                    code: "invalid_thinking",
                    ..
                })
            ),
            "{r:?}"
        );
    }

    #[test]
    fn green_beta_without_tools_keeps_strict_budget_rule() {
        let body = json!({
            "model": "m",
            "max_tokens": 1024,
            "thinking": { "type": "enabled", "budget_tokens": 1024 },
            "messages": []
        });
        let r = to_chat_payload(&body, INTERLEAVED_THINKING_BETA);
        assert!(
            matches!(
                r,
                Err(BridgeError {
                    code: "invalid_thinking",
                    ..
                })
            ),
            "the beta only relaxes the rule when tools are offered; {r:?}"
        );
    }

    #[test]
    fn green_thinking_disabled_is_a_no_op() {
        let p = conv(&json!({
            "model": "m",
            "thinking": { "type": "disabled", "budget_tokens": 0 },
            "messages": []
        }));
        assert_eq!(p.thinking, ThinkingMeta::default());
        assert!(!p.thinking.enabled);
    }

    #[test]
    fn red_thinking_budget_below_floor() {
        let r = to_chat_payload(
            &json!({
                "model": "m",
                "thinking": { "type": "enabled", "budget_tokens": 512 },
                "messages": []
            }),
            "",
        );
        assert!(
            matches!(
                r,
                Err(BridgeError {
                    code: "invalid_thinking",
                    ..
                })
            ),
            "{r:?}"
        );
    }

    #[test]
    fn red_thinking_enabled_requires_budget() {
        for v in [
            json!({ "type": "enabled" }),
            json!(true),
            json!({ "enabled": true }),
        ] {
            let r = to_chat_payload(&json!({ "model": "m", "thinking": v, "messages": [] }), "");
            assert!(
                matches!(
                    r,
                    Err(BridgeError {
                        code: "invalid_thinking",
                        ..
                    })
                ),
                "missing budget: {v:?} -> {r:?}"
            );
        }
    }

    #[test]
    fn red_thinking_malformed_shape_or_type() {
        for v in [json!("always"), json!({ "type": "sometimes" }), json!(null)] {
            let r = to_chat_payload(&json!({ "model": "m", "thinking": v, "messages": [] }), "");
            assert!(
                matches!(
                    r,
                    Err(BridgeError {
                        code: "invalid_thinking",
                        ..
                    })
                ),
                "malformed thinking: {v:?} -> {r:?}"
            );
        }
        // A null type reads as an object without an `enabled` flag: disabled.
        let p = conv(&json!({ "model": "m", "thinking": { "type": null }, "messages": [] }));
        assert!(!p.thinking.enabled);
    }

    #[test]
    fn red_thinking_rejects_temperature_and_top_p() {
        let r = to_chat_payload(
            &json!({
                "model": "m",
                "thinking": { "type": "enabled", "budget_tokens": 2048 },
                "temperature": 0.2,
                "messages": []
            }),
            "",
        );
        assert!(
            matches!(
                r,
                Err(BridgeError {
                    code: "invalid_thinking",
                    ..
                })
            ),
            "any temperature is rejected in thinking mode; {r:?}"
        );

        let r = to_chat_payload(
            &json!({
                "model": "m",
                "thinking": { "type": "enabled", "budget_tokens": 2048 },
                "top_p": 0.5,
                "messages": []
            }),
            "",
        );
        assert!(
            matches!(
                r,
                Err(BridgeError {
                    code: "invalid_thinking",
                    ..
                })
            ),
            "top_p below 0.95 is rejected; {r:?}"
        );

        // The 0.95 boundary is inclusive.
        let p = conv(&json!({
            "model": "m",
            "thinking": { "type": "enabled", "budget_tokens": 2048 },
            "top_p": 0.95,
            "messages": []
        }));
        assert!(p.thinking.enabled);
    }

    #[test]
    fn red_thinking_rejects_forced_tool_choice() {
        let r = to_chat_payload(
            &json!({
                "model": "m",
                "thinking": { "type": "enabled", "budget_tokens": 2048 },
                "tools": [{ "name": "f" }],
                "tool_choice": "any",
                "messages": []
            }),
            "",
        );
        assert!(
            matches!(
                r,
                Err(BridgeError {
                    code: "invalid_thinking",
                    ..
                })
            ),
            "forced tool choice is unsupported with thinking; {r:?}"
        );
    }

    #[test]
    fn red_thinking_rejects_assistant_prefill_but_not_redacted() {
        let r = to_chat_payload(
            &json!({
                "model": "m",
                "thinking": { "type": "enabled", "budget_tokens": 2048 },
                "messages": [
                    { "role": "user", "content": "q" },
                    { "role": "assistant", "content": [
                        { "type": "text", "text": "half-answered" }
                    ] }
                ]
            }),
            "",
        );
        assert!(
            matches!(
                r,
                Err(BridgeError {
                    code: "invalid_thinking",
                    ..
                })
            ),
            "assistant prefill as the last message is rejected; {r:?}"
        );

        // A lone redacted_thinking block is the one permitted prefill.
        let p = conv(&json!({
            "model": "m",
            "thinking": { "type": "enabled", "budget_tokens": 2048 },
            "messages": [
                { "role": "user", "content": "q" },
                { "role": "assistant", "content": [
                    { "type": "redacted_thinking", "data": "opaque" }
                ] }
            ]
        }));
        assert!(p.thinking.enabled);
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

    // T7: client tool schemas.

    /// Every emitted OpenAI tool must hold the function shape: `type`
    /// function, a string name, and an object parameter schema.
    fn assert_openai_function_shape(t: &Value, label: &str) {
        assert_eq!(t["type"], "function", "{label}");
        assert!(
            t["function"]["name"].is_string(),
            "{label}: name must be a string"
        );
        let params = &t["function"]["parameters"];
        assert_eq!(params["type"], "object", "{label}");
        assert!(
            params.get("properties").is_some_and(|p| p.is_object()),
            "{label}: object parameters carry a properties map"
        );
    }

    #[test]
    fn green_client_families_carry_generated_schemas() {
        let cases: &[(&str, &str, &str)] = &[
            ("bash_20250124", "bash", "command"),
            (
                "text_editor_20250728",
                "str_replace_based_edit_tool",
                "command",
            ),
            ("memory_20250818", "memory", "command"),
            ("computer_use_20250124", "computer", "action"),
        ];
        for &(type_str, default_name, prop) in cases {
            let p = conv(&json!({
                "model": "m",
                "tools": [{ "type": type_str }],
                "messages": []
            }));
            let tool = &p.tools[0];
            assert_openai_function_shape(tool, type_str);
            assert_eq!(
                tool["function"]["name"], default_name,
                "{type_str}: the family default name wins when the offer names nothing"
            );
            let desc = tool["function"]["description"]
                .as_str()
                .expect("description");
            assert!(
                !desc.is_empty(),
                "{type_str}: catalog description present: {desc}"
            );
            let props = tool["function"]["parameters"]["properties"]
                .as_object()
                .expect("properties");
            assert!(props.contains_key(prop), "{type_str}");
            // The generated schema is picked over the empty-object fallback.
            assert!(
                props.len() > 1,
                "{type_str}: generated family schema, not the empty fallback"
            );
        }
    }

    #[test]
    fn green_text_editor_undo_edit_only_for_legacy_types() {
        for (type_str, legacy) in [
            ("text_editor_20241022", true),
            ("text_editor_20250124", true),
            ("text_editor_20250728", false),
        ] {
            let p = conv(&json!({
                "model": "m",
                "tools": [{ "type": type_str }],
                "messages": []
            }));
            let commands = p.tools[0]["function"]["parameters"]["properties"]["command"]["enum"]
                .as_array()
                .expect("command enum");
            assert_eq!(
                commands.iter().any(|c| c == "undo_edit"),
                legacy,
                "{type_str}"
            );
        }
    }

    #[test]
    fn green_computer_zoom_only_for_20251124() {
        for (type_str, zoom) in [
            ("computer_use_20250124", false),
            ("computer_use_20251124", true),
        ] {
            let p = conv(&json!({
                "model": "m",
                "tools": [{ "type": type_str }],
                "messages": []
            }));
            let actions = p.tools[0]["function"]["parameters"]["properties"]["action"]["enum"]
                .as_array()
                .expect("action enum");
            assert_eq!(actions.iter().any(|a| a == "zoom"), zoom, "{type_str}");
        }
    }

    #[test]
    fn green_explicit_input_schema_and_description_win() {
        let p = conv(&json!({
            "model": "m",
            "tools": [{
                "type": "bash_20250124",
                "name": "my_bash",
                "description": "house flavor",
                "input_schema": { "type": "object", "properties": { "cmd": { "type": "string" } } }
            }],
            "messages": []
        }));
        let tool = &p.tools[0];
        assert_eq!(tool["function"]["name"], "my_bash", "explicit name wins");
        assert_eq!(
            tool["function"]["description"], "house flavor",
            "explicit description wins"
        );
        let props = tool["function"]["parameters"]["properties"]
            .as_object()
            .expect("props");
        assert_eq!(
            props.len(),
            1,
            "explicit input_schema wins over the generated one"
        );
        assert!(props.contains_key("cmd"));
    }

    #[test]
    fn green_caller_direct_echoed_for_client_tools() {
        let completion = json!({
            "id": "chatcmpl-c",
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": { "name": "bash", "arguments": "{\"command\":\"ls\"}" }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let meta = conv(&json!({
            "model": "m",
            "tools": [{ "type": "bash_20250124", "name": "bash" }],
            "messages": []
        }));
        let m = chat_to_message(&completion, "client-model", &meta.tool_meta);
        assert_eq!(
            m["content"],
            json!([{
                "type": "tool_use",
                "id": "call_1",
                "name": "bash",
                "input": { "command": "ls" },
                "caller": { "type": "direct" }
            }]),
            "a client-offered tool echoes the direct caller"
        );
        assert_eq!(m["stop_reason"], "tool_use");
    }

    #[test]
    fn green_caller_absent_without_client_tools() {
        let completion = json!({
            "id": "chatcmpl-c2",
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_2",
                        "function": { "name": "bash", "arguments": "{}" }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        // No tool offers at all: nothing names the call as caller-executed.
        let m = chat_to_message(&completion, "client-model", &[]);
        assert!(
            m["content"][0].get("caller").is_none(),
            "no client offer, no caller echo: {m}"
        );
        // A web_search offer is reserved for in-gateway execution: no echo.
        let web = conv(&json!({
            "model": "m",
            "tools": [{ "type": "web_search_20250305", "name": "bash" }],
            "messages": []
        }));
        let m = chat_to_message(&completion, "client-model", &web.tool_meta);
        assert!(
            m["content"][0].get("caller").is_none(),
            "web family offers never echo caller: {m}"
        );
    }

    #[test]
    fn green_tool_use_tool_calls_round_trip_is_identity() {
        // Anthropic tool_use -> OpenAI tool_calls -> Anthropic tool_use.
        let anthropic = json!({
            "type": "tool_use",
            "id": "toolu_abc",
            "name": "bash",
            "input": { "command": "ls -la", "restart": false }
        });
        let meta = vec![ToolMeta {
            name: "bash".into(),
            family: ToolFamily::Bash,
        }];
        let mut ids = IdState::new();
        let call = tool_use_to_call(&anthropic, &mut ids);
        assert_eq!(call["id"], "toolu_abc");
        assert_eq!(call["function"]["name"], "bash");
        assert_eq!(
            call["function"]["arguments"], r#"{"command":"ls -la","restart":false}"#,
            "arguments is the JSON-string input object"
        );
        let completion = json!({
            "id": "chatcmpl-rt",
            "choices": [{
                "message": { "tool_calls": [call] },
                "finish_reason": "tool_calls"
            }]
        });
        let m = chat_to_message(&completion, "client-model", &meta);
        assert_eq!(
            m["content"],
            json!([{
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "bash",
                "input": { "command": "ls -la", "restart": false },
                "caller": { "type": "direct" }
            }]),
            "the round-trip reproduces the tool_use block (plus the caller echo)"
        );
    }

    #[test]
    fn green_multi_turn_tool_round_trip_preserves_ordering() {
        // Anthropic -> chat: 3 parallel calls, results out of order (with a
        // block-array + is_error result), suffixed tool_result variants.
        let anthropic = json!({
            "model": "m",
            "max_tokens": 1024,
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
                {"name": "read_file"},
                {"name": "write_file"}
            ],
            "messages": [
              { "role": "user", "content": "list and read" },
              { "role": "assistant", "content": [
                  { "type": "text", "text": "running three" },
                  { "type": "tool_use", "id": "toolu_a", "name": "bash", "input": {"command": "ls"} },
                  { "type": "tool_use", "id": "toolu_b", "name": "read_file", "input": {"path": "a"} },
                  { "type": "tool_use", "id": "toolu_c", "name": "write_file", "input": {"path": "b", "text": "x"} }
              ]},
              { "role": "user", "content": [
                  // Out of order: c first, as a block array with is_error.
                  { "type": "write_file_tool_result", "tool_use_id": "toolu_c",
                    "content": [{"type": "text", "text": "write failed"}, {"type": "text", "text": "disk full"}],
                    "is_error": true },
                  { "type": "tool_result", "tool_use_id": "toolu_a", "content": "a\nb" },
                  { "type": "tool_result", "tool_use_id": "toolu_b", "content": "file contents" }
              ]}
            ]
        });
        let payload = conv(&anthropic);
        let msgs = payload.json["messages"].as_array().expect("chat messages");
        assert_eq!(msgs.len(), 5, "{msgs:?}");
        // Layout: user, assistant+tool_calls, tool, tool, tool — the results
        // out of request order stay in sequence.
        let roles: Vec<&str> = msgs.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(
            roles,
            ["user", "assistant", "tool", "tool", "tool"],
            "{msgs:?}"
        );
        let calls = msgs[1]["tool_calls"].as_array().expect("calls");
        assert_eq!(
            calls
                .iter()
                .map(|c| c["function"]["name"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["bash", "read_file", "write_file"],
            "parallel calls keep their request order"
        );
        let tool_msgs = msgs[2..].to_vec();
        let ids: Vec<&str> = tool_msgs
            .iter()
            .map(|m| m["tool_call_id"].as_str().unwrap())
            .collect();
        assert_eq!(
            ids,
            ["toolu_c", "toolu_a", "toolu_b"],
            "results keep their out-of-order sequence: {msgs:?}"
        );
        assert_eq!(
            tool_msgs[0]["content"], "tool returned an error: write failed\ndisk full",
            "block-array content folds; is_error prefixes: {msgs:?}"
        );
        assert_eq!(tool_msgs[1]["content"], "a\nb");
        assert_eq!(tool_msgs[2]["content"], "file contents");

        // chat -> Anthropic: the follow-up completion's three calls round-trip.
        let completion = json!({
            "id": "chatcmpl-multi",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "again",
                    "tool_calls": [
                        {"id": "toolu_a", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}},
                        {"id": "toolu_b", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"a\"}"}},
                        {"id": "toolu_c", "type": "function", "function": {"name": "write_file", "arguments": "{\"path\":\"b\"}"}}
                    ]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3}
        });
        let m = chat_to_message(&completion, "client-model", &payload.tool_meta);
        assert_eq!(m["role"], "assistant");
        assert_eq!(m["stop_reason"], "tool_use");
        let blocks = m["content"].as_array().expect("content");
        assert_eq!(blocks.len(), 4, "text plus the three tool_use blocks: {m}");
        assert_eq!(blocks[0], json!({"type": "text", "text": "again"}));
        for (block, id, name) in [
            (1, "toolu_a", "bash"),
            (2, "toolu_b", "read_file"),
            (3, "toolu_c", "write_file"),
        ] {
            assert_eq!(blocks[block]["type"], "tool_use");
            assert_eq!(blocks[block]["id"], id, "ids preserved in order");
            assert_eq!(blocks[block]["name"], name);
            assert_eq!(
                blocks[block]["caller"],
                json!({"type": "direct"}),
                "all three were client offers"
            );
        }
        // And feed the emitted assistant message straight back through:
        // tool_result pairing by id must be identity (stateless round-trip).
        let back = conv(&json!({
            "model": "m",
            "messages": [
              { "role": "user", "content": "list and read" },
              m.clone(),
              { "role": "user", "content": [
                  { "type": "tool_result", "tool_use_id": "toolu_c", "content": "ok now" },
                  { "type": "tool_result", "tool_use_id": "toolu_a", "content": "c1" },
                  { "type": "tool_result", "tool_use_id": "toolu_b", "content": "c2" }
              ]}
            ]
        }));
        let back_msgs = back.json["messages"].as_array().expect("messages");
        let assistant = back_msgs
            .iter()
            .find(|x| x["role"] == "assistant")
            .expect("assistant");
        assert_eq!(
            assistant["tool_calls"].as_array().map(|c| c.len()),
            Some(3),
            "the emitted tool_use blocks convert back to three calls"
        );
        let tool_ids: Vec<&str> = back_msgs
            .iter()
            .filter(|x| x["role"] == "tool")
            .map(|x| x["tool_call_id"].as_str().unwrap())
            .collect();
        assert_eq!(tool_ids, ["toolu_c", "toolu_a", "toolu_b"]);
    }

    #[test]
    fn red_computer_programmatic_callers_rejected() {
        let r = to_chat_payload(
            &json!({
                "model": "m",
                "tools": [{
                    "type": "computer_use_20250124",
                    "name": "computer",
                    "allowed_callers": ["programmatic"]
                }],
                "messages": []
            }),
            "",
        );
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
}
