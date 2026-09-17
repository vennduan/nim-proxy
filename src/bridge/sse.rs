//! Streaming half of the Anthropic Messages bridge (T5).
//!
//! `StreamTranslator` turns upstream OpenAI chat-completions SSE bytes
//! into the Anthropic event sequence: `message_start`, one
//! `content_block_start` / `content_block_delta` / `content_block_stop`
//! group per text, thinking, or tool block, then `message_delta`
//! (stop reason + usage) and `message_stop`. It is a byte-exact
//! transducer over `Bytes`: arbitrary chunk boundaries (mid-line,
//! mid-JSON) and upstream comment lines never leak through, and the
//! retained state is bounded — a stream that never commits to an
//! Anthropic event replays its upstream bytes untouched instead.

use bytes::Bytes;
use serde_json::{json, Value};

/// Retention cap for a stream the translator has not committed to
/// Anthropic events: below the first event, upstream bytes are replayed
/// byte-exact if the stream never produces one (the pre-T5 passthrough
/// contract). Once committed, retention stops and the state stays
/// bounded.
const RETAIN_CAP: usize = 1024 * 1024;

/// A tool-call block in flight, keyed by the upstream `tool_calls` index.
#[derive(Default, Clone, Debug)]
pub(crate) struct ToolBlock {
    id: Value,
    name: String,
    input: String,
}

/// Which block kind is open at a given content index.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum OpenKind {
    Text,
    Thinking,
    Tool,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct OpenBlock {
    kind: OpenKind,
}

/// Request-side metadata the translator needs to render tool blocks
/// exactly the way the buffered transform does (the T7 caller echo).
#[derive(Default, Clone, Debug)]
pub struct StreamMeta {
    /// Tool names offered client-side: their `tool_use` blocks echo
    /// `caller: {"type": "direct"}`.
    pub client_tool_names: Vec<String>,
    /// The client's request model, echoed in `message_start`.
    pub model: String,
}

pub struct StreamTranslator {
    retained: Vec<Bytes>,
    retained_len: usize,
    pending: Vec<u8>,

    committed: bool,
    start_emitted: bool,
    finishing: bool,
    finish_reason: Option<String>,
    input_tokens: u64,
    output_tokens: u64,

    open: Vec<OpenBlock>,
    tools: Vec<ToolBlock>,
    /// Upstream `tool_calls` index -> translated content index + 1; 0 is
    /// the sentinel "not opened yet".
    tool_slot: Vec<usize>,

    meta: StreamMeta,
}

impl Default for StreamTranslator {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamTranslator {
    pub fn new() -> Self {
        Self {
            retained: Vec::new(),
            retained_len: 0,
            pending: Vec::new(),
            committed: false,
            start_emitted: false,
            finishing: false,
            finish_reason: None,
            input_tokens: 0,
            output_tokens: 0,
            open: Vec::new(),
            tools: Vec::new(),
            tool_slot: Vec::new(),
            meta: StreamMeta::default(),
        }
    }

    pub fn with_meta(meta: StreamMeta) -> Self {
        Self {
            meta,
            ..Self::new()
        }
    }

    /// Translate one chunk of upstream bytes; returns the emitted
    /// Anthropic events this chunk yielded.
    pub fn push(&mut self, chunk: &Bytes) -> Vec<Bytes> {
        if !self.committed && self.retained_len < RETAIN_CAP {
            self.retained.push(chunk.clone());
            self.retained_len += chunk.len();
        }
        self.pending.extend_from_slice(chunk);
        let mut out: Vec<Bytes> = vec![];
        self.decode(&mut out);
        if self.committed {
            self.retained.clear();
            self.retained_len = 0;
        }
        out
    }

    /// True once an Anthropic event committed: proxy control frames stop
    /// reaching the client and upstream errors must surface as Anthropic
    /// `error` events, not raw OpenAI frames.
    pub fn committed(&self) -> bool {
        self.committed
    }

    /// End of stream: close the open blocks, emit `message_delta` +
    /// `message_stop`. For a stream that never committed to an event,
    /// instead replay every retained upstream byte byte-exact (the
    /// pre-T5 passthrough contract) — including control frames.
    /// Safe to call more than once: only the first call yields events.
    pub fn finish(&mut self) -> Vec<Bytes> {
        if !self.committed {
            let out = std::mem::take(&mut self.retained);
            self.committed = true;
            return out;
        }
        if self.finishing {
            return Vec::new();
        }
        self.finishing = true;
        let mut out: Vec<Bytes> = vec![];
        self.decode(&mut out);
        self.close_open_blocks(&mut out);
        if self.start_emitted {
            out.push(self.message_delta());
            out.push(self.event("message_stop", &json!({ "type": "message_stop" })));
        }
        out
    }
}

impl StreamTranslator {
    fn caller_direct(&self, name: &str) -> bool {
        self.meta.client_tool_names.iter().any(|n| n == name)
    }

    fn event(&self, name: &str, data: &Value) -> Bytes {
        Bytes::from(format!(
            "event: {name}\ndata: {}\n\n",
            serde_json::to_string(data).unwrap_or_else(|_| "null".into())
        ))
    }

    /// An Anthropic in-stream `error` event, emitted when the upstream
    /// fails after `message_start` committed. The terminal event of the
    /// stream in this protocol: no `message_delta`/`message_stop` follow.
    pub fn error_event(&self, code: &str, message: &str) -> Bytes {
        self.event(
            "error",
            &json!({
                "type": "error",
                "error": {
                    "type": "upstream_error",
                    "code": code,
                    "message": message
                }
            }),
        )
    }

    /// `message_start`, emitted once on the first upstream event.
    fn start(&mut self, v: &Value, out: &mut Vec<Bytes>) {
        self.start_emitted = true;
        self.committed = true;
        let requested = (!self.meta.model.is_empty()).then(|| self.meta.model.clone());
        let model = requested
            .or_else(|| {
                v.get("model")
                    .and_then(Value::as_str)
                    .filter(|m| !m.is_empty())
                    .map(str::to_owned)
            })
            .unwrap_or_default();
        out.push(self.event(
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_stream",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": { "input_tokens": 0, "output_tokens": 0 }
                }
            }),
        ));
    }

    fn content_block_start(&mut self, index: usize, block: &Value, kind: OpenKind) -> Bytes {
        let ev = self.event(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": block
            }),
        );
        self.open.push(OpenBlock { kind });
        ev
    }

    fn content_block_delta(
        &self,
        index: usize,
        delta_type: &str,
        field: &str,
        value: &Value,
    ) -> Bytes {
        let mut delta = json!({
            "type": "content_block_delta",
            "index": index,
            "delta": { "type": delta_type }
        });
        delta["delta"][field] = value.clone();
        self.event("content_block_delta", &delta)
    }

    pub(crate) fn content_block_stop(&self, index: usize) -> Bytes {
        self.event(
            "content_block_stop",
            &json!({ "type": "content_block_stop", "index": index }),
        )
    }

    /// `message_delta`: stop reason + the observed usage. Usage arrives
    /// on the upstream usage chunk (usage injection), so this fires at
    /// stream end with whatever was observed.
    fn message_delta(&self) -> Bytes {
        let stop_reason = self.finish_reason.clone().map(Value::String);
        self.event(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": { "stop_reason": stop_reason, "stop_sequence": Value::Null },
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens
                }
            }),
        )
    }

    fn close_open_blocks(&mut self, out: &mut Vec<Bytes>) {
        for i in (0..self.open.len()).rev() {
            out.push(self.content_block_stop(i));
            self.open.remove(i);
        }
    }

    /// Open or reuse the stream's text block.
    fn text_index(&mut self, out: &mut Vec<Bytes>) -> usize {
        if self.open.last().is_some_and(|b| b.kind == OpenKind::Text) {
            return self.open.len() - 1;
        }
        let index = self.open.len();
        out.push(self.content_block_start(
            index,
            &json!({ "type": "text", "text": "" }),
            OpenKind::Text,
        ));
        index
    }

    /// Open or reuse the stream's thinking block.
    fn thinking_index(&mut self, out: &mut Vec<Bytes>) -> usize {
        if self
            .open
            .last()
            .is_some_and(|b| b.kind == OpenKind::Thinking)
        {
            return self.open.len() - 1;
        }
        let index = self.open.len();
        out.push(self.content_block_start(
            index,
            &json!({ "type": "thinking", "thinking": "", "signature": Value::Null }),
            OpenKind::Thinking,
        ));
        index
    }

    /// Tool-call index -> translated content index, opening the block on
    /// first sight.
    fn tool_slot_index(&mut self, call_index: usize, out: &mut Vec<Bytes>) -> usize {
        while self.tools.len() <= call_index {
            self.tools.push(ToolBlock::default());
            self.tool_slot.push(0);
        }
        if self.tool_slot[call_index] != 0 {
            return self.tool_slot[call_index] - 1;
        }
        let index = self.open.len();
        let tool = &self.tools[call_index];
        let id = tool
            .id
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| format!("toolu_stream_{call_index}"));
        let mut block = json!({
            "type": "tool_use",
            "id": id,
            "name": tool.name,
            "input": {}
        });
        if self.caller_direct(&tool.name) {
            block["caller"] = json!({ "type": "direct" });
        }
        let block_json = block.clone();
        out.push(self.content_block_start(index, &block_json, OpenKind::Tool));
        self.tool_slot[call_index] = index + 1;
        index
    }

    fn decode(&mut self, out: &mut Vec<Bytes>) {
        // Drain every complete event (its data lines + the blank-line
        // terminator) from the pending bytes; partial tails wait.
        while let Some(end) = find_event_end(&self.pending) {
            let event_bytes: Vec<u8> = self.pending.drain(..=end).collect();
            let Ok(event_text) = std::str::from_utf8(&event_bytes) else {
                continue;
            };
            self.event_text(event_text, out);
        }
    }

    fn event_text(&mut self, event: &str, out: &mut Vec<Bytes>) {
        // `: comment` lines (upstream keep-alives, proxy control frames)
        // carry no Anthropic meaning — dropped, never forwarded.
        if event.lines().all(|l| l.is_empty() || l.starts_with(':')) {
            return;
        }
        let data = event
            .lines()
            .filter_map(|l| l.strip_prefix("data:"))
            .collect::<Vec<_>>()
            .join("\n");
        let data = data.trim();
        if data.is_empty() {
            return;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return;
        };
        self.data_value(&v, out);
    }

    fn data_value(&mut self, v: &Value, out: &mut Vec<Bytes>) {
        // The usage-injection chunk carries `usage` (often with empty
        // `choices`); observe it regardless of when it lands.
        if let Some(usage) = v.get("usage") {
            self.input_tokens = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.input_tokens);
            self.output_tokens = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.output_tokens);
        }
        let Some(choices) = v.get("choices").and_then(Value::as_array) else {
            return;
        };
        if choices.is_empty() {
            return;
        }
        if !self.start_emitted {
            self.start(v, out);
        }
        for c in choices {
            let delta = c.get("delta").cloned().unwrap_or(Value::Null);

            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    let index = self.text_index(out);
                    out.push(self.content_block_delta(
                        index,
                        "text_delta",
                        "text",
                        &Value::String(text.to_string()),
                    ));
                }
            }

            if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str) {
                if !reasoning.is_empty() {
                    let index = self.thinking_index(out);
                    out.push(self.content_block_delta(
                        index,
                        "thinking_delta",
                        "thinking",
                        &Value::String(reasoning.to_string()),
                    ));
                }
            }

            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    let Some(func) = call.get("function") else {
                        continue;
                    };
                    let call_index =
                        call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                    while self.tools.len() <= call_index {
                        self.tools.push(ToolBlock::default());
                        self.tool_slot.push(0);
                    }
                    let tool = &mut self.tools[call_index];
                    if call.get("id").is_some() {
                        tool.id = call["id"].clone();
                    }
                    if let Some(name) = func.get("name").and_then(Value::as_str) {
                        if tool.name.is_empty() {
                            tool.name = name.to_string();
                        }
                    }
                    let args = func
                        .get("arguments")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty());
                    if let Some(args) = args {
                        tool.input.push_str(args);
                    }
                    let index = self.tool_slot_index(call_index, out);
                    if let Some(args) = args {
                        out.push(self.content_block_delta(
                            index,
                            "input_json_delta",
                            "partial_json",
                            &Value::String(args.to_string()),
                        ));
                    }
                }
            }

            if let Some(r) = c.get("finish_reason").and_then(Value::as_str) {
                if !r.is_empty() {
                    self.finish_reason = Some(map_stop_reason(r));
                    self.close_open_blocks(out);
                }
            }
        }
    }
}

/// Upstream finish reasons -> the Anthropic stop vocabulary.
fn map_stop_reason(reason: &str) -> String {
    match reason {
        "length" => "max_tokens".into(),
        "tool_calls" => "tool_use".into(),
        "stop" => "end_turn".into(),
        other => other.to_string(),
    }
}

/// The byte index of the final `\n` of the next complete SSE event
/// (a blank-line terminator), or None when the event is incomplete.
pub(crate) fn find_event_end(buf: &[u8]) -> Option<usize> {
    for i in 0..buf.len().saturating_sub(1) {
        if buf[i] == b'\n' && buf[i + 1] == b'\n' {
            return Some(i + 1);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(b: &str) -> Bytes {
        Bytes::copy_from_slice(b.as_bytes())
    }

    /// Feed one byte at a time to prove boundary tolerance.
    fn byte_at_a_time(chunk: &str) -> Vec<Bytes> {
        let mut t = StreamTranslator::new();
        let mut out: Vec<Bytes> = Vec::new();
        for b in chunk.bytes() {
            let one = Bytes::copy_from_slice(&[b]);
            out.extend(t.push(&one));
        }
        out.extend(t.finish());
        out
    }

    fn events(chunk: &str) -> Vec<Bytes> {
        let mut t = StreamTranslator::new();
        let mut out = t.push(&s(chunk));
        out.extend(t.finish());
        out
    }

    fn joined(evs: &[Bytes]) -> String {
        evs.iter()
            .map(|b| String::from_utf8_lossy(b).to_string())
            .collect()
    }

    #[test]
    fn green_text_stream_produces_full_event_sequence() {
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"he\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"llo\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2}}\n\n"
        );
        let evs = events(stream);
        let joined = joined(&evs);
        assert!(
            joined.contains("event: message_start"),
            "message_start opens the stream: {joined:?}"
        );
        assert!(
            joined.contains("\"type\":\"text_delta\"") && joined.contains("\"index\":0"),
            "text deltas carry the content index: {joined:?}"
        );
        assert!(
            joined.contains("event: content_block_stop")
                && joined.contains("event: content_block_start"),
            "the block group opens and closes: {joined:?}"
        );
        let delta = joined
            .split("event: message_delta")
            .nth(1)
            .expect("a message_delta");
        assert!(
            delta.contains("\"stop_reason\":\"end_turn\""),
            "stop maps to end_turn: {delta:?}"
        );
        assert!(
            delta.contains("\"input_tokens\":7") && delta.contains("\"output_tokens\":2"),
            "usage is forwarded: {delta:?}"
        );
        assert!(
            joined.contains("event: message_stop"),
            "message_stop terminates: {joined:?}"
        );
    }

    #[test]
    fn green_length_finish_maps_to_max_tokens() {
        let stream = "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"length\"}]}\n\n";
        let joined = joined(&events(stream));
        let delta = joined
            .split("event: message_delta")
            .nth(1)
            .expect("a message_delta");
        assert!(
            delta.contains("\"stop_reason\":\"max_tokens\""),
            "length maps to max_tokens: {delta:?}"
        );
    }

    #[test]
    fn green_tool_call_stream_emits_tool_use_blocks() {
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Paris\\\"\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}\n\n"
        );
        let evs = events(stream);
        let plain = joined(&evs);
        assert!(
            plain.contains("\"type\":\"tool_use\"") && plain.contains("\"id\":\"call_1\""),
            "the call's id is echoed on the block: {plain:?}"
        );
        assert!(
            plain.contains("\"type\":\"input_json_delta\""),
            "arguments stream as input_json_delta: {plain:?}"
        );
        let delta = plain
            .split("event: message_delta")
            .nth(1)
            .expect("a message_delta");
        assert!(
            delta.contains("\"stop_reason\":\"tool_use\""),
            "tool_calls finish maps to tool_use: {delta:?}"
        );
        // The caller echo: a client-offered tool names its direct caller.
        let mut t = StreamTranslator::with_meta(StreamMeta {
            client_tool_names: vec!["get_weather".into()],
            model: "m".into(),
        });
        let mut out = t.push(&s(stream));
        out.extend(t.finish());
        let caller_joined = joined(&out);
        assert!(
            caller_joined.contains("\"caller\":{\"type\":\"direct\"}"),
            "client-offered tools echo the direct caller: {caller_joined:?}"
        );
    }

    #[test]
    fn green_reasoning_content_becomes_thinking_block() {
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"why \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"42\",\"content\":\"\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n"
        );
        let joined = joined(&events(stream));
        assert!(
            joined.contains("\"type\":\"thinking\"") && joined.contains("\"thinking_delta\""),
            "reasoning opens a thinking block: {joined:?}"
        );
    }

    #[test]
    fn green_uncommitted_stream_replays_bytes_exact() {
        let evs = events(": connected\n\n");
        assert_eq!(evs.len(), 1, "{evs:?}");
        assert_eq!(
            evs[0].as_ref(),
            b": connected\n\n",
            "byte-exact, not translated"
        );
    }

    #[test]
    fn green_comment_lines_are_dropped() {
        let stream = concat!(
            ": keepalive\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n"
        );
        let joined = joined(&events(stream));
        assert!(
            !joined.contains("keepalive"),
            "comments never reach the client: {joined:?}"
        );
        assert!(joined.contains("hi"), "{joined:?}");
    }

    #[test]
    fn green_hostile_byte_split_produces_identical_output() {
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hostile\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\n\n"
        );
        let a = joined(&byte_at_a_time(stream));
        let b = joined(&events(stream));
        assert_eq!(a, b, "byte-at-a-time == whole-chunk: {a} vs {b}");
    }

    #[test]
    fn green_malformed_json_is_ignored_not_forwarded() {
        let stream = concat!(
            "data: {oops-not-json\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n"
        );
        let joined = joined(&events(stream));
        assert!(
            !joined.contains("oops-not-json"),
            "upstream data never reaches the client raw: {joined:?}"
        );
        assert!(joined.contains("ok"), "{joined:?}");
    }

    #[test]
    fn red_empty_input_tool_call_omits_blank_json_fragment() {
        // A call with no arguments at all: the block opens with `{}`
        // and no input_json_delta fires for an empty fragment.
        let stream = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n";
        let joined = joined(&events(stream));
        assert!(
            joined.contains("\"type\":\"tool_use\"") && joined.contains("\"id\":\"c\""),
            "the block still opens: {joined:?}"
        );
        assert!(
            !joined.contains("partial_json\":\"\""),
            "no empty JSON fragment: {joined:?}"
        );
    }
}
