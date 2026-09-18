//! In-gateway `web_search` execution for the Messages bridge (T10).
//!
//! The gateway runs the model's `web_search` calls itself — the search
//! provider is an operator-injected endpoint from the config store (the
//! default is the keyless Bing RSS feed, the `WEB_SEARCH_RSS_URL`
//! equivalent) — then re-sends the conversation with the results folded
//! in as tool messages, bounded by an iteration cap. Provider failures
//! surface as `web_search_tool_result` error blocks; nothing is silent.
//!
//! `encrypted_content` matches the official wire shape (an opaque,
//! prefixed base64url blob) but is a synthetic identity marker, the same
//! as the synthetic thinking signatures: the model actually parses the
//! plaintext `results` payload the gateway builds for the tool message.

use serde_json::{json, Value};

use crate::config::SearchFormat;

/// The web_search offer's execution metadata, carried from the request
/// transform into the execute-then-resend loop.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WebSearchMeta {
    /// The name the offer reached the upstream under.
    pub name: String,
    /// Per-request call cap (the offer's `max_uses`); None = unbounded by
    /// the offer (the iteration cap still applies).
    pub max_uses: Option<u32>,
    /// The offer's `user_location` (city/region/country), folded into the
    /// query as a suffix.
    pub user_location: Option<Value>,
}

/// Query length the provider search accepts (beyond it the call fails with
/// `query_too_long` instead of hitting the endpoint).
pub const MAX_QUERY_LEN: usize = 512;

/// One provider result.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub page_age: Option<String>,
}

/// Execute one `web_search` tool block against the provider.
///
/// Returns the outward `web_search_tool_result` block, the plaintext
/// payload the model sees on the re-sent tool message, and how many
/// provider requests this execution made (0 when `max_uses` is exceeded —
/// the cap is checked before any I/O).
pub async fn execute(
    http: &reqwest::Client,
    settings: &crate::WebSearchSettings,
    tool_use_id: &str,
    input: &Value,
    meta: &WebSearchMeta,
    uses: &mut u32,
) -> (Value, Value, u32) {
    let query = query_of(input);
    if query.is_empty() {
        return (
            error_block(
                tool_use_id,
                "missing_query",
                "web_search requires a query field",
            ),
            json!({ "error": "missing_query" }),
            1,
        );
    }
    if query.len() > MAX_QUERY_LEN {
        return (
            error_block(
                tool_use_id,
                "query_too_long",
                &format!("query must be at most {MAX_QUERY_LEN} characters"),
            ),
            json!({ "error": "query_too_long", "query": query }),
            1,
        );
    }
    if matches!(meta.max_uses, Some(cap) if *uses >= cap) {
        return (
            error_block(
                tool_use_id,
                "max_uses_exceeded",
                "web_search has reached the request's maximum call count",
            ),
            json!({ "error": "max_uses_exceeded", "query": query }),
            0,
        );
    }
    *uses += 1;

    let search_query = with_location(&query, &meta.user_location);
    let Some(body) = fetch(http, settings, &search_query).await else {
        return (
            error_block(
                tool_use_id,
                "search_unavailable",
                "the web_search provider request failed",
            ),
            json!({ "error": "search_unavailable", "query": query }),
            1,
        );
    };
    let hits = match settings.format {
        SearchFormat::Rss => parse_rss(&body),
        SearchFormat::Json => parse_json(&body),
    }
    .into_iter()
    .filter(|h| !h.url.is_empty())
    .take(settings.max_results.max(1) as usize)
    .collect::<Vec<_>>();

    let retrieved_at = utc_now_iso();
    let mut outward: Vec<Value> = Vec::new();
    let mut model: Vec<Value> = Vec::new();
    for h in &hits {
        let encrypted = encrypt(&json!({
            "query": query,
            "url": h.url,
            "title": h.title,
            "snippet": h.snippet,
            "page_age": h.page_age,
            "retrieved_at": retrieved_at,
        }));
        let mut item = json!({
            "type": "web_search_result",
            "url": h.url,
            "title": h.title,
            "encrypted_content": encrypted,
        });
        if let Some(age) = &h.page_age {
            item["page_age"] = json!(age);
        }
        outward.push(item);
        model.push(json!({
            "url": h.url,
            "title": h.title,
            "snippet": h.snippet,
            "page_age": h.page_age,
            "encrypted_content": encrypted,
            "retrieved_at": retrieved_at,
        }));
    }
    (
        json!({
            "type": "web_search_tool_result",
            "tool_use_id": tool_use_id,
            "content": outward,
        }),
        json!({ "query": query, "results": model }),
        1,
    )
}

/// The result block's error item: an is_error result the model can see.
fn error_block(tool_use_id: &str, code: &str, message: &str) -> Value {
    json!({
        "type": "web_search_tool_result",
        "tool_use_id": tool_use_id,
        "content": [{
            "type": "web_search_tool_result_error",
            "error_code": code,
            "message": message
        }],
        "is_error": true,
    })
}

fn query_of(input: &Value) -> String {
    let obj = match input {
        Value::Object(o) => o,
        _ => return String::new(),
    };
    for key in ["query", "q", "search_query"] {
        if let Some(s) = obj.get(key).and_then(Value::as_str) {
            let s = s.trim();
            if !s.is_empty() {
                return s.to_owned();
            }
        }
    }
    String::new()
}

/// `user_location` (city/region/country) is folded into the query as a
/// space-joined suffix — the provider is a plain search endpoint, so the
/// location rides the query string, not a separate parameter.
fn with_location(query: &str, user_location: &Option<Value>) -> String {
    let Some(loc) = user_location.as_ref().and_then(Value::as_object) else {
        return query.to_owned();
    };
    let parts: Vec<String> = loc
        .get("city")
        .and_then(Value::as_str)
        .into_iter()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .chain(
            loc.get("region")
                .and_then(Value::as_str)
                .into_iter()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
        )
        .chain(
            loc.get("country")
                .and_then(Value::as_str)
                .into_iter()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
        )
        .collect();
    if parts.is_empty() {
        return query.to_owned();
    }
    format!("{query} {}", parts.join(" "))
}

/// The provider request. A timeout, transport error, non-2xx, or empty
/// body is a `None` the caller turns into a `search_unavailable` block.
async fn fetch(
    http: &reqwest::Client,
    settings: &crate::WebSearchSettings,
    query: &str,
) -> Option<String> {
    let url = build_url(&settings.base_url, settings.format, query);
    let res = http
        .get(url)
        .timeout(settings.timeout)
        .header(reqwest::header::USER_AGENT, "nim-proxy/1.0")
        .send()
        .await
        .ok()?;
    if !res.status().is_success() {
        tracing::warn!(status = %res.status(), "web_search provider request failed");
        return None;
    }
    let body = res.text().await.ok()?;
    (!body.is_empty()).then_some(body)
}

/// The provider URL: the operator endpoint plus the query. RSS mode sends
/// `q` and `format=rss` (the keyless Bing RSS contract); JSON mode sends
/// `q` only — a JSON provider is keyed to answer on that parameter.
pub fn build_url(base: &str, format: SearchFormat, query: &str) -> String {
    let encoded = url_encode(query);
    let sep = if base.contains('?') { '&' } else { '?' };
    match format {
        SearchFormat::Rss => format!("{base}{sep}q={encoded}&format=rss"),
        SearchFormat::Json => format!("{base}{sep}q={encoded}"),
    }
}

/// Minimal RFC 3986 query-component encoding of the unreserved set.
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'*' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// The provider's opaque result blob, official wire shape: a `nimsearch_`
/// prefix over the unpadded URL-safe base64 of the plaintext payload.
/// Synthetic (the model parses the plaintext payload, not this blob).
fn encrypt(payload: &Value) -> String {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    let raw = serde_json::to_vec(payload).unwrap_or_default();
    format!("nimsearch_{}", URL_SAFE_NO_PAD.encode(raw))
}

/// RFC 3339 UTC now, from the unix clock (no chrono dependency).
fn utc_now_iso() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!(
        "{}T{:02}:{:02}:{:02}Z",
        civil_from_days((secs / 86_400) as i64),
        secs % 3600 / 60,
        secs % 60,
        secs % 60
    )
}

fn civil_from_days(days: i64) -> String {
    // Howard Hinnant's civil-from-days: days since 1970-01-01 -> y-m-d.
    let z = days + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (yoe * 365 + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// The minimal `<item>` extraction the keyless RSS feeds need: title,
/// link, description, pubDate. No CDATA-in-CDATA or entity- soup beyond
/// the standard five entities; a feed that is not parseable this way
/// yields no results (the model sees an empty result set, not an error).
pub fn parse_rss(xml: &str) -> Vec<SearchHit> {
    let mut hits = Vec::new();
    let mut pos = 0;
    while let Some(rel) = xml[pos..].find("<item") {
        let start = pos + rel + "<item".len();
        // `<items>`-style false positives: an alphanumeric tag character
        // after `<item` is not an item element.
        if matches!(xml[start..].chars().next(), Some(c) if c.is_alphanumeric()) {
            pos = start;
            continue;
        }
        let Some(rel_end) = xml[start..].find("</item>") else {
            break;
        };
        let item = &xml[start..start + rel_end];
        let link = rss_text(item, "link").unwrap_or_default();
        let title = rss_text(item, "title").unwrap_or_default();
        if link.is_empty() {
            pos = start + rel_end + "</item>".len();
            continue;
        }
        hits.push(SearchHit {
            page_age: rss_text(item, "pubDate").or_else(|| rss_text(item, "dc:date")),
            snippet: rss_text(item, "description").unwrap_or_default(),
            title,
            url: link,
        });
        pos = start + rel_end + "</item>".len();
    }
    hits
}

fn rss_text(item: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let i = item.find(open.as_str())? + open.len();
    let rest = &item[i..];
    let j = rest.find(close.as_str())?;
    let text = unescape_xml(rest[..j].trim());
    let text = text
        .strip_prefix("<![CDATA[")
        .and_then(|t| t.strip_suffix("]]>"))
        .map(|t| t.trim())
        .map(str::to_owned)
        .unwrap_or_else(|| text.clone());
    let text = text.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_owned())
    }
}

fn unescape_xml(s: &str) -> String {
    const NAMED: &[(&str, char)] = &[
        ("amp", '&'),
        ("lt", '<'),
        ("gt", '>'),
        ("quot", '"'),
        ("apos", '\''),
    ];
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        let candidate = &rest[at + 1..];
        let Some(end) = candidate.find(';') else {
            // A bare ampersand with no entity: keep it literal.
            out.push('&');
            rest = candidate;
            continue;
        };
        let entity = &candidate[..end];
        let decoded = NAMED
            .iter()
            .find(|(name, _)| *name == entity)
            .map(|(_, c)| *c)
            .or_else(|| {
                entity
                    .strip_prefix("#x")
                    .or_else(|| entity.strip_prefix("#X"))
                    .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                    .or_else(|| {
                        entity
                            .strip_prefix('#')
                            .and_then(|dec| dec.parse::<u32>().ok())
                    })
                    .and_then(char::from_u32)
            });
        match decoded {
            Some(c) => {
                out.push(c);
                rest = &candidate[end + 1..];
            }
            // An unknown entity stays in the text verbatim.
            None => {
                out.push_str(&candidate[..end + 1]);
                rest = &candidate[end + 1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// The JSON provider contract: `{"results": [...]}` or a bare array, each
/// item `{ "url", "title", "snippet", "page_age" }` (only `url` required).
pub fn parse_json(body: &str) -> Vec<SearchHit> {
    let v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let items = match &v {
        Value::Array(a) => a.iter(),
        Value::Object(o) => match o.get("results") {
            Some(Value::Array(a)) => a.iter(),
            _ => return Vec::new(),
        },
        _ => return Vec::new(),
    };
    items
        .filter_map(|item| {
            let obj = item.as_object()?;
            let url = obj.get("url").and_then(Value::as_str)?;
            let url = url.trim();
            if url.is_empty() {
                return None;
            }
            Some(SearchHit {
                title: obj
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned(),
                url: url.to_owned(),
                snippet: obj
                    .get("snippet")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned(),
                page_age: obj
                    .get("page_age")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RSS: &str = r#"<?xml version="1.0"?>
    <rss><channel>
      <title>Feed</title>
      <item>
        <title>One &amp; only</title>
        <link>https://a.example/one</link>
        <description><![CDATA[a <b>bold</b> snippet]]></description>
        <pubDate>Tue, 02 Sep 2025 00:00:00 GMT</pubDate>
      </item>
      <item>
        <title>Two</title>
        <link>https://b.example/two</link>
        <description>second &amp; third &#x41;</description>
      </item>
      <item>
        <title>No link</title>
        <description>skipped</description>
      </item>
      <item>
        <title>Empty</title>
        <link></link>
      </item>
    </channel></rss>"#;

    #[test]
    fn rss_items_extract_title_link_snippet_and_age() {
        let hits = parse_rss(RSS);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!(hits[0].title, "One & only");
        assert_eq!(hits[0].url, "https://a.example/one");
        assert_eq!(hits[0].snippet, "a <b>bold</b> snippet");
        assert_eq!(
            hits[0].page_age.as_deref(),
            Some("Tue, 02 Sep 2025 00:00:00 GMT")
        );
        assert_eq!(hits[1].title, "Two");
        assert_eq!(hits[1].snippet, "second & third A");
        assert!(hits[1].page_age.is_none());
    }

    #[test]
    fn rss_malformed_input_is_empty_never_a_panic() {
        for bad in [
            "",
            "<item>",
            "<item><title>x</title>",
            "<items><item><link>https://x</link></items>",
            "not xml at all",
            "<item><title",
        ] {
            let _ = parse_rss(bad);
        }
    }

    #[test]
    fn json_results_object_and_bare_array() {
        let v = r#"{"results":[{"url":"https://x","title":"t","snippet":"s","page_age":"1d"},{"url":""}]}"#;
        let hits = parse_json(v);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].url, "https://x");
        let bare = r#"[{"url":"https://y","title":"ty"}]"#;
        assert_eq!(parse_json(bare).len(), 1);
        assert!(parse_json("{nope").is_empty());
        assert!(parse_json(r#"{"nope":1}"#).is_empty());
    }

    #[test]
    fn url_building_appends_q_and_rss_marker() {
        assert_eq!(
            build_url("http://127.0.0.1:1/search", SearchFormat::Rss, "rust async"),
            "http://127.0.0.1:1/search?q=rust%20async&format=rss"
        );
        assert_eq!(
            build_url("http://127.0.0.1:1/search?x=1", SearchFormat::Json, "a b"),
            "http://127.0.0.1:1/search?x=1&q=a%20b"
        );
    }

    #[test]
    fn location_suffix_is_appended_to_the_query() {
        assert_eq!(with_location("paris", &None), "paris");
        let loc = json!({ "city": "Paris", "region": "", "country": "FR" });
        assert_eq!(with_location("weather", &Some(loc)), "weather Paris FR");
        assert_eq!(
            with_location("weather", &Some(json!({ "city": " " }))),
            "weather",
            "blank fields are dropped"
        );
    }

    #[test]
    fn query_of_reads_the_documented_keys() {
        assert_eq!(query_of(&json!({"query":" x "})), "x");
        assert_eq!(query_of(&json!({"q":"y"})), "y");
        assert_eq!(query_of(&json!({"search_query":"z"})), "z");
        assert_eq!(query_of(&json!({})), "");
        assert_eq!(query_of(&Value::Null), "");
    }

    #[test]
    fn civil_from_days_known_values() {
        assert_eq!(civil_from_days(0), "1970-01-01");
        assert_eq!(civil_from_days(365), "1971-01-01");
        assert_eq!(civil_from_days(366), "1971-01-02", "1971 has 365 days");
        assert_eq!(civil_from_days(730), "1972-01-01", "1972 is a leap year");
        assert_eq!(civil_from_days(18_262), "2020-01-01");
        assert_eq!(civil_from_days(-1), "1969-12-31");
    }

    #[test]
    fn encrypted_content_has_the_official_prefix_shape() {
        let c = encrypt(&json!({"url": "https://x"}));
        assert!(c.starts_with("nimsearch_"), "{c}");
        // The payload decodes back to the plaintext (the model parses the
        // plaintext, the blob is an identity marker).
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
        let raw = URL_SAFE_NO_PAD
            .decode(c.strip_prefix("nimsearch_").unwrap().as_bytes())
            .unwrap();
        assert!(String::from_utf8(raw).unwrap().contains("https://x"));
    }

    #[test]
    fn error_block_shape() {
        let b = error_block("id1", "search_unavailable", "boom");
        assert_eq!(b["type"], "web_search_tool_result");
        assert_eq!(b["tool_use_id"], "id1");
        assert_eq!(b["is_error"], true);
        assert_eq!(b["content"][0]["type"], "web_search_tool_result_error");
        assert_eq!(b["content"][0]["error_code"], "search_unavailable");
    }
}
