#!/usr/bin/env python3
"""
Test suite for OpenAI-compatible tool calling against hyprstream's OAI endpoint.

Tests the standard OpenAI tool calling contract:
  https://platform.openai.com/docs/guides/function-calling

Run:
  python3 call_tools.py [--model MODEL] [--base-url URL]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import requests

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_BASE = "http://localhost:6789/oai/v1"
DEFAULT_MODEL = "Qwen3-0.6B:main"
TIMEOUT = 90  # seconds per request (small models are slow)

# ── tool definitions (reused across tests) ────────────────────────────────────
TOOL_GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. 'San Francisco, CA'",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    },
}

TOOL_SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            "required": ["query"],
        },
    },
}

TOOL_CALCULATOR = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '2 + 2'",
                },
            },
            "required": ["expression"],
        },
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def get_token(base_url: str) -> str:
    """Issue a JWT via the CLI."""
    result = subprocess.run(
        [os.path.expanduser("~/.local/bin/hyprstream"),
         "tool", "policy", "issue-token",
         "--subject", "rturk",
         "--requested-scopes", "model:*:infer",
         "--ttl", "3600"],
        capture_output=True, text=True,
    )
    for line in result.stdout.strip().splitlines():
        if line.startswith("token"):
            return line.split(None, 1)[1].strip()
    raise RuntimeError(f"Failed to get token: {result.stderr}")


def chat(base_url: str, token: str, payload: dict, stream: bool = False) -> dict | list:
    """Send a chat completion request. Returns parsed JSON or list of SSE chunks."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload.setdefault("stream", stream)

    if not stream:
        r = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=TIMEOUT,
        )
        return r.json()
    else:
        r = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=TIMEOUT,
            stream=True,
        )
        chunks = []
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunks.append(json.loads(data))
                except json.JSONDecodeError:
                    chunks.append({"_raw": data})
        return chunks


# ── result tracking ───────────────────────────────────────────────────────────

class Results:
    def __init__(self):
        self.tests = []

    def record(self, name: str, passed: bool, details: str = "", raw: object = None, xfail: bool = False):
        self.tests.append({
            "name": name,
            "passed": passed,
            "details": details,
            "raw": raw,
            "xfail": xfail,
        })
        if xfail and not passed:
            icon = "\033[33mXFAIL\033[0m"
        elif xfail and passed:
            icon = "\033[36mXPASS\033[0m"
        elif passed:
            icon = "\033[32mPASS\033[0m"
        else:
            icon = "\033[31mFAIL\033[0m"
        print(f"  [{icon}] {name}")
        if details:
            for line in details.splitlines():
                print(f"         {line}")

    def summary(self):
        total = len(self.tests)
        passed = sum(1 for t in self.tests if t["passed"])
        xfail = sum(1 for t in self.tests if t["xfail"] and not t["passed"])
        real_failures = sum(1 for t in self.tests if not t["passed"] and not t["xfail"])
        print(f"\n{'='*60}")
        msg = f"  RESULTS: {passed}/{total} passed"
        if xfail:
            msg += f", {xfail} expected failures"
        if real_failures:
            msg += f", {real_failures} failed"
        print(msg)
        print(f"{'='*60}")
        if real_failures:
            print("\n  Failed tests:")
            for t in self.tests:
                if not t["passed"] and not t["xfail"]:
                    print(f"    - {t['name']}: {t['details']}")
        print()
        return real_failures == 0


# ── test cases ────────────────────────────────────────────────────────────────

def test_basic_no_tools(base_url, token, model, results):
    """Baseline: chat without tools works."""
    resp = chat(base_url, token, {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'hello'. One word only."}],
        "max_tokens": 60,
        "temperature": 0.1,
    })
    has_error = "error" in resp
    has_content = False
    content = ""
    if not has_error:
        try:
            content = resp["choices"][0]["message"]["content"] or ""
            has_content = len(content) > 0
        except (KeyError, IndexError):
            pass

    results.record(
        "baseline: chat without tools",
        not has_error and has_content,
        f"content={content[:120]!r}" if has_content else f"error={resp.get('error', 'no content')}",
        raw=resp,
    )
    return not has_error


def test_single_tool_call(base_url, token, model, results):
    """Send a single tool and a prompt that should trigger it."""
    resp = chat(base_url, token, {
        "model": model,
        "messages": [
            {"role": "user", "content": "What's the weather in Paris right now?"},
        ],
        "tools": [TOOL_GET_WEATHER],
        "max_tokens": 512,
        "temperature": 0.1,
    })

    if "error" in resp:
        results.record("single tool call", False, f"error: {resp['error']}", raw=resp)
        return

    msg = resp.get("choices", [{}])[0].get("message", {})
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls")
    finish = resp.get("choices", [{}])[0].get("finish_reason")

    has_tool_calls = tool_calls is not None and len(tool_calls) > 0

    # Check structure of tool_calls if present
    struct_ok = True
    struct_details = []
    if has_tool_calls:
        tc = tool_calls[0]
        if "id" not in tc:
            struct_ok = False
            struct_details.append("missing 'id'")
        if tc.get("type") != "function":
            struct_ok = False
            struct_details.append(f"type={tc.get('type')!r} (expected 'function')")
        fn = tc.get("function", {})
        if "name" not in fn:
            struct_ok = False
            struct_details.append("missing function.name")
        if "arguments" not in fn:
            struct_ok = False
            struct_details.append("missing function.arguments")
        elif isinstance(fn["arguments"], str):
            try:
                json.loads(fn["arguments"])
            except json.JSONDecodeError:
                struct_ok = False
                struct_details.append(f"arguments not valid JSON: {fn['arguments'][:80]!r}")

    results.record(
        "single tool: model returns tool_calls",
        has_tool_calls,
        f"tool_calls={json.dumps(tool_calls, indent=2)[:300]}" if has_tool_calls
        else f"no tool_calls in response. content={content[:200]!r}",
        raw=resp,
    )

    results.record(
        "single tool: finish_reason='tool_calls'",
        finish == "tool_calls" and has_tool_calls,
        f"finish_reason={finish!r}",
        raw=resp,
    )

    if has_tool_calls:
        results.record(
            "single tool: response structure (id, type, function.name, function.arguments)",
            struct_ok,
            ", ".join(struct_details) if struct_details else "all fields present and valid",
            raw=tool_calls[0],
        )

        # Check the function name is correct
        fn_name = tool_calls[0].get("function", {}).get("name", "")
        results.record(
            "single tool: correct function name",
            fn_name == "get_weather",
            f"name={fn_name!r}",
        )

        # Check arguments contain location
        args_str = tool_calls[0].get("function", {}).get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
            has_location = "location" in args
            results.record(
                "single tool: arguments contain 'location'",
                has_location,
                f"args={json.dumps(args)}",
            )
        except json.JSONDecodeError:
            results.record(
                "single tool: arguments contain 'location'",
                False,
                f"could not parse arguments: {args_str[:100]!r}",
            )

    return has_tool_calls


def test_multi_tool_selection(base_url, token, model, results):
    """Send multiple tools, check model picks the right one."""
    resp = chat(base_url, token, {
        "model": model,
        "messages": [
            {"role": "user", "content": "What is 15 * 37?"},
        ],
        "tools": [TOOL_GET_WEATHER, TOOL_SEARCH, TOOL_CALCULATOR],
        "max_tokens": 512,
        "temperature": 0.1,
    })

    if "error" in resp:
        results.record("multi-tool selection", False, f"error: {resp['error']}", raw=resp)
        return

    msg = resp.get("choices", [{}])[0].get("message", {})
    tool_calls = msg.get("tool_calls")
    content = msg.get("content") or ""

    has_tool_calls = tool_calls is not None and len(tool_calls) > 0
    chose_calculator = False
    if has_tool_calls:
        chose_calculator = tool_calls[0].get("function", {}).get("name") == "calculator"

    results.record(
        "multi-tool: model picks 'calculator' for math",
        chose_calculator,
        f"tool_calls={json.dumps(tool_calls)[:200]}" if has_tool_calls
        else f"no tool_calls. content={content[:200]!r}",
        raw=resp,
    )


def test_tool_choice_none(base_url, token, model, results):
    """tool_choice='none' should suppress tool calls.

    XFAIL: tool_choice is parsed but not server-side enforced yet.
    """
    resp = chat(base_url, token, {
        "model": model,
        "messages": [
            {"role": "user", "content": "What's the weather in London?"},
        ],
        "tools": [TOOL_GET_WEATHER],
        "tool_choice": "none",
        "max_tokens": 150,
        "temperature": 0.1,
    })

    if "error" in resp:
        results.record("tool_choice=none", False, f"error: {resp['error']}", raw=resp)
        return

    msg = resp.get("choices", [{}])[0].get("message", {})
    tool_calls = msg.get("tool_calls")
    content = msg.get("content") or ""

    # With tool_choice=none, model should NOT return tool_calls
    no_tools = tool_calls is None or len(tool_calls) == 0
    results.record(
        "tool_choice='none': no tool_calls returned",
        no_tools,
        f"content={content[:120]!r}" if no_tools
        else f"tool_calls returned despite tool_choice=none: {json.dumps(tool_calls)[:150]}",
        raw=resp,
        xfail=True,  # tool_choice not server-side enforced yet
    )


def test_tool_choice_required(base_url, token, model, results):
    """tool_choice='required' should force a tool call even for a non-tool prompt.

    XFAIL: tool_choice is parsed but not server-side enforced yet.
    """
    resp = chat(base_url, token, {
        "model": model,
        "messages": [
            {"role": "user", "content": "Tell me a joke."},
        ],
        "tools": [TOOL_GET_WEATHER],
        "tool_choice": "required",
        "max_tokens": 512,
        "temperature": 0.3,
    })

    if "error" in resp:
        results.record("tool_choice=required", False, f"error: {resp['error']}", raw=resp)
        return

    msg = resp.get("choices", [{}])[0].get("message", {})
    tool_calls = msg.get("tool_calls")
    content = msg.get("content") or ""

    has_tool_calls = tool_calls is not None and len(tool_calls) > 0
    results.record(
        "tool_choice='required': tool_calls present",
        has_tool_calls,
        f"tool_calls={json.dumps(tool_calls)[:150]}" if has_tool_calls
        else f"no tool_calls. content={content[:150]!r}",
        raw=resp,
        xfail=True,  # tool_choice not server-side enforced yet
    )


def test_tool_choice_specific(base_url, token, model, results):
    """tool_choice={type: function, function: {name: ...}} should force a specific tool.

    XFAIL: tool_choice is parsed but not server-side enforced yet.
    """
    resp = chat(base_url, token, {
        "model": model,
        "messages": [
            {"role": "user", "content": "What is the meaning of life?"},
        ],
        "tools": [TOOL_GET_WEATHER, TOOL_CALCULATOR],
        "tool_choice": {"type": "function", "function": {"name": "calculator"}},
        "max_tokens": 512,
        "temperature": 0.3,
    })

    if "error" in resp:
        results.record("tool_choice=specific", False, f"error: {resp['error']}", raw=resp)
        return

    msg = resp.get("choices", [{}])[0].get("message", {})
    tool_calls = msg.get("tool_calls")
    content = msg.get("content") or ""

    has_tool_calls = tool_calls is not None and len(tool_calls) > 0
    chose_right = False
    if has_tool_calls:
        chose_right = tool_calls[0].get("function", {}).get("name") == "calculator"

    results.record(
        "tool_choice=specific: forced to 'calculator'",
        chose_right,
        f"tool_calls={json.dumps(tool_calls)[:150]}" if has_tool_calls
        else f"no tool_calls. content={content[:150]!r}",
        raw=resp,
        xfail=True,  # tool_choice not server-side enforced yet
    )


def test_multi_turn_tool_round_trip(base_url, token, model, results):
    """Full round trip: user → assistant (tool_call) → tool result → assistant answer."""

    # Turn 1: user asks, model should call tool
    resp1 = chat(base_url, token, {
        "model": model,
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"},
        ],
        "tools": [TOOL_GET_WEATHER],
        "max_tokens": 512,
        "temperature": 0.1,
    })

    if "error" in resp1:
        results.record("multi-turn: turn 1 (tool call)", False, f"error: {resp1['error']}", raw=resp1)
        return

    msg1 = resp1.get("choices", [{}])[0].get("message", {})
    tool_calls = msg1.get("tool_calls")

    if not tool_calls or len(tool_calls) == 0:
        results.record(
            "multi-turn: turn 1 (tool call)",
            False,
            f"no tool_calls in turn 1. content={msg1.get('content', '')[:150]!r}",
            raw=resp1,
        )
        return

    results.record("multi-turn: turn 1 (tool call)", True, f"got tool_call: {tool_calls[0].get('function', {}).get('name')}")

    tc_id = tool_calls[0].get("id", "call_test_123")

    # Turn 2: send tool result back, model should produce final answer
    resp2 = chat(base_url, token, {
        "model": model,
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": msg1.get("content"),
                "tool_calls": tool_calls,
            },
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": json.dumps({
                    "temperature": 22,
                    "unit": "celsius",
                    "condition": "partly cloudy",
                    "location": "Tokyo, Japan",
                }),
            },
        ],
        "tools": [TOOL_GET_WEATHER],
        "max_tokens": 512,
        "temperature": 0.3,
    })

    if "error" in resp2:
        results.record("multi-turn: turn 2 (tool result → answer)", False, f"error: {resp2['error']}", raw=resp2)
        return

    msg2 = resp2.get("choices", [{}])[0].get("message", {})
    content2 = msg2.get("content") or ""
    tool_calls2 = msg2.get("tool_calls")

    # Turn 2 should have text content and no further tool calls
    has_answer = len(content2) > 0
    no_more_tools = tool_calls2 is None or len(tool_calls2) == 0

    results.record(
        "multi-turn: turn 2 (final text answer)",
        has_answer,
        f"content={content2[:200]!r}",
        raw=resp2,
    )
    results.record(
        "multi-turn: turn 2 (no further tool calls)",
        no_more_tools,
        "clean" if no_more_tools else f"unexpected tool_calls: {json.dumps(tool_calls2)[:100]}",
    )


def test_parallel_tool_calls(base_url, token, model, results):
    """Ask something that should trigger multiple tool calls in one response."""
    resp = chat(base_url, token, {
        "model": model,
        "messages": [
            {"role": "user", "content": "What's the weather in both Paris and London?"},
        ],
        "tools": [TOOL_GET_WEATHER],
        "max_tokens": 300,
        "temperature": 0.1,
    })

    if "error" in resp:
        results.record("parallel tool calls", False, f"error: {resp['error']}", raw=resp)
        return

    msg = resp.get("choices", [{}])[0].get("message", {})
    tool_calls = msg.get("tool_calls")
    content = msg.get("content") or ""

    count = len(tool_calls) if tool_calls else 0

    results.record(
        "parallel tool calls: multiple tool_calls in one response",
        count >= 2,
        f"got {count} tool_calls" + (f": {json.dumps(tool_calls)[:200]}" if tool_calls else f". content={content[:150]!r}"),
        raw=resp,
    )

    if count >= 2:
        ids = [tc.get("id") for tc in tool_calls]
        unique_ids = len(set(ids)) == len(ids)
        results.record(
            "parallel tool calls: unique IDs",
            unique_ids,
            f"ids={ids}",
        )


def test_streaming_with_tools(base_url, token, model, results):
    """Streaming mode with tools — check tool_calls appear in SSE chunks."""
    chunks = chat(base_url, token, {
        "model": model,
        "messages": [
            {"role": "user", "content": "What's the weather in Berlin?"},
        ],
        "tools": [TOOL_GET_WEATHER],
        "max_tokens": 512,
        "temperature": 0.1,
    }, stream=True)

    if not chunks:
        results.record("streaming with tools", False, "no SSE chunks received")
        return

    # Check for errors in chunks
    for c in chunks:
        if "error" in c or "_raw" in c:
            results.record("streaming with tools", False, f"error in stream: {c}", raw=chunks)
            return

    # Look for tool_calls in any delta
    found_tool_calls = []
    all_content = ""
    finish_reason = None
    for c in chunks:
        choices = c.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        if delta.get("tool_calls"):
            found_tool_calls.extend(delta["tool_calls"])
        if delta.get("content"):
            all_content += delta["content"]
        fr = choices[0].get("finish_reason")
        if fr:
            finish_reason = fr

    has_tools = len(found_tool_calls) > 0

    results.record(
        "streaming: tool_calls in SSE deltas",
        has_tools,
        f"found {len(found_tool_calls)} tool_call(s)" if has_tools
        else f"no tool_calls in stream. content={all_content[:150]!r}",
        raw={"chunks_count": len(chunks), "tool_calls": found_tool_calls, "finish_reason": finish_reason},
    )

    results.record(
        "streaming: finish_reason='tool_calls'",
        finish_reason == "tool_calls" and has_tools,
        f"finish_reason={finish_reason!r}",
    )


def test_no_tool_when_unnecessary(base_url, token, model, results):
    """Tools are available but prompt doesn't need them — model should just answer."""
    resp = chat(base_url, token, {
        "model": model,
        "messages": [
            {"role": "user", "content": "What is 2 + 2? Just answer the number."},
        ],
        "tools": [TOOL_GET_WEATHER],  # irrelevant tool
        "max_tokens": 100,
        "temperature": 0.1,
    })

    if "error" in resp:
        results.record("no tool when unnecessary", False, f"error: {resp['error']}", raw=resp)
        return

    msg = resp.get("choices", [{}])[0].get("message", {})
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls")

    no_tools = tool_calls is None or len(tool_calls) == 0
    has_content = len(content) > 0

    results.record(
        "irrelevant tool: model answers without tool call",
        no_tools and has_content,
        f"content={content[:120]!r}" if no_tools
        else f"model called tool unnecessarily: {json.dumps(tool_calls)[:100]}",
        raw=resp,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test OpenAI tool calling against hyprstream")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to test (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE, help=f"OAI base URL (default: {DEFAULT_BASE})")
    parser.add_argument("--token", default=None, help="JWT token (auto-issued if omitted)")
    parser.add_argument("--dump", action="store_true", help="Dump full JSON responses on failure")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  hyprstream tool calling test suite")
    print(f"  model:    {args.model}")
    print(f"  endpoint: {args.base_url}")
    print(f"{'='*60}\n")

    # Get token
    if args.token:
        token = args.token
    else:
        print("  Issuing JWT token...", end=" ", flush=True)
        try:
            token = get_token(args.base_url)
            print(f"ok (sub=rturk)\n")
        except RuntimeError as e:
            print(f"\n  ERROR: {e}")
            sys.exit(1)

    results = Results()

    # ── Run tests ──
    print("── Baseline ────────────────────────────────────────")
    if not test_basic_no_tools(args.base_url, token, args.model, results):
        print("\n  Baseline failed — model service may be down. Aborting.\n")
        results.summary()
        sys.exit(1)

    print("\n── Single Tool Call ─────────────────────────────────")
    test_single_tool_call(args.base_url, token, args.model, results)

    print("\n── Multi-Tool Selection ─────────────────────────────")
    test_multi_tool_selection(args.base_url, token, args.model, results)

    print("\n── tool_choice Parameter ────────────────────────────")
    test_tool_choice_none(args.base_url, token, args.model, results)
    test_tool_choice_required(args.base_url, token, args.model, results)
    test_tool_choice_specific(args.base_url, token, args.model, results)

    print("\n── Multi-Turn Round Trip ────────────────────────────")
    test_multi_turn_tool_round_trip(args.base_url, token, args.model, results)

    print("\n── Parallel Tool Calls ─────────────────────────────")
    test_parallel_tool_calls(args.base_url, token, args.model, results)

    print("\n── Streaming with Tools ─────────────────────────────")
    test_streaming_with_tools(args.base_url, token, args.model, results)

    print("\n── Tool Relevance ──────────────────────────────────")
    test_no_tool_when_unnecessary(args.base_url, token, args.model, results)

    # ── Summary ──
    all_passed = results.summary()

    # Dump failures if requested
    if args.dump and not all_passed:
        print("\n── Full responses for failed tests ──────────────────")
        for t in results.tests:
            if not t["passed"] and t.get("raw"):
                print(f"\n  ### {t['name']}")
                print(json.dumps(t["raw"], indent=2))

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
