import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from self_learning_assistant.model.responses import (
    LocalResponsesServer,
    parse_generation,
    prepare_request,
    response_events,
    transformers_generator,
)
from self_learning_assistant.product.model import LocalAPIModel

TOOLS = [
    {
        "type": "function",
        "name": "search_web",
        "description": "Search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "purpose": {"type": "string", "enum": ["neutral", "counter"]},
            },
            "required": ["query", "purpose"],
        },
    }
]
CALL = """<tool_call><function=search_web>
<parameter=query>Python defaults</parameter><parameter=purpose>counter</parameter>
</function></tool_call>"""


def test_qwen_tool_call_is_translated_to_responses_function_call():
    _, tools = prepare_request({"input": "Verify defaults", "tools": TOOLS})
    item = parse_generation(CALL, tools)[0]
    assert item["type"] == "function_call"
    assert json.loads(item["arguments"]) == {
        "query": "Python defaults",
        "purpose": "counter",
    }


def test_report_tool_choice_none_removes_all_tools():
    _, tools = prepare_request(
        {"input": "report", "tools": TOOLS, "tool_choice": "none"}
    )
    assert tools == []
    with pytest.raises(ValueError, match="unauthorized"):
        parse_generation(CALL, tools)


@pytest.mark.parametrize(
    "raw",
    [
        "<tool_call>",
        CALL + "<tool_call>",
        CALL + "unexpected suffix",
        CALL.replace("search_web", "exec_command"),
        CALL.replace("counter", "unsupported"),
        CALL.replace("<parameter=query>Python defaults</parameter>", ""),
        CALL.replace("<parameter=query>Python defaults</parameter>", "garbage"),
        CALL.replace(
            "</function>", "<parameter=query>duplicate</parameter></function>"
        ),
        "<tool_call>not a function</tool_call>",
    ],
)
def test_invalid_or_unauthorized_qwen_tools_fail_closed(raw):
    _, tools = prepare_request({"input": "claim", "tools": TOOLS})
    with pytest.raises(ValueError):
        parse_generation(raw, tools)


def test_history_and_schema_are_translated_and_unapproved_tools_hidden():
    messages, tools = prepare_request(
        {
            "instructions": "Verify",
            "tools": [
                *TOOLS,
                {"type": "web_search"},
                {"type": "function", "name": "exec_command"},
            ],
            "text": {"format": {"schema": {"type": "object"}}},
            "input": [
                {"role": "developer", "content": "Only evidence"},
                {"role": "user", "content": [{"type": "input_text", "text": "claim"}]},
                {"type": "reasoning", "summary": []},
                {
                    "type": "function_call",
                    "name": "search_web",
                    "arguments": '{"query":"x"}',
                },
                {
                    "type": "function_call_output",
                    "output": [{"type": "input_text", "text": "source"}],
                },
            ],
        }
    )
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert [t["function"]["name"] for t in tools] == ["search_web"]
    assert "Only evidence" in messages[0]["content"]


@pytest.mark.parametrize(
    "payload",
    [
        {"input": [{"type": "web_search_call"}]},
        {"input": []},
        {"input": [{"role": "tool", "content": "x"}]},
        {"input": [{"role": "user", "content": [{"type": "input_image"}]}]},
        {"input": [{"role": "user", "content": None}]},
        {
            "input": [
                {"type": "function_call", "name": "exec_command", "arguments": "{}"}
            ]
        },
    ],
)
def test_unsupported_responses_inputs_are_rejected(payload):
    with pytest.raises(ValueError):
        prepare_request(payload)


def test_stream_events_include_complete_items_and_completion():
    response = {
        "id": "r",
        "output": parse_generation("<think>hidden</think>answer", []),
    }
    events = list(response_events(response))
    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "response.completed"
    assert events[2]["delta"] == "answer"


def test_transformers_generator_supplies_native_tool_template():
    class Model:
        def generate_with_options(self, messages, **kwargs):
            assert kwargs["tools"] == [{"name": "search_web"}]
            assert kwargs["enable_thinking"] is False
            return "answer"

    assert (
        transformers_generator(Model(), max_new_tokens=512)(
            [{"role": "user", "content": "claim"}], [{"name": "search_web"}]
        )
        == "answer"
    )


def test_local_http_bridge_supports_text_streams_and_errors(tmp_path):
    def generate(messages, tools):
        if messages[-1]["content"] == "crash":
            raise RuntimeError("private diagnostic")
        return CALL if tools else "answer"

    generate.diagnostics = lambda: {"cuda": {"allocated_bytes": 1234}}
    trace_path = tmp_path / "trace.jsonl"
    server = LocalResponsesServer(("127.0.0.1", 0), generate, trace_path=trace_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/health") as result:
            assert json.load(result)["backend"] == "local"
        with urlopen(base + "/v1/models") as result:
            assert json.load(result)["data"][0]["id"] == "self-learning-assistant-local"
        assert (
            LocalAPIModel(base + "/v1", "self-learning-assistant-local").generate(
                [{"role": "user", "content": "claim"}]
            )
            == "answer"
        )
        request = Request(
            base + "/v1/responses",
            data=json.dumps(
                {
                    "input": "claim",
                    "tools": TOOLS,
                    "stream": True,
                }
            ).encode(),
        )
        with urlopen(request) as result:
            events = [
                json.loads(line[6:])
                for line in result.read().decode().splitlines()
                if line.startswith("data: ")
            ]
        assert events[-1]["response"]["output"][0]["name"] == "search_web"
        for path, body, code in [
            ("/unknown", b"{}", 404),
            ("/v1/responses", b"{}", 400),
            ("/v1/responses", b'{"input":"crash"}', 500),
        ]:
            with pytest.raises(HTTPError) as error:
                urlopen(Request(base + path, data=body))
            assert error.value.code == code
            assert b"private diagnostic" not in error.value.read()
        records = [json.loads(line) for line in trace_path.read_text().splitlines()]
        failure = next(r for r in records if r["event"] == "generation_failed")
        assert failure["error_type"] == "RuntimeError"
        assert failure["error"] == "private diagnostic"
        assert "RuntimeError: private diagnostic" in failure["traceback"]
        assert (
            failure["resources_before"]
            == failure["resources_after"]
            == {"cuda": {"allocated_bytes": 1234}}
        )
        assert failure["queue_seconds"] >= 0
        assert failure["generation_seconds"] >= 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_cancellation_after_registering_owner_is_not_lost_when_inference_starts():
    class Model:
        def generate_with_options(self, messages, **kwargs):
            assert kwargs["cancel_event"].is_set()
            return "cancelled"

    generator = transformers_generator(Model(), max_new_tokens=128)
    generator.begin_generation()
    generator.cancel_current()
    assert generator([{"role": "user", "content": "claim"}], []) == "cancelled"


def test_local_chat_endpoint_supports_product_schema_and_rejects_streams():
    from self_learning_assistant.product.model import LocalAPIModel, structured

    seen = []

    def generate(messages, tools):
        seen.append((messages, tools))
        return '{"ok":true}'

    server = LocalResponsesServer(("127.0.0.1", 0), generate)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        result = structured(
            LocalAPIModel(base, "local"),
            [{"role": "user", "content": "Return ok true"}],
            {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        )
        assert result == {"ok": True}
        assert seen[0][1] == []
        assert '"ok"' in seen[0][0][0]["content"]
        assert seen[0][0][0]["content"].count('"properties"') == 1
        # A regular chat-completions client may supply only response_format.
        # In that case the bridge must still add the schema exactly once.
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": "Return ok true"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "check",
                        "schema": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}},
                        },
                    },
                },
            }
        ).encode()
        with urlopen(
            Request(
                base + "/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
        ) as response:
            assert (
                json.load(response)["choices"][0]["message"]["content"] == '{"ok":true}'
            )
        assert seen[1][0][0]["content"].count('"properties"') == 1
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/chat/completions", data=b'{"stream":true}'))
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_transformers_chat_generation_uses_request_budget_and_no_tools():
    class Model:
        def generate_with_options(self, messages, **kwargs):
            assert kwargs["max_new_tokens"] == 64
            assert kwargs["tools"] == []
            assert kwargs["enable_thinking"] is False
            assert kwargs["cancel_event"] is not None
            return "text"

    generator = transformers_generator(Model(), max_new_tokens=512)
    assert generator.generate_chat([{"role": "user", "content": "hello"}], 64) == "text"
