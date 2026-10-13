import json
import math
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ovos_switchboard.adapters.openai_llm import LLMClient, LLMEndpoint, parse_yes_no
from ovos_switchboard.core.ports import LLMUnavailable
from ovos_switchboard.data import prompt as bundled_prompt


TOOL_REPLY = {"choices": [{"message": {"tool_calls": [{"function": {"name": "ignore", "arguments": "{}"}}]}}],
              "usage": {"prompt_tokens": 42}}
GATE_REPLY = {"choices": [{"logprobs": {"content": [{"top_logprobs": [
    {"token": "Yes", "logprob": math.log(0.6)}, {"token": " no", "logprob": math.log(0.2)},
    {"token": "maybe", "logprob": math.log(0.2)}]}]}}], "usage": {"prompt_tokens": 7}}


@pytest.fixture(name="server")
def server_fixture():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # pylint: disable=invalid-name  # BaseHTTPRequestHandler's dispatch name
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((body, self.headers.get("Authorization")))
            reply = GATE_REPLY if body.get("logprobs") else TOOL_REPLY
            payload = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions", seen
    httpd.shutdown()


def client(url, cache_path=None, **endpoint):
    return LLMClient(LLMEndpoint(url, "test-model", **endpoint), bundled_prompt("fallback_strict.txt"),
                     bundled_prompt("addressee_gate.txt"), cache_path)


def test_route_and_gate_against_an_openai_compatible_endpoint(server):
    url, seen = server
    c = client(url, api_key="secret")
    tools = [{"type": "function", "function": {"name": "ignore", "parameters": {}}}]
    res = c.route("so anyway", tools)
    assert (res["tool"], res["args"], res["cached"]) == ("ignore", {}, False)
    assert c.route("so anyway", tools)["cached"], "identical requests are answered from the cache"
    assert c.p_addressed("so anyway")["p_yes"] == pytest.approx(0.75)
    body, auth = seen[0]
    assert (body["tool_choice"], body["temperature"], auth) == ("required", 0, "Bearer secret")
    assert len(seen) == 2


def test_cache_file_persists_answers(server, tmp_path):
    url, seen = server
    cache = tmp_path / "cache.jsonl"
    client(url, cache_path=str(cache)).p_addressed("hello")
    assert client(url, cache_path=str(cache)).p_addressed("hello")["cached"]
    assert len(seen) == 1


def test_unreachable_endpoint_raises_llm_unavailable():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # closed again before the request: connection refused
    with pytest.raises(LLMUnavailable):
        client(f"http://127.0.0.1:{port}/v1/chat/completions", timeout_s=2).p_addressed("hello")


def test_gate_without_yes_or_no_is_unusable():
    with pytest.raises(ValueError):
        parse_yes_no({"choices": [{"logprobs": {"content": [{"top_logprobs": [{"token": "hm", "logprob": 0.0}]}]}}]})
