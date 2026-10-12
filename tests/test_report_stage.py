import json
import threading
from urllib.request import Request, urlopen

from self_learning_assistant.model.responses import LocalResponsesServer


def test_cancellation_matches_research_owner_and_raw_output_is_recorded(tmp_path):
    class Generator:
        def __init__(self):
            self.started = threading.Event()
            self.cancelled = threading.Event()

        def __call__(self, messages, tools):
            self.started.set()
            assert self.cancelled.wait(2)
            return "stopped generation"

        def cancel_current(self):
            self.cancelled.set()

    generator = Generator()
    trace = tmp_path / "trace.jsonl"
    server = LocalResponsesServer(("127.0.0.1", 0), generator, trace_path=trace)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/v1"

    def request_generation():
        request = Request(
            base + "/responses",
            data=b'{"input":"claim"}',
            headers={"X-AE-Research-ID": "owner"},
        )
        with urlopen(request, timeout=3) as response:
            response.read()

    worker = threading.Thread(target=request_generation)
    worker.start()
    try:
        assert generator.started.wait(2)
        for owner, expected in [("unrelated", False), ("owner", True)]:
            with urlopen(
                Request(
                    base + "/cancel", data=b"{}", headers={"X-AE-Research-ID": owner}
                )
            ) as response:
                assert json.load(response)["cancelled"] is expected
        worker.join(timeout=2)
        assert not worker.is_alive()
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        response = next(row for row in rows if row["event"] == "response")
        assert response["research_id"] == "owner"
        assert response["raw_output"] == "stopped generation"
    finally:
        generator.cancelled.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
