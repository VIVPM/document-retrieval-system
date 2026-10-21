"""Check evaluation traces without sending data to Langfuse."""
from contextlib import contextmanager

import observability


class FakeSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class FakeTracer:
    def __init__(self):
        self.span = FakeSpan()

    @contextmanager
    def start_as_current_span(self, name):
        assert name == "evaluation-case"
        yield self.span


def test_evaluation_trace():
    previous = observability._llm_tracer
    try:
        tracer = FakeTracer()
        observability._llm_tracer = tracer
        with observability.trace_evaluation("What is the loan amount?", "Q1-flash") as span:
            observability.set_evaluation_output(span, "$380,000", True)
        assert tracer.span.attributes == {
            "langfuse.trace.name": "evaluation-case",
            "langfuse.observation.input": "What is the loan amount?",
            "langfuse.observation.metadata.case_id": "Q1-flash",
            "langfuse.observation.output": "$380,000",
            "langfuse.observation.metadata.passed": True,
        }
        observability._llm_tracer = None
        with observability.trace_evaluation("question", "Q2") as span:
            assert span is None
            observability.set_evaluation_output(span, "answer")
    finally:
        observability._llm_tracer = previous


if __name__ == "__main__":
    test_evaluation_trace()
