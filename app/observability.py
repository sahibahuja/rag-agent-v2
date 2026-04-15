import os
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace

def setup_tracing():
    # 1. Cleaner check: If the global tracer isn't the default 'ProxyTracer', 
    # then we've already set up Phoenix.
    if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        return 

    # 2. Use the HTTP endpoint (Port 6006) for maximum stability on Windows/Docker
    tracer_provider = register(
        project_name="rag-agent-v2",
        endpoint="http://localhost:6006/v1/traces",
    )

    # 3. Instrument LangChain
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    print("🚀 Phoenix Tracing Active (via OTLP/HTTP)")