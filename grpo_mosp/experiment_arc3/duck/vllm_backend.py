"""Serve the existing NF4 checkpoint with vLLM and normalize its streamed replies."""

import asyncio
import json as jsonlib
import os
import shlex
import signal
import socket
import subprocess
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from ...experiment_downstream_tasks.serve_qwen import DEADLINE_ERROR, DEADLINE_HEADER


@contextmanager
def serve_vllm(python, snapshot, name, log, *, context_limit, concurrent_jobs):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    address = f"http://127.0.0.1:{port}/v1"
    prefix = Path(python).resolve().parent.parent
    env = {
        **os.environ,
        "PATH": str(prefix / "bin") + os.pathsep + os.environ.get("PATH", ""),
        "CONDA_PREFIX": str(prefix), "CONDA_DEFAULT_ENV": prefix.name,
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
    }
    if not env.get("CUDA_HOME"):
        compiler = next(prefix.glob("lib/python*/site-packages/nvidia/cu*/bin/nvcc"), None)
        if compiler is not None:
            env["CUDA_HOME"] = str(compiler.parent.parent)
    command = [
        python, "-B", "-m", "vllm.entrypoints.cli.main", "serve", str(snapshot),
        "--host", "127.0.0.1", "--port", str(port), "--served-model-name", name,
        "--dtype", "bfloat16", "--quantization", "bitsandbytes", "--generation-config", "vllm",
        "--max-model-len", str(context_limit), "--max-num-seqs", str(concurrent_jobs),
        "--no-enable-prefix-caching", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
    ]
    with log.open("w", encoding="utf-8") as stream:
        print(shlex.join(command), file=stream, flush=True)
        with subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                              env=env, start_new_session=True) as process:
            try:
                deadline = time.monotonic() + 600
                while process.poll() is None and time.monotonic() < deadline:
                    try:
                        with urlopen(f"{address}/models", timeout=2) as response:
                            models = jsonlib.load(response)["data"]
                    except (URLError, TimeoutError):
                        time.sleep(0.5)
                        continue
                    if name not in {model["id"] for model in models}:
                        raise RuntimeError(f"Unexpected model at {address}")
                    yield address
                    return
                raise RuntimeError(f"vLLM server failed to start. See {log}")
            finally:
                # The API server may spawn an engine process that also owns GPU memory.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=15)
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)


def stream_payload(payload):
    options = dict(payload.get("chat_template_kwargs", {}))
    options["reasoning_effort"] = payload["reasoning_effort"]
    messages = []
    for message in payload["messages"]:
        if message["role"] == "assistant" and message.get("reasoning_content"):
            message = dict(message)
            message["reasoning"] = message.pop("reasoning_content")
        messages.append(message)
    data = {**payload, "messages": messages, "chat_template_kwargs": options, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    # The checkpoint template defines the effort levels, including xhigh.
    data.pop("reasoning_effort")
    return data


class StreamReply:
    def __init__(self):
        self.text = []
        self.reasoning = []
        self.tools = {}
        self.usage = {"completion_tokens": 0}
        self.finish = None

    def add(self, chunk):
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices", ()):
            self.finish = choice.get("finish_reason") or self.finish
            delta = choice["delta"]
            if content := delta.get("content"):
                self.text.append(content)
            if reasoning := delta.get("reasoning") or delta.get("reasoning_content"):
                self.reasoning.append(reasoning)
            for item in delta.get("tool_calls") or ():
                tool = self.tools.setdefault(item["index"], {
                    "type": "function", "function": {"name": "", "arguments": ""},
                })
                if item.get("id"):
                    tool["id"] = item["id"]
                for key, segment in (item.get("function") or {}).items():
                    if key in tool["function"] and segment:
                        tool["function"][key] += segment

    def body(self):
        message = {"role": "assistant", "content": "".join(self.text) or None}
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        if self.tools:
            message["tool_calls"] = [tool for _, tool in sorted(self.tools.items())]
        return {"choices": [{"message": message, "finish_reason": self.finish}], "usage": self.usage}


def vllm_post(url, *, json, timeout, headers=None):
    """Return a requests-style result while enforcing an absolute deadline."""
    import httpx
    import requests

    def reply(status, body, reason=None):
        response = requests.Response()
        response.status_code = status
        response.reason = reason
        response.url = url
        response.encoding = "utf-8"
        response._content = jsonlib.dumps(body).encode("utf-8")
        return response

    cutoff = float(headers[DEADLINE_HEADER]) if headers is not None else None
    collected = StreamReply()

    async def submit():
        remaining = max(0.0, cutoff - time.monotonic()) if cutoff is not None else timeout
        async with asyncio.timeout(remaining), httpx.AsyncClient(trust_env=False, timeout=None) as client:
            async with client.stream("POST", url, json=stream_payload(json)) as response:
                if response.is_error:
                    await response.aread()
                    try:
                        body = response.json()
                    except ValueError:
                        body = {"error": {"message": response.text}}
                    return reply(response.status_code, body, response.reason_phrase)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        if collected.finish is None:
                            break
                        return reply(200, collected.body())
                    chunk = jsonlib.loads(data)
                    if "error" in chunk:
                        return reply(502, chunk, "vLLM stream error")
                    collected.add(chunk)
        raise requests.ConnectionError("vLLM stream ended before completing the reply")

    try:
        return asyncio.run(submit())
    except TimeoutError as error:
        if cutoff is None:
            raise requests.Timeout("vLLM request timed out") from error
        return reply(504, {"error": {"code": DEADLINE_ERROR, "message": "vLLM stream closed at its deadline"},
                           "usage": collected.usage})
    except httpx.TransportError as error:
        raise requests.ConnectionError(str(error)) from error
