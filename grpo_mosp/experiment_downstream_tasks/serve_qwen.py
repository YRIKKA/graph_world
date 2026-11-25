"""Serve the pinned NF4 checkpoint and optionally a GRPO adapter on loopback."""

import argparse
import base64
import io
import json
import math
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

DEADLINE_HEADER = "X-Generation-Deadline"
DEADLINE_ERROR = "generation_deadline"
REASONING_EFFORTS = ("low", "medium", "xhigh")


class GenerationDeadline(Exception):
    def __init__(self, message, completion_tokens=0):
        super().__init__(message)
        self.completion_tokens = completion_tokens


def load_checkpoint(snapshot, adapter):
    import torch
    from bitsandbytes.nn import Linear4bit
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    if isinstance(config.quantization_config, dict):
        config.quantization_config["bnb_4bit_compute_dtype"] = "bfloat16"
    else:
        config.quantization_config.bnb_4bit_compute_dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot, config=config, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
    )
    quantized = [module for module in model.modules() if isinstance(module, Linear4bit)]
    if not quantized or any(
        module.weight.quant_state.quant_type != "nf4" or not module.weight.quant_state.nested
        for module in quantized
    ):
        raise RuntimeError("Expected the pinned prequantized NF4 checkpoint")
    if adapter:
        from peft import PeftConfig, PeftModel
        from peft.utils import get_peft_model_state_dict
        from safetensors import safe_open

        # Training wraps the text-only model. The same layers live under
        # language_model when the vision tower is present.
        peft_config = PeftConfig.from_pretrained(adapter)
        if not all(name.startswith("model.layers.") for name in peft_config.target_modules):
            raise ValueError("Expected a text-only MOSP attention adapter")
        peft_config.target_modules = {
            name.replace("model.layers.", "model.language_model.layers.", 1)
            for name in peft_config.target_modules
        }
        model = PeftModel.from_pretrained(
            model, adapter, config=peft_config, is_trainable=False,
            key_mapping={r"^model\.layers\.": "model.language_model.layers."},
        )
        loaded = get_peft_model_state_dict(model, save_embedding_layers=False)
        with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as saved:
            keys = {
                key.replace("base_model.model.model.layers.",
                            "base_model.model.model.language_model.layers.", 1): key
                for key in saved.keys()
            }
            if set(loaded) != set(keys) or any(
                not torch.equal(tensor.detach().cpu(), saved.get_tensor(keys[key]).to(tensor.dtype))
                for key, tensor in loaded.items()
            ):
                raise RuntimeError("The multimodal model did not load the saved LoRA tensors exactly")
    model.eval()
    return model, processor


def prepare_messages(messages):
    from PIL import Image

    converted = []
    for message in messages:
        # Duck replays vLLM's reasoning field. The Qwen template reads reasoning_content.
        reasoning = message.get("reasoning")
        if message.get("role") == "assistant" and not message.get("reasoning_content") and isinstance(reasoning, str):
            message = {**message, "reasoning_content": reasoning}
        content = message.get("content")
        calls = message.get("tool_calls")
        if calls:
            # The pinned Qwen template expects decoded argument objects in history.
            message = {**message, "tool_calls": [
                {**call, "function": {
                    **call["function"], "arguments": json.loads(call["function"]["arguments"]),
                }}
                for call in calls
            ]}
        if content is None or isinstance(content, str):
            converted.append({**message, "content": content or ""})
            continue
        parts = []
        for part in content:
            if part["type"] == "text":
                parts.append(part)
            elif part["type"] == "image_url":
                url = part["image_url"]["url"]
                header, _, data = url.partition(",")
                if not header.startswith("data:image/") or not header.endswith(";base64"):
                    raise ValueError("Expected an inline base64 screenshot")
                with Image.open(io.BytesIO(base64.b64decode(data, validate=True))) as image:
                    parts.append({"type": "image", "image": image.convert("RGB")})
            else:
                raise ValueError(f"Unsupported message content: {part['type']}")
        converted.append({**message, "content": parts})
    return converted


TOOL_BLOCK = re.compile(r"<tool_call>\s*<function=([\w.-]+)>\s*(.*?)</function>\s*</tool_call>", re.DOTALL)
PARAMETER = re.compile(r"<parameter=([\w.-]+)>(.*?)</parameter>", re.DOTALL)


def tool_argument(value, schema):
    value = value.strip()
    if schema.get("type") == "string":
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Preserve malformed values so the benchmark can report tool errors.
        return value


def parse_tool_calls(content, tools):
    schemas = {
        tool["function"]["name"]: tool["function"].get("parameters", {}).get("properties", {})
        for tool in tools
    }
    calls = []

    def replace(match):
        name, parameters = match.groups()
        if PARAMETER.sub("", parameters).strip():
            return match[0]
        schema = schemas.get(name, {})
        arguments = {
            key: tool_argument(value, schema.get(key, {}))
            for key, value in PARAMETER.findall(parameters)
        }
        calls.append({
            "id": f"call_{uuid.uuid4().hex}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        })
        return ""

    return TOOL_BLOCK.sub(replace, content).strip(), calls


def complete(model, processor, request, model_name, *, context_limit=32768, completion_limit=4096,
             fit_completion_to_context=False, default_completion_to_context=False, deadline=None):
    import torch

    if request.get("stream") or request.get("model") != model_name:
        raise ValueError("Use the served model name and disable streaming")
    tools = request.get("tools") if request.get("tool_choice") != "none" else None
    template_options = request.get("chat_template_kwargs", {})
    if not isinstance(template_options, dict):
        raise ValueError("chat_template_kwargs must be an object")
    enable_thinking = template_options.get("enable_thinking", request.get("enable_thinking", False))
    thinking_options = {"enable_thinking": enable_thinking}
    if "preserve_thinking" in template_options:
        thinking_options["preserve_thinking"] = template_options["preserve_thinking"]
    if any(not isinstance(value, bool) for value in thinking_options.values()):
        raise ValueError("Thinking options must be boolean")
    if "reasoning_effort" in request:
        effort = request["reasoning_effort"]
        if effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}")
        thinking_options["reasoning_effort"] = effort
    inputs = processor.apply_chat_template(
        prepare_messages(request["messages"]), add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
        tools=tools, **thinking_options,
    )
    prompt_tokens = inputs["input_ids"].shape[-1]
    remaining_tokens = context_limit - prompt_tokens
    requested_tokens = request.get("max_completion_tokens") or request.get("max_tokens")
    if requested_tokens is None and default_completion_to_context:
        max_tokens = min(completion_limit, remaining_tokens)
    else:
        max_tokens = int(requested_tokens or 1024)
    if not 1 <= max_tokens <= completion_limit:
        raise ValueError(f"Completion must request between 1 and {completion_limit} tokens")
    if fit_completion_to_context:
        max_tokens = min(max_tokens, remaining_tokens)
    if not 1 <= max_tokens <= remaining_tokens:
        raise ValueError(f"Prompt and completion exceed the {context_limit}-token context limit")
    inputs = inputs.to(model.device)
    temperature = float(request.get("temperature", 0))
    sampling = {}
    if temperature > 0:
        sampling = {"temperature": temperature, "top_p": request.get("top_p", 1)}
        if "top_k" in request:
            sampling["top_k"] = request["top_k"]
    stopping = {}
    if deadline is not None:
        from transformers import StoppingCriteriaList

        def expired(input_ids, scores, **kwargs):
            return time.monotonic() >= deadline

        if expired(None, None):
            raise GenerationDeadline("Request expired before decoding")
        stopping["stopping_criteria"] = StoppingCriteriaList([expired])
    with torch.inference_mode():
        tokens = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=bool(sampling),
            use_cache=True, **sampling, **stopping,
        )[0, prompt_tokens:]
    if deadline is not None and time.monotonic() >= deadline:
        raise GenerationDeadline("Generation stopped at its deadline", len(tokens))
    output = processor.decode(tokens, skip_special_tokens=True).strip()
    reasoning, separator, content = output.removeprefix("<think>").partition("</think>")
    if separator:
        reasoning, content = reasoning.strip(), content.strip()
    elif enable_thinking:
        reasoning, content = reasoning.strip(), ""
    else:
        reasoning, content = "", output
    finish_reason = "length" if len(tokens) == max_tokens else "stop"
    message = {"role": "assistant", "content": content or None}
    if tools and content:
        content, calls = parse_tool_calls(content, tools)
        if calls:
            message.update(content=content or None, tool_calls=calls)
            finish_reason = "tool_calls"
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-local", "object": "chat.completion", "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(tokens),
                  "total_tokens": prompt_tokens + len(tokens)},
    }


def serve(snapshot, adapter, model_name, port, ready_file, *, generation_timeout=None, **token_limits):
    model, processor = load_checkpoint(snapshot, adapter)
    generation_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, response):
            payload = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            try:
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                self.log_error("Client disconnected before receiving the reply")

        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, {"model": model_name})
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            started = time.monotonic()
            try:
                deadline = None
                if generation_timeout is not None:
                    deadline = started + generation_timeout
                    client_deadline = float(self.headers.get(DEADLINE_HEADER, deadline))
                    if not math.isfinite(client_deadline):
                        raise ValueError("Invalid generation deadline")
                    deadline = min(deadline, client_deadline)
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 32 * 1024**2:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(size))
                if deadline is None:
                    response = complete(model, processor, request, model_name, **token_limits)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not generation_lock.acquire(timeout=remaining):
                        raise GenerationDeadline("Request expired waiting for the model")
                    try:
                        if time.monotonic() >= deadline:
                            raise GenerationDeadline("Request expired waiting for the model")
                        response = complete(model, processor, request, model_name, deadline=deadline, **token_limits)
                    finally:
                        generation_lock.release()
            except GenerationDeadline as error:
                self.log_message("stop=%s tokens=%d server_elapsed=%.3fs",
                                 DEADLINE_ERROR, error.completion_tokens, time.monotonic() - started)
                self.send_json(504, {
                    "error": {"code": DEADLINE_ERROR, "message": str(error)},
                    "usage": {"completion_tokens": error.completion_tokens},
                })
            except (KeyError, TypeError, ValueError) as error:
                self.send_json(400, {"error": {"message": str(error)}})
            except Exception as error:
                self.log_error("%s", error)
                self.send_json(500, {"error": {"message": str(error)}})
            else:
                if deadline is not None:
                    self.log_message("stop=%s tokens=%d server_elapsed=%.3fs",
                                     response["choices"][0]["finish_reason"], response["usage"]["completion_tokens"],
                                     time.monotonic() - started)
                self.send_json(200, response)

    server_type = ThreadingHTTPServer if generation_timeout is not None else HTTPServer
    with server_type(("127.0.0.1", port), Handler) as server:
        if ready_file:
            ready_file.write_text(str(server.server_port))
        else:
            print(server.server_port, flush=True)
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--context-limit", type=int, default=32768)
    parser.add_argument("--completion-limit", type=int, default=4096)
    parser.add_argument("--fit-completion-to-context", action="store_true")
    parser.add_argument("--default-completion-to-context", action="store_true")
    parser.add_argument("--generation-timeout", type=float, metavar="SECONDS")
    args = parser.parse_args()
    if not 1 <= args.completion_limit <= args.context_limit:
        parser.error("Token limits must be positive and the completion limit must fit the context")
    if args.generation_timeout is not None and not (0 < args.generation_timeout < math.inf):
        parser.error("--generation-timeout must be a finite positive number")
    serve(
        args.snapshot, args.adapter, args.model, args.port, args.ready_file,
        context_limit=args.context_limit, completion_limit=args.completion_limit,
        fit_completion_to_context=args.fit_completion_to_context,
        default_completion_to_context=args.default_completion_to_context,
        generation_timeout=args.generation_timeout,
    )
