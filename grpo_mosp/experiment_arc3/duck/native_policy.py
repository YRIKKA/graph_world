"""Run the pinned Duck policy with the evaluation's local inference client."""

import json
import time

from inference.agent import tool_agent


class NativeDuckAnalyzer(tool_agent.ToolAgent):
    def __init__(self, client, endpoint, *, record=None):
        settings = client.config["analyzer"]
        super().__init__(model=getattr(client, "model", "base"), base_url=endpoint, provider="vllm", api_key="local",
                         timeout=0, save_request_logs=settings["save_request_logs"])
        self.client = client
        self.model_actions = 0
        self.record = record
        self._failure = None
        self._analysis_deadline = None
        self._max_output_tokens = client.max_output_tokens or settings["max_output"] or None
        self._reply_reserve_tokens = self._max_output_tokens or 512
        self._context_budget_tokens = max(
            1024, client.config["shared"]["context_window"]
            - self._reply_reserve_tokens - self._request_safety_margin_tokens,
        )
        self._tool_steps = settings["tool_steps"] or None
        self._python_timeout = settings["tool_timeout"]
        self._tool_output_tokens = settings["tool_output_tokens"]
        self._tool_output_chars = self._tool_output_tokens * 4
        self._system_prompt = tool_agent._build_system_prompt(tool_output_tokens=self._tool_output_tokens)

    @property
    def generated_tokens(self):
        return self.client.generated_tokens

    @property
    def model_calls(self):
        return self.client.model_calls

    @property
    def inference_seconds(self):
        return self.client.inference_seconds

    def _chat_completion(self, messages, *, tools, request_timeout_seconds=None):
        remaining = None
        if self._analysis_deadline is not None:
            remaining = max(0, self._analysis_deadline - time.monotonic())
        try:
            response = self.client.completion(messages, tools, remaining, record=self.record)
        except RuntimeError as error:
            # ToolAgent logs exceptions internally and may return None. Pass the
            # original inference failure back to Tufa's pass reporter.
            self._failure = error
            raise
        choice = response["choices"][0]
        return tool_agent._ChatCompletionResult(choice["message"], choice["finish_reason"], response["usage"])

    def analyze(self, state_path, action_num, *, step_env, transcript_path,
                analysis_step, request_timeout_seconds, **kwargs):
        self._analysis_deadline = (None if request_timeout_seconds is None
                                   else time.monotonic() + request_timeout_seconds)
        self._failure = None
        prior_record = self.record

        def record(event):
            with transcript_path.open("a", encoding="utf-8") as stream:
                stream.write(f"Decision {analysis_step} inference: {json.dumps(event)}\n")
            if prior_record is not None:
                prior_record(event)

        def act(payload):
            self.model_actions += len(payload["actions"])
            return step_env(payload)

        self.record = record
        try:
            result = super().analyze(
                state_path, action_num, step_env=act, transcript_path=transcript_path,
                analysis_step=analysis_step, **kwargs,
            )
        finally:
            self.record = prior_record
        if self._failure is not None:
            raise self._failure
        return result
