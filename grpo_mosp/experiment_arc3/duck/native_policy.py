"""Run the pinned Duck policy with the evaluation's local inference client."""

import json
import time

from inference.agent import tool_agent


class NativeDuckAnalyzer(tool_agent.ToolAgent):
    def __init__(self, client, endpoint, *, record=None, yield_seconds=None):
        settings = client.config["analyzer"]
        super().__init__(model=getattr(client, "model", "base"), base_url=endpoint, provider="vllm", api_key="local",
                         timeout=0, save_request_logs=settings["save_request_logs"])
        self.client = client
        self.move_budget = client.move_budget
        self.model_actions = 0
        self.record = record
        self._failure = None
        self._analysis_deadline = None
        self._current_task = None
        self._max_output_tokens = client.max_output_tokens or settings["max_output"] or None
        self._reply_reserve_tokens = self._max_output_tokens or 512
        self._context_budget_tokens = max(
            1024, client.config["shared"]["context_window"]
            - self._reply_reserve_tokens - self._request_safety_margin_tokens,
        )
        self._yield_seconds = yield_seconds
        self._tool_steps = settings["tool_steps"] or None
        self._python_timeout = settings["tool_timeout"]
        self._tool_output_tokens = settings["tool_output_tokens"]
        self._tool_output_chars = self._tool_output_tokens * 4
        self._system_prompt = tool_agent._build_system_prompt(tool_output_tokens=self._tool_output_tokens)

    def refresh_after_reset(self, state_path, payload):
        self._ensure_session(state_path)
        self._last_action_result = self._compact_action_result(payload)
        self._last_step_summary = self._summarize_step_sequence([self._last_action_result])

    def _build_user_message(self, user_prompt, current_frame):
        self._current_task = super()._build_user_message(user_prompt, current_frame)
        return self._current_task

    def _persistent_history_messages(self, messages, *, tools=None):
        # The previous turn can be discarded when storing history for a new task.
        task, self._current_task = self._current_task, None
        try:
            return super()._persistent_history_messages(messages, tools=tools)
        finally:
            self._current_task = task

    def _force_reduce_messages(self, messages, *, preserve_recent=1):
        if self._current_task is None:
            return super()._force_reduce_messages(messages, preserve_recent=preserve_recent)
        excess = self._context_budget_tokens - self._estimate_request_input_tokens(messages) + 1
        return self._trim_messages_for_context(messages, extra_safety_tokens=max(0, excess))

    def _trim_messages_for_context(self, messages, *, tools=None, preserve_recent=1, extra_safety_tokens=0):
        task = self._current_task
        if task is None:
            return super()._trim_messages_for_context(
                messages, tools=tools, preserve_recent=preserve_recent, extra_safety_tokens=extra_safety_tokens,
            )

        task_index = next((index for index, message in enumerate(messages) if message is task), None)
        if task_index is None:
            self._fail_context("current task is missing")
        system = messages[0]
        previous = list(messages[1:task_index])
        replies = list(messages[task_index + 1:])
        budget = max(1, self._context_budget_tokens - max(0, extra_safety_tokens))

        def candidate():
            return [system, *previous, task, *replies]

        def fits():
            return self._estimate_request_input_tokens(candidate(), tools=tools) <= budget

        while previous and not fits():
            self._drop_oldest_history_block(previous, preserve_recent=0)
        if fits():
            return candidate()

        # Pin the most recent assistant tool call and its results within this task.
        latest_exchange_size = next(
            (len(replies) - index for index in range(len(replies) - 2, -1, -1)
             if replies[index].get("tool_calls") and replies[index + 1]["role"] == "tool"),
            0,
        )
        while self._drop_oldest_history_block(replies, preserve_recent=latest_exchange_size):
            if fits():
                return candidate()

        if latest_exchange_size:
            end = next((index for index in range(1, len(replies)) if replies[index]["role"] != "tool"), len(replies))
            assistant, *results = replies[:end]
            replies = [assistant, *results]
            if fits():
                return candidate()
            limit = max(len(message["content"]) for message in results)
            while limit:
                limit //= 2
                replies = [assistant, *(
                    {**message, "content": message["content"][:limit] + "..."}
                    if len(message["content"]) > limit else message
                    for message in results
                )]
                if fits():
                    return candidate()

        self._fail_context("current task and latest tool exchange exceed the context budget")

    def _fail_context(self, reason):
        self._failure = RuntimeError(f"context_budget_exhausted: {reason}")
        raise self._failure

    @property
    def generated_tokens(self):
        return self.client.generated_tokens

    @property
    def model_calls(self):
        return self.client.model_calls

    @property
    def inference_seconds(self):
        return self.client.inference_seconds

    def _dispatch_tool(self, state_path, name, arguments):
        timeout = self._python_timeout
        remaining = self.move_budget.deadline - time.monotonic()
        # The upstream Python worker accepts timeouts in whole seconds.
        self._python_timeout = min(timeout, max(1, int(remaining)))
        try:
            return super()._dispatch_tool(state_path, name, arguments)
        finally:
            self._python_timeout = timeout

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
            self._current_task = None
        if self._failure is not None:
            raise self._failure
        return result
