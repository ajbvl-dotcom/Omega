import os
import lib_llm_ext as llm
import providers
from src.logger import get_logger
from config import config_get_by_key

logger = get_logger(__name__)

class OpenAIProvider(providers.LLMProvider):

    def __init__(self):
        super().__init__()

    def start(self) -> None:
        openai_model = config_get_by_key("openai_model", "gpt-5.5")
        model = config_get_by_key("model", openai_model)
        self.delegate = OpenAIProviderImpl("OpenAI", "OPENAI_API_KEY",
                                           model, "https://api.openai.com/v1")

    def stop(self) -> None:
        self.delegate.stop()

    def respond(self, prompt, max_tokens=6000, reasoning_mode="medium", tools=None):
        return self.delegate.respond(prompt, max_tokens, reasoning_mode, tools)

    def tool_results(self, results):
        return self.delegate.tool_results(results)

    def end_turn(self, status):
        return self.delegate.end_turn(status)

    def chat(self, prompt: str, max_tokens: int = 6000, reasoning_mode: str = "medium") -> str:
        return self.delegate.chat(prompt, max_tokens, reasoning_mode)

def loadOmegaPlugin():
    providers.registerLLMProvider("OpenAI", OpenAIProvider())

class OpenAIProviderImpl(llm.AIProvider):
    """OpenAI provider using the Responses API (reasoning models)."""

    def respond(self, content, max_tokens=6000, reasoning="medium", tools=None):
        """Complete Responses transport; no command synthesis or text cleaning."""
        self._ensure_client()
        if self._client is None:
            raise RuntimeError(f"{self.name} not configured (set {self._var_name})")
        sysmsg, usermsg = llm._split_system_user(content)
        import copy
        replay = copy.deepcopy(getattr(self, "_replay", []))
        replay.append({"role": "user", "content": usermsg})
        kwargs = {"model": self._model_name, "instructions": sysmsg,
                  "input": replay, "max_output_tokens": int(max_tokens),
                  "reasoning": {"effort": reasoning}, "tools": tools or [],
                  "include": ["reasoning.encrypted_content"]}
        self._replay = copy.deepcopy(replay)
        response = self._client.responses.create(**kwargs)
        raw = response.model_dump(mode="json")
        # Replay is transport state, never a dispatch decision. Keep every output
        # item in its original order, including reasoning and assistant phase.
        self._replay = replay + copy.deepcopy(raw.get("output", []))
        return raw

    def tool_results(self, results):
        import json
        import copy
        replay = copy.deepcopy(getattr(self, "_replay", []))
        # Commit a complete batch atomically and tolerate mechanical redelivery
        # after an uncertain receipt. Do not flatten native result sets.
        for call_id, result in results:
            output = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
            existing = [item for item in replay if item.get("type") == "function_call_output"
                        and item.get("call_id") == call_id]
            if existing:
                if any(item["output"] != output for item in existing):
                    raise ValueError("Conflicting replay result for call ID")
            else:
                replay.append({"type": "function_call_output", "call_id": call_id,
                               "output": output})
        self._replay = replay

    def end_turn(self, status):
        # The forwarder first delivers any retained native results. Remaining
        # missing results carry no proof of non-execution (e.g. interrupted
        # native work); report uncertainty, never synthesize not_executed=True.
        replay = getattr(self, "_replay", [])
        answered = {item.get("call_id") for item in replay
                    if item.get("type") == "function_call_output"}
        pending = [item.get("call_id") for item in replay
                   if item.get("type") == "function_call" and item.get("call_id") not in answered]
        self.tool_results([[call_id, {"execution": "unknown", "native_result": None,
                                     "turn_status": status}]
                           for call_id in dict.fromkeys(pending) if call_id])

    def chat(self, content: str, max_tokens: int = 6000, reasoning: str = "medium", **kwargs) -> str:
        """Send chat request via the Responses API, initializing client if needed."""
        self._ensure_client()

        if self._client is None:
            raise RuntimeError(f"{self.name} not configured (set {self._var_name})")

        sysmsg, usermsg = llm._split_system_user(content)

        try:
            create_kwargs = {
                "instructions": sysmsg,
                "model": self._model_name,
                "input": usermsg,
                "max_output_tokens": max_tokens,
                "reasoning": {"effort": reasoning},
                "prompt_cache_key": config_get_by_key("OPENAI_PROMPT_CACHE_KEY", llm._stable_cache_key("openai", self._model_name, sysmsg)),
            }
            # GPT-5.5 supports only 24h; GPT-5.4 also supports extended retention.
            if self._model_name.startswith(("gpt-5.5", "gpt-5.4")):
                create_kwargs["prompt_cache_retention"] = "24h"

            create_kwargs.update(kwargs)

            response = self._client.responses.create(**create_kwargs)

            raw = response.output_text or ""
            incomplete_details = getattr(response, "incomplete_details", None)
            incomplete_reason = getattr(incomplete_details, "reason", None)
            llm._log_raw(self._name, self._model_name, raw)
            llm._log_responses_completion(self._name, self._model_name, response)
            if not raw:
                logger.warning("LLM returned an empty response")
                if incomplete_reason == "max_output_tokens":
                    raw = llm._llm_empty_response_command()
            return self._clean_text(raw)
        except Exception as e:
            logger.exception(f"[OpenAIProviderImpl.chat]: Exception while communicating with LLM: {e}")
            return ""
