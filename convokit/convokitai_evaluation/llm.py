"""
LLM access for the evaluation Transformers.

``convokit.genai``'s ``GPTClient`` always sends ``temperature`` and ``max_tokens``, which
OpenAI's reasoning models (gpt-5, gpt-5-mini) reject. ``GPT5Client`` keeps the
``convokit.genai`` ``LLMClient`` interface and adds ``reasoning_effort``, an optional fixed
``seed`` and JSON mode.
"""

import json
import re
import time
from typing import Optional

from convokit.genai import GenAIConfigManager, LLMClient, LLMResponse

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

RETRY_AFTER = 10  # seconds


def parse_json(text: str):
    """
    Parse the JSON object or array in a model reply, tolerating Markdown code fences and control
    characters.

    :param text: the model reply
    :return: the parsed JSON (dict or list)
    :raises json.JSONDecodeError: if the reply is not valid JSON
    """
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text, strict=False)


class GPT5Client(LLMClient):
    """
    Client for OpenAI reasoning models (the gpt-5 family) with the ``convokit.genai``
    ``LLMClient`` interface.

    Failed calls (and, in ``generate_json``, unparseable replies) are retried after a
    10-second wait.

    :param model: model name, e.g. ``"gpt-5"`` or ``"gpt-5-mini"``
    :param config_manager: GenAIConfigManager holding the ``"gpt"`` API key (default: a
        GenAIConfigManager reading ``~/.convokit/config.yml``)
    :param reasoning_effort: OpenAI ``reasoning_effort`` (``"minimal"``, ``"low"``,
        ``"medium"`` or ``"high"``)
    :param seed: OpenAI ``seed`` for best-effort determinism (not sent if None)
    :param json_mode: request ``response_format={"type": "json_object"}``
    :param max_retries: how many times to retry a failed call or an unparseable JSON reply
    :raises ImportError: if the ``openai`` package is not installed
    :raises ValueError: if no OpenAI API key is configured
    """

    def __init__(
        self,
        model: str,
        config_manager: GenAIConfigManager = None,
        reasoning_effort: str = "minimal",
        seed: Optional[int] = None,
        json_mode: bool = False,
        max_retries: int = 1,
    ):
        if OpenAI is None:
            raise ImportError(
                "OpenAI client not available. Please install via `pip install convokit[genai]`."
            )
        if config_manager is None:
            config_manager = GenAIConfigManager()
        api_key = config_manager.get_api_key("gpt")
        if not api_key:
            raise ValueError("OpenAI API key is required. ")

        self.config_manager = config_manager
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self.json_mode = json_mode
        self.max_retries = max_retries

    def _call(self, prompt: str):
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": self.reasoning_effort,
        }
        if self.seed is not None:
            request["seed"] = self.seed
        if self.json_mode:
            request["response_format"] = {"type": "json_object"}
        return self.client.chat.completions.create(**request)

    def _retry(self, attempt):
        for n in range(self.max_retries + 1):
            try:
                return attempt()
            except Exception as e:
                if n == self.max_retries:
                    raise
                print(f"{type(e).__name__}: {e}. Retrying in {RETRY_AFTER}s...")
                time.sleep(RETRY_AFTER)

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a reply to a prompt.

        :param prompt: the prompt
        :param kwargs: ignored; accepted for compatibility with the ``LLMClient`` interface
        :return: an LLMResponse with the reply text, total tokens used (-1 if unknown), latency
            in seconds and the raw OpenAI response
        """
        start = time.time()
        raw = self._retry(lambda: self._call(prompt))
        tokens = raw.usage.total_tokens if raw.usage else -1
        return LLMResponse(
            text=raw.choices[0].message.content,
            tokens=tokens,
            latency=time.time() - start,
            raw=raw,
        )

    def generate_json(self, prompt: str):
        """
        Generate a reply to a prompt and parse it as JSON.

        :param prompt: the prompt
        :return: the parsed JSON (dict or list)
        """
        return self._retry(
            lambda: parse_json(self._call(prompt).choices[0].message.content)
        )
