"""Chat model abstraction.

The agent depends only on the small ``LLM`` protocol, so tests can inject a
scripted fake and the OpenAI-backed implementation stays swappable.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from app.config import Settings
from app.errors import LLMError, LLMUnavailableError

logger = logging.getLogger(__name__)


class LLM(Protocol):
    available: bool

    def complete(self, system: str, user: str, *, config: dict | None = None) -> str: ...


class OpenAIChatLLM:
    """LangChain ``ChatOpenAI`` wrapper with uniform error handling."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.available = settings.llm_configured
        self._model = None

    @property
    def model(self):
        if not self.available:
            raise LLMUnavailableError("OPENAI_API_KEY is not configured.")
        if self._model is None:
            from langchain_openai import ChatOpenAI

            kwargs: dict[str, Any] = dict(
                model=self.settings.openai_model,
                api_key=self.settings.openai_api_key.get_secret_value(),
                timeout=self.settings.openai_timeout_s,
                max_retries=self.settings.openai_max_retries,
                temperature=self.settings.llm_temperature,
            )
            if self.settings.openai_base_url:
                kwargs["base_url"] = self.settings.openai_base_url
            self._model = ChatOpenAI(**kwargs)
        return self._model

    def complete(self, system: str, user: str, *, config: dict | None = None) -> str:
        import openai
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            response = self.model.invoke([SystemMessage(system), HumanMessage(user)], config=config)
        except LLMUnavailableError:
            raise
        except openai.AuthenticationError as exc:
            raise LLMError("The OpenAI API rejected the configured credentials.") from exc
        except openai.RateLimitError as exc:
            raise LLMError("The OpenAI API rate limit or quota was exceeded.") from exc
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise LLMError("Could not reach the OpenAI API (timeout or network error).") from exc
        except openai.APIError as exc:
            raise LLMError(f"The OpenAI API returned an error: {exc.__class__.__name__}.") from exc
        content = response.content
        if isinstance(content, list):  # some models return content parts
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        return str(content).strip()
