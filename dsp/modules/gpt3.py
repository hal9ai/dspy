import functools
import json
from typing import Any, Literal, Optional, cast

import backoff
import openai
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion import Completion, CompletionChoice

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.modules.lm import LM


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details)
    )


class GPT3(LM):
    """Wrapper around OpenAI's GPT API. Supports both the OpenAI and Azure APIs.

    Args:
        model (str, optional): OpenAI or Azure supported LLM model to use. Defaults to "text-davinci-002".
        api_key (Optional[str], optional): API provider Authentication token. use Defaults to None.
        api_provider (Literal["openai", "azure"], optional): The API provider to use. Defaults to "openai".
        model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "text".
        **kwargs: Additional arguments to pass to the API provider.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo-instruct",
        api_key: Optional[str] = None,
        api_provider: Literal["openai", "azure"] = "openai",
        model_type: Literal["chat", "text"] | None = None,
        client: openai.AzureOpenAI | openai.OpenAI | None = None,
        **kwargs,
    ):
        super().__init__(model)
        self.provider = "openai"

        default_model_type = (
            "chat"
            if ("gpt-3.5" in model or "turbo" in model or "gpt-4" in model)
            and ("instruct" not in model)
            else "text"
        )
        self.model_type = model_type if model_type else default_model_type

        if api_provider == "azure":
            assert (
                "engine" in kwargs or "deployment_id" in kwargs
            ), "Must specify engine or deployment_id for Azure API instead of model."
            assert "api_version" in kwargs, "Must specify api_version for Azure API"
            assert (
                "azure_endpoint" in kwargs
            ), "Must specify azure_endpoint for Azure API"
            openai.api_type = "azure"
            if kwargs.get("api_version"):
                openai.api_version = kwargs["api_version"]
        if client:
            self.client = client
        else:
            if api_provider == "azure":
                self.client = openai.AzureOpenAI(
                    azure_endpoint=kwargs.pop('azure_endpoint'),
                    api_key=api_key,
                )
            else:
                self.client = openai.OpenAI(api_key=api_key)
        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }  # TODO: add kwargs above for </s>
        if api_key:
            kwargs["api_key"] = api_key
        if api_provider != "azure":
            self.kwargs["model"] = model
        self.history: list[dict[str, Any]] = []

    def _openai_client(self):
        return self.client

    def basic_request(self, prompt: str, **kwargs) -> ChatCompletion | Completion:
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}
        if self.model_type == "chat":
            # caching mechanism requires hashable kwargs
            kwargs["messages"] = [{"role": "user", "content": prompt}]
            kwargs = {"stringify_request": json.dumps(kwargs)}
            response = cached_gpt3_turbo_request(**kwargs)

        else:
            kwargs["prompt"] = prompt
            response = cached_gpt3_request(**kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.InternalServerError, openai.APIError),
        max_time=1000,
        on_backoff=backoff_hdlr,
    )
    def request(self, prompt: str, **kwargs) -> Completion| ChatCompletion:
        """Handles retreival of GPT-3 completions whilst handling rate limiting and caching."""
        if "model_type" in kwargs:
            del kwargs["model_type"]

        return self.basic_request(prompt, **kwargs)

    def _get_choice_text(self, choice: Choice| CompletionChoice) -> str:
        if isinstance(choice, Choice):
            return str(choice.message.content)
        else:
            return choice.text

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[str]:
        """Retrieves completions from GPT-3.

        Args:
            prompt (str): prompt to send to GPT-3
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        # if kwargs.get("n", 1) > 1:
        #     if self.model_type == "chat":
        #         kwargs = {**kwargs}
        #     else:
        #         kwargs = {**kwargs, "logprobs": 5}

        response = self.request(prompt, **kwargs)
        choices = response.choices

        completed_choices = [c for c in choices if c.finish_reason != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices #type:ignore

        completions = [self._get_choice_text(c) for c in choices]

        if return_sorted and kwargs.get("n", 1) > 1:
            scored_completions = []

            for c in choices:
                tokens, logprobs = (
                    c.logprobs.tokens,
                    c.logprobs.token_logprobs,
                )

                if "<|endoftext|>" in tokens:
                    index = tokens.index("<|endoftext|>") + 1
                    tokens, logprobs = tokens[:index], logprobs[:index]

                avglog = sum(logprobs) / len(logprobs)
                scored_completions.append((avglog, self._get_choice_text(c)))

            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]

        return completions


@CacheMemory.cache
def cached_gpt3_request_v2(client: openai.OpenAI| openai.AzureOpenAI, **kwargs):
    return client.completions.create(**kwargs)


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def cached_gpt3_request_v2_wrapped(**kwargs):
    return cached_gpt3_request_v2(**kwargs)


cached_gpt3_request = cached_gpt3_request_v2_wrapped


@CacheMemory.cache
def _cached_gpt3_turbo_request_v2(client: openai.OpenAI| openai.AzureOpenAI, **kwargs) -> ChatCompletion:
    if "stringify_request" in kwargs:
        kwargs = json.loads(kwargs["stringify_request"])
    return client.chat.completions.create(**kwargs)


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def _cached_gpt3_turbo_request_v2_wrapped(**kwargs) -> ChatCompletion:
    return _cached_gpt3_turbo_request_v2(**kwargs)


cached_gpt3_turbo_request = _cached_gpt3_turbo_request_v2_wrapped
