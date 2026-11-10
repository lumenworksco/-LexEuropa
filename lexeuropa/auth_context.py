"""Which Hugging Face token pays for the model calls of the request being served.

Local use: the owner's `HF_TOKEN` from `.env`, as always.

Public demo (`LEXEUROPA_PUBLIC=1`): the Space holds NO credentials of its own. Every model call must
use the token of the visitor who signed in with Hugging Face ("Sign in with HF", scope `inference-api`),
so their own free inference credit pays. In that mode `current_token()` never falls back to a global
token: a call site that forgets to bind one fails loudly instead of silently billing the owner.

The token lives in a `ContextVar`, bound per request. `ContextVar`s do not flow into thread-pool
workers by themselves, so fan-out code must use `map_in_context` / `submit_in_context`, which give
each task its own copy of the caller's context.
"""
from __future__ import annotations

import contextlib
import contextvars
from concurrent.futures import Executor, Future
from typing import Callable, Iterable, Iterator, TypeVar

from lexeuropa.config import HF_TOKEN, PUBLIC_MODE

T = TypeVar("T")
R = TypeVar("R")

_request_token: contextvars.ContextVar[str | None] = contextvars.ContextVar("lexeuropa_request_token", default=None)


class SignInRequired(RuntimeError):
    """A model call was attempted in public mode with no visitor token bound."""


@contextlib.contextmanager
def bound_token(token: str | None) -> Iterator[None]:
    marker = _request_token.set(token)
    try:
        yield
    finally:
        _request_token.reset(marker)


def current_token() -> str | None:
    """The token to authenticate with, or None. In public mode only a bound visitor token counts."""
    bound = _request_token.get()
    if bound:
        return bound
    return None if PUBLIC_MODE else HF_TOKEN


def submit_in_context(pool: Executor, fn: Callable[..., R], *args) -> Future:
    """pool.submit, but the task runs in its own copy of the current context (so it sees the bound token)."""
    return pool.submit(contextvars.copy_context().run, fn, *args)


def map_in_context(pool: Executor, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
    """Like list(pool.map(fn, items)), each call in its own copy of the current context; order is kept."""
    futures = [submit_in_context(pool, fn, item) for item in items]
    return [f.result() for f in futures]
