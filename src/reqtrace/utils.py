import time

import litellm
from rich.console import Console

console = Console(force_terminal=True)

_NON_RETRYABLE = (
    litellm.BadRequestError,
    litellm.AuthenticationError,
    litellm.PermissionDeniedError,
    litellm.NotFoundError,
    litellm.UnprocessableEntityError,
)


def with_retry(fn, *args, max_retries: int = 5, base_delay: float = 0.3):
    """Call fn(*args) with exponential backoff on transient failures.

    Delays between attempts: 300ms → 600ms → 1.2s → 2.4s → 4.8s.
    Client errors (4xx) are re-raised immediately without retrying.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn(*args)
        except _NON_RETRYABLE:
            raise
        except Exception as e:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            console.print(
                f"    [yellow]Attempt {attempt + 1} failed ({type(e).__name__}: {e}), "
                f"retrying in {delay:.1f}s…[/yellow]"
            )
            time.sleep(delay)
