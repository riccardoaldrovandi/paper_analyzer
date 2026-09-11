import time


class QuotaExceededError(Exception):
    """Raised when a Gemini API call fails with HTTP 429 / RESOURCE_EXHAUSTED (free-tier quota exhausted)."""
    pass


def call_gemini_with_retry(func, max_retries: int = 5, base_delay: float = 5.0):
    """
    Runs a zero-argument callable performing a Gemini API request, retrying with
    linear backoff on transient server errors (HTTP 503 / UNAVAILABLE, i.e. the
    model being temporarily overloaded). A 429 / RESOURCE_EXHAUSTED quota error is
    NOT retried, since hammering an exhausted daily quota with the same key just
    burns time, and is instead raised as QuotaExceededError so callers can fall
    back to a different provider.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_text = str(e)
            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                raise QuotaExceededError(error_text) from e

            last_error = e
            if ("503" in error_text or "UNAVAILABLE" in error_text) and attempt < max_retries:
                delay = base_delay * attempt
                print(f"[~] Gemini temporarily overloaded (503), attempt {attempt}/{max_retries}. Retrying in {delay:.0f}s...")
                time.sleep(delay)
                continue
            raise

    raise last_error
