import time


class QuotaExceededError(Exception):
    """Raised when a Gemini API call fails with HTTP 429 / RESOURCE_EXHAUSTED (free-tier quota exhausted)."""
    pass


class GeminiOverloadedError(Exception):
    """Raised when a Gemini API call keeps failing with HTTP 503 / UNAVAILABLE after retries are exhausted."""
    pass


# Circuit breaker shared by every call_gemini_with_retry() call in the process.
# Google exposes no "is this model available right now" endpoint -- the only
# way to find out is to actually call generate_content and see what comes
# back. So once a call confirms Gemini is down (quota exhausted or
# persistently overloaded), we remember that verdict for a cooldown window
# instead of re-discovering it (and burning another request) on every one of
# the possibly dozens of figures/pages still left to process in this run.
_unavailable_until = 0.0
_unavailable_error = None
QUOTA_COOLDOWN_SECONDS = 900   # daily quota won't refill within a run either way
OVERLOAD_COOLDOWN_SECONDS = 60  # 503 spikes are usually brief


def call_gemini_with_retry(func, max_retries: int = 2, base_delay: float = 5.0):
    """
    Runs a zero-argument callable performing a Gemini API request, retrying with
    linear backoff on transient server errors (HTTP 503 / UNAVAILABLE, i.e. the
    model being temporarily overloaded). A 429 / RESOURCE_EXHAUSTED quota error is
    NOT retried, since hammering an exhausted daily quota with the same key just
    burns time, and is instead raised as QuotaExceededError so callers can fall
    back to a different provider.

    max_retries is deliberately kept low: every attempt (including retries) is a
    real request that Google counts against the free-tier daily quota (as low as
    20 requests/day/model) even when it fails with a 503, so retrying an
    overloaded model 5 times could burn a quarter of the day's whole quota on a
    single call that was never going to succeed. Once retries are exhausted,
    GeminiOverloadedError is raised so callers can fall back to a different
    provider the same way they already do for QuotaExceededError.

    Before even attempting a request, checks the module-level circuit breaker
    (see above) and immediately re-raises the last known failure -- with zero
    extra requests spent -- if Gemini was confirmed unavailable within the
    cooldown window.
    """
    global _unavailable_until, _unavailable_error

    if time.time() < _unavailable_until:
        remaining = int(_unavailable_until - time.time())
        print(f"[~] Skipping Gemini call: still in cooldown after a recent failure ({remaining}s left). Using fallback directly.")
        raise _unavailable_error

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = func()
            _unavailable_until = 0.0  # a call went through: clear any stale breaker
            return result
        except Exception as e:
            error_text = str(e)
            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                err = QuotaExceededError(error_text)
                _unavailable_until = time.time() + QUOTA_COOLDOWN_SECONDS
                _unavailable_error = err
                raise err from e

            is_overloaded = "503" in error_text or "UNAVAILABLE" in error_text
            last_error = e
            if is_overloaded and attempt < max_retries:
                delay = base_delay * attempt
                print(f"[~] Gemini temporarily overloaded (503), attempt {attempt}/{max_retries}. Retrying in {delay:.0f}s...")
                time.sleep(delay)
                continue
            if is_overloaded:
                err = GeminiOverloadedError(error_text)
                _unavailable_until = time.time() + OVERLOAD_COOLDOWN_SECONDS
                _unavailable_error = err
                raise err from e
            raise

    raise last_error
