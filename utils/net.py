import random
import time
from typing import Any

import requests


def request_with_retry(method: str, url: str, *, timeout: int = 30,
                       max_attempts: int = 5, backoff_base: int = 2,
                       jitter: bool = True, **kwargs: Any) -> requests.Response:
    """Execute a ``requests`` call with exponential backoff and optional jitter.

    Parameters
    ----------
    method : str
        HTTP method such as ``"get"`` or ``"post"``.
    url : str
        Target URL.
    timeout : int, optional
        Timeout for the request in seconds, by default 30.
    max_attempts : int, optional
        Maximum number of attempts, by default 5.
    backoff_base : int, optional
        Base value for exponential backoff, by default 2.
    jitter : bool, optional
        When ``True`` apply random jitter to the sleep interval.
    **kwargs : Any
        Additional parameters forwarded to :func:`requests.request`.

    Returns
    -------
    requests.Response
        The successful HTTP response.

    Raises
    ------
    requests.RequestException
        If all attempts fail or a non-retryable HTTP error is encountered.
    """

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            # Don't retry for 4xx client errors
            if (
                isinstance(exc, requests.HTTPError)
                and exc.response is not None
                and exc.response.status_code < 500
            ):
                raise
            if attempt == max_attempts:
                raise
            sleep = backoff_base ** (attempt - 1)
            if jitter:
                sleep *= random.uniform(0, 1)
            time.sleep(sleep)
    # Should not reach here, but raise for safety
    raise RuntimeError("request_with_retry reached unreachable state")
