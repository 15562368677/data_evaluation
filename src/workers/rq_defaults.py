"""Shared RQ retention defaults for background jobs."""

import os


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# Execution timeout for a single PnP batch job.
RQ_JOB_TIMEOUT = _get_int_env("RQ_JOB_TIMEOUT", 3600)

# Keep successful job metadata briefly for troubleshooting, then let Redis reclaim it.
RQ_RESULT_TTL = _get_int_env("RQ_RESULT_TTL", 3600)

# Keep failed jobs for a few days so we can inspect recent errors without holding them for a year.
RQ_FAILURE_TTL = _get_int_env("RQ_FAILURE_TTL", 7 * 24 * 3600)
