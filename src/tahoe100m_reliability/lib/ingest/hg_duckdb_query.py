"""
Handles querying HuggingFace datasets with DuckDB.

Implements a staggered batch process to deal with HF rate limits.
"""
import duckdb
import os
import time
import logging
import random
import re

from concurrent.futures import ThreadPoolExecutor
from itertools import count

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def execute_with_retry(
    con,
    query,
    batch,
    max_retries=10,
    initial_backoff=30,
    max_backoff=300,
):
    """Execute query, retrying with server-provided or exponential backoff."""

    backoff = initial_backoff

    for attempt in range(max_retries + 1):
        try:
            return con.execute(query, [batch]).fetchall()

        except (duckdb.HTTPException, duckdb.InvalidInputException, duckdb.InterruptException) as e:
            error = str(e)
            lower_error = error.lower()

            rate_limited = any(
                x in lower_error for x in
                [
                    '429', 'rate limit', 'too many requests',
                    'could not resolve hostname error for http head to',
                    'interrupted'
                ]
            )

            if not rate_limited:
                raise

            if attempt == max_retries:
                raise

            # Try to extract Retry-After if DuckDB included it
            # in the exception message.
            retry_after = None

            match = re.search(
                r"Retry-After[:\s]+(\d+)",
                error,
                re.IGNORECASE,
            )

            if match:
                retry_after = int(match.group(1))

            if retry_after is not None:
                delay = retry_after
                reason = "Retry-After"
            else:
                delay = min(
                    backoff * random.uniform(0.8, 1.2),
                    max_backoff,
                )
                reason = "exponential backoff"

            log.warning(
                "Rate limit or network error detected. "
                "Sleeping %.1fs (%s, attempt %d/%d)",
                delay,
                reason,
                attempt + 1,
                max_retries,
            )

            time.sleep(delay)

            if retry_after is None:
                backoff = min(backoff * 2, max_backoff)


def batch_duckdb_query(
    process_batch,
    parquet_files: list[str],
    query: str,
    batch_size: int = 45,
    start_index: int = 0,
    max_workers: int = 1,
):
    """Batch DuckDB queries with adaptive rate-limit handling."""

    con = duckdb.connect()

    hf_token = os.environ.get("HF_TOKEN")
    con.execute(f"""
        CREATE SECRET HF_TOKEN (
            TYPE HUGGINGFACE,
            TOKEN '{hf_token}'
        )
    """)

    start_time = time.perf_counter()
    parquet_files = parquet_files[start_index:]

    log.info(
        "Processing %d parquet files in batches of %d...",
        len(parquet_files),
        batch_size,
    )

    batch_id = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for index in range(0, len(parquet_files), batch_size):
            batch = parquet_files[index:index + batch_size]

            log.info(
                f"Processing batch {batch_id} starting at file {index + start_index}...",
            )

            rows = execute_with_retry(con, query, batch)

            futures.append(
                executor.submit(
                    process_batch,
                    rows,
                    batch_id,
                )
            )
            batch_id += 1

            del rows

        for future in futures:
            future.result()

    con.close()
    elapsed = time.perf_counter() - start_time

    log.info(
        "DuckDB query for all samples took %.2f seconds",
        elapsed,
    )
