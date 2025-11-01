"""
scheduler_weekly.py
--------------------
Thin wrapper around `scheduler_daily` that configures the digest pipeline for a weekly cadence.

Usage:
    $ python src/scheduler_weekly.py --dry-run
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from typing import Optional

from scheduler_daily import DigestConfig, ordinal, run_daily_job


def _format_weekly_period(run_dt: datetime, start_date: date) -> str:
    """Return formatted period string like 'Oct 01, 2025 - Oct 07, 2025 (41ST WEEK)'."""
    end_date = run_dt.date()
    span = f"{start_date:%b %d, %Y} - {end_date:%b %d, %Y}"
    week_label = f"{ordinal(run_dt.isocalendar().week).upper()} WEEK"
    return f"{span} ({week_label})"


WEEKLY_CONFIG = DigestConfig(
    label="Weekly",
    default_days_back=7,
    intro_line=lambda run_dt, start_date, topic: f"[Weekly RAG Digest] {_format_weekly_period(run_dt, start_date)} – {topic}",
    html_heading=lambda run_dt, _start_date: "Weekly RAG Digest",
    html_date_line=lambda run_dt, start_date: _format_weekly_period(run_dt, start_date),
    schedule_note="You’re receiving this because you enabled the GitHub Action for weekly research summaries.",
    subject_line=lambda run_dt, start_date, topic: (
        f"[Weekly RAG Digest] {_format_weekly_period(run_dt, start_date)} – {topic}"
    ),
    markdown_title=lambda run_dt, start_date, body: (
        f"# 🧠 Weekly LLM Research Digest — {_format_weekly_period(run_dt, start_date)}\n\n{body}\n"
    ),
)


def run_weekly_job(days_back: Optional[int] = None, dry_run: bool = False) -> None:
    """
    Execute the weekly digest pipeline using the shared scheduler implementation.

    Args:
        days_back (int | None): Historical window to fetch papers from (defaults to config).
        dry_run (bool): Whether to skip emailing and instead save artifacts locally.
    """
    run_daily_job(days_back=days_back, dry_run=dry_run, config=WEEKLY_CONFIG)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--days-back",
        type=int,
        default=WEEKLY_CONFIG.default_days_back,
        help=f"Number of past days to include (default: {WEEKLY_CONFIG.default_days_back}, covers the past week).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending email; saves digest artifacts to ./artifacts/ instead.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_weekly_job(days_back=args.days_back, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
