"""
scheduler_daily.py
-------------------
Automates the daily RAG pipeline end-to-end and dispatches a themed research digest.

Responsibilities:
- Issue the generic large language model research query for daily summaries.
- Fetch the newest arXiv papers, stream and chunk PDFs, and rebuild the FAISS index.
- Call rag_answer.py to produce the templated Summary/Overall response used in reports.
- Parse either HTML or Markdown bullets into structured digest items with links.
- Write the markdown digest to ./reports/YYYY-MM-DD-llm-digest.md.
- Optionally email both plain-text and HTML versions (with `--dry-run` capturing artifacts).

Example:
    $ python src/scheduler_daily.py --days-back 1 --dry-run

Output:
    ./reports/YYYY-MM-DD-llm-digest.md
    ./artifacts/digest.{html,txt} (when --dry-run is supplied)
"""

import argparse
import html
import json
import logging
import os
import smtplib
import ssl
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Callable, Optional
import re

from rag_answer import answer_question

GENERIC_LLM_QUERY = (
    "Summarize the latest research on large language models (methods, training, alignment/safety, evaluation, "
    "efficiency, multimodal, and applications). Provide concise, linked bullets."
)


def ordinal(n: int) -> str:
    """Convert an integer into an ordinal string (1 -> '1st')."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


@dataclass(frozen=True)
class DigestConfig:
    """Configuration bundle controlling digest labeling, defaults, and formatting."""

    label: str
    default_days_back: int
    intro_line: Callable[[datetime, date, str], str]
    html_heading: Callable[[datetime, date], str]
    html_date_line: Callable[[datetime, date], str]
    schedule_note: str
    subject_line: Callable[[datetime, date, str], str]
    markdown_title: Callable[[datetime, date, str], str]


DAILY_CONFIG = DigestConfig(
    label="Daily",
    default_days_back=6,
    intro_line=lambda run_dt, _start_date, topic: f"[Daily RAG Digest] {run_dt:%A} – {topic}",
    html_heading=lambda run_dt, _start_date: "Daily RAG Digest",
    html_date_line=lambda run_dt, _start_date: run_dt.strftime("%A, %b %d, %Y"),
    schedule_note="You’re receiving this because you enabled the GitHub Action for daily research summaries.",
    subject_line=lambda run_dt, _start_date, topic: f"[Daily RAG Digest] {run_dt:%A} – {topic}",
    markdown_title=lambda run_dt, _start_date, body: (
        f"# 🧠 LLM Research Digest — {run_dt.date().isoformat()}\n\n{body}\n"
    ),
)


def get_daily_query() -> str:
    """
    Return the templated summarization query used for the daily digest.

    Returns:
        str: Scheduler prompt covering the current landscape of LLM research.
    """
    return GENERIC_LLM_QUERY


def _run_module_main(module, argv: list[str]) -> None:
    """
    Import `module` and invoke its `main()` while temporarily overriding `sys.argv`.

    Args:
        module (str): Module name to import (e.g., "fetch_arxiv").
        argv (list[str]): Argument vector to expose to the module during execution.
    """
    mod = __import__(module, fromlist=["main"])
    main_func = getattr(mod, "main")
    argv_backup = sys.argv.copy()
    try:
        sys.argv = argv
        main_func()
    finally:
        sys.argv = argv_backup


def topic_from_query(q: str) -> str:
    """
    Convert a scheduler query into a concise digest topic.

    Args:
        q (str): Full scheduler query string.

    Returns:
        str: Human-friendly topic text capped at 120 characters.
    """
    t = q.strip()
    for pref in ("Summarize latest ", "summarize latest ", "Summarize the latest ", "summarize the latest "):
        if t.startswith(pref):
            t = t[len(pref) :]
    return t.rstrip(".")[:120]


SUMMARY_ITEM_PATTERN = re.compile(
    r"<li[^>]*>\s*<b>\s*<a href=\"([^\"]+)\"[^>]*>(.*?)</a>\s*</b>\s*:\s*(.*?)</li>",
    re.IGNORECASE | re.DOTALL,
)


def parse_summary_items(answer_html: str) -> list[dict]:
    """
    Parse HTML list items produced by `rag_answer.py` into structured summaries.

    Args:
        answer_html (str): Model response expected to contain `<li>` elements.

    Returns:
        list[dict]: Each entry contains `title`, `url`, and `summary` keys.
    """
    items: list[dict] = []
    if not answer_html:
        return items
    for match in SUMMARY_ITEM_PATTERN.finditer(answer_html):
        url = match.group(1).strip()
        title = html.unescape(match.group(2).strip())
        raw_summary = match.group(3).strip()
        clean_summary = html.unescape(re.sub(r"<[^>]+>", "", raw_summary)).strip()
        items.append({"title": title, "url": url, "summary": clean_summary})
    return items


def build_digest_text(
    summary_items: list[dict],
    topic: str,
    run_dt: datetime,
    period_start: date,
    config: DigestConfig,
    fallback: str | None = None,
) -> str:
    """
    Render the plain-text digest body shared via email and reports.

    Args:
        summary_items (list[dict]): Structured summaries from `parse_summary_items`.
        topic (str): Digest topic label (e.g., "LLM retrieval methods").
        run_dt (datetime): Execution timestamp (UTC).
        period_start (date): Beginning of the reporting window.
        config (DigestConfig): Formatting configuration controlling labels and headings.
        fallback (str | None): Raw summary text to use when parsing fails.

    Returns:
        str: Plain-text digest containing heading and bullet list.
    """
    lines: list[str] = []
    lines.append(config.intro_line(run_dt, period_start, topic))
    lines.append("")
    lines.append("Summary:")
    if summary_items:
        for item in summary_items:
            bullet = f"- {item['title']}: {item['summary']}"
            if item.get("url"):
                bullet += f" ({item['url']})"
            lines.append(bullet)
    elif fallback:
        clean = html.unescape(re.sub(r"<[^>]+>", "", fallback))
        lines.append(clean)
    else:
        lines.append("No structured summary available.")
    return "\n".join(lines)


def build_digest_html(
    summary_items: list[dict],
    topic: str,
    run_dt: datetime,
    period_start: date,
    config: DigestConfig,
    fallback: str | None = None,
    context: str | None = None,
) -> str:
    """
    Render the HTML digest body with lightweight styling for email clients.

    Args:
        summary_items (list[dict]): Structured summaries from `parse_summary_items`.
        topic (str): Digest topic label.
        run_dt (datetime): Execution timestamp (UTC).
        period_start (date): Beginning of the reporting window.
        config (DigestConfig): Formatting configuration controlling labels and headings.
        fallback (str | None): Raw summary text to display when parsing fails.
        context (str | None): Retrieved context to surface during dry runs.

    Returns:
        str: Sanitized HTML string ready to embed in an email.
    """
    heading_title = html.escape(config.html_heading(run_dt, period_start))
    date_line = html.escape(config.html_date_line(run_dt, period_start))
    schedule_note = html.escape(config.schedule_note)
    if summary_items:
        bullet_entries: list[str] = []
        for item in summary_items:
            title = html.escape(item.get("title") or "Untitled")
            summary = html.escape(item.get("summary") or "")
            url = item.get("url") or ""
            if url:
                url = html.escape(url)
                bullet_entries.append(
                    f'<li style="margin-bottom:12px; line-height:1.5;"><b><a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a></b>: {summary}</li>'
                )
            else:
                bullet_entries.append(
                    f'<li style="margin-bottom:12px; line-height:1.5;"><b>{title}</b>: {summary}</li>'
                )
        bullet_html = "\n".join(bullet_entries)
    elif fallback:
        bullet_html = f'<li style="margin-bottom:12px; line-height:1.5;">{html.escape(fallback)}</li>'
    else:
        bullet_html = '<li style="margin-bottom:12px; line-height:1.5;">No structured summary available.</li>'

    context_block = ""
    if context:
        context_block = (
            "<h2 style=\"font-size:16px;margin:24px 0 8px 0;\">Retrieved Context (Dry Run)</h2>"
            f"<pre style=\"background:#f5f5f5;padding:12px;border-radius:8px;white-space:pre-wrap;font-size:12px;line-height:1.4;\">{html.escape(context)}</pre>"
        )

    return f"""<!doctype html>
    <html>
    <head><meta charset=\"utf-8\"></head>
    <body style=\"margin:0;padding:0;background:#f6f7fb;\">
      <table role=\"presentation\" width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" style=\"background:#f6f7fb;\">
        <tr><td align=\"center\" style=\"padding:24px;\">
          <table width=\"640\" cellpadding=\"0\" cellspacing=\"0\" style=\"background:#ffffff;border-radius:12px;padding:24px;font-family:Arial,Helvetica,sans-serif;color:#111;\">
            <tr><td>
              <h1 style=\"margin:0 0 8px 0;font-size:20px;\">{heading_title}</h1>
              <div style=\"color:#666;font-size:12px;margin-bottom:4px;\">{date_line}</div>
              <h2 style=\"font-size:16px;margin:0 0 12px 0;\">Summary</h2>
              <ul style=\"padding-left:18px;margin:0;list-style-type:disc;\">
                {bullet_html}
              </ul>
              {context_block}
              <hr style=\"margin:20px 0;border:none;border-top:1px solid #eee;\">
              <div style=\"font-size:12px;color:#666;\">{schedule_note}</div>
            </td></tr>
          </table>
        </td></tr>
      </table>
    </body>
    </html>"""


def send_email(subject: str, text_body: str, html_body: str) -> None:
    """
    Deliver the digest email using SMTP with plain-text and HTML alternatives.

    Args:
        subject (str): Email subject line.
        text_body (str): Plain-text representation of the digest.
        html_body (str): HTML representation of the digest.

    Side Effects:
        Sends email via credentials stored in EMAIL_* environment variables.
    """
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("EMAIL_RECEIVER")
    host = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("EMAIL_SMTP_PORT", "587"))

    if not all([sender, password, receiver]):
        print("Email not sent: missing EMAIL_SENDER/EMAIL_PASSWORD/EMAIL_RECEIVER")
        return
    recipients = [r.strip() for r in receiver.split(",") if r.strip()]
    if not recipients:
        print("Email not sent: EMAIL_RECEIVER did not contain any addresses.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(sender, password)
        server.sendmail(sender, recipients, msg.as_string())


def extract_answer(summary_text: str) -> str:
    """
    Remove the leading "Answer:" marker occasionally emitted by models.

    Args:
        summary_text (str): Raw stdout captured from `rag_answer.py`.

    Returns:
        str: Cleaned summary text without the prefix.
    """
    if not summary_text:
        return ""
    marker = "Answer:"
    idx = summary_text.find(marker)
    if idx == -1:
        return summary_text.strip()
    return summary_text[idx + len(marker) :].strip()


def run_daily_job(
    days_back: Optional[int] = None,
    dry_run: bool = False,
    *,
    config: DigestConfig = DAILY_CONFIG,
) -> None:
    """
    Execute the end-to-end digest pipeline and optionally send/email the digest.

    Args:
        days_back (int | None): Number of historical days to include when fetching papers.
        dry_run (bool): If True, skip email send and persist artifacts locally.
        config (DigestConfig): Formatting and schedule configuration; defaults to daily digest.

    Side Effects:
        Writes artifacts under `data/`, `indexes/`, `reports/`, and optionally `artifacts/`.
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    effective_days_back = config.default_days_back if days_back is None else days_back

    if dry_run:
        logger.info("[DRY-RUN] Running scheduler without sending email (artifacts will be saved).")

    logger.info("[FETCH] Fetching latest papers (days_back=%s)", effective_days_back)
    fetch_args = [
        "fetch_arxiv.py",
        "--days-back",
        str(effective_days_back),
        "--output",
        "data/arxiv_results.json",
    ]
    _run_module_main("fetch_arxiv", fetch_args)

    metadata_path = Path("data/arxiv_results.json")
    if not metadata_path.exists():
        logger.error("[FETCH] Metadata file %s not found.", metadata_path)
        return

    try:
        papers_raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("[FETCH] Failed to parse metadata JSON: %s", exc)
        return

    if not isinstance(papers_raw, list):
        logger.error("[FETCH] Unexpected metadata format; expected list of papers.")
        return

    if not papers_raw:
        print(f"No new papers found in the last {effective_days_back} day(s).")
        return

    prepared_papers = []
    for paper in papers_raw:
        if not isinstance(paper, dict):
            continue
        url = paper.get("pdf_url") or paper.get("id") or ""
        summary = (paper.get("summary") or "").replace("\n", " ").strip()
        prepared_papers.append(
            {
                "title": paper.get("title") or "Untitled",
                "url": url,
                "summary": summary,
            }
        )

    if not prepared_papers:
        print(f"No new papers found in the last {effective_days_back} day(s).")
        return

    num_papers = len(prepared_papers)

    logger.info("[INGEST] Streaming PDFs and chunking text")
    ingest_args = [
        "ingest_stream.py",
        "--input",
        "data/arxiv_results.json",
        "--chunk-size",
        "1200",
        "--chunk-overlap",
        "300",
        "--output",
        "data/chunks.jsonl",
    ]
    _run_module_main("ingest_stream", ingest_args)

    chunks_path = Path("data/chunks.jsonl")
    if not chunks_path.exists() or chunks_path.stat().st_size == 0:
        logger.info("[EMBED] No chunks produced; skipping embedding and summary.")
        return

    logger.info("[EMBED] Building FAISS index")
    embed_args = [
        "embed_index.py",
        "--build",
        "--input",
        "data/chunks.jsonl",
        "--index-path",
        "indexes/faiss.index",
        "--metadata-path",
        "indexes/docs_meta.json",
    ]
    _run_module_main("embed_index", embed_args)

    logger.info("[SUMMARIZE] Generating summary via OpenAI gpt-4o-mini")
    query = get_daily_query()
    retrieval_context = ""
    result = answer_question(
        query,
        provider="openai",
        openai_model="gpt-4o-mini",
        k=num_papers,
        include_context=True,
    )
    if isinstance(result, tuple):
        summary_text, retrieval_context = result
    else:
        summary_text = result
        retrieval_context = ""
    summary_text = (summary_text or "").strip()
    if not summary_text:
        logger.warning("[SUMMARIZE] Summary output was empty.")
        return

    run_dt = datetime.now(timezone.utc)
    period_end = run_dt.date()
    if effective_days_back and effective_days_back > 1:
        period_start = period_end - timedelta(days=effective_days_back - 1)
    else:
        period_start = period_end
    topic = topic_from_query(query)
    summary_items = parse_summary_items(summary_text)
    fallback_summary = summary_text if summary_items else None

    logger.info("[SAVE] Writing digest to reports directory")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    digest_path = reports_dir / f"{run_dt.date().isoformat()}-llm-digest.md"
    if summary_items:
        bullet_lines = [
            f"- [{item['title']}]({item['url']}): {item['summary']}"
            if item.get("url")
            else f"- {item['title']}: {item['summary']}"
            for item in summary_items
        ]
        digest_body = "Summary:\n" + "\n".join(bullet_lines)
    else:
        digest_body = fallback_summary or "No structured summary available."
    digest_content = config.markdown_title(run_dt, period_start, digest_body)
    digest_path.write_text(digest_content, encoding="utf-8")
    logger.info("[SAVE] Digest written to %s", digest_path)

    subject = config.subject_line(run_dt, period_start, topic)
    text_body = build_digest_text(summary_items, topic, run_dt, period_start, config, fallback_summary)
    html_body = build_digest_html(
        summary_items,
        topic,
        run_dt,
        period_start,
        config,
        fallback_summary,
        retrieval_context if dry_run else None,
    )

    if dry_run:
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "digest.html").write_text(html_body, encoding="utf-8")
        (artifacts_dir / "digest.txt").write_text(text_body, encoding="utf-8")
        logger.info("[DRY-RUN] Digest artifacts saved to %s; email not sent.", artifacts_dir)
        return

    send_email(subject, text_body, html_body)
    print("Email digest sent.")


def run_daily_summary() -> None:
    """
    Backwards-compatible helper that proxies directly to `rag_answer.py`.

    Invokes the current digest query using the OpenAI provider and prints the result.
    """
    query = get_daily_query()
    _run_module_main(
        "rag_answer",
        [
            "rag_answer.py",
            "--provider",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--query",
            query,
        ],
    )


def _parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments controlling the scheduler runtime.

    Returns:
        argparse.Namespace: Namespace containing `days_back` and `dry_run`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--days-back",
        type=int,
        default=DAILY_CONFIG.default_days_back,
        help=f"Number of past days to include (default: {DAILY_CONFIG.default_days_back}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without executing the full pipeline (preview only).",
    )
    return parser.parse_args()


def main() -> None:
    """
    CLI entry point: parse arguments and run the digest job using those settings.
    """
    args = _parse_args()
    run_daily_job(days_back=args.days_back, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
