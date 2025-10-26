"""
scheduler_daily.py
-------------------
Automates the daily RAG pipeline and generates a summarized research digest.
Responsibilities:
- Schedule daily execution (e.g., via cron or GitHub Actions).
- Run the full pipeline: fetch → ingest → embed → summarize.
- Generate a markdown "Daily LLM Research Digest" with summaries of newly published papers.
- Save the digest in ./reports with date-based filenames (e.g., 2025-10-13-llm-digest.md).
Example:
    $ python src/scheduler_daily.py
Output:
    ./reports/YYYY-MM-DD-llm-digest.md
"""

import argparse
import html
import io
import json
import logging
import os
import smtplib
import ssl
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
import re

DAILY_QUERIES = {
    0: "Summarize latest LLM retrieval methods.",
    1: "Summarize latest multi-modal LLM research.",
    2: "Summarize latest LLM fine-tuning and alignment papers.",
    3: "Summarize latest reinforcement learning or policy optimization in LLMs.",
    4: "Summarize latest evaluation benchmarks for large language models.",
    5: "Summarize latest LLM efficiency and inference optimization research.",
    6: "Summarize latest applications of LLMs in reasoning and agents.",
}


def get_daily_query() -> str:
    """Return the scheduled query for the current (UTC) weekday."""
    return DAILY_QUERIES[datetime.utcnow().weekday()]


def _run_module_main(module, argv: list[str]) -> None:
    mod = __import__(module, fromlist=["main"])
    main_func = getattr(mod, "main")
    argv_backup = sys.argv.copy()
    try:
        sys.argv = argv
        main_func()
    finally:
        sys.argv = argv_backup


def topic_from_query(q: str) -> str:
    t = q.strip()
    for pref in ("Summarize latest ", "summarize latest "):
        if t.startswith(pref):
            t = t[len(pref) :]
    return t.rstrip(".")[:120]


def render_inline_markdown(text: str) -> str:
    escaped = html.escape(text or "")
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def build_digest_text(papers, topic: str, run_dt: datetime, answer: str) -> str:
    lines: list[str] = []
    lines.append(f"[Daily RAG Digest] {run_dt:%A} – {topic}")
    lines.append("")
    if answer:
        lines.append("Summary:")
        lines.append(answer)
        lines.append("")
    lines.append(f"{len(papers)} new papers summarized:")
    for i, paper in enumerate(papers, 1):
        title = paper.get("title") or "Untitled"
        url = paper.get("url") or ""
        summary = paper.get("summary") or ""
        lines.append(f"{i}. {title}")
        if url:
            lines.append(f"   {url}")
        if summary:
            lines.append(f"   – {summary}")
    return "\n".join(lines)


def build_digest_html(papers, topic: str, run_dt: datetime, answer: str) -> str:
    safe_topic = html.escape(topic)
    date_str = html.escape(run_dt.strftime("%A, %b %d, %Y"))
    rows = []
    for i, paper in enumerate(papers, 1):
        title = html.escape(paper.get("title") or "Untitled")
        url = paper.get("url") or ""
        summary_text = render_inline_markdown(paper.get("summary") or "")
        link = (
            f'<a href="{html.escape(url)}" target="_blank" rel="noopener noreferrer">{title}</a>'
            if url
            else title
        )
        summary_block = f'<div style="margin-top:4px;">{summary_text}</div>' if summary_text else ""
        rows.append(
            f"""
          <li style=\"margin-bottom:12px; line-height:1.4;\">
            <div style=\"font-weight:600;\">{link}</div>
            {summary_block}
          </li>
        """
        )
    items = "\n".join(rows)
    answer_html = render_inline_markdown(answer).replace("\n", "<br>")
    summary_block = (
        f"<div style=\"margin-bottom:20px;\"><h2 style=\"font-size:16px;margin:0 0 8px 0;\">Summary</h2><div>{answer_html}</div></div>"
        if answer
        else ""
    )

    return f"""<!doctype html>
    <html>
    <head><meta charset=\"utf-8\"></head>
    <body style=\"margin:0;padding:0;background:#f6f7fb;\">
      <table role=\"presentation\" width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" style=\"background:#f6f7fb;\">
        <tr><td align=\"center\" style=\"padding:24px;\">
          <table width=\"640\" cellpadding=\"0\" cellspacing=\"0\" style=\"background:#ffffff;border-radius:12px;padding:24px;font-family:Arial,Helvetica,sans-serif;color:#111;\">
            <tr><td>
              <h1 style=\"margin:0 0 8px 0;font-size:20px;\">Daily RAG Digest — {safe_topic}</h1>
              <div style=\"color:#666;font-size:12px;margin-bottom:16px;\">{date_str}</div>
              {summary_block}
              <ol style=\"padding-left:18px;margin:0;\">
                {items}
              </ol>
              <hr style=\"margin:20px 0;border:none;border-top:1px solid #eee;\">
              <div style=\"font-size:12px;color:#666;\">
                You’re receiving this because you enabled the GitHub Action for daily research summaries.
              </div>
            </td></tr>
          </table>
        </td></tr>
      </table>
    </body>
    </html>"""


def send_email(subject: str, text_body: str, html_body: str) -> None:
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("EMAIL_RECEIVER")
    host = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("EMAIL_SMTP_PORT", "587"))

    if not all([sender, password, receiver]):
        print("Email not sent: missing EMAIL_SENDER/EMAIL_PASSWORD/EMAIL_RECEIVER")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(sender, password)
        server.sendmail(sender, [receiver], msg.as_string())


def extract_answer(summary_text: str) -> str:
    if not summary_text:
        return ""
    marker = "Answer:"
    idx = summary_text.find(marker)
    if idx == -1:
        return summary_text.strip()
    return summary_text[idx + len(marker) :].strip()


def run_daily_job(days_back: int = 1, dry_run: bool = False) -> None:
    """
    Execute the daily pipeline: fetch new LLM papers, ingest/chunk, build index, and
    generate a “Daily LLM Research Digest.”
    Args:
        days_back (int): How many past days of papers to include in the update.
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    if dry_run:
        logger.info("[DRY-RUN] Running scheduler without sending email (artifacts will be saved).")

    logger.info("[FETCH] Fetching latest papers (days_back=%s)", days_back)
    fetch_args = [
        "fetch_arxiv.py",
        "--days-back",
        str(days_back),
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
        print(f"No new papers found in the last {days_back} day(s).")
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
        print(f"No new papers found in the last {days_back} day(s).")
        return

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
    summary_buffer = io.StringIO()
    summarize_args = [
        "rag_answer.py",
        "--provider",
        "openai",
        "--model",
        "gpt-4o-mini",
        "--query",
        query,
    ]
    with redirect_stdout(summary_buffer):
        _run_module_main("rag_answer", summarize_args)

    summary_text = summary_buffer.getvalue().strip()
    if not summary_text:
        logger.warning("[SUMMARIZE] Summary output was empty.")
        return

    run_dt = datetime.now(timezone.utc)
    topic = topic_from_query(query)
    answer = extract_answer(summary_text)

    logger.info("[SAVE] Writing digest to reports directory")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    digest_path = reports_dir / f"{run_dt.date().isoformat()}-llm-digest.md"
    digest_body = answer if answer else summary_text
    digest_content = f"# 🧠 LLM Research Digest — {run_dt.date().isoformat()}\n\n{digest_body}\n"
    digest_path.write_text(digest_content, encoding="utf-8")
    logger.info("[SAVE] Digest written to %s", digest_path)

    subject = f"[Daily RAG Digest] {run_dt:%A} - {topic}"
    text_body = build_digest_text(prepared_papers, topic, run_dt, answer)
    html_body = build_digest_html(prepared_papers, topic, run_dt, answer)

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
    """Temporary stub to invoke rag_answer CLI with a fixed query."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-back", type=int, default=1, help="Number of past days to include.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without executing the full pipeline (preview only).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_daily_job(days_back=args.days_back, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
