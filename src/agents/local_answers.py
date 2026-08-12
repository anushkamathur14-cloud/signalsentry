"""Local, offline answers for the Ask panel when LLM is mock or auth fails."""

from __future__ import annotations

from typing import Any


def _wants_flagged_issues(question: str) -> bool:
    q = question.lower()
    keys = (
        "critical",
        "flagged",
        "alert",
        "issue",
        "risk",
        "anomaly",
        "what should i",
        "needs attention",
        "overview",
        "churn",
        "campaign",
    )
    return any(k in q for k in keys)


def _wants_architecture(question: str) -> bool:
    q = question.lower()
    keys = ("langchain", "nemoclaw", "openclaw", "how does", "backend", "structure", "architecture")
    return any(k in q for k in keys)


def answer_from_local_context(question: str, context: dict[str, Any] | None = None) -> str | None:
    """
    Return a useful demo answer from synthetic alerts already on the page.
    None if we should fall through to the generic architecture blurb / LLM.
    """
    ctx = context or {}
    issues = ctx.get("flagged_issues") or []

    if _wants_flagged_issues(question) and issues:
        critical = [i for i in issues if str(i.get("severity", "")).lower() == "critical"]
        high = [i for i in issues if str(i.get("severity", "")).lower() == "high"]
        focus = critical or high or issues[:5]
        lines = [
            f"Here’s what is flagged in this demo world right now "
            f"({len(issues)} total alerts"
            + (f", {len(critical)} critical" if critical else "")
            + "):",
            "",
        ]
        for i, item in enumerate(focus[:6], 1):
            lines.append(
                f"{i}. **{item.get('kind', 'Item')} {item.get('entity_id')}** "
                f"({item.get('severity')}) — {item.get('headline')}"
            )
            if item.get("recommended_action"):
                lines.append(f"   - Action: {item['recommended_action']}")
            if item.get("insight"):
                lines.append(f"   - Why: {item['insight']}")
            lines.append("")
        lines.append(
            "Open **Home** to expand a card, or **Account risks** / **Campaign issues** "
            "for the full brief, chart, and source rows."
        )
        return "\n".join(lines)

    if _wants_architecture(question):
        return (
            "SignalSentry flow: **synthetic data → Python detectors → LangChain structured "
            "investigators → dashboard briefs**.\n\n"
            "- **Detectors** find anomalies (no LLM).\n"
            "- **LangChain** explains them via `ChatOpenAI.with_structured_output`.\n"
            "- **NemoClaw** is the local `inference.local` route; **OpenClaw** is the local chat UI.\n"
            "- This hosted site defaults to **mock**; optional BYOK uses NVIDIA’s public API.\n\n"
            "See **How it works** for the full map."
        )

    return None


def architecture_fallback() -> str:
    return (
        "SignalSentry's backend is: synthetic generators → YAML-threshold detectors → "
        "LangChain `ChatOpenAI.with_structured_output` investigators. "
        "Locally, LangChain talks to NemoClaw at `https://inference.local/v1` (OpenClaw "
        "is the sandbox chat/gateway). On this hosted demo, investigations default to "
        "mock templates. Open **How it works** for the system map, or ask about "
        "**critical / flagged issues** to get a local summary from the current demo alerts."
    )
