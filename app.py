"""SignalSentry Streamlit dashboard."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Local / NemoClaw live mode is controlled by .env (USE_MOCK_MODEL=false).
# Streamlit Community Cloud cannot reach inference.local — default mock until BYOK.
from src.models.llm import (
    force_mock_if_hosted_demo,
    is_hosted_demo_environment,
    load_model_config,
    resolve_model_config,
)

force_mock_if_hosted_demo()
os.environ.setdefault("MODEL_BASE_URL", "https://inference.local/v1")
os.environ.setdefault("MODEL_API_KEY", "nemoclaw-local-placeholder")
os.environ.setdefault("MODEL_NAME", "nvidia/nemotron-mini")
os.environ.setdefault("SEED", "42")

from src.agents.investigators import ask_assistant, investigate_churn
from src.generation.generate_all import META_PATH, generate_all
from src.models.schemas import CandidateAlert, Severity
from src.paths import GENERATED_DIR, GROUND_TRUTH_DIR, OUTPUTS_DIR, ROOT
from src.presentation import build_churn_briefing, build_media_briefing
from src.privacy import (
    list_readable_files,
    read_investigation_log,
    synthetic_data_confirmation,
)
from src.run_analysis import run_analysis

st.set_page_config(
    page_title="SignalSentry",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _read_generation_meta() -> dict:
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def ensure_demo_data() -> None:
    """Generate synthetic data + analysis if outputs are missing (Streamlit Cloud cold start)."""
    alerts_path = OUTPUTS_DIR / "churn_alerts.json"
    churn_path = GENERATED_DIR / "churn_metrics.parquet"
    if alerts_path.exists() and churn_path.exists():
        return
    with st.spinner("Preparing synthetic demo data (first run only)..."):
        # Cold start always uses mock so Cloud boots without an API key.
        generate_all(seed=int(os.getenv("SEED", "42")))
        run_analysis(max_investigations=25, config=load_model_config())


def regenerate_demo_world(
    *,
    seed: int,
    n_accounts: int,
    n_campaigns: int,
    max_investigations: int,
) -> None:
    """Build a new synthetic world and re-run detect → investigate → evaluate."""
    generate_all(seed=seed, n_accounts=n_accounts, n_campaigns=n_campaigns)
    run_analysis(max_investigations=max_investigations, config=active_model_config())
    _load_metrics.clear()


def active_model_config():
    """Session-aware config: BYOK live path, else hosted mock / local .env."""
    return resolve_model_config(visitor_api_key=st.session_state.get("byok_api_key") or None)


def byok_sidebar() -> None:
    """Portfolio policy: your key is never required; visitors may bring their own."""
    with st.sidebar.expander("Live LangChain (optional)", expanded=False):
        st.caption(
            "Default is mock (no API spend). Paste your own NVIDIA key for live LangChain. "
            "Session-only — not saved to the repo."
        )
        key = st.text_input(
            "NVIDIA API key (BYOK)",
            type="password",
            value=st.session_state.get("byok_api_key", ""),
            help="Create a key at build.nvidia.com.",
            key="byok_input",
        )
        cols = st.columns(2)
        if cols[0].button("Use key", use_container_width=True):
            st.session_state.byok_api_key = key.strip()
            st.rerun()
        if cols[1].button("Clear key", use_container_width=True):
            st.session_state.byok_api_key = ""
            os.environ.pop("SIGNAL_SENTRY_BYOK_ACTIVE", None)
            st.rerun()
        cfg = active_model_config()
        if cfg.use_mock:
            st.caption("Active: mock")
        elif cfg.is_nemoclaw_route:
            st.caption(f"Active: NemoClaw · {cfg.model_name}")
        else:
            st.caption(f"Active: BYOK · {cfg.model_name}")


def _render_trace(trace: dict | None) -> None:
    if not trace:
        st.caption("No LangChain trace for this run.")
        return
    st.markdown(
        f"**Run** `{trace.get('run_id')}` · `{trace.get('kind')}` · path `{trace.get('path_label')}`"
    )
    for step in trace.get("steps") or []:
        dur = step.get("duration_ms")
        dur_s = f" · {dur} ms" if dur is not None else ""
        st.markdown(f"- `{step.get('status')}` **{step.get('name')}** — {step.get('detail')}{dur_s}")
    with st.expander("Full trace JSON"):
        st.json(trace)


def synthetic_data_sidebar() -> None:
    """Controls to mint a fresh synthetic demo dataset on demand."""
    meta = _read_generation_meta()
    if "demo_seed" not in st.session_state:
        st.session_state.demo_seed = int(meta.get("seed") or os.getenv("SEED", "42"))

    with st.sidebar.expander("Advanced · demo data", expanded=False):
        if meta:
            st.caption(
                f"Seed `{meta.get('seed')}` · {meta.get('n_accounts')} accounts · "
                f"{meta.get('n_campaigns')} campaigns"
            )
        seed = st.number_input(
            "World seed",
            min_value=0,
            max_value=2_147_483_647,
            value=int(st.session_state.demo_seed),
            step=1,
        )
        st.session_state.demo_seed = int(seed)
        if st.button("Random seed", use_container_width=True):
            st.session_state.demo_seed = secrets.randbelow(1_000_000)
            st.rerun()
        n_accounts = st.number_input(
            "Accounts", min_value=20, max_value=200, value=int(meta.get("n_accounts") or 100)
        )
        n_campaigns = st.number_input(
            "Campaigns", min_value=15, max_value=80, value=int(meta.get("n_campaigns") or 40)
        )
        max_inv = st.slider("Investigations to run", min_value=5, max_value=40, value=25)
        if st.button("Regenerate world", type="primary", use_container_width=True):
            with st.spinner(
                f"Generating world (seed={st.session_state.demo_seed}) and re-running analysis..."
            ):
                regenerate_demo_world(
                    seed=int(st.session_state.demo_seed),
                    n_accounts=int(n_accounts),
                    n_campaigns=int(n_campaigns),
                    max_investigations=int(max_inv),
                )
            st.success(f"Ready (seed {st.session_state.demo_seed}).")
            st.rerun()


# Visual direction: cool slate + teal (avoid purple/cream AI clichés)
st.markdown(
    """
    <style>
      :root {
        --ss-bg: #0f1720;
        --ss-panel: #16202c;
        --ss-accent: #2dd4bf;
        --ss-text: #e8eef5;
        --ss-muted: #93a4b5;
      }
      .stApp { background: radial-gradient(1200px 600px at 10% -10%, #1b3a3a 0%, #0f1720 45%, #0b1219 100%); color: var(--ss-text); }
      h1, h2, h3 { font-family: "IBM Plex Sans", "Segoe UI", sans-serif !important; letter-spacing: -0.02em; }
      div[data-testid="stMetric"] {
        background: rgba(22, 32, 44, 0.85);
        border: 1px solid rgba(45, 212, 191, 0.25);
        padding: 0.75rem 1rem;
        border-radius: 0.4rem;
      }
      .ss-banner {
        border-left: 3px solid #2dd4bf;
        padding: 0.6rem 0.9rem;
        background: rgba(22,32,44,0.7);
        margin-bottom: 1rem;
        color: #c9d6e3;
      }
      .ss-brief {
        background: rgba(22, 32, 44, 0.92);
        border: 1px solid rgba(45, 212, 191, 0.28);
        border-radius: 0.5rem;
        padding: 1rem 1.1rem;
        margin-bottom: 0.85rem;
      }
      .ss-brief h4 {
        margin: 0 0 0.35rem 0;
        color: #2dd4bf;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }
      .ss-brief p, .ss-brief li {
        color: #e8eef5;
        font-size: 0.95rem;
        line-height: 1.45;
      }
      .ss-brief ul { margin: 0.25rem 0 0 1.1rem; padding: 0; }
      .ss-kicker {
        color: #93a4b5;
        font-size: 0.8rem;
        margin-bottom: 0.35rem;
      }
      .ss-story {
        background: linear-gradient(145deg, rgba(22,32,44,0.95), rgba(27,58,58,0.35));
        border: 1px solid rgba(45, 212, 191, 0.35);
        border-radius: 0.6rem;
        padding: 1.15rem 1.25rem 0.4rem 1.25rem;
        margin-bottom: 1rem;
      }
      .ss-story h2 {
        margin: 0.15rem 0 0.75rem 0 !important;
        font-size: 1.45rem !important;
        color: #e8eef5 !important;
      }
      .ss-chip {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.75rem;
        margin-right: 0.35rem;
        background: rgba(45, 212, 191, 0.15);
        color: #2dd4bf;
        border: 1px solid rgba(45, 212, 191, 0.35);
      }
      .ss-chip-warn {
        background: rgba(251, 146, 60, 0.15);
        color: #fdba74;
        border-color: rgba(251, 146, 60, 0.35);
      }
      .ss-chip-crit {
        background: rgba(248, 113, 113, 0.15);
        color: #fca5a5;
        border-color: rgba(248, 113, 113, 0.35);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_outputs():
    return {
        "churn_alerts": _load_json(OUTPUTS_DIR / "churn_alerts.json", []),
        "media_alerts": _load_json(OUTPUTS_DIR / "media_alerts.json", []),
        "churn_investigations": _load_json(OUTPUTS_DIR / "churn_investigations.json", []),
        "media_investigations": _load_json(OUTPUTS_DIR / "media_investigations.json", []),
        "evaluation": _load_json(OUTPUTS_DIR / "evaluation.json", {}),
        "summary": _load_json(OUTPUTS_DIR / "run_summary.json", {}),
    }


@st.cache_data(show_spinner=False)
def _load_metrics():
    churn = pd.read_parquet(GENERATED_DIR / "churn_metrics.parquet") if (GENERATED_DIR / "churn_metrics.parquet").exists() else pd.DataFrame()
    media = pd.read_parquet(GENERATED_DIR / "media_metrics.parquet") if (GENERATED_DIR / "media_metrics.parquet").exists() else pd.DataFrame()
    churn_gt = pd.read_parquet(GROUND_TRUTH_DIR / "churn_labels.parquet") if (GROUND_TRUTH_DIR / "churn_labels.parquet").exists() else pd.DataFrame()
    media_gt = pd.read_parquet(GROUND_TRUTH_DIR / "media_labels.parquet") if (GROUND_TRUTH_DIR / "media_labels.parquet").exists() else pd.DataFrame()
    return churn, media, churn_gt, media_gt


def overview_page(data, show_eval: bool):
    st.title("SignalSentry")
    st.markdown(
        '<div class="ss-banner">Early warning for churn and paid-media issues — '
        "plain English first, numbers underneath. Recommendations need a human.</div>",
        unsafe_allow_html=True,
    )

    churn_alerts = data["churn_alerts"]
    media_alerts = data["media_alerts"]
    active = len(churn_alerts) + len(media_alerts)

    c1, c2, c3 = st.columns(3)
    c1.metric("Things to review", active)
    c2.metric("Account risks", len(churn_alerts))
    c3.metric("Campaign issues", len(media_alerts))

    st.subheader("Needs attention")
    st.caption("Open Churn or Campaigns in the sidebar to act on one.")

    inv_churn = {
        row["alert"]["entity_id"] + "|" + row["alert"]["alert_type"]: row.get("investigation", {})
        for row in data["churn_investigations"]
        if "alert" in row
    }
    inv_media = {
        row["alert"]["entity_id"] + "|" + row["alert"]["alert_type"]: row.get("investigation", {})
        for row in data["media_investigations"]
        if "alert" in row
    }

    spotlight = []
    for a in churn_alerts:
        brief = build_churn_briefing(a, inv_churn.get(a["entity_id"] + "|" + a["alert_type"], {}))
        spotlight.append(
            (
                {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(str(a.get("severity")).lower(), 9),
                "Account",
                a.get("entity_id"),
                brief["headline"],
                brief["recommended_action"],
                a.get("severity"),
            )
        )
    for a in media_alerts:
        brief = build_media_briefing(a, inv_media.get(a["entity_id"] + "|" + a["alert_type"], {}))
        spotlight.append(
            (
                {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(str(a.get("severity")).lower(), 9),
                "Campaign",
                a.get("entity_id"),
                brief["headline"],
                brief["recommended_action"],
                a.get("severity"),
            )
        )
    spotlight.sort(key=lambda r: r[0])
    for _, kind, entity, headline, action, sev in spotlight[:6]:
        st.markdown(
            f'<div class="ss-brief"><h4>{kind} · {entity} · {sev}</h4>'
            f"<p><b>{headline}</b></p><p>{action}</p></div>",
            unsafe_allow_html=True,
        )
    if not spotlight:
        st.info("No alerts yet — use Regenerate in the sidebar Advanced section if needed.")

    with st.expander("Charts & distribution (optional)", expanded=False):
        sev_rows = []
        for domain, alerts in [("churn", churn_alerts), ("media", media_alerts)]:
            for a in alerts:
                sev_rows.append({"domain": domain, "severity": a.get("severity", "unknown")})
        if sev_rows:
            sev_df = pd.DataFrame(sev_rows)
            fig = px.histogram(
                sev_df,
                x="severity",
                color="domain",
                barmode="group",
                title="Alerts by severity",
                color_discrete_sequence=["#2dd4bf", "#38bdf8"],
            )
            st.plotly_chart(_style_chart(fig), use_container_width=True)

        risks = []
        for row in data["churn_investigations"]:
            inv = row.get("investigation", {})
            risks.append(
                {
                    "account_id": inv.get("account_id"),
                    "risk_score": inv.get("risk_score"),
                }
            )
        if risks:
            rdf = pd.DataFrame(risks)
            fig2 = px.histogram(
                rdf,
                x="risk_score",
                nbins=12,
                title="Churn risk score distribution",
                color_discrete_sequence=["#2dd4bf"],
            )
            st.plotly_chart(_style_chart(fig2), use_container_width=True)

    if show_eval and data["evaluation"]:
        with st.expander("Evaluation vs ground truth", expanded=False):
            st.json(data["evaluation"])


def _severity_chip(severity: str) -> str:
    sev = str(severity or "unknown").lower()
    cls = "ss-chip"
    if sev in {"high", "medium"}:
        cls = "ss-chip ss-chip-warn"
    if sev == "critical":
        cls = "ss-chip ss-chip-crit"
    return f'<span class="{cls}">{sev}</span>'


def _brief_block(title: str, body: str) -> None:
    st.markdown(
        f'<div class="ss-brief"><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def _brief_list(title: str, items: list) -> None:
    if not items:
        return
    lis = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(
        f'<div class="ss-brief"><h4>{title}</h4><ul>{lis}</ul></div>',
        unsafe_allow_html=True,
    )


def _render_story_summary(brief: dict) -> None:
    """Plain-English story first — what a CSM would say in a meeting."""
    chips = (
        f'{_severity_chip(brief.get("severity"))}'
        f'<span class="ss-chip">{brief.get("entity_id")}</span>'
        f'<span class="ss-chip">{brief.get("window")}</span>'
    )
    st.markdown(
        f'<div class="ss-story">{chips}<h2>{brief.get("headline")}</h2></div>',
        unsafe_allow_html=True,
    )
    _brief_block("What's going on", brief.get("insight") or "")
    if brief.get("opportunity"):
        _brief_block("Why it matters", brief.get("opportunity") or "")
    _brief_block("Recommended action", brief.get("recommended_action") or "")
    cols = st.columns(2)
    with cols[0]:
        _brief_list("Next steps", brief.get("next_steps") or [])
    with cols[1]:
        _brief_list("Expected impact", brief.get("expected_impact") or [])
    if brief.get("immediate_review"):
        st.error("Flag for immediate human review — advisory only; nothing is changed automatically.")


def _render_technical_section(
    *,
    brief: dict,
    alert: dict,
    chart_factory,
    extra_expanders: list | None = None,
) -> None:
    """Mechanics below the fold: chart math, detector payload, hypotheses."""
    st.markdown("---")
    with st.expander("Numbers & chart", expanded=False):
        st.caption("Detector math and the trend behind the story.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Current", f"{float(alert.get('current_value') or 0):.2f}")
        c2.metric("Expected", f"{float(alert.get('expected_value') or 0):.2f}")
        c3.metric("Change", brief.get("what_changed") or "—")
        chart_factory()

    with st.expander("How the detector decided this", expanded=False):
        st.write(
            {
                "alert_type": alert.get("alert_type"),
                "metrics_involved": alert.get("metrics_involved"),
                "supporting_calculations": alert.get("supporting_calculations"),
                "window": brief.get("window"),
            }
        )
        st.json(alert)

    with st.expander("Hypotheses, metrics & confidence", expanded=False):
        _brief_list("Diagnosis (hypotheses)", brief.get("diagnosis") or [])
        _brief_list("Success metrics", brief.get("success_metrics") or [])
        _brief_list("Guardrails", brief.get("guardrails") or [])
        _brief_list("Evidence", brief.get("evidence") or [])
        meta = []
        if brief.get("confidence") is not None:
            meta.append(f"Confidence {float(brief['confidence']):.0%}")
        if brief.get("risk_score") is not None:
            meta.append(f"Risk score {brief['risk_score']}")
        if meta:
            st.caption(" · ".join(meta))
        if brief.get("data_limitations"):
            st.markdown("**Data confidence**")
            for lim in brief["data_limitations"]:
                st.write(f"- {lim}")

    for title, render_fn in extra_expanders or []:
        with st.expander(title, expanded=False):
            render_fn()


def _pick_alert(alerts: list[dict], inv_map: dict, builder, *, label: str) -> tuple[dict, dict, dict]:
    """One human-readable dropdown + optional severity filter."""
    severities = sorted({str(a.get("severity", "unknown")) for a in alerts})
    filt = st.selectbox(
        "Show severity",
        options=["All"] + severities,
        index=0,
        help="Narrow the list when there are many alerts.",
    )
    filtered = alerts if filt == "All" else [a for a in alerts if str(a.get("severity")) == filt]
    if not filtered:
        filtered = alerts

    options = []
    for a in filtered:
        key = a["entity_id"] + "|" + a["alert_type"]
        inv = inv_map.get(key, {}).get("investigation", {})
        brief = builder(a, inv)
        sev = a.get("severity", "?")
        options.append(
            {
                "key": key,
                "label": f"{brief['headline']} · {a['entity_id']} ({sev})",
                "alert": a,
                "inv": inv,
                "brief": brief,
                "sort_sev": {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(str(sev).lower(), 9),
                "sort_risk": -(inv.get("risk_score") or 0),
            }
        )
    options.sort(key=lambda o: (o["sort_sev"], o["sort_risk"], o["label"]))
    labels = [o["label"] for o in options]
    chosen = st.selectbox(label, labels, index=0)
    row = options[labels.index(chosen)]
    return row["alert"], row["inv"], row["brief"]


def _style_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8eef5",
        legend_title_text="",
        margin=dict(l=10, r=10, t=40, b=10),
        height=360,
    )
    return fig


def _add_anomaly_window(fig, start, end, label: str = "Flagged window"):
    if not start or not end:
        return fig
    fig.add_vrect(
        x0=start,
        x1=end,
        fillcolor="rgba(45, 212, 191, 0.12)",
        line_width=0,
        annotation_text=label,
        annotation_position="top left",
        annotation_font_color="#93a4b5",
    )
    return fig


def churn_page(data, churn_df, churn_gt, show_eval: bool):
    st.title("Account risks")
    st.caption("Read the summary. Open the sections below only if you want the numbers.")

    alerts = data["churn_alerts"]
    if not alerts:
        st.info("No churn alerts yet. Use Regenerate demo data under Advanced in the sidebar.")
        return

    inv_map = {
        row["alert"]["entity_id"] + "|" + row["alert"]["alert_type"]: row
        for row in data["churn_investigations"]
        if "alert" in row
    }
    alert, _inv, brief = _pick_alert(
        alerts, inv_map, build_churn_briefing, label="Pick a risk"
    )
    account = alert["entity_id"]

    _render_story_summary(brief)

    def _chart():
        if churn_df.empty:
            st.info("No metric history loaded.")
            return
        hist = churn_df[churn_df["account_id"] == account].sort_values("date")
        metrics = [m for m in (alert.get("metrics_involved") or []) if m in hist.columns]
        if not metrics:
            metrics = [
                c
                for c in ["weekly_sessions", "active_users", "key_feature_adoption", "nps_score"]
                if c in hist.columns
            ]
        fig = px.line(hist, x="date", y=metrics, title="Metric trend (flagged window shaded)")
        fig = _add_anomaly_window(fig, alert.get("start_date"), alert.get("end_date"))
        st.plotly_chart(_style_chart(fig), use_container_width=True)

    extras = []
    if show_eval and not churn_gt.empty:
        extras.append(
            (
                "Evaluation ground truth",
                lambda: st.dataframe(
                    churn_gt[churn_gt["account_id"] == account], use_container_width=True
                ),
            )
        )

    _render_technical_section(
        brief=brief,
        alert=alert,
        chart_factory=_chart,
        extra_expanders=extras,
    )


def media_page(data, media_df, media_gt, show_eval: bool):
    st.title("Campaign issues")
    st.caption("Read the summary. Charts and detector math are folded underneath.")

    alerts = data["media_alerts"]
    if not alerts:
        st.info("No campaign alerts yet. Use Regenerate demo data under Advanced in the sidebar.")
        return

    inv_map = {
        row["alert"]["entity_id"] + "|" + row["alert"]["alert_type"]: row
        for row in data["media_investigations"]
        if "alert" in row
    }
    alert, _inv, brief = _pick_alert(
        alerts, inv_map, build_media_briefing, label="Pick an issue"
    )
    campaign = alert["entity_id"]

    _render_story_summary(brief)

    def _chart():
        if media_df.empty:
            st.info("No metric history loaded.")
            return
        hist = media_df[media_df["campaign_id"] == campaign].sort_values("date")
        metrics = [m for m in (alert.get("metrics_involved") or []) if m in hist.columns]
        if not metrics:
            metrics = [
                c
                for c in ["spend", "conversions", "cpc", "conversion_rate", "frequency"]
                if c in hist.columns
            ]
        fig = px.line(
            hist, x="date", y=metrics[:4], title="Campaign trend (flagged window shaded)"
        )
        fig = _add_anomaly_window(fig, alert.get("start_date"), alert.get("end_date"))
        st.plotly_chart(_style_chart(fig), use_container_width=True)

        primary = (alert.get("metrics_involved") or ["value"])[0]
        cmp = pd.DataFrame(
            {
                "series": ["Expected", "Current"],
                "value": [float(alert["expected_value"]), float(alert["current_value"])],
            }
        )
        fig2 = px.bar(
            cmp,
            x="series",
            y="value",
            title=f"{primary.replace('_', ' ').title()} — expected vs current",
            color="series",
            color_discrete_sequence=["#38bdf8", "#2dd4bf"],
        )
        st.plotly_chart(_style_chart(fig2), use_container_width=True)

    extras = []
    if show_eval and not media_gt.empty:
        extras.append(
            (
                "Evaluation ground truth",
                lambda: st.dataframe(
                    media_gt[media_gt["campaign_id"] == campaign], use_container_width=True
                ),
            )
        )

    _render_technical_section(
        brief=brief,
        alert=alert,
        chart_factory=_chart,
        extra_expanders=extras,
    )


def backend_traces_page(data):
    st.title("Backend & LangChain traces")
    st.markdown(
        '<div class="ss-banner">How SignalSentry actually runs: detectors are Python; '
        "investigators are LangChain structured calls over an OpenAI-compatible route "
        "(NemoClaw locally, or NVIDIA public BYOK on this hosted demo).</div>",
        unsafe_allow_html=True,
    )
    cfg = active_model_config()

    st.subheader("Architecture")
    st.code(
        "synthetic generators\n"
        "        ↓\n"
        "deterministic detectors (YAML thresholds, z-scores)  ← no LLM\n"
        "        ↓\n"
        "candidate alerts + metric context JSON\n"
        "        ↓\n"
        "LangChain ChatOpenAI.with_structured_output(Pydantic)\n"
        "        ↓\n"
        "NemoClaw inference.local  OR  NVIDIA integrate.api (BYOK)  OR  mock\n"
        "        ↓\n"
        "investigation report + audit / LangChain step trace",
        language="text",
    )
    st.markdown(
        """
**NemoClaw / OpenClaw vs this site**
- **NemoClaw** fronts `https://inference.local/v1` inside your local sandbox — that is the intended live backend.
- **OpenClaw** is the local agent/chat gateway UI (`127.0.0.1:18789`). It is not embedded here (localhost-only, token-gated).
- **This portfolio site** shows the same LangChain investigator contract. Default = mock. Optional BYOK = live NVIDIA endpoint with the same client code.
"""
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Active path", cfg.path_label)
    c2.metric("Model", cfg.model_name)
    c3.metric("Mock?", str(cfg.use_mock))

    st.subheader("Replay one investigation (shows a fresh trace)")
    churn_alerts = data.get("churn_alerts") or []
    if not churn_alerts:
        st.info("No alerts yet — generate demo data first.")
    else:
        labels = [f"{a.get('entity_id')} · {a.get('alert_type')}" for a in churn_alerts[:20]]
        pick = st.selectbox("Alert", labels)
        if st.button("Run LangChain investigator on this alert", type="primary"):
            alert_raw = churn_alerts[labels.index(pick)]
            alert = CandidateAlert(
                entity_id=alert_raw["entity_id"],
                start_date=alert_raw["start_date"],
                end_date=alert_raw["end_date"],
                alert_type=alert_raw["alert_type"],
                severity=Severity(alert_raw["severity"]),
                metrics_involved=alert_raw.get("metrics_involved") or [],
                current_value=float(alert_raw.get("current_value") or 0),
                expected_value=float(alert_raw.get("expected_value") or 0),
                supporting_calculations=alert_raw.get("supporting_calculations") or {},
                domain=alert_raw.get("domain") or "churn",
            )
            with st.spinner("Investigating…"):
                result, preview = investigate_churn(alert, config=active_model_config())
            st.success("Done — scroll for payload + trace.")
            st.json(result.model_dump(mode="json"))
            with st.expander("Inference payload sent to LangChain"):
                st.json(preview)

    st.subheader("Recent LangChain traces")
    log_rows = read_investigation_log(limit=30)
    traced = [r for r in reversed(log_rows) if r.get("langchain_trace")]
    if not traced:
        st.info("No traces yet. Run an investigation above or ask a question on the Ask page.")
    else:
        for row in traced[:8]:
            with st.expander(
                f"{row.get('timestamp', '')} · {row.get('domain')} · {row.get('entity_id')} · {row.get('mode')}"
            ):
                _render_trace(row.get("langchain_trace"))
                if row.get("payload_preview"):
                    st.markdown("**Payload preview**")
                    st.json(row["payload_preview"])


def ask_page(data):
    st.title("Ask SignalSentry")
    st.markdown(
        '<div class="ss-banner">Ask how the backend works. Mock answers need no key; '
        "BYOK runs the same LangChain `ChatOpenAI` client used by investigators.</div>",
        unsafe_allow_html=True,
    )
    cfg = active_model_config()
    st.caption(f"Active path: `{cfg.path_label}` · model `{cfg.model_name}`")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trace"):
                with st.expander("LangChain trace"):
                    _render_trace(msg["trace"])

    prompt = st.chat_input("e.g. How does LangChain talk to NemoClaw?")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        context = {
            "churn_alert_count": len(data.get("churn_alerts") or []),
            "media_alert_count": len(data.get("media_alerts") or []),
            "path_label": cfg.path_label,
            "hosted": is_hosted_demo_environment(),
        }
        with st.spinner("Thinking…"):
            answer, payload = ask_assistant(prompt, config=cfg, context=context)
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer,
                "trace": payload.get("langchain_trace"),
            }
        )
        st.rerun()


def privacy_page():
    st.title("Privacy and Safety")
    cfg = active_model_config()
    confirm = synthetic_data_confirmation()
    st.success(confirm["message"])

    if cfg.use_mock:
        st.warning(
            "Currently in **mock** mode (no LangChain model calls). "
            "Paste a NVIDIA API key in the sidebar (BYOK) for live LangChain, "
            "or run locally inside NemoClaw with `USE_MOCK_MODEL=false`."
        )
    else:
        st.info(
            f"**Live LangChain** → `{cfg.base_url}` · model `{cfg.model_name}` · path `{cfg.path_label}` "
            f"({'NemoClaw route' if cfg.is_nemoclaw_route else 'OpenAI-compatible / BYOK'})."
        )

    st.write(
        {
            "inference_destination": cfg.destination_label,
            "model_name": cfg.model_name,
            "mode": "mock" if cfg.use_mock else "live-langchain",
            "path_label": cfg.path_label,
            "nemoclaw_route": cfg.is_nemoclaw_route,
            "synthetic_only": confirm["synthetic_only"],
            "byok_session": bool(st.session_state.get("byok_api_key")),
        }
    )
    st.subheader("Files the application reads")
    st.dataframe(pd.DataFrame(list_readable_files()), use_container_width=True)

    st.subheader("Investigation audit log")
    log_rows = read_investigation_log()
    if log_rows:
        st.dataframe(pd.DataFrame([{k: v for k, v in row.items() if k != "payload_preview" and k != "result"} for row in log_rows]), use_container_width=True)
        st.subheader("Exact inference payload preview (latest)")
        st.json(log_rows[-1].get("payload_preview", {}))
    else:
        st.info("No investigations logged yet.")

    st.caption("SignalSentry never sends outbound customer messages or applies automatic campaign changes.")


def main():
    st.sidebar.title("SignalSentry")
    if "byok_api_key" not in st.session_state:
        st.session_state.byok_api_key = ""

    cfg = active_model_config()
    st.sidebar.caption("Mock demo" if cfg.use_mock else f"Live · {cfg.path_label}")

    page = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Account risks",
            "Campaign issues",
            "How it works",
            "Ask",
            "Privacy",
        ],
    )
    byok_sidebar()
    with st.sidebar.expander("Developer options", expanded=False):
        show_eval = st.toggle("Show ground truth", value=False)
    synthetic_data_sidebar()

    ensure_demo_data()

    if not (OUTPUTS_DIR / "churn_alerts.json").exists():
        st.warning("No analysis outputs found yet.")
        st.code("python -m src.generation.generate_all\npython -m src.run_analysis", language="bash")

    data = _load_outputs()
    churn_df, media_df, churn_gt, media_gt = _load_metrics()

    if page == "Overview":
        overview_page(data, show_eval)
    elif page == "Account risks":
        churn_page(data, churn_df, churn_gt, show_eval)
    elif page == "Campaign issues":
        media_page(data, media_df, media_gt, show_eval)
    elif page == "How it works":
        backend_traces_page(data)
    elif page == "Ask":
        ask_page(data)
    else:
        privacy_page()


if __name__ == "__main__":
    main()
