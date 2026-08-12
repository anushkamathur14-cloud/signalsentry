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
    st.sidebar.markdown("---")
    st.sidebar.subheader("Live LangChain (optional)")
    st.sidebar.caption(
        "Default demo is mock (no API spend). "
        "Paste **your own** NVIDIA API key to run the real LangChain path. "
        "The key stays in this browser session only — not written to the repo."
    )
    key = st.sidebar.text_input(
        "NVIDIA API key (BYOK)",
        type="password",
        value=st.session_state.get("byok_api_key", ""),
        help="Create a key at build.nvidia.com. Leave blank for mock investigators.",
        key="byok_input",
    )
    cols = st.sidebar.columns(2)
    if cols[0].button("Use key", use_container_width=True):
        st.session_state.byok_api_key = key.strip()
        st.rerun()
    if cols[1].button("Clear key", use_container_width=True):
        st.session_state.byok_api_key = ""
        os.environ.pop("SIGNAL_SENTRY_BYOK_ACTIVE", None)
        st.rerun()

    cfg = active_model_config()
    if cfg.use_mock:
        st.sidebar.info("Path: **mock** · add BYOK key for live LangChain")
    elif cfg.is_nemoclaw_route:
        st.sidebar.success(f"Path: **NemoClaw** · `{cfg.model_name}`")
    else:
        st.sidebar.success(f"Path: **BYOK live** · `{cfg.model_name}`")


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
    st.sidebar.markdown("---")
    st.sidebar.subheader("Synthetic data")
    meta = _read_generation_meta()
    if meta:
        st.sidebar.caption(
            f"Current world · seed `{meta.get('seed')}` · "
            f"{meta.get('n_accounts')} accounts · {meta.get('n_campaigns')} campaigns"
        )
    else:
        st.sidebar.caption("No generation metadata yet — cold start will create seed 42.")

    if "demo_seed" not in st.session_state:
        st.session_state.demo_seed = int(meta.get("seed") or os.getenv("SEED", "42"))

    seed = st.sidebar.number_input(
        "World seed",
        min_value=0,
        max_value=2_147_483_647,
        value=int(st.session_state.demo_seed),
        step=1,
        help="Same seed → same demo world. New seed → different accounts/campaigns with injected anomalies.",
    )
    st.session_state.demo_seed = int(seed)

    if st.sidebar.button("Random seed", use_container_width=True):
        st.session_state.demo_seed = secrets.randbelow(1_000_000)
        st.rerun()

    n_accounts = st.sidebar.number_input(
        "Accounts", min_value=20, max_value=200, value=int(meta.get("n_accounts") or 100)
    )
    n_campaigns = st.sidebar.number_input(
        "Campaigns", min_value=15, max_value=80, value=int(meta.get("n_campaigns") or 40)
    )
    max_inv = st.sidebar.slider("Investigations to run", min_value=5, max_value=40, value=25)

    if st.sidebar.button("Regenerate world + reanalyze", type="primary", use_container_width=True):
        with st.spinner(
            f"Generating synthetic world (seed={st.session_state.demo_seed}) and re-running analysis..."
        ):
            regenerate_demo_world(
                seed=int(st.session_state.demo_seed),
                n_accounts=int(n_accounts),
                n_campaigns=int(n_campaigns),
                max_investigations=int(max_inv),
            )
        st.sidebar.success(f"Fresh synthetic world ready (seed {st.session_state.demo_seed}).")
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
        '<div class="ss-banner">Local-first early warning for customer churn and paid-media anomalies. '
        "Recommendations are advisory and require human review.</div>",
        unsafe_allow_html=True,
    )

    churn_alerts = data["churn_alerts"]
    media_alerts = data["media_alerts"]
    active = len(churn_alerts) + len(media_alerts)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active alerts", active)
    c2.metric("Churn alerts", len(churn_alerts))
    c3.metric("Campaign alerts", len(media_alerts))
    mode = data.get("summary", {}).get("model_mode", load_model_config().destination_label)
    c4.metric("Inference mode", str(mode))

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
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8eef5")
        st.plotly_chart(fig, use_container_width=True)

    # Churn risk distribution from investigations
    risks = []
    for row in data["churn_investigations"]:
        inv = row.get("investigation", {})
        risks.append({"account_id": inv.get("account_id"), "risk_score": inv.get("risk_score"), "risk_level": inv.get("risk_level")})
    if risks:
        rdf = pd.DataFrame(risks)
        fig2 = px.histogram(rdf, x="risk_score", nbins=12, title="Churn risk score distribution", color_discrete_sequence=["#2dd4bf"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8eef5")
        st.plotly_chart(fig2, use_container_width=True)

    type_rows = [{"alert_type": a.get("alert_type"), "count": 1} for a in media_alerts]
    if type_rows:
        tdf = pd.DataFrame(type_rows).groupby("alert_type", as_index=False).sum()
        fig3 = px.bar(tdf, x="alert_type", y="count", title="Campaign anomalies by type", color_discrete_sequence=["#38bdf8"])
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8eef5", xaxis_tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)

    if show_eval and data["evaluation"]:
        st.subheader("Detection performance vs synthetic ground truth")
        st.json(data["evaluation"])


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


def _render_briefing_panel(brief: dict) -> None:
    st.markdown(
        f'<div class="ss-kicker">{brief.get("window")} · '
        f'severity <b>{brief.get("severity")}</b> · '
        f'{brief.get("what_changed")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"### {brief.get('headline')}")
    _brief_block("What's the anomaly", brief.get("insight") or "")
    _brief_block("Opportunity", brief.get("opportunity") or "")
    _brief_list("Diagnosis (hypotheses)", brief.get("diagnosis") or [])
    _brief_block("Recommended action", brief.get("recommended_action") or "")
    _brief_list("Next steps", brief.get("next_steps") or [])
    _brief_list("Expected impact (directional)", brief.get("expected_impact") or [])
    _brief_list("Success metrics to watch", brief.get("success_metrics") or [])
    _brief_list("Guardrails", brief.get("guardrails") or [])
    conf = brief.get("confidence")
    risk = brief.get("risk_score")
    meta = []
    if conf is not None:
        meta.append(f"Confidence {float(conf):.0%}")
    if risk is not None:
        meta.append(f"Risk score {risk}")
    if brief.get("immediate_review"):
        meta.append("Immediate human review recommended")
    if meta:
        st.caption(" · ".join(meta))
    if brief.get("data_limitations"):
        with st.expander("Data confidence & limitations"):
            for lim in brief["data_limitations"]:
                st.write(f"- {lim}")


def _style_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8eef5",
        legend_title_text="",
    )
    return fig


def _add_anomaly_window(fig, start, end, label: str = "Anomaly window"):
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
    st.title("Customer Churn Risks")
    st.markdown(
        '<div class="ss-banner">Pick an account to see the chart and a plain-language brief: '
        "what broke, what to do next, and what impact to expect.</div>",
        unsafe_allow_html=True,
    )
    inv_map = {
        row["alert"]["entity_id"] + "|" + row["alert"]["alert_type"]: row
        for row in data["churn_investigations"]
        if "alert" in row
    }

    rows = []
    for a in data["churn_alerts"]:
        key = a["entity_id"] + "|" + a["alert_type"]
        inv = inv_map.get(key, {}).get("investigation", {})
        brief = build_churn_briefing(a, inv)
        rows.append(
            {
                "account_id": a["entity_id"],
                "alert_type": a["alert_type"],
                "severity": a["severity"],
                "plain_english": brief["headline"],
                "risk_score": inv.get("risk_score"),
                "confidence": inv.get("confidence"),
                "current": a["current_value"],
                "expected": a["expected_value"],
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        st.info("No churn alerts yet. Run `python -m src.run_analysis`.")
        return
    table = table.sort_values(["risk_score", "severity"], ascending=False, na_position="last")
    st.dataframe(table, use_container_width=True)

    account = st.selectbox("Account", options=list(dict.fromkeys(table["account_id"].tolist())))
    account_alerts = [a for a in data["churn_alerts"] if a["entity_id"] == account]
    alert_labels = [f"{a['alert_type']} ({a['severity']})" for a in account_alerts]
    chosen = st.selectbox("Signal", alert_labels) if alert_labels else None
    alert = account_alerts[alert_labels.index(chosen)] if chosen else account_alerts[0]
    inv = inv_map.get(alert["entity_id"] + "|" + alert["alert_type"], {}).get("investigation", {})
    brief = build_churn_briefing(alert, inv)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current", f"{float(alert['current_value']):.2f}")
    m2.metric("Expected", f"{float(alert['expected_value']):.2f}")
    m3.metric("Severity", str(alert.get("severity")))
    m4.metric("Risk score", f"{brief.get('risk_score') if brief.get('risk_score') is not None else '—'}")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Trend")
        if not churn_df.empty:
            hist = churn_df[churn_df["account_id"] == account].sort_values("date")
            metrics = [m for m in (alert.get("metrics_involved") or []) if m in hist.columns]
            if not metrics:
                metrics = [c for c in ["weekly_sessions", "active_users", "key_feature_adoption", "nps_score"] if c in hist.columns]
            fig = px.line(
                hist,
                x="date",
                y=metrics,
                title=f"{brief['headline']} — {account}",
            )
            fig = _add_anomaly_window(fig, alert.get("start_date"), alert.get("end_date"))
            st.plotly_chart(_style_chart(fig), use_container_width=True)
        else:
            st.info("No metric history loaded.")
        with st.expander("Detector details (raw)"):
            st.json(alert)

    with right:
        st.subheader("Brief")
        _render_briefing_panel(brief)

    if show_eval and not churn_gt.empty:
        st.subheader("Ground-truth label")
        st.dataframe(churn_gt[churn_gt["account_id"] == account], use_container_width=True)


def media_page(data, media_df, media_gt, show_eval: bool):
    st.title("Campaign Anomalies")
    st.markdown(
        '<div class="ss-banner">Media brief in plain English: anomaly → opportunity → '
        "recommended action → next steps → expected impact (same shape as a CSM monthly review).</div>",
        unsafe_allow_html=True,
    )
    rows = []
    inv_map = {
        row["alert"]["entity_id"] + "|" + row["alert"]["alert_type"]: row
        for row in data["media_investigations"]
        if "alert" in row
    }
    for a in data["media_alerts"]:
        key = a["entity_id"] + "|" + a["alert_type"]
        inv = inv_map.get(key, {}).get("investigation", {})
        brief = build_media_briefing(a, inv)
        rows.append(
            {
                "campaign_id": a["entity_id"],
                "alert_type": a["alert_type"],
                "severity": a["severity"],
                "plain_english": brief["headline"],
                "current": a["current_value"],
                "expected": a["expected_value"],
                "confidence": inv.get("confidence"),
                "immediate_review": inv.get("requires_immediate_human_review"),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        st.info("No campaign alerts yet. Run `python -m src.run_analysis`.")
        return
    st.dataframe(table.sort_values("severity"), use_container_width=True)

    campaign = st.selectbox("Campaign", options=list(dict.fromkeys(table["campaign_id"].tolist())))
    camp_alerts = [a for a in data["media_alerts"] if a["entity_id"] == campaign]
    alert_labels = [f"{a['alert_type']} ({a['severity']})" for a in camp_alerts]
    chosen = st.selectbox("Anomaly", alert_labels) if alert_labels else None
    alert = camp_alerts[alert_labels.index(chosen)] if chosen else camp_alerts[0]
    inv = inv_map.get(alert["entity_id"] + "|" + alert["alert_type"], {}).get("investigation", {})
    brief = build_media_briefing(alert, inv)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current", f"{float(alert['current_value']):.2f}")
    m2.metric("Expected", f"{float(alert['expected_value']):.2f}")
    m3.metric("Severity", str(alert.get("severity")))
    m4.metric("Delta", brief.get("what_changed", "—"))

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Trend")
        if not media_df.empty:
            hist = media_df[media_df["campaign_id"] == campaign].sort_values("date")
            metrics = [m for m in (alert.get("metrics_involved") or []) if m in hist.columns]
            if not metrics:
                metrics = [c for c in ["spend", "conversions", "cpc", "conversion_rate", "frequency"] if c in hist.columns]
            # Prefer a focused dual-axis style story when spend + conversions both present
            fig = px.line(
                hist,
                x="date",
                y=metrics[:4],
                title=f"{brief['headline']} — {campaign}",
            )
            fig = _add_anomaly_window(fig, alert.get("start_date"), alert.get("end_date"))
            st.plotly_chart(_style_chart(fig), use_container_width=True)

            # Simple comparison bars for current vs expected on the primary metric
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
        else:
            st.info("No metric history loaded.")
        with st.expander("Detector details (raw)"):
            st.json(alert)

    with right:
        st.subheader("Brief")
        if brief.get("immediate_review"):
            st.error("Immediate human review recommended — advisory only; no auto changes.")
        _render_briefing_panel(brief)

    if show_eval and not media_gt.empty:
        st.subheader("Ground-truth label")
        st.dataframe(media_gt[media_gt["campaign_id"] == campaign], use_container_width=True)


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

    byok_sidebar()
    cfg = active_model_config()
    if cfg.use_mock:
        st.sidebar.caption("Mode: mock demo — optional BYOK for live LangChain")
    elif cfg.is_nemoclaw_route:
        st.sidebar.caption(f"Mode: live NemoClaw → {cfg.base_url}")
    else:
        st.sidebar.caption(f"Mode: live BYOK → {cfg.base_url}")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Customer Churn Risks",
            "Campaign Anomalies",
            "Backend & Traces",
            "Ask SignalSentry",
            "Privacy and Safety",
        ],
    )
    show_eval = st.sidebar.toggle("Evaluation toggle (show ground truth)", value=False)
    st.sidebar.caption(f"Project root: {ROOT}")
    synthetic_data_sidebar()

    ensure_demo_data()

    if not (OUTPUTS_DIR / "churn_alerts.json").exists():
        st.warning("No analysis outputs found. Run dataset generation and analysis first.")
        st.code("python -m src.generation.generate_all\npython -m src.run_analysis", language="bash")

    data = _load_outputs()
    churn_df, media_df, churn_gt, media_gt = _load_metrics()

    if page == "Overview":
        overview_page(data, show_eval)
    elif page == "Customer Churn Risks":
        churn_page(data, churn_df, churn_gt, show_eval)
    elif page == "Campaign Anomalies":
        media_page(data, media_df, media_gt, show_eval)
    elif page == "Backend & Traces":
        backend_traces_page(data)
    elif page == "Ask SignalSentry":
        ask_page(data)
    else:
        privacy_page()


if __name__ == "__main__":
    main()
