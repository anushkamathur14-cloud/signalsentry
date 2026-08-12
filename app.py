"""SignalSentry Streamlit dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Demo / Streamlit Cloud defaults: mock investigators, no live provider calls.
os.environ.setdefault("USE_MOCK_MODEL", "true")
os.environ.setdefault("MODEL_BASE_URL", "https://inference.local/v1")
os.environ.setdefault("MODEL_API_KEY", "nemoclaw-local-placeholder")
os.environ.setdefault("MODEL_NAME", "nvidia/nemotron-mini")
os.environ.setdefault("SEED", "42")

from src.models.llm import load_model_config
from src.paths import GENERATED_DIR, GROUND_TRUTH_DIR, OUTPUTS_DIR, ROOT
from src.privacy import (
    list_readable_files,
    read_investigation_log,
    synthetic_data_confirmation,
)

st.set_page_config(
    page_title="SignalSentry",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def ensure_demo_data() -> None:
    """Generate synthetic data + mock analysis if outputs are missing (Streamlit Cloud cold start)."""
    alerts_path = OUTPUTS_DIR / "churn_alerts.json"
    churn_path = GENERATED_DIR / "churn_metrics.parquet"
    if alerts_path.exists() and churn_path.exists():
        return
    with st.spinner("Preparing synthetic demo data (first run only)..."):
        from src.generation.generate_all import generate_all
        from src.run_analysis import run_analysis

        generate_all(seed=int(os.getenv("SEED", "42")))
        run_analysis(max_investigations=25)


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


def churn_page(data, churn_df, churn_gt, show_eval: bool):
    st.title("Customer Churn Risks")
    inv_map = {
        row["alert"]["entity_id"] + "|" + row["alert"]["alert_type"]: row
        for row in data["churn_investigations"]
        if "alert" in row
    }

    rows = []
    for a in data["churn_alerts"]:
        key = a["entity_id"] + "|" + a["alert_type"]
        inv = inv_map.get(key, {}).get("investigation", {})
        rows.append(
            {
                "account_id": a["entity_id"],
                "alert_type": a["alert_type"],
                "severity": a["severity"],
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
    st.subheader("Detected warning signals")
    st.json(account_alerts)

    if not churn_df.empty:
        hist = churn_df[churn_df["account_id"] == account].sort_values("date")
        fig = px.line(
            hist,
            x="date",
            y=["weekly_sessions", "active_users", "key_feature_adoption", "nps_score"],
            title=f"Metric trends — {account}",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8eef5")
        st.plotly_chart(fig, use_container_width=True)

    account_inv = [row for row in data["churn_investigations"] if row.get("alert", {}).get("entity_id") == account]
    if account_inv:
        inv = account_inv[0]["investigation"]
        st.subheader("Agent explanation")
        st.write(inv.get("evidence", []))
        st.write("Likely causes:", inv.get("likely_causes", []))
        st.success(f"Recommended CSM action: {inv.get('recommended_csm_action')}")
        st.caption(f"Confidence={inv.get('confidence')} · Limitations: {inv.get('data_limitations')}")

    if show_eval and not churn_gt.empty:
        st.subheader("Ground-truth label")
        st.dataframe(churn_gt[churn_gt["account_id"] == account], use_container_width=True)


def media_page(data, media_df, media_gt, show_eval: bool):
    st.title("Campaign Anomalies")
    rows = []
    inv_map = {
        row["alert"]["entity_id"] + "|" + row["alert"]["alert_type"]: row
        for row in data["media_investigations"]
        if "alert" in row
    }
    for a in data["media_alerts"]:
        key = a["entity_id"] + "|" + a["alert_type"]
        inv = inv_map.get(key, {}).get("investigation", {})
        rows.append(
            {
                "campaign_id": a["entity_id"],
                "alert_type": a["alert_type"],
                "severity": a["severity"],
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
    st.subheader("Actual vs expected / supporting calculations")
    st.json(camp_alerts)

    if not media_df.empty:
        hist = media_df[media_df["campaign_id"] == campaign].sort_values("date")
        fig = px.line(
            hist,
            x="date",
            y=["spend", "conversions", "cpc", "conversion_rate", "frequency"],
            title=f"Time series — {campaign}",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8eef5")
        st.plotly_chart(fig, use_container_width=True)

    camp_inv = [row for row in data["media_investigations"] if row.get("alert", {}).get("entity_id") == campaign]
    if camp_inv:
        inv = camp_inv[0]["investigation"]
        st.subheader("Agent investigation")
        st.write(inv.get("anomaly_summary"))
        st.write(inv.get("evidence", []))
        st.write("Likely causes:", inv.get("likely_causes", []))
        st.warning(f"Recommended action: {inv.get('recommended_action')}")
        if inv.get("requires_immediate_human_review"):
            st.error("Immediate human review required (advisory — no automated changes).")

    if show_eval and not media_gt.empty:
        st.subheader("Ground-truth label")
        st.dataframe(media_gt[media_gt["campaign_id"] == campaign], use_container_width=True)


def privacy_page():
    st.title("Privacy and Safety")
    cfg = load_model_config()
    confirm = synthetic_data_confirmation()
    st.success(confirm["message"])
    st.write(
        {
            "inference_destination": cfg.destination_label,
            "model_name": cfg.model_name,
            "mock_or_real": "mock" if cfg.use_mock else "real",
            "synthetic_only": confirm["synthetic_only"],
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
    st.sidebar.caption("Synthetic demo · advisory recommendations only")
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Customer Churn Risks", "Campaign Anomalies", "Privacy and Safety"],
    )
    show_eval = st.sidebar.toggle("Evaluation toggle (show ground truth)", value=False)
    st.sidebar.caption(f"Project root: {ROOT}")

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
    else:
        privacy_page()


if __name__ == "__main__":
    main()
